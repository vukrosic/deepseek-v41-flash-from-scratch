"""Two tiny language models.
Classic: every layer keeps its own keys and values for every token (the ordinary KV cache).
Flash:   the memory is built the way DeepSeek-V4.1-Flash builds it, one switch per idea."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH, HEADS, INDEX, WINDOW = 64, 4, 16, 8
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])      # every size a 4-bit E2M1 float can hold (plus a sign)


def heads(t):
    return t.view(*t.shape[:2], HEADS, -1).transpose(1, 2)


def feed_forward():
    return nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.SiLU(), nn.Linear(4 * WIDTH, WIDTH))


# ---------------------------------------------------------------------------------------------- classic

class ClassicLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.q, self.k, self.v = nn.Linear(WIDTH, WIDTH), nn.Linear(WIDTH, WIDTH), nn.Linear(WIDTH, WIDTH)
        self.proj, self.ff = nn.Linear(WIDTH, WIDTH), feed_forward()

    def forward(self, x, cache=None):
        """x holds only the new tokens. cache holds this layer's keys and values for the earlier tokens."""
        h = self.n1(x)
        k, v = self.k(h), self.v(h)
        if cache is not None:
            if 'k' in cache:
                k, v = torch.cat([cache['k'], k], 1), torch.cat([cache['v'], v], 1)
            cache['k'], cache['v'] = k, v
        new, old = x.shape[1], k.shape[1] - x.shape[1]
        may = torch.arange(k.shape[1])[None, :] <= old + torch.arange(new)[:, None]     # no looking ahead
        a = F.scaled_dot_product_attention(heads(self.q(h)), heads(k), heads(v), attn_mask=may)
        x = x + self.proj(a.transpose(1, 2).reshape(x.shape))
        return x + self.ff(self.n2(x))


class Classic(nn.Module):
    def __init__(self, vocab, length, layers=4):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, WIDTH), nn.Embedding(length, WIDTH)
        self.layers = nn.ModuleList(ClassicLayer() for _ in range(layers))
        self.norm, self.out = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, vocab)

    def forward(self, seq, cache=None, start=0):
        """Without a cache: seq is the whole text. With a cache: seq is only the new tokens, which begin at `start`."""
        x = self.tok(seq) + self.pos(start + torch.arange(seq.shape[1]))
        for i, layer in enumerate(self.layers):
            x = layer(x, None if cache is None else cache.setdefault(i, {}))
        return self.out(self.norm(x))


# ------------------------------------------------------------------------------------------------ flash

def e4m3(s):
    """Round a scale to an 8-bit float: 3 bits after the leading one."""
    m, e = torch.frexp(s)
    return torch.ldexp(torch.round(m * 16) / 16, e)


def fp4(x, block=16):
    """Store x in 4 bits per number: every block of 16 numbers shares one scale, each number snaps to the FP4 list."""
    g = x.reshape(*x.shape[:-1], -1, block)
    scale = e4m3(g.abs().amax(-1, keepdim=True) / 6).clamp_min(1e-8)
    y = g / scale
    q = FP4[(y.abs().unsqueeze(-1) - FP4).abs().argmin(-1)] * y.sign()
    return (q * scale).reshape(x.shape)


def one_bit(x, block=16):
    """The crudest storage: keep only the sign of every number, and one size per block of 16."""
    g = x.reshape(*x.shape[:-1], -1, block)
    return (g.sign() * g.abs().mean(-1, keepdim=True)).reshape(x.shape)


class Memory(nn.Module):
    """Turns a layer's input into the shared memory: one vector per `ratio` tokens, used as key and as value."""

    def __init__(self, ratio, index):
        super().__init__()
        self.ratio = ratio
        self.kv, self.norm = nn.Linear(WIDTH, WIDTH), nn.LayerNorm(WIDTH)
        if ratio > 1:
            self.gate = nn.Linear(WIDTH, WIDTH)
        if index:
            self.index_key = nn.Linear(WIDTH, INDEX)

    def forward(self, h):
        B, L, _ = h.shape
        P = L // self.ratio
        kv = self.kv(h)
        if self.ratio > 1:                                      # a gated blend of each group of tokens
            group = lambda t: t[:, :P * self.ratio].reshape(B, P, self.ratio, WIDTH)
            kv = (group(kv) * group(self.gate(h)).softmax(2)).sum(2)
        return self.norm(kv)


class FlashLayer(nn.Module):
    def __init__(self, memory=None, index=False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.q = nn.Linear(WIDTH, WIDTH)
        self.near_k, self.near_v = nn.Linear(WIDTH, WIDTH), nn.Linear(WIDTH, WIDTH)
        self.proj, self.ff = nn.Linear(WIDTH, WIDTH), feed_forward()
        self.memory = memory                                    # only a source layer owns one
        if memory is not None and index:
            self.index_query = nn.Linear(WIDTH, INDEX)

    def write(self, x, shared, cfg):
        """A source layer turns its input for every token into the shared memory. A projection, not a layer run."""
        mem = self.memory(self.n1(x))
        if cfg['fp4']:                                          # True: round, and let training see the rounding
            q = one_bit(mem) if cfg['fp4'] == 'sign' else fp4(mem)
            mem = mem + (q - mem).detach() if cfg['fp4'] is True else q      # 'after', 'sign': round a finished model
        shared.update(mem=mem, ratio=self.memory.ratio, pick=None)
        if cfg['topk']:
            key = self.memory.index_key(mem)
            shared['key'] = key + (fp4(key) - key).detach() if cfg['fp4'] else key

    def forward(self, x, shared, cfg, first=0, last=None):
        """x holds the tokens from position `first` on. Only the final `last` of them get an output."""
        B, n, _ = x.shape
        last = last or n
        h = self.n1(x)
        mem, P = shared['mem'], shared['mem'].shape[1]
        t = first + torch.arange(n - last, n)[:, None]          # the positions that get an output
        seen = ((torch.arange(P)[None, :] + 1) * shared['ratio'] - 1 <= t)[None].expand(B, -1, -1)   # finished groups
        if self.memory is not None and cfg['topk']:             # the index: a small score for every memory slot
            score = (self.index_query(h[:, -last:]) @ shared['key'].transpose(1, 2)).masked_fill(~seen, -1e9)
            shared['score'] = score
            if cfg['sparse']:
                if cfg['sparse'] == 'random':
                    score = torch.rand(score.shape).masked_fill(~seen, -1e9)
                best = score.topk(min(cfg['topk'], P), -1).indices
                shared['pick'] = torch.zeros(B, last, P, dtype=torch.bool).scatter_(2, best, True)
        if shared['pick'] is not None:                          # read only the slots the index chose
            seen = seen & shared['pick'][:, -last:]
        pos = first + torch.arange(n)[None, :]
        near_ok = ((pos <= t) & (pos > t - WINDOW))[None].expand(B, -1, -1)
        keys = heads(torch.cat([mem, self.near_k(h)], 1))       # a memory note is its own key and its own value;
        values = heads(torch.cat([mem, self.near_v(h)], 1))     # the last WINDOW tokens keep a key and a value each
        ok = torch.cat([seen, near_ok], 2)[:, None]
        att = heads(self.q(h[:, -last:])) @ keys.transpose(-1, -2) / math.sqrt(WIDTH // HEADS)
        att = att.masked_fill(~ok, -1e9).softmax(-1)
        if self.memory is not None and cfg['topk']:             # what the index should learn to predict
            far = att[..., :P].mean(1).detach()
            shared['target'] = far / far.sum(-1, keepdim=True).clamp_min(1e-9)
        x = x[:, -last:] + self.proj((att @ values).transpose(1, 2).reshape(B, last, WIDTH))
        return x + self.ff(self.n2(x))


class Flash(nn.Module):
    """sources: {layer: ratio}. A source layer builds the memory; the layers after it reuse it until the next source."""

    def __init__(self, vocab, length, layers=4, sources={0: 1, 2: 1}, topk=0, fp4=False):
        super().__init__()
        self.cfg = {'topk': topk, 'sparse': False, 'fp4': fp4}
        self.sources = dict(sources)
        self.tok, self.pos = nn.Embedding(vocab, WIDTH), nn.Embedding(length, WIDTH)
        self.layers = nn.ModuleList(
            FlashLayer(Memory(sources[i], bool(topk)) if i in sources else None, bool(topk)) for i in range(layers))
        self.norm, self.out = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, vocab)

    def forward(self, seq):
        x = self.tok(seq) + self.pos(torch.arange(seq.shape[1]))
        shared, self.index_loss = {}, 0.
        for i, layer in enumerate(self.layers):
            if i in self.sources:
                layer.write(x, shared, self.cfg)
            x = layer(x, shared, self.cfg)
            if i in self.sources and self.cfg['topk']:          # teach the index to point where attention looks
                logp = shared['score'].log_softmax(-1)
                self.index_loss = self.index_loss - (shared['target'] * logp).sum(-1).mean()
        return self.out(self.norm(x))

    def next_token(self, seq, half):
        """Scores for the token after seq. Layers below `half` read every token; the layers above run on as few
        tokens as they can. Returns the scores and the number of (token, layer) runs."""
        L, top = seq.shape[1], len(self.layers)
        assert all(i <= half for i in self.sources), 'the top layers must share one memory'
        x = self.tok(seq) + self.pos(torch.arange(L))
        shared, runs = {}, 0
        for i, layer in enumerate(self.layers):
            if i in self.sources:
                layer.write(x, shared, self.cfg)                # for i == half: built from the reader's last output
            rows = L if i < half else min(L, 1 + (WINDOW - 1) * (top - 1 - i))
            x = layer(x, shared, self.cfg, L - x.shape[1], rows)
            runs += rows
        return self.out(self.norm(x))[:, -1], runs

    def bytes_per_token(self):
        """Memory that grows with the text, in bytes per token."""
        number = 0.5 if self.cfg['fp4'] else 4                  # 4 bits, or a 32-bit float
        scale = 1 if self.cfg['fp4'] else 0                     # one 8-bit scale per block of 16 numbers
        total = 0
        for i, ratio in self.sources.items():
            total += (WIDTH * number + WIDTH // 16 * scale) / ratio
            if self.cfg['topk']:
                total += (INDEX * number + INDEX // 16 * scale) / ratio
        return total
