"""Fast checks with untrained models (a few seconds): python3 tests.py"""
import torch

from model import FP4, Classic, Flash, e4m3, fp4
from task import ANSWER_AT, ASK, LENGTH, RECORDS, VOCAB, lines

seq = lines(4, torch.Generator().manual_seed(0))

# the task: every question asks for a key that is in the notebook, and the token after it is that key's value
for row in seq.tolist():
    book = dict(zip(row[0:2 * RECORDS:2], row[1:2 * RECORDS:2]))
    assert len(book) == RECORDS
    for p in ANSWER_AT:
        assert row[p - 1] == ASK and book[row[p]] == row[p + 1]

with torch.no_grad():
    # the ordinary model: one token at a time with notes gives the same scores as the whole text at once
    torch.manual_seed(0)
    classic = Classic(VOCAB, LENGTH).eval()
    cache = {}
    parts = torch.cat([classic(seq[:, n:n + 1], cache, n) for n in range(LENGTH)], 1)
    assert (classic(seq) - parts).abs().max() < 1e-4

    # 4-bit storage: every stored number is one of the 8 sizes times its block's scale
    x = torch.randn(3, 64)
    scale = e4m3(x.reshape(3, 4, 16).abs().amax(-1, keepdim=True) / 6)
    sizes = (fp4(x).reshape(3, 4, 16) / scale).abs()
    assert (sizes.unsqueeze(-1) - FP4).abs().amin(-1).max() < 1e-5
    assert (fp4(x) - x).abs().mean() < 0.2 * x.abs().mean()

    for kw in [dict(sources={0: 1, 1: 1, 2: 1, 3: 1}), dict(sources={0: 2, 2: 1}),
               dict(sources={0: 2, 2: 1}, topk=4, fp4=True)]:
        torch.manual_seed(0)
        flash = Flash(VOCAB, LENGTH, **kw).eval()
        for sparse in [False, True] if kw.get('topk') else [False]:
            flash.cfg['sparse'] = sparse
            # no looking ahead: changing the last token must not change any earlier score
            other = seq.clone()
            other[:, -1] = (other[:, -1] + 1) % VOCAB
            assert (flash(seq)[:, :-1] - flash(other)[:, :-1]).abs().max() < 1e-5
            # two halves: skipping the top layers for old tokens gives the same next-token scores
            if 2 in kw['sources'] and 3 not in kw['sources']:
                scores, runs = flash.next_token(seq, half=2)
                assert (flash(seq)[:, -1] - scores).abs().max() < 1e-4 and runs < LENGTH * 4

    assert Flash(VOCAB, LENGTH, sources={0: 1, 1: 1, 2: 1, 3: 1}).bytes_per_token() == 1024
    assert Flash(VOCAB, LENGTH, sources={0: 2, 2: 1}).bytes_per_token() == 384
    assert Flash(VOCAB, LENGTH, sources={0: 2, 2: 1}, fp4=True).bytes_per_token() == 54
    assert Flash(VOCAB, LENGTH, sources={0: 2, 2: 1}, topk=4, fp4=True).bytes_per_token() == 67.5
print('all checks pass')
