"""Shared by every step: train a model on the notebook task and measure it."""
import json
import os
import time

import torch
import torch.nn.functional as F

from task import ANSWER_AT, VALUES, lines

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS, BATCH = 3000, 64
NEXT = [p + 1 for p in ANSWER_AT]


def loss_on(model, seq):
    """Only the answers are graded: at each question's key, the next token must be the right value."""
    return F.cross_entropy(model(seq[:, :-1])[:, ANSWER_AT].flatten(0, 1), seq[:, NEXT].flatten())


@torch.no_grad()
def accuracy(model, n=1000, seed=1):
    """Share of answers that are right on fresh notebooks. Guessing gives 1 in 32."""
    seq = lines(n, torch.Generator().manual_seed(seed))
    return float((model(seq[:, :-1])[:, ANSWER_AT].argmax(-1) == seq[:, NEXT]).float().mean())


def fit(model, steps=STEPS, seed=0, lr=2e-3, every=250):
    """Train. Returns the accuracy every `every` steps and the seconds it took."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr)
    warm = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200))
    curve, t = [], time.time()
    for step in range(steps):
        loss = loss_on(model, lines(BATCH, gen)) + getattr(model, 'index_loss', 0.)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        warm.step()
        if (step + 1) % every == 0:
            curve.append([step + 1, round(accuracy(model, 300, seed=2), 4)])
            print(f'    step {step + 1:5d}  accuracy {curve[-1][1]:.1%}', flush=True)
    return curve, time.time() - t


def trained(name, make, steps=STEPS):
    """Train make() once and keep it in runs/. A second call loads it. Returns the model and its record.
    Training a tiny model is partly luck, so a run that ends under 90% gets one more random start.
    record['starts'] holds the accuracy of every start; the better model is the one kept."""
    path = os.path.join(HERE, 'runs', name)
    rec = load(name) if os.path.exists(path + '.json') else {'starts': []}
    while not rec['starts'] or (max(rec['starts']) < 0.9 and len(rec['starts']) < 2):
        seed = len(rec['starts'])
        torch.manual_seed(seed)
        model = make()
        print(f'  training {name} for {steps} steps (random start {seed + 1})', flush=True)
        curve, seconds = fit(model, steps, seed)
        acc = accuracy(model.eval())
        if not rec['starts'] or acc > max(rec['starts']):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path + '.pt')
            rec.update(accuracy=acc, curve=curve, seconds=round(seconds, 1), start=seed,
                       weights=sum(p.numel() for p in model.parameters()))
        rec['starts'].append(acc)
        save(name, rec)
    torch.manual_seed(0)
    model = make()
    model.load_state_dict(torch.load(path + '.pt'))
    return model.eval(), rec


def save(name, data):
    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    with open(os.path.join(HERE, 'runs', name + '.json'), 'w') as f:
        json.dump(data, f, indent=1)


def load(name):
    with open(os.path.join(HERE, 'runs', name + '.json')) as f:
        return json.load(f)
