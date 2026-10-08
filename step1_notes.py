"""1. The notes. A model that keeps a key and a value for every old token never has to reread the text.
Train the ordinary model, then answer one line twice: rereading everything, and with notes."""
import time

import torch

from model import Classic
from task import LENGTH, RECORDS, VOCAB, lines, show
from train import save, trained

model, rec = trained('classic', lambda: Classic(VOCAB, LENGTH))
print(f"ordinary model: {rec['weights']:,} weights, right answers {rec['accuracy']:.1%}")

line = lines(1, torch.Generator().manual_seed(7))
print('one line:', show(line[0, :6].tolist()), '...', show(line[0, 2 * RECORDS:2 * RECORDS + 6].tolist()), '...')
layers, fed = len(model.layers), LENGTH - 1
with torch.no_grad():
    t, slow, slow_runs = time.time(), [], 0
    for n in range(1, LENGTH):                              # no notes: every new token rereads all the text
        slow.append(model(line[:, :n])[0, -1])
        slow_runs += n * layers
    slow_s = time.time() - t
    t, fast, fast_runs, cache = time.time(), [], 0, {}
    for n in range(1, LENGTH):                              # notes: only the new token goes through the layers
        fast.append(model(line[:, n - 1:n], cache, n - 1)[0, -1])
        fast_runs += layers
    fast_s = time.time() - t
gap = max(float((a - b).abs().max()) for a, b in zip(slow, fast))
numbers = sum(c['k'].numel() + c['v'].numel() for c in cache.values())
size = numbers * 4 / fed                                    # a 32-bit float is 4 bytes

print(f'no notes: {slow_runs:,} runs of a layer on a token, {slow_s * 1000:.0f} ms')
print(f'notes:    {fast_runs:,} runs, {fast_s * 1000:.0f} ms  ({slow_runs / fast_runs:.0f} times fewer runs)')
print(f'largest gap between the two sets of scores: {gap:.1e}')
print(f'the notes hold {numbers:,} numbers for {fed} tokens: {size:,.0f} bytes per token')
save('step1', {'accuracy': rec['accuracy'], 'weights': rec['weights'], 'curve': rec['curve'], 'slow_runs': slow_runs,
               'fast_runs': fast_runs, 'slow_ms': slow_s * 1000, 'fast_ms': fast_s * 1000, 'gap': gap,
               'numbers': numbers, 'bytes': size, 'line': line[0].tolist()})
