"""6. Two halves. The top layers read one memory, built from the bottom half's last output.
So a prompt token only has to go through the bottom half. Check: are the scores the same?"""
import torch

from model import WINDOW, Flash
from task import LENGTH, VOCAB, lines
from train import save, trained

model, _ = trained('index', lambda: Flash(VOCAB, LENGTH, sources={0: 2, 2: 1}, topk=4, fp4=True))
model.cfg.update(sparse=True, topk=4)
seq = lines(200, torch.Generator().manual_seed(3))
prompt, answer = seq[:, :-1], seq[:, -1]                    # everything up to the last question's key
with torch.no_grad():
    full = model(prompt)[:, -1]
    split, runs = model.next_token(prompt, half=2)
tokens, layers = prompt.shape[1], len(model.layers)
gap = float((full - split).abs().max())
print(f'every token through every layer: {tokens} x {layers} = {tokens * layers} runs')
print(f'prompt through the bottom half, the last {WINDOW} tokens replayed above: {runs} runs')
print(f'largest gap between the scores, over 200 prompts: {gap:.1e}')
print(f'right answers: {float((full.argmax(-1) == answer).float().mean()):.1%} and '
      f'{float((split.argmax(-1) == answer).float().mean()):.1%}')
save('step6', {'tokens': tokens, 'full_runs': tokens * layers, 'split_runs': runs, 'gap': gap, 'window': WINDOW,
               'full_acc': float((full.argmax(-1) == answer).float().mean()),
               'split_acc': float((split.argmax(-1) == answer).float().mean())})
