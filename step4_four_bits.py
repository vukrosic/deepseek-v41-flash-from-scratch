"""4. Four bits. Every number in a note is stored as a 4-bit float; 16 numbers share one 8-bit scale.
First by hand on 16 real numbers. Then round a finished model. Then train a model that sees the rounding."""
import torch

from model import FP4, Flash, e4m3, fp4
from task import LENGTH, VOCAB, lines
from train import accuracy, save, trained

model, rec = trained('pool_2_1', lambda: Flash(VOCAB, LENGTH, sources={0: 2, 2: 1}))
line = lines(1, torch.Generator().manual_seed(7))
with torch.no_grad():
    x = model.tok(line) + model.pos(torch.arange(LENGTH))
    shared = {}
    model.layers[0].write(x, shared, model.cfg)
note = shared['mem'][0, 0, :16]                             # the first 16 numbers of the first note
scale = e4m3(note.abs().max() / 6)
stored = fp4(note)
print('the 4-bit sizes:', FP4.tolist(), 'and a sign')
print('16 real numbers:', [round(v, 3) for v in note.tolist()])
print(f'largest {note.abs().max():.3f} / 6 = {note.abs().max() / 6:.4f}, kept as the scale {scale:.4f}')
print('in steps of the scale:', [round(v, 2) for v in (note / scale).tolist()])
print('snapped:', [round(v, 1) for v in (stored / scale).tolist()])
print('read back:', [round(v, 3) for v in stored.tolist()])
print(f'mean error {(stored - note).abs().mean():.3f} on a mean size of {note.abs().mean():.3f}')

exact = accuracy(model)
model.cfg['fp4'] = 'after'
after = accuracy(model)
model.cfg['fp4'] = 'sign'
sign = accuracy(model)
model.cfg['fp4'] = False
print(f'finished model, exact notes: {exact:.1%} · rounded to 4 bits: {after:.1%} · signs only (1 bit): {sign:.1%}')

rounded, rec4 = trained('fp4', lambda: Flash(VOCAB, LENGTH, sources={0: 2, 2: 1}, fp4=True))
print(f"trained with 4-bit notes: {rounded.bytes_per_token():.0f} bytes per token, right answers {rec4['accuracy']:.1%}")
save('step4', {'note': note.tolist(), 'scale': float(scale), 'steps': (note / scale).tolist(),
               'snapped': (stored / scale).tolist(), 'stored': stored.tolist(),
               'error': float((stored - note).abs().mean()), 'size': float(note.abs().mean()),
               'exact': exact, 'after': after, 'sign': sign, 'trained': rec4['accuracy'], 'curve': rec4['curve'],
               'bytes_before': model.bytes_per_token(), 'bytes': rounded.bytes_per_token()})
