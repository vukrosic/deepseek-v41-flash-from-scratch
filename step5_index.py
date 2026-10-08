"""5. The index. Every note also gets a tiny index key. A new token scores all index keys, cheaply,
and attention then reads only the best few notes. The index learns by copying where full attention looked."""
from model import Flash
from task import LENGTH, RECORDS, VOCAB
from train import accuracy, save, trained

model, rec = trained('index', lambda: Flash(VOCAB, LENGTH, sources={0: 2, 2: 1}, topk=4, fp4=True))
print(f"trained reading every note: right answers {rec['accuracy']:.1%}; "
      f"memory with index keys {model.bytes_per_token():.1f} bytes per token")
out = {'dense': rec['accuracy'], 'bytes': model.bytes_per_token(), 'curve': rec['curve'], 'notes': 2 * RECORDS,
       'top': {}}
model.cfg['sparse'] = 'random'
out['random'] = accuracy(model)
print(f"read 4 notes picked at random:  {out['random']:.1%}")
for k in [1, 2, 4, 8]:
    model.cfg.update(sparse=True, topk=k)
    out['top'][str(k)] = accuracy(model)
    print(f"read the {k} notes the index picks: {out['top'][str(k)]:.1%}")
save('step5', out)
