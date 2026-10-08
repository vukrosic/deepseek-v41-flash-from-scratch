"""2. One note. DeepSeek's config says: one key-value head, and only a few layers write memory.
First each layer keeps one vector that is its key and its value. Then two layers write and the others reuse."""
from model import Flash
from task import LENGTH, VOCAB
from train import save, trained

starts = lambda rec: ' (two random starts: ' + ', '.join(f'{a:.1%}' for a in rec['starts']) + ')' if len(rec['starts']) > 1 else ''

out = {}
for name, sources in [('own', {0: 1, 1: 1, 2: 1, 3: 1}), ('shared', {0: 1, 2: 1})]:
    model, rec = trained('flash_' + name, lambda: Flash(VOCAB, LENGTH, sources=sources))
    out[name] = {'bytes': model.bytes_per_token(), 'accuracy': rec['accuracy'], 'curve': rec['curve'], 'starts': rec['starts']}
    print(f"{name:7s} {len(sources)} layers write: {model.bytes_per_token():6.0f} bytes per token, "
          f"right answers {rec['accuracy']:.1%}" + starts(rec))
save('step2', out)
