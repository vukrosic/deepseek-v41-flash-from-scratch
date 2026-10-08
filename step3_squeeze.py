"""3. Squeeze. One note for several tokens: a learned blend of each group.
Push the group size up until the answers break. Then DeepSeek's layout: bottom half in pairs, top half whole."""
from model import Flash
from task import LENGTH, VOCAB
from train import save, trained

starts = lambda rec: ' (two random starts: ' + ', '.join(f'{a:.1%}' for a in rec['starts']) + ')' if len(rec['starts']) > 1 else ''

out = {}
for ratio in [2, 4, 8, 16]:
    model, rec = trained(f'pool_{ratio}', lambda: Flash(VOCAB, LENGTH, sources={0: ratio, 2: ratio}))
    out[str(ratio)] = {'bytes': model.bytes_per_token(), 'accuracy': rec['accuracy'], 'curve': rec['curve'], 'starts': rec['starts']}
    print(f"one note per {ratio:2d} tokens: {model.bytes_per_token():6.1f} bytes per token, "
          f"right answers {rec['accuracy']:.1%}" + starts(rec))
model, rec = trained('pool_2_1', lambda: Flash(VOCAB, LENGTH, sources={0: 2, 2: 1}))
out['2_1'] = {'bytes': model.bytes_per_token(), 'accuracy': rec['accuracy'], 'curve': rec['curve'], 'starts': rec['starts']}
print(f"bottom half in pairs, top half whole: {model.bytes_per_token():.0f} bytes per token, "
      f"right answers {rec['accuracy']:.1%}" + starts(rec))
save('step3', out)
