"""7. The bill. Count the real model's memory from the numbers in its config.json.
This is arithmetic on DeepSeek's published config, not a measurement of DeepSeek's model."""
from train import save

LAYERS, HIDDEN, HEAD_DIM, INDEX_DIM = 40, 5120, 512, 128
KV_SOURCES = [2, 8, 14, 20]                 # kv_source_layer_ids
RATIO = {2: 2, 8: 2, 14: 2, 20: 1}          # compress_ratios at those layers

ordinary = LAYERS * 2 * HIDDEN * 2          # a key and a value per layer, 16-bit numbers
one_vector = LAYERS * HEAD_DIM * 2          # one 512-number vector per layer is the key and the value
shared = len(KV_SOURCES) * HEAD_DIM * 2     # only 4 layers write
pooled = sum(HEAD_DIM * 2 / RATIO[i] for i in KV_SOURCES)
note = HEAD_DIM * 4 // 8 + HEAD_DIM // 16   # 4 bits per number, one scale byte per 16 numbers
four_bit = sum(note / RATIO[i] for i in KV_SOURCES)
key = INDEX_DIM * 4 // 8 + INDEX_DIM // 32  # index keys: 4 bits, one scale byte per 32 numbers
total = sum((note + key) / RATIO[i] for i in KV_SOURCES)

rows = [('ordinary notes', ordinary), ('one vector for key and value', one_vector), ('4 layers write', shared),
        ('bottom half in pairs', pooled), ('4-bit numbers', four_bit), ('plus index keys', total)]
for name, size in rows:
    print(f'{name:30s} {size:>9,.0f} bytes per token   {size * 1e6 / 1e9:>8,.2f} GB per million tokens')
print(f'one note {note} bytes, one index key {key} bytes; {ordinary / total:.0f} times smaller in total')
save('step7', {'rows': rows, 'note': note, 'key': key, 'times': ordinary / total})
