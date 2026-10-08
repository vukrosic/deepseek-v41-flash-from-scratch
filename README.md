# DeepSeek V4.1 Flash from scratch: the memory

Seven short PyTorch programs (about 550 lines with the shared files and the checks) that build, at toy size, the
way DeepSeek-V4.1-Flash keeps its memory of a long text small. They produce every toy number in the video
"DeepSeek V4.1 Flash From Scratch".

This is not DeepSeek's model and it reproduces none of DeepSeek's results. The real model has 40 layers and a
context of one million tokens. The model here has 4 layers and 213,569 weights, reads lines of 80 tokens and
trains on a laptop CPU.

## The question

A language model keeps notes about every token it has read (the KV cache), so that it does not read the text
again for each new word. Counted from DeepSeek's `config.json`, an ordinary model of this shape would keep
819,200 bytes of notes per token. The same count for V4.1 Flash gives about 890. The seven steps rebuild the five
changes that make up the difference, one at a time, and measure what each one costs in right answers.

## Run it

```bash
pip install torch
bash run_all.sh
```

Python 3.9+. `demo-output.txt` is one saved run. Trained models are saved in `runs/` and reused on the next call.

## The practice task

One line is 80 tokens: 16 records (a key, then its value), then 16 questions (a question mark, a key, and the
value the model must give).

```
k17 v12 k19 v31 k29 v5 ... ? k19 v31 ? k29 v5 ...
```

To answer, the model has to find a record far back in the line. That is the job the notes exist for, so a model
with broken notes fails in a way you can count: right answers out of all answers.

## The seven steps

| file | idea | result of the saved run |
| --- | --- | --- |
| `step1_notes.py` | keep notes (a key and a value per token, per layer) and run only the new token | 12,640 runs of a layer on a token without notes, 316 with; scores equal to 7.6e-06; 2,048 bytes per token |
| `step2_one_note.py` | one vector serves as key and value; then only 2 of the 4 layers write, and the rest read those notes | 1,024 then 512 bytes per token; 100.0 % both times |
| `step3_squeeze.py` | one note stands for 2, 4, 8 or 16 tokens | 2 and 4 tokens per note: 100.0 %. 8: 36.7 %. 16: 23.6 %. Bottom writer in pairs, top writer whole: 384 bytes, 100.0 % |
| `step4_four_bits.py` | store every number in 4 bits (8 sizes and a sign) plus one scale per 16 numbers | a finished model rounded to 4 bits: 100.0 %; signs only: 72.2 %; trained with 4-bit notes: 54 bytes, 100.0 % |
| `step5_index.py` | a small index key per note, so a layer reads only the notes the index picks | 4 notes at random: 17.5 %. The 1 note the index picks: 99.9 %. 2, 4 or 8: 100.0 %. 67.5 bytes per token with the index keys |
| `step6_two_halves.py` | the prompt goes through the bottom half only; the last 8 tokens are replayed through the top half | 167 runs against 316; scores equal to 3.8e-06 over 200 prompts; 100.0 % both ways |
| `step7_bill.py` | the same five changes counted for the real model from its config | 819,200 → 40,960 → 4,096 → 2,560 → 720 → 890 bytes per token |

Shared files: `task.py` (the practice lines), `model.py` (the model and its note writers), `train.py` (training
and measuring), `tests.py` (checks, run first by `run_all.sh`).

## Two things the runs showed that I did not expect

**One training run is not enough.** In step 2 the model with two writers ended at 26.8 % from one random start
and at 100.0 % from another. From there on, every model that ends under 90 % gets a second random start, and
both results are printed. The two claims that something breaks (8 and 16 tokens per note) rest on two starts
each: 35.7 % / 36.7 % and 23.4 % / 23.6 %.

**The sliding window needs its own key and value.** Every layer also keeps exact notes for the last 8 tokens.
With one vector as key and value in that window, every model without squeezing stalled near 27 %. The window
here keeps a separate key and value. It holds 8 tokens however long the text is, so it does not change the bytes
per token of a long text.

## What this does not show

- That any of this carries over to the real model. Each change works on a small lookup task; how well it works
  at full size is DeepSeek's result, not mine.
- That 4 tokens per note is safe. It was here. DeepSeek's config uses 2 in the bottom half and 1 in the top
  half.
- Speed, price or answer quality of the real model. I did not run it.
- The 890 bytes per token is my arithmetic from `config.json`, not a number DeepSeek publishes. DeepSeek's
  release note gives a different comparison: a quarter of the fast memory and an eighth of the disk of their
  previous model.
- How DeepSeek keeps the replay in step 6 short across 20 layers. An exact replay needs more tokens for every
  layer below the top one. The model card names the method ("SWA Bounded Replay"); the details are in the
  technical report, which I could not read.

Left out entirely: the mixture of experts, Engram, the draft layers, the vision part and all of training.

## Sources for the real model

- Model card and `config.json` of `deepseek-ai/DeepSeek-V4.1-Flash` on Hugging Face (MIT), read 2026-10-07.
- DeepSeek API release note of 2026-09-10 (`api-docs.deepseek.com/news/news260910`).

## An experiment to try

In `step3_squeeze.py` the notes break between 4 and 8 tokens per note. A record is 2 tokens. Make the records
longer or the model wider and find where the break moves.

MIT license.
