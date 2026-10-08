"""The practice task: a notebook and questions about it.
One line is 80 tokens: 16 records (a key, then its value), then 16 questions (a question mark, a key, the value)."""
import torch

KEYS, VALUES, RECORDS, QUESTIONS = 32, 32, 16, 16
ASK = KEYS + VALUES                          # tokens 0-31 are keys, 32-63 are values, 64 is "?"
VOCAB, LENGTH = KEYS + VALUES + 1, 2 * RECORDS + 3 * QUESTIONS
ANSWER_AT = [2 * RECORDS + 3 * i + 1 for i in range(QUESTIONS)]   # at these positions the next token is an answer


def lines(n, gen):
    """n random lines as one table of tokens. Every key appears once per line."""
    keys = torch.rand(n, KEYS, generator=gen).argsort(1)[:, :RECORDS]
    values = torch.randint(VALUES, (n, RECORDS), generator=gen) + KEYS
    asked = torch.rand(n, RECORDS, generator=gen).argsort(1)[:, :QUESTIONS]
    book = torch.stack([keys, values], 2).flatten(1)
    quiz = torch.stack([torch.full((n, QUESTIONS), ASK), keys.gather(1, asked), values.gather(1, asked)], 2).flatten(1)
    return torch.cat([book, quiz], 1)


def show(line):
    word = lambda t: '?' if t == ASK else (f'k{t}' if t < KEYS else f'v{t - KEYS}')
    return ' '.join(word(t) for t in line)


if __name__ == '__main__':
    line = lines(1, torch.Generator().manual_seed(0))[0].tolist()
    print(show(line[:8]), '...', show(line[2 * RECORDS - 2:2 * RECORDS + 6]), '...')
    print('tokens per line', LENGTH, '· answers at', ANSWER_AT)
