import random
import re
from collections import Counter

import numpy as np
import torch

PAD, UNK, SEP = 0, 1, 2
MIN_WIDTH = 5
TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text):
    return ["<nl>" if t == "\n" else t for t in TOKEN_RE.findall(text)]


def mode_columns(df, mode):
    if mode == "output":
        return [df["output"]]
    if mode == "input":
        return [df["input"]]
    return [df["input"], df["output"]]


def build_vocab(df, mode, max_size=30000, min_freq=2):
    counts = Counter()
    for column in mode_columns(df, mode):
        for text in column:
            counts.update(tokenize(text))
    words = [w for w, c in counts.most_common(max_size - 3) if c >= min_freq]
    return {w: i + 3 for i, w in enumerate(words)}


def encode_ids(text, vocab, limit):
    return [vocab.get(t, UNK) for t in tokenize(text)[:limit]]


def encode_frame(df, vocab, mode, max_len, max_in_len=128):
    seqs = []
    for inp, out in zip(df["input"], df["output"]):
        if mode == "output":
            ids = encode_ids(out, vocab, max_len)
        elif mode == "input":
            ids = encode_ids(inp, vocab, max_len)
        else:
            ids = encode_ids(inp, vocab, max_in_len) + [SEP] + encode_ids(out, vocab, max_len)
        seqs.append(ids or [UNK])
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    width = max(int(lengths.max()), MIN_WIDTH)
    x = np.zeros((len(seqs), width), dtype=np.int64)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = s
    return torch.from_numpy(x), torch.from_numpy(lengths)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for x, lengths, _ in loader:
        logits = model(x.to(device), lengths.to(device))
        preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()
