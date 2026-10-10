"""TSV dataset: one `relative/image/path<TAB>latex` per line."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from .preprocess import preprocess
from .tokenizer import Vocab, read_tsv


class FormulaDataset(Dataset):
    def __init__(
        self,
        tsv: str | Path,
        vocab: Vocab,
        img_height: int = 64,
        max_width: int = 512,
        max_len: int = 200,
        root: str | Path | None = None,
    ) -> None:
        self.tsv = Path(tsv)
        self.root = Path(root) if root else self.tsv.parent
        self.vocab = vocab
        self.img_height = img_height
        self.max_width = max_width
        self.max_len = max_len
        self.rows = [(img, f) for img, f in read_tsv(self.tsv) if (self.root / img).exists()]
        if not self.rows:
            raise ValueError(f"no usable samples in {self.tsv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        img_path, formula = self.rows[i]
        image = preprocess(self.root / img_path, self.img_height, self.max_width)
        ids = torch.tensor(self.vocab.encode(formula, self.max_len), dtype=torch.long)
        return image, ids, formula

    def approx_lengths(self) -> list[int]:
        return [len(f) for _, f in self.rows]


def collate(batch, pad_id: int = 0):
    images, ids, formulas = zip(*batch)
    h = images[0].shape[1]
    w = max(im.shape[2] for im in images)
    # background is 0 after preprocessing, so pad with 0
    imgs = torch.stack([F.pad(im, (0, w - im.shape[2], 0, h - im.shape[1]), value=0.0) for im in images])
    L = max(len(t) for t in ids)
    tgt = torch.full((len(ids), L), pad_id, dtype=torch.long)
    for i, t in enumerate(ids):
        tgt[i, : len(t)] = t
    return imgs, tgt, list(formulas)


class BucketSampler(Sampler):
    """Groups samples of similar formula length to reduce padding."""

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True, seed: int = 0) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [idx[i : i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
