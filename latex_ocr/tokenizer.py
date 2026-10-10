"""LaTeX tokenizer and vocabulary."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

# \command | \<symbol> (e.g. "\{", "\,") | any single non-space character
_TOKEN_RE = re.compile(r"\\[a-zA-Z]+\*?|\\.|[^\s]")
_COMMAND_RE = re.compile(r"^\\[a-zA-Z]+\*?$")


def tokenize(latex: str) -> list[str]:
    return _TOKEN_RE.findall(latex)


def detokenize(tokens: list[str]) -> str:
    """Join tokens with as few spaces as needed for valid LaTeX."""
    out: list[str] = []
    for tok in tokens:
        if out and _COMMAND_RE.match(out[-1]) and tok[:1].isalpha():
            out.append(" ")
        out.append(tok)
    return "".join(out)


class Vocab:
    def __init__(self, tokens: list[str]) -> None:
        self.itos = list(SPECIALS) + [t for t in tokens if t not in SPECIALS]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, latex: str, max_len: int | None = None) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize(latex)]
        if max_len is not None:
            ids = ids[: max_len - 2]
        return [self.bos_id] + ids + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        toks: list[str] = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            toks.append(self.itos[i] if 0 <= i < len(self.itos) else UNK)
        return detokenize(toks)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["itos"])

    @classmethod
    def from_list(cls, itos: list[str]) -> "Vocab":
        return cls(itos)

    @classmethod
    def build(cls, formulas, min_freq: int = 1) -> "Vocab":
        counter: Counter[str] = Counter()
        for f in formulas:
            counter.update(tokenize(f))
        tokens = sorted(t for t, c in counter.items() if c >= min_freq)
        return cls(tokens)


def read_tsv(path: str | Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            img, formula = line.split("\t", 1)
            rows.append((img, formula))
    return rows


def _main() -> None:
    p = argparse.ArgumentParser(description="LaTeX vocabulary tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build vocab.json from a dataset TSV")
    b.add_argument("tsv", nargs="+")
    b.add_argument("-o", "--out", required=True)
    b.add_argument("--min-freq", type=int, default=1)
    args = p.parse_args()

    if args.cmd == "build":
        formulas = [f for path in args.tsv for _, f in read_tsv(path)]
        vocab = Vocab.build(formulas, args.min_freq)
        vocab.save(args.out)
        print(f"{len(vocab)} tokens from {len(formulas)} formulas -> {args.out}")


if __name__ == "__main__":
    _main()
