#!/usr/bin/env python3
"""Render a synthetic image-to-LaTeX dataset with matplotlib mathtext.

    python scripts/render_dataset.py --formulas data/formulas.txt --out data/synth --variants 3
    python scripts/render_dataset.py --formulas data/formulas.txt --random 20000 --out data/synth

Output:
    <out>/images/000000.png ...
    <out>/train.tsv, <out>/val.tsv   (relative image path <TAB> latex)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import mathtext  # noqa: E402

FONTSETS = ["cm", "stix", "stixsans", "dejavuserif", "dejavusans"]

# ---------------------------------------------------------------- random formulas

VARS = list("abcdxyznkmtuvw") + [r"\alpha", r"\beta", r"\theta", r"\lambda", r"\mu", r"\sigma", r"\phi", r"\omega"]
FUNCS = [r"\sin", r"\cos", r"\log", r"\exp", r"\tan", r"\ln"]
OPS = ["+", "-", r"\cdot", r"\times"]
RELS = ["=", r"\leq", r"\geq", r"\approx", r"\neq", "<", ">"]


def _atom(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.55:
        return rng.choice(VARS)
    if r < 0.85:
        return str(rng.randint(0, 99))
    return rng.choice([r"\infty", r"\pi", "e", "i"])


def _expr(rng: random.Random, depth: int = 0) -> str:
    if depth > 2:
        return _atom(rng)
    r = rng.random()
    if r < 0.25:
        return _atom(rng)
    if r < 0.40:
        return f"{_atom(rng)}^{{{_expr(rng, depth + 1)}}}"
    if r < 0.50:
        return f"{rng.choice(VARS)}_{{{rng.choice(['i', 'j', 'k', 'n', '0', '1', 't'])}}}"
    if r < 0.62:
        return rf"\frac{{{_expr(rng, depth + 1)}}}{{{_expr(rng, depth + 1)}}}"
    if r < 0.70:
        return rf"\sqrt{{{_expr(rng, depth + 1)}}}"
    if r < 0.78:
        return rf"{rng.choice(FUNCS)}\left({_expr(rng, depth + 1)}\right)"
    if r < 0.85:
        big = rng.choice([r"\sum", r"\prod", r"\int"])
        lo, hi = ("0", "1") if big == r"\int" else ("i=1", "n")
        return rf"{big}_{{{lo}}}^{{{hi}}} {_expr(rng, depth + 1)}"
    return f"{_expr(rng, depth + 1)} {rng.choice(OPS)} {_expr(rng, depth + 1)}"


def random_formula(rng: random.Random) -> str:
    lhs = _expr(rng)
    if rng.random() < 0.6:
        return f"{lhs} {rng.choice(RELS)} {_expr(rng)}"
    return lhs


# ---------------------------------------------------------------- rendering


def can_parse(latex: str) -> bool:
    try:
        mathtext.MathTextParser("agg").parse(f"${latex}$", dpi=72)
        return True
    except Exception:
        return False


def render(latex: str, path: Path, fontset: str, fontsize: int, dpi: int) -> bool:
    with plt.rc_context({"mathtext.fontset": fontset}):
        fig = plt.figure(figsize=(0.01, 0.01))
        try:
            fig.text(0, 0, f"${latex}$", fontsize=fontsize)
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
            return True
        except Exception:
            return False
        finally:
            plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--formulas", type=Path, help="text file, one formula per line")
    p.add_argument("--random", type=int, default=0, help="number of random formulas to add")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variants", type=int, default=2, help="renders per formula (different fonts/sizes)")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    formulas: list[str] = []
    if args.formulas:
        for line in args.formulas.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("%"):
                formulas.append(line)
    formulas += [random_formula(rng) for _ in range(args.random)]
    formulas = list(dict.fromkeys(formulas))  # dedupe, keep order
    if not formulas:
        print("no formulas: pass --formulas and/or --random", file=sys.stderr)
        return 1

    valid = [f for f in formulas if can_parse(f)]
    print(f"{len(valid)}/{len(formulas)} formulas parse with mathtext")

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng.shuffle(valid)
    n_val = max(1, int(len(valid) * args.val_ratio))
    splits = {"val": valid[:n_val], "train": valid[n_val:]}

    counter = 0
    for split, items in splits.items():
        rows: list[str] = []
        for f in items:
            for _ in range(args.variants):
                name = f"{counter:06d}.png"
                ok = render(
                    f,
                    img_dir / name,
                    fontset=rng.choice(FONTSETS),
                    fontsize=rng.choice([14, 16, 18, 20, 24]),
                    dpi=rng.choice([100, 120, 150, 200]),
                )
                if ok:
                    rows.append(f"images/{name}\t{f}")
                    counter += 1
        (args.out / f"{split}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{split}: {len(rows)} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
