"""Evaluate a checkpoint on a TSV dataset.

    python -m latex_ocr.evaluate --checkpoint checkpoints/best.pt --data data/synth/val.tsv
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import load_checkpoint
from .dataset import FormulaDataset, collate
from .metrics import summarize


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate latex-ocr")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="TSV file")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dump", help="write predictions to this TSV")
    args = p.parse_args(argv)

    model, vocab, state = load_checkpoint(args.checkpoint, args.device)
    d = state.get("data", {})
    ds = FormulaDataset(args.data, vocab, d.get("img_height", 64), d.get("max_width", 512), d.get("max_len", 200))

    preds: list[str] = []
    refs: list[str] = []
    if args.beam > 1:
        for i in tqdm(range(len(ds)), desc="eval"):
            image, _, formula = ds[i]
            seq = model.beam_search(image.unsqueeze(0).to(args.device), vocab.bos_id, vocab.eos_id, beam=args.beam)
            preds.append(vocab.decode(seq))
            refs.append(formula)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
        for images, _, formulas in tqdm(loader, desc="eval"):
            out = model.greedy(images.to(args.device), vocab.bos_id, vocab.eos_id)
            preds.extend(vocab.decode(s.tolist()) for s in out)
            refs.extend(formulas)

    print(json.dumps(summarize(preds, refs), indent=2))

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for (img, _), pred, ref in zip(ds.rows, preds, refs):
                f.write(f"{img}\t{ref}\t{pred}\n")


if __name__ == "__main__":
    main()
