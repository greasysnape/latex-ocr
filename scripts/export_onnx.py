#!/usr/bin/env python3
"""Export a checkpoint to ONNX (encoder + decoder).

    python scripts/export_onnx.py --checkpoint checkpoints/best.pt --out checkpoints/onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latex_ocr.checkpoint import load_checkpoint  # noqa: E402


class DecoderStep(nn.Module):
    """Wraps the decoder so it returns logits for the last position only."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.decoder(tokens, memory)[:, -1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    model, vocab, state = load_checkpoint(args.checkpoint, "cpu")
    d = state.get("data", {})
    h, w = d.get("img_height", 64), min(d.get("max_width", 512), 256)
    args.out.mkdir(parents=True, exist_ok=True)

    image = torch.zeros(1, 1, h, w)
    torch.onnx.export(
        model.encoder,
        (image,),
        args.out / "encoder.onnx",
        input_names=["image"],
        output_names=["memory"],
        dynamic_axes={"image": {0: "batch", 3: "width"}, "memory": {0: "batch", 1: "seq"}},
        opset_version=args.opset,
    )

    with torch.no_grad():
        memory = model.encoder(image)
    tokens = torch.tensor([[vocab.bos_id, 5, 6]], dtype=torch.long)
    torch.onnx.export(
        DecoderStep(model.decoder),
        (tokens, memory),
        args.out / "decoder.onnx",
        input_names=["tokens", "memory"],
        output_names=["logits"],
        dynamic_axes={"tokens": {0: "batch", 1: "length"}, "memory": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )

    meta = {
        "vocab": vocab.itos,
        "bos_id": vocab.bos_id,
        "eos_id": vocab.eos_id,
        "img_height": h,
        "max_width": d.get("max_width", 512),
        "max_len": model.cfg.max_len,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"exported to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
