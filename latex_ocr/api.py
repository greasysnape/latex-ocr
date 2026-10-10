"""High-level inference API."""

from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .preprocess import preprocess

DEFAULT_CHECKPOINT = Path("checkpoints/latex_ocr_base.pt")


class LatexOCR:
    """
    >>> ocr = LatexOCR("checkpoints/latex_ocr_base.pt")
    >>> ocr("formula.png")
    '\\\\frac{a}{b}'
    """

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None, beam: int = 1) -> None:
        path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {path}. Train a model (see README) or pass --checkpoint."
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vocab, state = load_checkpoint(path, self.device)
        d = state.get("data", {})
        self.img_height = d.get("img_height", 64)
        self.max_width = d.get("max_width", 512)
        self.beam = beam

    @torch.no_grad()
    def __call__(self, image) -> str:
        x = preprocess(image, self.img_height, self.max_width).unsqueeze(0).to(self.device)
        if self.beam > 1:
            seq = self.model.beam_search(x, self.vocab.bos_id, self.vocab.eos_id, beam=self.beam)
        else:
            seq = self.model.greedy(x, self.vocab.bos_id, self.vocab.eos_id)[0].tolist()
        return self.vocab.decode(seq)

    def batch(self, images) -> list[str]:
        return [self(im) for im in images]
