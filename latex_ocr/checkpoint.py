"""Saving and loading checkpoints (model weights + config + vocab)."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import LatexOCRModel, ModelConfig
from .tokenizer import Vocab


def save_checkpoint(
    path: str | Path,
    model: LatexOCRModel,
    vocab: Vocab,
    data_cfg: dict,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    step: int = 0,
    metrics: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "model_config": model.cfg.to_dict(),
        "vocab": vocab.itos,
        "data": {k: data_cfg[k] for k in ("img_height", "max_width", "max_len") if k in data_cfg},
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**state["model_config"])
    model = LatexOCRModel(cfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    vocab = Vocab.from_list(state["vocab"])
    return model, vocab, state
