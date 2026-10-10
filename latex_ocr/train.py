"""Training loop.

    python -m latex_ocr.train --config configs/base.yaml [--resume checkpoints/last.pt]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .dataset import BucketSampler, FormulaDataset, collate
from .metrics import summarize
from .model import LatexOCRModel, ModelConfig, count_parameters
from .tokenizer import Vocab, read_tsv

log = logging.getLogger("latex_ocr.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(optimizer, warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model: LatexOCRModel, loader: DataLoader, vocab: Vocab, device) -> dict:
    model.eval()
    preds: list[str] = []
    refs: list[str] = []
    for images, _, formulas in tqdm(loader, desc="eval", leave=False):
        out = model.greedy(images.to(device), vocab.bos_id, vocab.eos_id)
        preds.extend(vocab.decode(seq.tolist()) for seq in out)
        refs.extend(formulas)
    model.train()
    return summarize(preds, refs)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train latex-ocr")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--resume", help="checkpoint to resume from")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    set_seed(tcfg["seed"])
    device = torch.device(args.device)

    vocab_path = Path(dcfg["vocab"])
    if vocab_path.exists():
        vocab = Vocab.load(vocab_path)
    else:
        log.info("vocab not found, building from %s", dcfg["train"])
        vocab = Vocab.build(f for _, f in read_tsv(dcfg["train"]))
        vocab.save(vocab_path)

    ds_kwargs = dict(img_height=dcfg["img_height"], max_width=dcfg["max_width"], max_len=dcfg["max_len"])
    train_ds = FormulaDataset(dcfg["train"], vocab, **ds_kwargs)
    val_ds = FormulaDataset(dcfg["val"], vocab, **ds_kwargs)
    if tcfg.get("eval_samples") and len(val_ds) > tcfg["eval_samples"]:
        val_ds = Subset(val_ds, list(range(tcfg["eval_samples"])))

    sampler = BucketSampler(train_ds.approx_lengths(), tcfg["batch_size"], shuffle=True, seed=tcfg["seed"])
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=dcfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], collate_fn=collate, num_workers=dcfg["num_workers"])

    model_cfg = ModelConfig(vocab_size=len(vocab), max_len=dcfg["max_len"], **mcfg)
    model = LatexOCRModel(model_cfg).to(device)
    log.info("model: %.1fM parameters, vocab %d", count_parameters(model) / 1e6, len(vocab))

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(train_loader)
    scheduler = warmup_cosine(optimizer, tcfg["warmup_steps"], total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=tcfg["label_smoothing"])
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, step, best = 0, 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch, step = state.get("epoch", 0), state.get("step", 0)
        best = state.get("metrics", {}).get("best_edit_distance", best)
        log.info("resumed from %s (epoch %d, step %d)", args.resume, start_epoch, step)

    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"

    model.train()
    for epoch in range(start_epoch, tcfg["epochs"]):
        sampler.set_epoch(epoch)
        t0 = time.time()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{tcfg['epochs']}")
        for images, tgt, _ in pbar:
            images, tgt = images.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, tgt_in, pad_id=vocab.pad_id)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
            pbar.set_postfix(loss=f"{running:.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        record = {"epoch": epoch + 1, "step": step, "train_loss": running, "time": round(time.time() - t0, 1)}
        if (epoch + 1) % tcfg["eval_every"] == 0:
            m = evaluate(model, val_loader, vocab, device)
            record.update({f"val_{k}": v for k, v in m.items()})
            log.info("epoch %d: loss %.4f, val edit %.4f, exact %.3f",
                     epoch + 1, running, m["edit_distance"], m["exact_match"])
            if m["edit_distance"] < best:
                best = m["edit_distance"]
                save_checkpoint(out_dir / "best.pt", model, vocab, dcfg, epoch=epoch + 1, step=step,
                                metrics={**m, "best_edit_distance": best})
                log.info("new best -> %s", out_dir / "best.pt")

        save_checkpoint(out_dir / "last.pt", model, vocab, dcfg, optimizer, scheduler,
                        epoch=epoch + 1, step=step, metrics={"best_edit_distance": best})
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
