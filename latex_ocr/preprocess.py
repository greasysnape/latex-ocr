"""Image preprocessing: crop to content, resize to fixed height, normalize."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps

STRIDE = 16  # total downsampling of the encoder stem


def load_image(src) -> Image.Image:
    img = src if isinstance(src, Image.Image) else Image.open(src)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # transparent background -> white
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


def crop_to_content(img: Image.Image, threshold: int = 200, margin: int = 4) -> Image.Image:
    arr = np.asarray(img)
    ink = arr < threshold
    if not ink.any():
        return img
    # dark background (e.g. dark-mode screenshots) -> invert first
    if ink.mean() > 0.5:
        img = ImageOps.invert(img)
        arr = np.asarray(img)
        ink = arr < threshold
    ys, xs = np.where(ink)
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, arr.shape[1])
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, arr.shape[0])
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def resize(img: Image.Image, height: int, max_width: int) -> Image.Image:
    w, h = img.size
    new_w = max(1, round(w * height / h))
    new_h = height
    if new_w > max_width:
        # too wide: shrink and pad vertically instead of squashing
        scale = max_width / new_w
        new_w = max_width
        new_h = max(1, round(height * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if new_h < height:
        canvas = Image.new("L", (new_w, height), 255)
        canvas.paste(img, (0, (height - new_h) // 2))
        img = canvas
    # pad width to a multiple of the encoder stride
    pad_w = -new_w % STRIDE
    if pad_w:
        canvas = Image.new("L", (new_w + pad_w, height), 255)
        canvas.paste(img, (0, 0))
        img = canvas
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    """(1, H, W) float tensor, ink = 1, background = 0."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(1.0 - arr).unsqueeze(0)


def preprocess(src, height: int = 64, max_width: int = 512) -> torch.Tensor:
    img = load_image(src)
    img = crop_to_content(img)
    img = resize(img, height, max_width)
    return to_tensor(img)
