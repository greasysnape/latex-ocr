"""CNN stem + Transformer encoder-decoder for image-to-LaTeX."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    dim_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 200

    def to_dict(self) -> dict:
        return asdict(self)


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


class ConvStem(nn.Module):
    """Downsamples by 16 in both directions."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(1, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
            _conv_block(256, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sinusoid_1d(length: int, dim: int, device=None) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def sinusoid_2d(h: int, w: int, dim: int, device=None) -> torch.Tensor:
    """(h*w, dim): first half encodes rows, second half columns."""
    half = dim // 2
    pe_h = sinusoid_1d(h, half, device)  # (h, half)
    pe_w = sinusoid_1d(w, half, device)  # (w, half)
    pe = torch.cat(
        [pe_h.unsqueeze(1).expand(h, w, half), pe_w.unsqueeze(0).expand(h, w, half)],
        dim=-1,
    )
    return pe.reshape(h * w, dim)


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.stem = ConvStem(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.enc_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.stem(images)  # (B, C, H', W')
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)  # (B, H'W', C)
        x = x + sinusoid_2d(h, w, c, x.device).unsqueeze(0)
        return self.norm(self.encoder(x))


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.register_buffer("pos", sinusoid_1d(cfg.max_len, cfg.d_model), persistent=False)
        layer = nn.TransformerDecoderLayer(
            cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, cfg.dec_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        L = tgt.size(1)
        x = self.embed(tgt) * math.sqrt(self.d_model) + self.pos[:L].unsqueeze(0)
        causal = torch.triu(torch.ones(L, L, device=tgt.device, dtype=torch.bool), diagonal=1)
        x = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=pad_mask)
        return self.out(self.norm(x))


class LatexOCRModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, images: torch.Tensor, tgt_in: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        memory = self.encoder(images)
        return self.decoder(tgt_in, memory, pad_mask=tgt_in.eq(pad_id))

    @torch.no_grad()
    def greedy(self, images: torch.Tensor, bos_id: int, eos_id: int, max_len: int | None = None) -> torch.Tensor:
        max_len = min(max_len or self.cfg.max_len, self.cfg.max_len)
        memory = self.encoder(images)
        b = images.size(0)
        seq = torch.full((b, 1), bos_id, dtype=torch.long, device=images.device)
        done = torch.zeros(b, dtype=torch.bool, device=images.device)
        for _ in range(max_len - 1):
            logits = self.decoder(seq, memory)[:, -1]
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            done |= nxt.eq(eos_id)
            if done.all():
                break
        return seq

    @torch.no_grad()
    def beam_search(
        self,
        image: torch.Tensor,
        bos_id: int,
        eos_id: int,
        beam: int = 5,
        max_len: int | None = None,
        length_penalty: float = 0.7,
    ) -> list[int]:
        """Beam search for a single image (1, 1, H, W)."""
        max_len = min(max_len or self.cfg.max_len, self.cfg.max_len)
        memory = self.encoder(image)
        beams: list[tuple[list[int], float]] = [([bos_id], 0.0)]
        finished: list[tuple[list[int], float]] = []

        for _ in range(max_len - 1):
            seqs = torch.tensor([s for s, _ in beams], dtype=torch.long, device=image.device)
            logp = self.decoder(seqs, memory.expand(len(beams), -1, -1))[:, -1].log_softmax(-1)
            scores = torch.tensor([sc for _, sc in beams], device=image.device).unsqueeze(1) + logp
            top = scores.flatten().topk(beam)
            vocab = logp.size(-1)
            new_beams = []
            for score, idx in zip(top.values.tolist(), top.indices.tolist()):
                bi, tok = divmod(idx, vocab)
                seq = beams[bi][0] + [tok]
                if tok == eos_id:
                    finished.append((seq, score / (len(seq) ** length_penalty)))
                else:
                    new_beams.append((seq, score))
            if len(finished) >= beam or not new_beams:
                break
            beams = new_beams

        if not finished:
            finished = [(s, sc / (len(s) ** length_penalty)) for s, sc in beams]
        return max(finished, key=lambda x: x[1])[0]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
