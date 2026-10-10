"""Evaluation metrics on token sequences."""

from __future__ import annotations

from .tokenizer import tokenize


def edit_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p and not r:
        return 0.0
    return edit_distance(p, r) / max(len(p), len(r))


def exact_match(pred: str, ref: str) -> bool:
    return tokenize(pred) == tokenize(ref)


def summarize(preds: list[str], refs: list[str]) -> dict:
    n = len(refs)
    if n == 0:
        return {"n": 0, "edit_distance": 0.0, "exact_match": 0.0}
    ed = sum(normalized_edit_distance(p, r) for p, r in zip(preds, refs)) / n
    em = sum(exact_match(p, r) for p, r in zip(preds, refs)) / n
    return {"n": n, "edit_distance": ed, "exact_match": em}
