"""Image-to-LaTeX OCR for math formulas."""

__version__ = "0.4.0"


def __getattr__(name):
    # lazy import so `python -m latex_ocr.tokenizer` does not need torch
    if name == "LatexOCR":
        from .api import LatexOCR

        return LatexOCR
    raise AttributeError(name)


__all__ = ["LatexOCR"]
