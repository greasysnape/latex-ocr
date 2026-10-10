"""Command-line inference: `latex-ocr image.png [...]`."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="latex-ocr", description="Convert formula images to LaTeX")
    p.add_argument("images", nargs="+", help="image files")
    p.add_argument("-c", "--checkpoint", help="model checkpoint (default: checkpoints/latex_ocr_base.pt)")
    p.add_argument("--beam", type=int, default=1, help="beam size (1 = greedy)")
    p.add_argument("--device", help="cpu or cuda (default: auto)")
    p.add_argument("--with-name", action="store_true", help="prefix each result with the file name")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)

    from .api import LatexOCR  # imported here so --help works without torch

    try:
        ocr = LatexOCR(args.checkpoint, device=args.device, beam=args.beam)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    status = 0
    for path in args.images:
        try:
            latex = ocr(path)
        except (OSError, ValueError) as e:
            print(f"{path}: error: {e}", file=sys.stderr)
            status = 1
            continue
        print(f"{path}\t{latex}" if args.with_name else latex)
    return status


if __name__ == "__main__":
    sys.exit(main())
