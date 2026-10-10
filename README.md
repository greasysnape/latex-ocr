# latex-ocr

Convert images of math formulas into LaTeX.

A compact image-to-sequence model (CNN stem + Transformer encoder/decoder) written in plain PyTorch.
Train it on your own data, or on a synthetic dataset generated from a list of formulas with the
included renderer, then run inference from the command line or Python.

```
$ latex-ocr samples/quadratic.png
x = \frac{-b \pm \sqrt{b^{2} - 4ac}}{2a}
```

## Features

- Small model (~10M parameters with `configs/base.yaml`), trains on a single GPU
- Synthetic data generator using matplotlib mathtext (no TeX installation needed)
- Greedy and beam-search decoding
- Token-level edit distance / exact-match evaluation
- ONNX export for deployment

## Installation

```bash
git clone <repository-url>
cd latex-ocr
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[render]"
```

Install the PyTorch build that matches your CUDA version first if you want GPU training
(see pytorch.org).

## Pretrained weights

Weights are not stored in git. Put a checkpoint at `checkpoints/latex_ocr_base.pt`
(the default path), or pass `--checkpoint path/to/model.pt`.
You can train one yourself as described below.

## Quick start: inference

```bash
latex-ocr formula.png
latex-ocr img1.png img2.png --beam 5
latex-ocr formula.png --checkpoint checkpoints/my_model.pt --device cpu
```

From Python:

```python
from latex_ocr import LatexOCR

ocr = LatexOCR("checkpoints/latex_ocr_base.pt")
print(ocr("formula.png"))
```

## Training

### 1. Build a dataset

Either bring your own data as a TSV file (`relative/image/path<TAB>latex` per line),
or render a synthetic dataset:

```bash
python scripts/render_dataset.py \
    --formulas data/formulas.txt \
    --out data/synth \
    --variants 3
```

This writes `data/synth/images/*.png`, `train.tsv` and `val.tsv`.
`data/formulas.txt` contains about a hundred seed formulas; add more, or use `--random N`
to mix in randomly generated expressions.

### 2. Build the vocabulary

```bash
python -m latex_ocr.tokenizer build data/synth/train.tsv -o data/synth/vocab.json
```

### 3. Train

```bash
python -m latex_ocr.train --config configs/base.yaml
```

Checkpoints are written to `checkpoints/` (`last.pt` every epoch, `best.pt` on the best
validation edit distance). Use `configs/small.yaml` for a quick CPU sanity run.

### 4. Evaluate

```bash
python -m latex_ocr.evaluate --checkpoint checkpoints/best.pt --data data/synth/val.tsv
```

## ONNX export

```bash
python scripts/export_onnx.py --checkpoint checkpoints/best.pt --out checkpoints/onnx
```

Produces `encoder.onnx` and `decoder.onnx`.

## Project layout

```
latex-ocr/
├── configs/            # training configs
├── data/formulas.txt   # seed formulas for the synthetic renderer
├── latex_ocr/
│   ├── tokenizer.py    # LaTeX tokenizer + vocabulary
│   ├── preprocess.py   # crop / resize / normalize
│   ├── dataset.py      # TSV dataset + collate
│   ├── model.py        # encoder-decoder model, greedy & beam search
│   ├── metrics.py      # edit distance, exact match
│   ├── train.py        # training loop
│   ├── evaluate.py
│   ├── api.py          # LatexOCR inference wrapper
│   └── cli.py
├── scripts/
│   ├── render_dataset.py
│   └── export_onnx.py
└── tests/
```

## Tips

- Crops that contain only the formula with a little margin work best.
- Rendered (digital) formulas are much easier than handwriting; this model is not trained
  for handwriting.
- Increase `max_width` in the config for very long single-line equations.
