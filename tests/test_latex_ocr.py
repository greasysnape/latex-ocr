import pytest
import torch
from PIL import Image, ImageDraw

from latex_ocr.dataset import collate
from latex_ocr.metrics import edit_distance, normalized_edit_distance, summarize
from latex_ocr.model import LatexOCRModel, ModelConfig
from latex_ocr.preprocess import STRIDE, preprocess
from latex_ocr.tokenizer import Vocab, detokenize, tokenize


def test_tokenize():
    assert tokenize(r"\frac{a}{b}") == [r"\frac", "{", "a", "}", "{", "b", "}"]
    assert tokenize(r"x_{i}^{2} \leq \alpha") == ["x", "_", "{", "i", "}", "^", "{", "2", "}", r"\leq", r"\alpha"]
    assert tokenize(r"a \, b \{ c") == ["a", r"\,", "b", r"\{", "c"]


@pytest.mark.parametrize("latex", [
    r"\frac{a}{b}",
    r"\sin x + \cos y",
    r"\alpha \beta",
    r"\mathrm{softmax}(z)_{i}",
])
def test_detokenize_roundtrip(latex):
    assert tokenize(detokenize(tokenize(latex))) == tokenize(latex)


def test_detokenize_keeps_command_separation():
    assert detokenize([r"\sin", "x"]) == r"\sin x"
    assert detokenize([r"\alpha", "+", "1"]) == r"\alpha+1"


def test_vocab_encode_decode(tmp_path):
    v = Vocab.build([r"\frac{a}{b}", "x^{2}"])
    ids = v.encode(r"\frac{a}{b}")
    assert ids[0] == v.bos_id and ids[-1] == v.eos_id
    assert v.decode(ids) == r"\frac{a}{b}"
    assert v.encode("q")[1] == v.unk_id

    v.save(tmp_path / "vocab.json")
    v2 = Vocab.load(tmp_path / "vocab.json")
    assert v2.itos == v.itos


def test_preprocess_shape():
    img = Image.new("L", (300, 80), 255)
    ImageDraw.Draw(img).text((40, 30), "x^2 + y^2", fill=0)
    t = preprocess(img, height=32, max_width=256)
    assert t.shape[0] == 1 and t.shape[1] == 32
    assert t.shape[2] % STRIDE == 0 and t.shape[2] <= 256 + STRIDE
    assert t.max() > 0.5  # ink present


def test_collate_pads():
    a = torch.zeros(1, 32, 64)
    b = torch.zeros(1, 32, 96)
    imgs, tgt, formulas = collate([(a, torch.tensor([1, 5, 2]), "a"), (b, torch.tensor([1, 5, 6, 7, 2]), "b")])
    assert imgs.shape == (2, 1, 32, 96)
    assert tgt.shape == (2, 5) and tgt[0, 3] == 0
    assert formulas == ["a", "b"]


def _tiny_model(vocab_size=20):
    cfg = ModelConfig(vocab_size=vocab_size, d_model=32, nhead=4, enc_layers=1, dec_layers=1, dim_ff=64, max_len=16)
    return LatexOCRModel(cfg).eval()


def test_model_forward_and_decode():
    model = _tiny_model()
    images = torch.rand(2, 1, 32, 64)
    tgt = torch.tensor([[1, 5, 6, 2], [1, 7, 2, 0]])
    logits = model(images, tgt)
    assert logits.shape == (2, 4, 20)

    out = model.greedy(images, bos_id=1, eos_id=2, max_len=10)
    assert out.shape[0] == 2 and out.shape[1] <= 10
    assert (out[:, 0] == 1).all()

    seq = model.beam_search(images[:1], bos_id=1, eos_id=2, beam=3, max_len=10)
    assert seq[0] == 1 and len(seq) <= 10


def test_metrics():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert normalized_edit_distance(r"\frac{a}{b}", r"\frac{a}{b}") == 0.0
    s = summarize([r"x^{2}", "y"], [r"x^{2}", "z"])
    assert s["n"] == 2 and s["exact_match"] == 0.5
