"""MaSC.score on tiny random SigLIP2 models (CPU, no downloads).

`score` encodes each image once: PF reuses the output's vision-tower hidden
state from the CP branch. These tests count the vision-tower calls and compare
every score with an independent reference that encodes the output a second
time for PF, as the paper's reproduction code does (prompt padded to the text
tower's length, output patches with mask weight >= 0.5 hidden from the pool).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

transformers = pytest.importorskip("transformers")

import masc.metric as metric  # noqa: E402
from masc import MaSC  # noqa: E402

VOCAB, TEXT_LEN = 100, 16


class _Processor:
    """Stand-in for the SigLIP2 processor: the real image processor plus a
    deterministic word-level tokenizer that, like the real processor, pads
    and truncates to the text tower's length."""

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, text, padding, return_tensors):
        assert padding == "max_length" and return_tensors == "pt"
        rows = []
        for t in text:
            ids = [sum(map(ord, w)) % (VOCAB - 1) + 1 for w in t.split()][:TEXT_LEN]
            rows.append(ids + [0] * (TEXT_LEN - len(ids)))
        return {"input_ids": torch.tensor(rows)}


def _tiny(kind: str):
    text = dict(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=TEXT_LEN,
    )
    vision = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        patch_size=16,
    )
    torch.manual_seed(0)
    if kind == "naflex":
        cfg = transformers.Siglip2Config(
            text_config=text, vision_config={**vision, "num_patches": 64}
        )
        model = transformers.Siglip2Model(cfg)
        proc = transformers.Siglip2ImageProcessor(patch_size=16, max_num_patches=64)
    else:
        cfg = transformers.SiglipConfig(
            text_config=text, vision_config={**vision, "image_size": 64}
        )
        model = transformers.SiglipModel(cfg)
        proc = transformers.SiglipImageProcessor(size={"height": 64, "width": 64})
    return model, proc


@pytest.fixture(params=["naflex", "fixed"])
def scorer(request, monkeypatch):
    model, proc = _tiny(request.param)
    monkeypatch.setattr(metric.AutoModel, "from_pretrained", lambda _id: model)
    monkeypatch.setattr(metric.AutoProcessor, "from_pretrained", lambda _id: _Processor(proc))
    model_id = "tiny-siglip2-naflex" if request.param == "naflex" else "tiny-siglip2-64"
    return MaSC(model_id=model_id, device="cpu", naflex_max_patches=64)


def _image(seed: int, w: int, h: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def _disk(w: int, h: int, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    return ((((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= (r * min(h, w)) ** 2) * 255).astype(
        np.uint8
    )


def _reference(m: MaSC, ref_img, ref_mask, out_img, out_mask, text) -> tuple[float, float]:
    """One forward per image for CP, then a second forward of the output for
    PF (the structure of the reproduction code and of masc-metric <= 0.2.0)."""

    def encode(img):
        kw = dict(images=[img.convert("RGB")], return_tensors="pt")
        if m.is_naflex:
            kw["max_num_patches"] = m.naflex_max_patches
        inp = m.processor.image_processor(**kw)
        if m.is_naflex:
            o = m.model.vision_model(
                pixel_values=inp["pixel_values"],
                pixel_attention_mask=inp["pixel_attention_mask"],
                spatial_shapes=inp["spatial_shapes"],
            )
            hp, wp = (int(x) for x in inp["spatial_shapes"][0])
            valid = inp["pixel_attention_mask"][0].bool()
        else:
            o = m.model.vision_model(pixel_values=inp["pixel_values"])
            n = o.last_hidden_state.shape[1]
            hp = wp = int(round(n**0.5))
            valid = torch.ones(n, dtype=torch.bool)
        return o.last_hidden_state, valid, hp, wp

    with torch.inference_mode():
        hr, vr, hpr, wpr = encode(ref_img)
        ho, vo, _, _ = encode(out_img)
        r = F.normalize(hr[0][vr], dim=-1)
        nn = (r @ F.normalize(ho[0][vo], dim=-1).T).max(dim=-1).values
        fg_r = m._downsample_mask(ref_mask, hpr, wpr).flatten().to(r.dtype) > m.fg_threshold
        cp = float(nn[fg_r].mean()) if fg_r.any() else float("nan")

        h2, v2, hp, wp = encode(out_img)  # the second output forward that 0.3.0 removed
        fg_o = m._downsample_mask(out_mask, hp, wp).flatten() >= m.fg_threshold
        kpm = torch.ones(h2.shape[1], dtype=torch.bool)
        kpm[v2.nonzero(as_tuple=False).squeeze(-1)] = fg_o
        if not (~kpm).any():
            return cp, float("nan")
        img = m._pool_with_kpm(m.model.vision_model.head, h2, kpm)
        ids = m.processor(text=[text], padding="max_length", return_tensors="pt")
        txt = m.model.text_model(input_ids=ids["input_ids"]).pooler_output
        pf = float((F.normalize(img, dim=-1) * F.normalize(txt, dim=-1)).sum(-1)[0])
    return cp, pf


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


CASES = {  # name -> (ref size, ref mask radius, out size, out mask radius); radius None = empty
    "square": ((64, 64), 0.3, (64, 64), 0.25),
    "non-square": ((96, 48), 0.35, (48, 112), 0.2),
    "empty output mask": ((64, 64), 0.3, (80, 64), None),
    "full output mask": ((64, 64), 0.3, (64, 64), 2.0),
    "empty reference mask": ((64, 64), None, (64, 64), 0.25),
}


@pytest.mark.parametrize("case", list(CASES))
def test_score_equals_two_pass_reference(scorer, case):
    (rw, rh), rr, (ow, oh), orr = CASES[case]
    ref_img, out_img = _image(1, rw, rh), _image(2, ow, oh)
    ref_mask = _disk(rw, rh, rr) if rr is not None else np.zeros((rh, rw), np.uint8)
    out_mask = _disk(ow, oh, orr) if orr is not None else np.zeros((oh, ow), np.uint8)
    prompt = "a photo of a kitten chasing a butterfly in a sunny garden"
    res = scorer.score(ref_img, ref_mask, out_img, out_mask, prompt, object_name="kitten")
    cp, pf = _reference(scorer, ref_img, ref_mask, out_img, out_mask, res.extras["prompt_text"])
    assert _same(res.cp, cp), (res.cp, cp)
    assert _same(res.pf, pf), (res.pf, pf)
    if case == "full output mask":
        assert math.isnan(res.pf) and res.extras["pf_n_bg_pooled"] == 0
    if case == "empty reference mask":
        assert math.isnan(res.cp) and res.extras["cp_n_ref_fg"] == 0
    if case == "empty output mask":
        assert res.extras["pf_n_bg_pooled"] == res.extras["pf_n_total"]


def test_each_image_is_encoded_once(scorer):
    calls = []
    vm = scorer.model.vision_model
    forward = vm.forward
    vm.forward = lambda *a, **k: calls.append(1) or forward(*a, **k)
    tm = scorer.model.text_model
    tforward = tm.forward
    text_lengths = []
    tm.forward = lambda *a, **k: text_lengths.append(k["input_ids"].shape[1]) or tforward(*a, **k)
    scorer.score(
        _image(3, 64, 64),
        _disk(64, 64, 0.3),
        _image(4, 64, 64),
        _disk(64, 64, 0.2),
        "a dog on the beach",
        object_name="dog",
    )
    assert len(calls) == 2
    # One text forward, padded to the text tower's trained length.
    assert text_lengths == [scorer.model.config.text_config.max_position_embeddings]


def test_pf_hides_patches_at_the_threshold(scorer):
    """A mask edge on a patch boundary gives weight exactly 0.5 on the patch
    grid; PF hides those patches (>= 0.5) while CP does not count them as
    reference foreground (> 0.5)."""
    img = _image(5, 64, 64)
    mask = np.zeros((64, 64), np.uint8)
    mask[:, :24] = 255  # columns 16-31 (patch column 1) are half covered
    res = scorer.score(img, mask, img, mask, "a dog on the beach", object_name="dog")
    w = scorer._downsample_mask(mask, 4, 4).flatten() if not scorer.is_naflex else None
    if w is not None:
        assert (w == 0.5).any()
        assert res.extras["pf_n_fg"] == int((w >= 0.5).sum())
        assert res.extras["cp_n_ref_fg"] == int((w > 0.5).sum())
