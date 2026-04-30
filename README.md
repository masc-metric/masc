# MaSC

**A Masked Similarity Metric for Evaluating Concept-Driven Generation.**

MaSC is a non-LLM evaluation metric for single-concept text-to-image
personalization. One forward pass of `google/siglip2-so400m-patch16-naflex`
per image plus pre-computed segmentation masks yields **two scores** from
the same patch-token tensor:

- **Concept Preservation (CP)** — masked-maxcos over the foreground
  concept region.
- **Prompt Following (PF)** — cosine between a subject-stripped prompt
  embedding and a background-pooled image embedding.

## Install

```bash
pip install masc-metric
```

You will also need a SigLIP2 backbone, which the package auto-downloads
from HuggingFace on first use. The default checkpoint
(`google/siglip2-so400m-patch16-naflex`) is gated — accept the model's
license on HF and `huggingface-cli login` once.

## Usage

```python
import numpy as np
from PIL import Image
from masc import MaSC

masc = MaSC()  # default: siglip2-so400m-patch16-naflex; auto-picks cuda > mps > cpu

ref_image = Image.open("ref.png")
out_image = Image.open("out.png")
ref_mask = np.array(Image.open("ref_mask.png").convert("L"))  # H, W uint8
out_mask = np.array(Image.open("out_mask.png").convert("L"))

result = masc.score(
    ref_image=ref_image,
    ref_mask=ref_mask,
    out_image=out_image,
    out_mask=out_mask,
    prompt="a photo of a cat on the beach",
    object_name="cat",  # subject stripped from prompt for the PF head
)

print(result.cp, result.pf)
print(result.extras)
```

A runnable end-to-end version of the snippet above is in
[`examples/score_sample.py`](examples/score_sample.py), which scores a
real (reference, output, prompt) sample shipped under
[`examples/sample_hat/`](examples/sample_hat/).

`MaSC.score(...)` returns a `MaSCResult` with:

- `cp: float` — concept preservation score in `[-1, 1]` (higher = better).
- `pf: float` — prompt following score in `[-1, 1]` (higher = better).
- `extras: dict` — patch counts, fg fractions, the (possibly stripped)
  prompt actually fed to the text encoder.

### Masks

MaSC expects **pre-computed binary masks** as `np.ndarray` (H, W). It does
not bundle a segmenter — produce masks however you like (SAM3, Grounded
SAM, hand-labeled). Foreground = non-zero. Values can be uint8 (0..255)
or float (0..1).

The masks must mark the concept region; in the reference image the
concept is the subject being preserved, in the output image the concept
is the same subject in its new scene.

### Choosing a backbone

Any SigLIP2 checkpoint works (fixed-size or NaFlex):

```python
masc = MaSC(model_id="google/siglip2-base-patch16-512")
```

### Device

By default MaSC auto-picks the best available torch device: CUDA →
Apple MPS → CPU. Override with `device="cuda"`, `device="mps"`, or
`device="cpu"` if you want to pin it.

Smaller backbones are faster but lose accuracy. The paper's headline
numbers are on `siglip2-so400m-patch16-naflex` at 1024 patches.

## Citation

If you use MaSC, please cite:

```bibtex
@inproceedings{bartkowiak2026masc,
  title     = {{MaSC}: A Masked Similarity Metric for Evaluating Concept-Driven Generation},
  author    = {Bartkowiak, Patryk},
  year      = {2026}
}
```

## License

Apache-2.0. See `LICENSE`.
