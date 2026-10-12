# MaSC: A Masked Similarity Metric for Evaluating Concept-Driven Generation

[**Project page**](https://masc-metric.github.io/) ·
[**GitHub**](https://github.com/masc-metric/masc) ·
[**PyPI**](https://pypi.org/project/masc-metric/)

![MaSC method diagram](https://raw.githubusercontent.com/masc-metric/masc/main/assets/method_diagram.png)

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

### Subject stripping (PF prompt)

Passing `object_name` removes the subject from the prompt before the PF text
encoding. Two strippers are available:

```python
masc = MaSC()                   # stripper="regex": default, behind every MaSC number in the paper
masc = MaSC(stripper="spacy")   # syntax-aware, optional extra
```

The default `regex` stripper removes `(a|an|the) <object_name>`, or the bare
name when no article precedes it. It needs no extra dependency, but it misses
inflected forms ("kittens" for "kitten") and leaves fragments such as the
possessive `'s` or a stranded adjective ("a vintage motorcycle" becomes
"a vintage").

The `spacy` stripper parses the prompt and removes every mention of the
subject together with its noun-phrase pre-modifiers (determiner, possessive,
numeral, adjective, compound), matching by lemma so plurals and capitalized
forms are found. It keeps everything attached to the right of the subject
(prepositional phrases, participial clauses, relations to other objects):

```python
from masc import strip_subject_from_prompt
strip_subject_from_prompt("Two vintage motorcycles parked on a lonely desert road",
                          "motorcycle", stripper="spacy")
# -> 'parked on a lonely desert road'
```

Install it with:

```bash
pip install "masc-metric[spacy]"
python -m spacy download en_core_web_sm
```

On DreamBench++ the two strippers produce different text for 162 of the 1,179
non-style prompts, but PF agreement with the human ratings does not change:
Krippendorff's α differs by −0.001 (95% CI [−0.004, +0.001], paired bootstrap
over 6,605 rated images). The regex stripper therefore stays the default. The
spaCy stripper gives cleaner text when prompts mention the subject in the
plural, with a possessive or with adjectives; whether that changes agreement on
such prompt sets is untested.

Known limits of the `spacy` stripper:

- Pronouns that refer to the subject ("its", "her") stay; the pipeline has no
  coreference component.
- A subject mentioned only by a synonym or a different noun ("a corgi" for
  `object_name="corgi dog"`) is not found, and the prompt is kept.
- Coordination keeps the conjunction: "a kitten and a puppy" becomes "and a
  puppy".
- Style prompts keep their "in the style of" scaffold.
- The small English model often parses a subject followed by a participle as
  one compound noun ("a dolphin swimming"). The stripper then removes the
  determiner and adjectives with the subject, unless a noun not ending in
  "-ing" follows directly ("a motorcycle rider" becomes "a rider"). A
  participle without "-ing" can therefore leave them behind: "a ripe avocado
  split open" becomes "a ripe split open".

Outputs depend on the spaCy model. The DreamBench++ comparison above and the
expected strings in `tests/test_text.py` use `en_core_web_sm` 3.8.0 (spaCy
3.8); loading another version of it warns. `result.extras` records the
`stripper`, the spaCy pipeline (`stripper_model`, e.g.
`"en_core_web_sm-3.8.0"`) and whether the subject was found and removed
(`prompt_stripped`).

### Choosing a backbone

Any SigLIP2 checkpoint works (fixed-size or NaFlex):

```python
masc = MaSC(model_id="google/siglip2-base-patch16-512")
```

Smaller backbones are faster but lose accuracy. The paper's headline
numbers are on `siglip2-so400m-patch16-naflex` at 1024 patches.

### Device

By default MaSC auto-picks the best available torch device: CUDA →
Apple MPS → CPU. Override with `device="cuda"`, `device="mps"`, or
`device="cpu"` if you want to pin it.

## Citation

If you use MaSC, please cite:

```bibtex
@article{bartkowiak2026masc,
  title={MaSC: A Masked Similarity Metric for Evaluating Concept-Driven Generation},
  author={Bartkowiak, Patryk and Petersen, Lennart and Kotrys, Bartosz and Michels, Dominik and Pirk, Soren and Palubicki, Wojtek},
  journal={arXiv preprint arXiv:2605.22469},
  year={2026}
}
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
