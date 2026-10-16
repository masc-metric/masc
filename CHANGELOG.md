# Changelog

## 0.3.0 - unreleased

### Fixed

- PF now matches the paper. Versions 0.1.0 to 0.2.0 encoded the prompt
  without padding it: the tokenizer's `model_max_length` is unset for
  SigLIP2, so `padding="max_length"` added nothing, and the text tower saw a
  few tokens instead of the 64 it was trained with. Its pooled embedding
  then differs strongly from the padded one, so PF scores from those
  versions do not reproduce the paper's PF results. The prompt now goes
  through the SigLIP2 processor, which pads it to 64 tokens, as the paper's
  reproduction code does. PF scores change: on 120 DreamBench++ pairs by
  0.129 on average and by up to 0.352. CP is unchanged.
- PF hides output patches whose mask weight is at or above `fg_threshold`
  (`>=`), as in the paper's code; it used `>`. Mask edges that fall on a
  patch boundary give weight exactly 0.5, so this changes `pf_n_fg` and PF
  for such masks. CP keeps reference patches with weight above the
  threshold, as before.

With the default checkpoint, CP and PF are now bit-identical to the
reproduction pipeline (masc-metric/masc-reproduction) on the same inputs and
hardware.

### Changed

- `MaSC.score` encodes each image once: PF reuses the output's vision-tower
  hidden state from the CP branch instead of running the vision tower on the
  output a second time, as the paper describes (two image forwards and one
  text forward per call, down from three image forwards). CP is
  bit-identical to 0.2.0 with the default NaFlex checkpoint, in fp32 and
  bf16.
- Fixed-size checkpoints (e.g. `google/siglip2-base-patch16-512`) take the
  image processor's preprocessing for both scores. CP used a manual resize
  and normalisation that differs from the processor by at most 6e-8 per
  pixel value; CP moves by less than 1e-6.

### Removed

- The unused `MaSC.image_mean`, `MaSC.image_std` and `MaSC.tokenizer`
  attributes; the processor encodes the text.

## 0.2.0 - 2026-10-12

### Added

- A syntax-aware subject stripper for the PF prompt: `MaSC(stripper="spacy")`
  and `strip_subject_from_prompt(..., stripper="spacy")`, installed with the
  `spacy` extra (`pip install "masc-metric[spacy]"`, then
  `python -m spacy download en_core_web_sm`). It matches plural and
  capitalized mentions and removes the subject's noun-phrase modifiers; the
  README describes its rule and known limits. The regex stripper stays the
  default and scores are unchanged.
- `stripper="none"` keeps the prompt as given.
- `MaSCResult.extras` records the `stripper` and, for spaCy, its pipeline
  (`stripper_model`, e.g. `"en_core_web_sm-3.8.0"`).

### Changed

- `extras["prompt_stripped"]` is `True` only when the subject was found and
  removed; it used to be `True` whenever `object_name` was given.
- The citation and the package metadata list all six authors of the paper.

### Fixed

- `strip_subject_from_prompt` with an empty `object_name` removed every
  article from the prompt; it now removes nothing.

## 0.1.1 - 2026-04-30

- README updates and a method diagram; dropped the legacy License classifier.

## 0.1.0 - 2026-04-30

- First release.
