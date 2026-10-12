# Changelog

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
