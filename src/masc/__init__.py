"""MaSC: a Masked Similarity Metric for concept-driven generation.

A single SigLIP2 forward pass per image yields two scores from the same
patch-token tensor:

- Concept Preservation (CP): masked-maxcos over fg-region patch tokens.
- Prompt Following (PF): BG-pooled image embedding vs subject-stripped
  prompt embedding, both in SigLIP2's joint contrastive space.

Reference: "MaSC: A Masked Similarity Metric for Evaluating
Concept-Driven Generation".
"""
from .metric import MaSC, MaSCResult
from .text import strip_subject_from_prompt

__version__ = "0.1.1"
__all__ = ["MaSC", "MaSCResult", "strip_subject_from_prompt", "__version__"]
