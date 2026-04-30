"""Smoke tests that don't require downloading the SigLIP2 backbone.

The MaSC class itself is GPU/network-heavy, so we only test the pure
helpers here. Real end-to-end tests should be run manually with a
backbone available; see README for usage.
"""
from __future__ import annotations

import numpy as np

from masc import strip_subject_from_prompt
from masc import __version__


def test_version():
    assert __version__ == "0.1.1"


def test_strip_subject_with_article():
    out = strip_subject_from_prompt(
        "A photo of a kitten chasing a butterfly in a sunny garden",
        "kitten",
    )
    assert out == "A photo of chasing a butterfly in a sunny garden"


def test_strip_subject_bare():
    out = strip_subject_from_prompt(
        "kitten on the beach at sunset",
        "kitten",
    )
    assert out == "on the beach at sunset"


def test_strip_subject_case_insensitive():
    out = strip_subject_from_prompt(
        "A photo of THE Kitten in a basket",
        "kitten",
    )
    assert out == "A photo of in a basket"


def test_strip_subject_multiword():
    out = strip_subject_from_prompt(
        "a photo of a piggy bank on a desk",
        "piggy bank",
    )
    assert out == "a photo of on a desk"


def test_strip_subject_no_match_returns_collapsed():
    out = strip_subject_from_prompt("a photo of a dog", "cat")
    assert out == "a photo of a dog"


def test_imports():
    """The package surface should import cleanly without a model download."""
    import masc

    assert hasattr(masc, "MaSC")
    assert hasattr(masc, "MaSCResult")
    assert hasattr(masc, "strip_subject_from_prompt")
