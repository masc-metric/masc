"""Subject strippers: the default regex stripper and the optional spaCy one.

The spaCy output cases depend on a statistical parser: their expected strings
were checked against en_core_web_sm 3.8.0, and they are skipped for any other
version or when spaCy or the model is not installed (`pip install
"masc-metric[spacy]"` and `python -m spacy download en_core_web_sm`). The
loading and dispatch tests use a stand-in spaCy module and always run.
"""
from __future__ import annotations

import sys
import types
import warnings

import pytest

import masc.text
from masc import MaSC, strip_subject_from_prompt
from masc.text import (
    DEFAULT_SPACY_MODEL,
    REFERENCE_SPACY_MODEL_VERSION,
    STRIPPERS,
    SpacySubjectStripper,
)


def _reference_model_missing() -> str | None:
    """Why the exact-output spaCy tests cannot run here, or None."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nlp = SpacySubjectStripper.load().nlp
    except (ImportError, OSError):
        return "spaCy or en_core_web_sm not installed"
    if nlp.meta.get("version") != REFERENCE_SPACY_MODEL_VERSION:
        return (f"expected strings were checked with en_core_web_sm "
                f"{REFERENCE_SPACY_MODEL_VERSION}, found {nlp.meta.get('version')}")
    return None


_MISSING = _reference_model_missing()
needs_spacy = pytest.mark.skipif(_MISSING is not None, reason=_MISSING or "")


def spacy_strip(prompt: str, name: str) -> str:
    return strip_subject_from_prompt(prompt, name, stripper="spacy")


class _StandInPipeline:
    def __init__(self, version: str) -> None:
        self.meta = {"lang": "en", "name": "core_web_sm", "version": version}


def stand_in_spacy(monkeypatch, version: str = REFERENCE_SPACY_MODEL_VERSION,
                   model_installed: bool = True) -> list[str]:
    """Replace `spacy` with a stand-in and empty the pipeline cache; returns
    the models the stand-in was asked to load."""
    loaded: list[str] = []

    def load(model, disable=()):
        loaded.append(model)
        if not model_installed:
            raise OSError(f"[E050] Can't find model '{model}'.")
        return _StandInPipeline(version)

    monkeypatch.setitem(sys.modules, "spacy", types.SimpleNamespace(load=load))
    monkeypatch.setattr(masc.text, "_PIPELINES", {})
    return loaded


def no_spacy(monkeypatch) -> None:
    """Make `import spacy` fail, as without the extra."""
    monkeypatch.setitem(sys.modules, "spacy", None)
    monkeypatch.setattr(masc.text, "_PIPELINES", {})


# --- dispatch ---------------------------------------------------------------

def test_strippers_listed():
    assert STRIPPERS == ("regex", "spacy", "none")


def test_regex_is_default():
    p = "A photo of a kitten chasing a butterfly in a sunny garden"
    assert strip_subject_from_prompt(p, "kitten") == strip_subject_from_prompt(
        p, "kitten", stripper="regex")


def test_none_returns_prompt_unchanged():
    p = "A photo of a  kitten in a basket "
    assert strip_subject_from_prompt(p, "kitten", stripper="none") == p


def test_unknown_stripper_raises():
    with pytest.raises(ValueError, match="stripper"):
        strip_subject_from_prompt("a photo of a cat", "cat", stripper="llm")


@pytest.mark.parametrize("stripper", ["regex", "spacy"])
@pytest.mark.parametrize("name", ["", "   "])
def test_empty_object_name_removes_nothing(monkeypatch, stripper, name):
    # The regex stripper of 0.1.x removed every article for an empty name.
    no_spacy(monkeypatch)  # nothing to strip, so spaCy is not needed
    assert strip_subject_from_prompt("A photo of  a cat on the mat", name,
                                     stripper=stripper) == "A photo of a cat on the mat"


# --- spaCy loading ----------------------------------------------------------

def test_spacy_missing_raises_install_hint(monkeypatch):
    no_spacy(monkeypatch)
    with pytest.raises(ImportError, match=r"pip install 'masc-metric\[spacy\]'"):
        strip_subject_from_prompt("a photo of a cat", "cat", stripper="spacy")


def test_spacy_model_missing_raises_download_hint(monkeypatch):
    stand_in_spacy(monkeypatch, model_installed=False)
    with pytest.raises(OSError, match="python -m spacy download en_core_web_sm"):
        SpacySubjectStripper.load()


def test_missing_spacy_model_raises_with_install_hint():
    pytest.importorskip("spacy")
    with pytest.raises(OSError, match="spacy download"):
        SpacySubjectStripper.load("en_core_web_doesnotexist")


def test_pipeline_loads_once_per_model(monkeypatch):
    loaded = stand_in_spacy(monkeypatch)
    a = SpacySubjectStripper.load()
    b = SpacySubjectStripper.load(DEFAULT_SPACY_MODEL)
    c = SpacySubjectStripper.load(model=DEFAULT_SPACY_MODEL)
    assert loaded == [DEFAULT_SPACY_MODEL]
    assert a.nlp is b.nlp is c.nlp


def test_other_model_version_warns(monkeypatch):
    stand_in_spacy(monkeypatch, version="3.7.1")
    with pytest.warns(UserWarning, match=r"en_core_web_sm 3\.7\.1.*tested with 3\.8\.0"):
        SpacySubjectStripper.load()


def test_reference_model_version_loads_silently(monkeypatch):
    stand_in_spacy(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert SpacySubjectStripper.load().model_name == "en_core_web_sm-3.8.0"


# --- regex stripper: behavior the paper's numbers rely on ----------------

def test_regex_misses_plural_and_leaves_possessive_marker():
    # Documented regex limits that the spaCy stripper addresses.
    assert strip_subject_from_prompt("Two kittens sleeping in a basket", "kitten") == \
        "Two kittens sleeping in a basket"
    assert strip_subject_from_prompt("The dog's ball lies on the grass", "dog") == \
        "'s ball lies on the grass"


def test_regex_strips_only_article_mentions_when_one_exists():
    assert strip_subject_from_prompt("A kitten next to another kitten on a sofa", "kitten") == \
        "next to another kitten on a sofa"


# --- spaCy stripper ---------------------------------------------------------

@needs_spacy
def test_spacy_ordinary_subject():
    assert spacy_strip("A photo of a kitten chasing a butterfly in a sunny garden",
                       "kitten") == "A photo of chasing a butterfly in a sunny garden"


@needs_spacy
def test_spacy_plural_with_numeral():
    assert spacy_strip("Two kittens sleeping in a basket", "kitten") == "sleeping in a basket"


@needs_spacy
def test_spacy_possessor_keeps_possessed_object():
    assert spacy_strip("The dog's ball lies on the grass", "dog") == "ball lies on the grass"


@needs_spacy
def test_spacy_multiword_subjects():
    assert spacy_strip("a photo of a piggy bank on a desk", "piggy bank") == "a photo of on a desk"
    assert spacy_strip("A Rubik's cube on a wooden desk", "rubik's cube") == "on a wooden desk"
    assert spacy_strip("Cherry blossoms falling over a quiet temple", "cherry blossoms") == \
        "falling over a quiet temple"


@needs_spacy
def test_spacy_hyphenated_and_case_insensitive():
    assert spacy_strip("A t-shirt hanging on a clothesline", "t-shirt") == "hanging on a clothesline"
    assert spacy_strip("A photo of THE Kitten in a basket", "kitten") == "A photo of in a basket"


@needs_spacy
def test_spacy_repeated_mentions_all_removed():
    assert spacy_strip("A kitten next to another kitten on a sofa", "kitten") == "next to on a sofa"


@needs_spacy
def test_spacy_removes_adjectival_premodifiers():
    assert spacy_strip("A photo of a vintage motorcycle on a lonely desert road",
                       "motorcycle") == "A photo of on a lonely desert road"
    assert spacy_strip("A small, fluffy kitten with big eyes on a windowsill", "kitten") == \
        "with big eyes on a windowsill"


@needs_spacy
def test_spacy_keeps_relation_bearing_phrases():
    assert spacy_strip("A cat sitting on top of a backpack in the snow", "backpack") == \
        "A cat sitting on top of in the snow"
    # The subject as a compound modifier of another noun: that noun and its
    # determiner stay.
    assert spacy_strip("A photograph of a motorcycle rider racing on a track",
                       "motorcycle") == "A photograph of a rider racing on a track"
    assert spacy_strip("A spider-man figure standing on a rooftop", "spider-man") == \
        "A figure standing on a rooftop"


@needs_spacy
def test_spacy_subject_misparsed_as_modifier_of_participle():
    # en_core_web_sm often reads "a dolphin swimming" as the compound noun
    # "dolphin swimming"; the determiner must go with the subject.
    assert spacy_strip("A photo of a dolphin swimming next to a small wooden boat", "dolphin") == \
        "A photo of swimming next to a small wooden boat"
    assert spacy_strip("A drawing of a parrot resting on a branch at dusk", "parrot") == \
        "A drawing of resting on a branch at dusk"
    assert spacy_strip("An airplane ascending through cloudy skies", "airplane") == \
        "ascending through cloudy skies"
    # Chains: subject -> participle -> object noun ("a girl planting tulips").
    assert spacy_strip("A photograph of a girl planting tulips in a backyard", "girl") == \
        "A photograph of planting tulips in a backyard"
    assert spacy_strip("A UFO radiating beams of light over a desert", "UFO") == \
        "radiating beams of light over a desert"


@needs_spacy
def test_spacy_participle_without_ing_leaves_modifiers():
    # Known limit, documented in the README.
    assert spacy_strip("a ripe avocado split open", "avocado") == "a ripe split open"


@needs_spacy
def test_spacy_genuine_compound_keeps_its_determiner():
    assert spacy_strip("A photo of a cactus garden in the afternoon sun", "cactus") == \
        "A photo of a garden in the afternoon sun"


@needs_spacy
def test_spacy_head_noun_fallback_for_multiword_names():
    assert spacy_strip("The bouquet sits on a wooden table", "bouquet of flowers") == \
        "sits on a wooden table"


@needs_spacy
def test_spacy_pronouns_are_not_resolved():
    # Known limit: no coreference; pronouns referring to the subject stay.
    assert spacy_strip("A girl wearing her favorite hat in the rain", "girl") == \
        "wearing her favorite hat in the rain"
    assert spacy_strip(
        "An ink sketch of a motorcycle with detailed mechanical parts, emphasizing its sleek design",
        "motorcycle") == "An ink sketch of with detailed mechanical parts, emphasizing its sleek design"


@needs_spacy
def test_spacy_coordination_keeps_conjunction():
    # Known limit, documented.
    assert spacy_strip("A kitten and a puppy playing in the yard", "kitten") == \
        "and a puppy playing in the yard"


@needs_spacy
def test_spacy_style_prompt_keeps_scaffold():
    # Style references are out of scope for PF; behavior is documented.
    assert spacy_strip("A cityscape in the style of watercolor sketch", "watercolor sketch") == \
        "A cityscape in the style of"


@needs_spacy
def test_spacy_unmatched_name_returns_prompt():
    assert spacy_strip("a photo of a dog", "cat") == "a photo of a dog"
    # A mention by a different noun ("corgi" for "corgi dog") is not found.
    assert spacy_strip("A photo of a corgi running on the beach", "corgi dog") == \
        "A photo of a corgi running on the beach"


@needs_spacy
def test_spacy_readme_example():
    assert strip_subject_from_prompt(
        "Two vintage motorcycles parked on a lonely desert road", "motorcycle",
        stripper="spacy") == "parked on a lonely desert road"


# --- MaSC: the stripper option and its result fields --------------------------

def test_masc_rejects_unknown_stripper_before_loading_a_model():
    with pytest.raises(ValueError, match="stripper"):
        MaSC(stripper="gemini")


def test_masc_spacy_without_extra_fails_before_loading_a_model(monkeypatch):
    no_spacy(monkeypatch)
    with pytest.raises(ImportError, match=r"masc-metric\[spacy\]"):
        MaSC(stripper="spacy")


def _scorer(stripper: str, stripper_model: str | None = None) -> MaSC:
    """A MaSC without a backbone: enough for the PF prompt logic."""
    scorer = MaSC.__new__(MaSC)
    scorer.stripper, scorer.stripper_model = stripper, stripper_model
    return scorer


@pytest.mark.parametrize(("stripper", "object_name", "text", "stripped"), [
    ("regex", "kitten", "A photo of in a basket", True),
    ("regex", "puppy", "A photo of a kitten in a basket", False),  # not found
    ("regex", None, "A photo of a  kitten in a basket", False),
    ("regex", " ", "A photo of a  kitten in a basket", False),
    ("none", "kitten", "A photo of a  kitten in a basket", False),
])
def test_pf_prompt_and_result_fields(stripper, object_name, text, stripped):
    out, extras = _scorer(stripper)._pf_prompt("A photo of a  kitten in a basket", object_name)
    assert out == text
    assert extras == {
        "prompt_stripped": stripped,
        "stripper": stripper if object_name and object_name.strip() else None,
        "stripper_model": None,
    }


@needs_spacy
def test_pf_prompt_records_spacy_pipeline():
    scorer = _scorer("spacy", SpacySubjectStripper.load().model_name)
    out, extras = scorer._pf_prompt("Two kittens sleeping in a basket", "kitten")
    assert out == "sleeping in a basket"
    assert extras == {"prompt_stripped": True, "stripper": "spacy",
                      "stripper_model": "en_core_web_sm-3.8.0"}
