"""Subject stripping for the PF prompt.

The PF lift in MaSC comes from removing the subject token from the
prompt before encoding it. The subject is invariant across prompts in
single-concept personalization — keeping it in the prompt makes the
text-image cosine partly measure subject identity rather than scene
adherence. Stripping it isolates the *scene* signal, which is what PF
should actually measure.

Three strippers are available through `strip_subject_from_prompt(...,
stripper=...)` and `MaSC(stripper=...)`:

- ``"regex"`` (default): the stripper behind every MaSC number in the paper.
  Removes ``(a|an|the) <object>``, or the bare ``<object>`` when no article
  precedes it. Needs no extra dependency.
- ``"spacy"``: syntax-aware (optional extra, ``pip install
  "masc-metric[spacy]"`` plus ``python -m spacy download en_core_web_sm``).
  See `SpacySubjectStripper` for the exact rule and its known limits.
- ``"none"``: returns the prompt unchanged.
"""
from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

STRIPPERS = ("regex", "spacy", "none")
DEFAULT_SPACY_MODEL = "en_core_web_sm"
# The en_core_web_sm version the spaCy stripper was developed and tested with;
# the DreamBench++ stripper comparison in the README used it too.
REFERENCE_SPACY_MODEL_VERSION = "3.8.0"

# spaCy pipelines by model name, so each loads once per process.
_PIPELINES: dict[str, Language] = {}


def strip_subject_from_prompt(
    prompt: str,
    object_name: str,
    stripper: str = "regex",
    *,
    spacy_model: str = DEFAULT_SPACY_MODEL,
) -> str:
    """Remove the canonical subject name from a prompt.

    Parameters
    ----------
    prompt
        The generation prompt.
    object_name
        Canonical subject name, e.g. ``"kitten"`` or ``"piggy bank"``. An
        empty name matches nothing.
    stripper
        ``"regex"`` (default, the paper's stripper), ``"spacy"`` or ``"none"``.
    spacy_model
        spaCy pipeline used by ``stripper="spacy"``.

    Returns
    -------
    str
        The prompt without the subject, whitespace collapsed; with
        ``stripper="none"``, the prompt as given.

    Examples
    --------
    >>> strip_subject_from_prompt(
    ...     "A photo of a kitten chasing a butterfly in a sunny garden",
    ...     "kitten",
    ... )
    'A photo of chasing a butterfly in a sunny garden'
    """
    if stripper not in STRIPPERS:
        raise ValueError(f"Unknown stripper {stripper!r}; expected one of {STRIPPERS}")
    if stripper == "none":
        return prompt
    if not object_name.strip():
        return _collapse_whitespace(prompt)
    if stripper == "regex":
        return _strip_regex(prompt, object_name)
    return SpacySubjectStripper.load(spacy_model).strip(prompt, object_name)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_regex(prompt: str, object_name: str) -> str:
    """Tries `(a|an|the) <object>` first, then bare `<object>`. Whitespace
    is collapsed at the end. Object names containing hyphens, spaces, or
    other regex metacharacters are handled via `re.escape`."""
    obj = re.escape(object_name.strip())
    for pattern in (rf"\b(?:a|an|the)\s+{obj}\b", rf"\b{obj}\b"):
        if re.search(pattern, prompt, flags=re.IGNORECASE):
            prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE)
            break
    return _collapse_whitespace(prompt)


# Left dependents of the subject's head noun that belong to its noun phrase.
_PRE_MODIFIER_DEPS = frozenset(
    {"det", "predet", "poss", "nummod", "quantmod", "amod", "compound", "nmod", "nn"}
)
# Relations by which a mention can hang off a later word as a pre-modifier.
_MODIFIER_OF_LATER_DEPS = frozenset({"amod", "compound", "nmod", "nn"})
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_LEADING_PUNCT = re.compile(r"^[\s,;:]+")


class SpacySubjectStripper:
    """Syntax-aware subject stripper.

    Rule, per prompt:

    1. Mentions. Parse prompt and object name with the same spaCy pipeline.
       A mention is a token span whose lower-cased lemmas equal the object
       name's lemmas, or whose lower-cased words equal the name's words, so
       inflected forms match ("kittens", "Kitten", "cherry blossoms").
       If the name has several words and no full mention exists, a single
       NOUN/PROPN token whose lemma equals the lemma of the name's syntactic
       head also counts ("the bouquet" for "bouquet of flowers"). Every
       mention in the prompt is handled, not only the first.
    2. Noun phrase. For each mention, take its head token and remove the
       mention together with the head's left dependents of the noun phrase
       (determiner, possessive, numeral, adjective, compound, including
       their own modifiers): "a fluffy little kitten" is removed whole. If
       the mention is itself a possessor ("the dog's ball"), its possessive
       marker goes too, leaving "ball". If the parser reads the mention as a
       modifier of a later word (the frequent mis-parse of "a dolphin
       swimming" as the compound noun "dolphin swimming"), the determiner and
       adjectives before the mention are removed with it, unless the later
       word directly follows and is a noun in its own right that does not
       end in "-ing" ("a motorcycle rider" becomes "a rider").
    3. Everything else is kept, including right-hand attachments that
       carry scene content: prepositional phrases ("on a beach"),
       participial and relative clauses ("chasing a butterfly"), and
       relations to other objects ("next to a cat").

    Known limits (documented, not handled):

    - Pronouns that refer to the subject ("its", "her", "they") are kept.
      The pipeline has no coreference component, so the stripper does not
      resolve them.
    - Coordination keeps the conjunction: "a kitten and a puppy" becomes
      "and a puppy".
    - A mention that uses neither the name nor its head noun (a synonym,
      "the feline" for "kitten", or "a corgi" for "corgi dog") is not
      found; the prompt is returned unchanged, as with the regex stripper.
    - Style references (DreamBench++ `style_*`) have no object to strip;
      "in the style of <name>" loses the name and keeps the scaffold. MaSC
      excludes style prompts from PF.
    - Parses come from a statistical model; a mis-parse can attach a
      modifier to the wrong head. Results are deterministic for a fixed
      spaCy and model version; loading an en_core_web_sm other than
      `REFERENCE_SPACY_MODEL_VERSION` warns.
    """

    def __init__(self, nlp: Language) -> None:
        self.nlp = nlp

    @classmethod
    def load(cls, model: str = DEFAULT_SPACY_MODEL) -> SpacySubjectStripper:
        """A stripper on the spaCy pipeline `model`, loaded once per process."""
        if model not in _PIPELINES:
            try:
                import spacy
            except ImportError as e:
                raise ImportError(
                    "The spaCy stripper needs the optional extra: "
                    "pip install 'masc-metric[spacy]' && python -m spacy download "
                    f"{model}"
                ) from e
            try:
                nlp = spacy.load(model, disable=["ner"])
            except OSError as e:
                raise OSError(
                    f"spaCy model {model!r} is not installed: python -m spacy download {model}"
                ) from e
            version = nlp.meta.get("version")
            if model == DEFAULT_SPACY_MODEL and version != REFERENCE_SPACY_MODEL_VERSION:
                warnings.warn(
                    f"Loaded {model} {version}; the spaCy stripper was tested with "
                    f"{REFERENCE_SPACY_MODEL_VERSION}, so stripped prompts can differ "
                    "from the paper's.",
                    stacklevel=2,
                )
            _PIPELINES[model] = nlp
        return cls(_PIPELINES[model])

    @property
    def model_name(self) -> str:
        meta = self.nlp.meta
        return f"{meta.get('lang', '?')}_{meta.get('name', '?')}-{meta.get('version', '?')}"

    def find_mentions(self, doc: Doc, object_name: str) -> list[tuple[int, int]]:
        """Token spans [start, end) of subject mentions in `doc`."""
        name = self.nlp(object_name.strip())
        n = len(name)
        if n == 0:
            return []
        name_lemmas = [t.lemma_.lower() for t in name]
        name_words = [t.lower_ for t in name]
        spans = []
        i = 0
        while i + n <= len(doc):
            window = doc[i : i + n]
            if ([t.lemma_.lower() for t in window] == name_lemmas
                    or [t.lower_ for t in window] == name_words):
                spans.append((i, i + n))
                i += n
            else:
                i += 1
        if spans or n == 1:
            return spans
        root = name[:].root
        root_lemma = root.lemma_.lower()
        return [(t.i, t.i + 1) for t in doc
                if t.pos_ in ("NOUN", "PROPN") and t.lemma_.lower() == root_lemma]

    def _removal_indices(self, doc: Doc, spans: list[tuple[int, int]]) -> set[int]:
        remove: set[int] = set()
        for start, end in spans:
            span = doc[start:end]
            head = span.root
            left = start
            right = max(end, head.i + 1)
            for child in head.lefts:
                if child.dep_ in _PRE_MODIFIER_DEPS:
                    left = min(left, child.left_edge.i)
            # A mention parsed as a modifier of a later word ("a dolphin
            # swimming" read as the compound noun "dolphin swimming") leaves its
            # determiner and adjectives attached to that word. They belong to
            # the mention, unless the later word follows directly and is a noun
            # in its own right ("a motorcycle rider" keeps "a").
            outer = head.head
            own_noun = (outer.i == right and outer.pos_ in ("NOUN", "PROPN")
                        and not outer.lower_.endswith("ing"))
            node = head
            while (not own_noun and node.dep_ in _MODIFIER_OF_LATER_DEPS
                   and node.head.i > node.i):
                node = node.head
                for child in node.lefts:
                    if child.dep_ in _PRE_MODIFIER_DEPS and child.right_edge.i < start:
                        left = min(left, child.left_edge.i)
            # The noun phrase's left part is contiguous: remove the whole range,
            # including commas between stacked adjectives ("a small, fluffy kitten").
            remove.update(range(left, right))
            if head.dep_ == "poss":
                remove.update(c.i for c in head.rights if c.dep_ == "case")
        return remove

    def strip(self, prompt: str, object_name: str) -> str:
        """`prompt` without the mentions of `object_name` and their noun-phrase
        pre-modifiers, whitespace collapsed."""
        doc = self.nlp(prompt)
        spans = self.find_mentions(doc, object_name)
        if not spans:
            return _collapse_whitespace(prompt)
        remove = self._removal_indices(doc, spans)
        kept = "".join(t.text_with_ws for t in doc if t.i not in remove)
        kept = re.sub(r"\s+", " ", kept)
        kept = _SPACE_BEFORE_PUNCT.sub(r"\1", kept).strip()
        return _LEADING_PUNCT.sub("", kept)
