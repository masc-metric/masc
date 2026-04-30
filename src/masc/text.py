"""Subject stripping for the PF prompt.

The PF lift in MaSC comes from removing the subject token from the
prompt before encoding it. The subject is invariant across prompts in
single-concept personalization — keeping it in the prompt makes the
text-image cosine partly measure subject identity rather than scene
adherence. Stripping it isolates the *scene* signal, which is what PF
should actually measure.
"""
from __future__ import annotations

import re


def strip_subject_from_prompt(prompt: str, object_name: str) -> str:
    """Remove the canonical subject name from a prompt.

    Tries `(a|an|the) <object>` first, then bare `<object>`. Whitespace
    is collapsed at the end. Object names containing hyphens, spaces, or
    other regex metacharacters are handled via `re.escape`.

    Examples
    --------
    >>> strip_subject_from_prompt(
    ...     "A photo of a kitten chasing a butterfly in a sunny garden",
    ...     "kitten",
    ... )
    'A photo of chasing a butterfly in a sunny garden'
    """
    obj = re.escape(object_name.strip())
    for pattern in (rf"\b(?:a|an|the)\s+{obj}\b", rf"\b{obj}\b"):
        if re.search(pattern, prompt, flags=re.IGNORECASE):
            prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE)
            break
    return re.sub(r"\s+", " ", prompt).strip()
