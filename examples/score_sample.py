"""End-to-end MaSC example.

Scores one (reference, output, prompt) sample using a real SigLIP2
backbone. Requires:

- A CUDA-capable GPU (or set device="cpu" for a slow run).
- A HuggingFace token with access to the gated `google/siglip2-*`
  checkpoints (`huggingface-cli login`).

Run from the repo root:

    python examples/score_sample.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from masc import MaSC


def main() -> None:
    sample_dir = Path(__file__).parent / "sample_hat"

    ref_image = Image.open(sample_dir / "reference.jpg")
    out_image = Image.open(sample_dir / "output.jpg")
    ref_mask = np.array(Image.open(sample_dir / "reference_mask.png").convert("L"))
    out_mask = np.array(Image.open(sample_dir / "output_mask.png").convert("L"))

    object_name, prompt = (sample_dir / "prompt.txt").read_text().strip().splitlines()

    masc = MaSC()  # auto-picks cuda > mps > cpu; pass device="..." to override
    result = masc.score(
        ref_image=ref_image,
        ref_mask=ref_mask,
        out_image=out_image,
        out_mask=out_mask,
        prompt=prompt,
        object_name=object_name,
    )

    print(f"subject:      {object_name}")
    print(f"prompt:       {prompt}")
    print(f"prompt (PF):  {result.extras['prompt_text']}")
    print(f"CP:           {result.cp:+.4f}")
    print(f"PF:           {result.pf:+.4f}")


if __name__ == "__main__":
    main()
