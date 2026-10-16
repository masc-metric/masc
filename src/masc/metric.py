"""Unified single-pass MaSC metric.

Per (ref_image, out_image, prompt) call:

1. One vision-tower forward on the reference → ref patch tokens.
2. One vision-tower forward on the output    → out patch tokens.
3. One text-tower forward on the (subject-stripped) prompt.

CP and PF both consume those tensors; PF reuses the output's patch tokens,
so no image is encoded twice:

- CP = mean over fg-ref patches of max-cosine to any out patch
       (`masked_maxcos`), with masks downsampled to the patch grid and
       thresholded at 0.5.
- PF = cosine between (a) the output's patches pooled through SigLIP2's
       trained `MultiheadAttentionPoolingHead` with a `key_padding_mask`
       that hides the foreground patches, leaving the BG-only pool, and
       (b) the prompt's pooled text embedding.

NaFlex and fixed-size SigLIP2 checkpoints are both supported. Default
backbone matches the paper: `google/siglip2-so400m-patch16-naflex` at
1024 patches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .text import STRIPPERS, SpacySubjectStripper, strip_subject_from_prompt

DEFAULT_MODEL_ID = "google/siglip2-so400m-patch16-naflex"
DEFAULT_NAFLEX_MAX_PATCHES = 1024
DEFAULT_FG_THRESHOLD = 0.5


def _auto_device() -> str:
    """Pick the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class _Encoding:
    """One image through the vision tower, kept in the layout the pooler needs."""

    hidden: torch.Tensor
    """Last hidden state `[1, N_max, D]`, padded positions included (NaFlex)."""

    valid: torch.Tensor
    """`[N_max]` bool: the positions that hold a real patch."""

    grid: tuple[int, int]
    """Patch grid `(Hp, Wp)`; `valid.sum() == Hp * Wp`."""

    fg: torch.Tensor
    """`[Hp * Wp]` mask weights in `[0, 1]` (float32), row-major over the grid."""

    @property
    def patches(self) -> torch.Tensor:
        """`[Hp * Wp, D]` patch tokens (SigLIP2 has no CLS token)."""
        return self.hidden[0][self.valid]


@dataclass
class MaSCResult:
    """Result of a single MaSC scoring call."""

    cp: float
    """Concept Preservation score in [-1, 1] (higher = better identity match)."""

    pf: float
    """Prompt Following score in [-1, 1] (higher = better scene adherence).

    NaN if no background patches remained after masking (the entire
    output was foreground)."""

    extras: dict = field(default_factory=dict)
    """Diagnostics: patch counts, fg fractions, stripped prompt, etc."""


class MaSC:
    """Single-pass MaSC scorer over a frozen SigLIP2 backbone.

    Parameters
    ----------
    model_id
        HuggingFace model id. Default: `google/siglip2-so400m-patch16-naflex`.
        Any SigLIP2 checkpoint is supported (fixed-size or NaFlex).
    device
        `"cuda"`, `"mps"`, `"cpu"`, or a specific torch device string.
        If `None` (default), picks the best available: cuda > mps > cpu.
    naflex_max_patches
        Patch budget for NaFlex variants. Ignored for fixed-size
        checkpoints. Default 1024 (32x32 grid for square inputs).
    fg_threshold
        Threshold applied to the patch-grid-downsampled mask to decide
        which patches are foreground. Default 0.5. As in the paper's code,
        CP keeps the reference patches whose weight is above it and PF hides
        the output patches whose weight is at or above it.
    dtype
        Optional torch dtype to cast the model to (e.g. `torch.bfloat16`).
        Default: keep the checkpoint's native dtype.
    stripper
        How `score(..., object_name=...)` removes the subject from the PF
        prompt: `"regex"` (default; the paper's stripper), `"spacy"`
        (syntax-aware; needs the `spacy` extra and `en_core_web_sm`, see
        `masc.text.SpacySubjectStripper`), or `"none"` (keep the prompt).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        naflex_max_patches: int = DEFAULT_NAFLEX_MAX_PATCHES,
        fg_threshold: float = DEFAULT_FG_THRESHOLD,
        dtype: Optional[torch.dtype] = None,
        stripper: str = "regex",
    ) -> None:
        if stripper not in STRIPPERS:
            raise ValueError(f"stripper must be one of {STRIPPERS}, got {stripper!r}")
        self.stripper = stripper
        # Loading spaCy here fails before the backbone download, not mid-evaluation.
        self.stripper_model = (
            SpacySubjectStripper.load().model_name if stripper == "spacy" else None
        )
        self.model_id = model_id
        self.device = device if device is not None else _auto_device()
        self.fg_threshold = float(fg_threshold)
        self.is_naflex = "naflex" in model_id.lower()
        self.naflex_max_patches = int(naflex_max_patches)

        self.processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        if dtype is not None:
            model = model.to(dtype=dtype)
        self.model = model.eval().to(self.device)

        vision_cfg = self.model.vision_model.config
        self.patch_size = int(vision_cfg.patch_size)
        # Native input size for fixed-size checkpoints; ignored for NaFlex.
        self.input_size = int(getattr(vision_cfg, "image_size", 0) or 0)

    # --------------- public API ---------------

    @torch.inference_mode()
    def score(
        self,
        ref_image: Image.Image,
        ref_mask: np.ndarray,
        out_image: Image.Image,
        out_mask: np.ndarray,
        prompt: str,
        object_name: Optional[str] = None,
    ) -> MaSCResult:
        """Score one (ref, out, prompt) sample.

        Parameters
        ----------
        ref_image, out_image
            PIL images. Any size; the processor handles resizing /
            patchification.
        ref_mask, out_mask
            Binary `np.ndarray` masks the same H, W as their image (or
            any size — they are bilinearly downsampled to the patch
            grid). Values are taken to be in `[0, 255]` (uint8) or
            `[0, 1]` (float). Foreground = non-zero.
        prompt
            The full prompt used to generate `out_image`.
        object_name
            Canonical subject name to strip from `prompt` before text
            encoding (e.g. `"cat"`), with the constructor's `stripper`.
            If `None`, the prompt is used as-is. Stripping is
            recommended; the paper's PF numbers use it.

        Returns
        -------
        MaSCResult
            Its `extras` record the encoded text (`prompt_text`), whether
            the subject was found and removed (`prompt_stripped`), the
            `stripper` used and, for spaCy, its pipeline (`stripper_model`).
        """
        ref = self._encode_image(ref_image, ref_mask)
        out = self._encode_image(out_image, out_mask)

        cp, cp_extras = self._concept_preservation(ref, out)

        text_prompt, strip_extras = self._pf_prompt(prompt, object_name)
        pf, pf_extras = self._prompt_following(out, text_prompt)

        return MaSCResult(
            cp=cp,
            pf=pf,
            extras={
                "prompt_text": text_prompt,
                **strip_extras,
                **cp_extras,
                **pf_extras,
            },
        )

    def _pf_prompt(self, prompt: str, object_name: Optional[str]) -> tuple[str, dict]:
        """The text the PF head encodes, and how it was obtained."""
        if not (object_name and object_name.strip()):
            return prompt, {"prompt_stripped": False, "stripper": None, "stripper_model": None}
        text = strip_subject_from_prompt(prompt, object_name, stripper=self.stripper)
        return text, {
            # Whitespace-insensitive: True only if the subject was found.
            "prompt_stripped": text.split() != prompt.split(),
            "stripper": self.stripper,
            "stripper_model": self.stripper_model,
        }

    # --------------- vision encoding ---------------

    def _encode_image(self, image: Image.Image, fg_mask: np.ndarray) -> _Encoding:
        """One vision-tower forward; the image processor resizes (and, for
        NaFlex, patchifies) the image exactly as the trained pooler expects."""
        kwargs = dict(images=[image.convert("RGB")], return_tensors="pt")
        if self.is_naflex:
            kwargs["max_num_patches"] = self.naflex_max_patches
        inputs = self.processor.image_processor(**kwargs)
        pixel_values = inputs["pixel_values"].to(self.device)

        if self.is_naflex:
            pixel_attention_mask = inputs["pixel_attention_mask"].to(self.device)
            vis_out = self.model.vision_model(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=inputs["spatial_shapes"].to(self.device),
            )
            Hp, Wp = (int(x) for x in inputs["spatial_shapes"][0].tolist())
            valid = pixel_attention_mask[0] > 0
        else:
            vis_out = self.model.vision_model(pixel_values=pixel_values)
            if self.input_size <= 0 or self.input_size % self.patch_size != 0:
                raise ValueError(
                    f"Bad input_size={self.input_size} for patch_size={self.patch_size}"
                )
            Hp = Wp = self.input_size // self.patch_size
            n = vis_out.last_hidden_state.shape[1]
            valid = torch.ones(n, dtype=torch.bool, device=self.device)

        n_valid = int(valid.sum())
        if n_valid != Hp * Wp:
            raise RuntimeError(f"Valid token count {n_valid} vs Hp*Wp={Hp * Wp}")
        fg = self._downsample_mask(fg_mask, Hp, Wp).flatten()
        return _Encoding(hidden=vis_out.last_hidden_state, valid=valid, grid=(Hp, Wp), fg=fg)

    # --------------- CP head ---------------

    def _concept_preservation(self, ref: _Encoding, out: _Encoding) -> tuple[float, dict]:
        ref_patches = ref.patches
        ref_unit = F.normalize(ref_patches, dim=-1)
        out_unit = F.normalize(out.patches, dim=-1)
        # Per-ref-patch max cosine to any out patch.
        sims = ref_unit @ out_unit.T  # [N_ref, N_out]
        nn_cos_per_ref = sims.max(dim=-1).values  # [N_ref]

        ref_fg = ref.fg.to(ref_patches.dtype)
        ref_in_fg = ref_fg > self.fg_threshold
        n_fg = int(ref_in_fg.sum())
        if n_fg == 0:
            return float("nan"), {
                "cp_n_ref_fg": 0,
                "cp_n_ref_total": int(ref_unit.shape[0]),
                "cp_fg_frac_ref": float(ref_fg.mean()),
            }
        score = float(nn_cos_per_ref[ref_in_fg].mean())
        return score, {
            "cp_n_ref_fg": n_fg,
            "cp_n_ref_total": int(ref_unit.shape[0]),
            "cp_fg_frac_ref": float(ref_fg.mean()),
        }

    # --------------- PF head ---------------

    def _prompt_following(self, out: _Encoding, prompt: str) -> tuple[float, dict]:
        """BG-pooled image cosine vs pooled text embedding.

        Reuses the output's vision-tower hidden state from the CP branch: the
        trained attention pooler runs on it with a `key_padding_mask` that
        hides padded slots and foreground patches.
        """
        Hp, Wp = out.grid
        n_max = out.hidden.shape[1]
        fg_real = out.fg >= self.fg_threshold
        valid_idx = out.valid.nonzero(as_tuple=False).squeeze(-1)

        # key_padding_mask: True = exclude. Exclude all padded slots, then
        # additionally exclude foreground patches → BG-only attention pool.
        kpm = torch.ones(n_max, dtype=torch.bool, device=self.device)
        kpm[valid_idx] = fg_real  # True (excluded) where fg
        n_fg = int(fg_real.sum())
        n_bg = int((~kpm).sum())
        if n_bg == 0:
            return float("nan"), {
                "pf_n_fg": n_fg,
                "pf_n_total": Hp * Wp,
                "pf_n_bg_pooled": 0,
            }

        pooled_img = self._pool_with_kpm(self.model.vision_model.head, out.hidden, kpm)  # [1, D]

        # The processor pads to the text tower's trained length (64 tokens
        # for SigLIP2); the pooled text embedding depends on that padding.
        input_ids = self.processor(text=[prompt], padding="max_length", return_tensors="pt")[
            "input_ids"
        ].to(self.device)
        txt_out = self.model.text_model(input_ids=input_ids)
        pooled_txt = txt_out.pooler_output  # [1, D]

        img_n = F.normalize(pooled_img, dim=-1)
        txt_n = F.normalize(pooled_txt, dim=-1)
        score = float((img_n * txt_n).sum(-1)[0].cpu())
        return score, {
            "pf_n_fg": n_fg,
            "pf_n_total": Hp * Wp,
            "pf_n_bg_pooled": n_bg,
        }

    @staticmethod
    def _pool_with_kpm(
        head: torch.nn.Module, patches: torch.Tensor, kpm: torch.Tensor
    ) -> torch.Tensor:
        """SigLIP2's `MultiheadAttentionPoolingHead.forward`, threading a
        `key_padding_mask` through the attention so excluded positions
        are invisible to the learned probe query."""
        probe = head.probe.repeat(patches.shape[0], 1, 1)
        attn_out = head.attention(
            probe, patches, patches, key_padding_mask=kpm.unsqueeze(0)
        )[0]
        residual = attn_out
        hidden = head.layernorm(attn_out)
        hidden = residual + head.mlp(hidden)
        return hidden[:, 0]

    # --------------- mask helpers ---------------

    def _downsample_mask(
        self, fg_mask: np.ndarray, Hp: int, Wp: int
    ) -> torch.Tensor:
        """Bilinear-downsample a 2D mask to (Hp, Wp), values clamped to
        [0, 1]. Accepts uint8 (0..255) or float (0..1)."""
        arr = np.asarray(fg_mask)
        if arr.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {arr.shape}")
        arr = arr.astype(np.float32)
        if arr.max() > 1.0 + 1e-6:
            arr = arr / 255.0
        m = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        m = F.interpolate(m, size=(Hp, Wp), mode="bilinear", align_corners=False)
        return m[0, 0].clamp(0.0, 1.0).to(self.device)
