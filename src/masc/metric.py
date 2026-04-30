"""Unified single-pass MaSC metric.

Per (ref_image, out_image, prompt) call:

1. One vision-tower forward on the reference → ref patch tokens.
2. One vision-tower forward on the output    → out patch tokens.
3. One text-tower forward on the (subject-stripped) prompt.

CP and PF both consume those tensors:

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
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .text import strip_subject_from_prompt

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
        which patches are foreground. Default 0.5.
    dtype
        Optional torch dtype to cast the model to (e.g. `torch.bfloat16`).
        Default: keep the checkpoint's native dtype.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        naflex_max_patches: int = DEFAULT_NAFLEX_MAX_PATCHES,
        fg_threshold: float = DEFAULT_FG_THRESHOLD,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device if device is not None else _auto_device()
        self.fg_threshold = float(fg_threshold)
        self.is_naflex = "naflex" in model_id.lower()
        self.naflex_max_patches = int(naflex_max_patches)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        if dtype is not None:
            model = model.to(dtype=dtype)
        self.model = model.eval().to(self.device)

        vision_cfg = self.model.vision_model.config
        self.patch_size = int(vision_cfg.patch_size)
        self.image_mean = list(self.processor.image_processor.image_mean)
        self.image_std = list(self.processor.image_processor.image_std)
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
            encoding (e.g. `"cat"`). If `None`, the prompt is used
            as-is. Stripping is recommended — it is what produces the
            PF lift over CLIP-T on the same backbone.
        """
        ref_patches, ref_fg, _ = self._encode_image(ref_image, ref_mask)
        out_patches, out_fg, _ = self._encode_image(out_image, out_mask)

        cp, cp_extras = self._concept_preservation(ref_patches, ref_fg, out_patches)

        text_prompt = (
            strip_subject_from_prompt(prompt, object_name) if object_name else prompt
        )
        pf, pf_extras = self._prompt_following(out_image, out_mask, text_prompt)

        return MaSCResult(
            cp=cp,
            pf=pf,
            extras={
                "prompt_text": text_prompt,
                "prompt_stripped": object_name is not None,
                **cp_extras,
                **pf_extras,
            },
        )

    # --------------- vision encoding ---------------

    def _encode_image(
        self, image: Image.Image, fg_mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Return (patch_tokens [N, D], fg_weights [N] in [0, 1], (Hp, Wp))."""
        if self.is_naflex:
            return self._encode_image_naflex(image, fg_mask)
        return self._encode_image_fixed(image, fg_mask)

    def _encode_image_fixed(
        self, image: Image.Image, fg_mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        size = self.input_size
        if size <= 0 or size % self.patch_size != 0:
            raise ValueError(
                f"Bad input_size={size} for patch_size={self.patch_size}"
            )
        img = image.convert("RGB").resize((size, size), Image.BILINEAR)
        t = TF.to_tensor(img)
        t = TF.normalize(t, mean=self.image_mean, std=self.image_std)
        pixel_values = t.unsqueeze(0).to(self.device)

        out = self.model.vision_model(pixel_values=pixel_values)
        patches = out.last_hidden_state[0]  # [N, D] — SigLIP2 has no CLS

        Hp = Wp = size // self.patch_size
        if patches.shape[0] != Hp * Wp:
            raise RuntimeError(
                f"Unexpected token count {patches.shape[0]} vs Hp*Wp={Hp * Wp}"
            )
        fg = self._downsample_mask(fg_mask, Hp, Wp).flatten().to(patches.dtype)
        return patches, fg, (Hp, Wp)

    def _encode_image_naflex(
        self, image: Image.Image, fg_mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        inputs = self.processor.image_processor(
            images=image.convert("RGB"),
            max_num_patches=self.naflex_max_patches,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        pixel_attention_mask = inputs["pixel_attention_mask"].to(self.device)
        spatial_shapes = inputs["spatial_shapes"].to(self.device)
        Hp, Wp = (int(x) for x in inputs["spatial_shapes"][0].tolist())

        out = self.model.vision_model(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        valid = pixel_attention_mask[0] > 0
        patches = out.last_hidden_state[0][valid]  # [Hp*Wp, D]
        if patches.shape[0] != Hp * Wp:
            raise RuntimeError(
                f"Valid token count {patches.shape[0]} vs Hp*Wp={Hp * Wp}"
            )
        fg = self._downsample_mask(fg_mask, Hp, Wp).flatten().to(patches.dtype)
        return patches, fg, (Hp, Wp)

    # --------------- CP head ---------------

    def _concept_preservation(
        self, ref_patches: torch.Tensor, ref_fg: torch.Tensor, out_patches: torch.Tensor
    ) -> tuple[float, dict]:
        ref_unit = F.normalize(ref_patches, dim=-1)
        out_unit = F.normalize(out_patches, dim=-1)
        # Per-ref-patch max cosine to any out patch.
        sims = ref_unit @ out_unit.T  # [N_ref, N_out]
        nn_cos_per_ref = sims.max(dim=-1).values  # [N_ref]

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

    def _prompt_following(
        self, out_image: Image.Image, out_mask: np.ndarray, prompt: str
    ) -> tuple[float, dict]:
        """BG-pooled image cosine vs pooled text embedding.

        We can't directly reuse the patch tensor from `_encode_image`
        because the vision pooler needs the full padded sequence
        (NaFlex) or the unprojected last hidden state with its layout
        intact. We rerun the vision tower here to get both the pooler
        head and a layout that the head expects. The text encode is one
        forward; the vision recompute is the same forward already done
        once for CP, so on a real eval loop you can cache image
        features per-image upstream and amortize.
        """
        # Re-run vision with the full processor output so we have the
        # padded tensor structure the pooler head was trained on.
        kwargs = dict(images=[out_image.convert("RGB")], return_tensors="pt")
        if self.is_naflex:
            kwargs["max_num_patches"] = self.naflex_max_patches
        try:
            inputs = self.processor.image_processor(**kwargs)
        except TypeError:
            kwargs.pop("max_num_patches", None)
            inputs = self.processor.image_processor(**kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.is_naflex:
            vis_out = self.model.vision_model(
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
            )
            Hp, Wp = (int(x) for x in inputs["spatial_shapes"][0].tolist())
            valid_mask = inputs["pixel_attention_mask"][0].bool()
        else:
            vis_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            N = vis_out.last_hidden_state.shape[1]
            Hp = Wp = int(round(N**0.5))
            if Hp * Wp != N:
                raise RuntimeError(f"Non-square fixed grid (n={N}).")
            valid_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        patches = vis_out.last_hidden_state  # [1, N_max, D]
        N_max = patches.shape[1]

        fg_real = self._downsample_mask(out_mask, Hp, Wp).flatten() > self.fg_threshold
        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() != Hp * Wp:
            raise RuntimeError(
                f"valid count {valid_idx.numel()} != Hp*Wp {Hp * Wp}"
            )

        # key_padding_mask: True = exclude. Exclude all padded slots, then
        # additionally exclude foreground patches → BG-only attention pool.
        kpm = torch.ones(N_max, dtype=torch.bool, device=self.device)
        kpm[valid_idx] = fg_real  # True (excluded) where fg
        n_fg = int(fg_real.sum())
        n_bg = int((~kpm).sum())
        if n_bg == 0:
            return float("nan"), {
                "pf_n_fg": n_fg,
                "pf_n_total": Hp * Wp,
                "pf_n_bg_pooled": 0,
            }

        pooled_img = self._pool_with_kpm(
            self.model.vision_model.head, patches, kpm
        )  # [1, D]

        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        txt_out = self.model.text_model(input_ids=text_inputs["input_ids"])
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
