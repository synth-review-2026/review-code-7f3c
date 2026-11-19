"""Enhancement predictor with a lightweight full-resolution refinement stage."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import normalization
from .script_util import create_model


def _zero_init(module: nn.Module) -> None:
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)


class FullResolutionRefiner(nn.Module):
    def __init__(self, in_channels: int = 6, width: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, padding=1),
            normalization(width, 8),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1),
            normalization(width, 8),
            nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1),
            normalization(width, 8),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1),
            normalization(width, 8),
            nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1),
            normalization(width, 8),
        )
        self.activation = nn.SiLU()
        self.intensity_head = nn.Conv3d(width, 1, 1)
        self.mask_head = nn.Conv3d(width, 1, 1)
        _zero_init(self.intensity_head)
        _zero_init(self.mask_head)

    def forward(
        self,
        t1n: torch.Tensor,
        t2w: torch.Tensor,
        t2f: torch.Tensor,
        coarse_intensity: torch.Tensor,
        coarse_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = t1n.shape[2:]
        intensity_up = F.interpolate(
            coarse_intensity, size=size, mode="trilinear", align_corners=False
        )
        mask_up = F.interpolate(
            coarse_mask, size=size, mode="trilinear", align_corners=False
        ).clamp(1e-4, 1.0 - 1e-4)
        mask_logit_up = torch.log(mask_up) - torch.log1p(-mask_up)
        features = self.stem(
            torch.cat(
                [t1n, t2w, t2f, intensity_up, mask_up, intensity_up * mask_up],
                dim=1,
            )
        )
        features = features + self.block1(features)
        features = features + self.block2(features)
        features = self.activation(features)
        intensity_delta = 0.25 * torch.tanh(self.intensity_head(features))
        refined_intensity = torch.clamp(intensity_up + intensity_delta, 0.0, 1.0)
        refined_mask_logit = mask_logit_up + self.mask_head(features)
        return refined_intensity, refined_mask_logit


class EnhancementPredictor(nn.Module):
    """Coarse wavelet predictor plus full-resolution intensity/mask refinement."""

    def __init__(self, output_mode: str = "soft_mask"):
        super().__init__()
        if output_mode not in (
            "intensity",
            "soft_mask",
            "offset_mask",
            "smooth_mask",
        ):
            raise ValueError(f"Unknown enhancement output mode: {output_mode}")
        self.output_mode = output_mode
        self.base = create_model(
            image_size=64,
            num_channels=32,
            num_res_blocks=2,
            channel_mult=(1, 2, 2, 4),
            learn_sigma=False,
            class_cond=False,
            use_checkpoint=False,
            attention_resolutions="",
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            dropout=0.0,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
            dims=3,
            num_groups=16,
            in_channels=24,
            out_channels=3,
            bottleneck_attention=False,
            resample_2d=False,
            additive_skips=False,
            use_freq=False,
        )
        self.refiner = FullResolutionRefiner()

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        base_device = next(self.base.parameters()).device
        self.base.devices = [base_device, base_device]
        return result

    def forward(
        self,
        cond_dwt: torch.Tensor,
        timesteps: torch.Tensor,
        t1n: torch.Tensor | None = None,
        t2w: torch.Tensor | None = None,
        t2f: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.base(cond_dwt, timesteps)
        coarse_intensity = torch.sigmoid(logits[:, 0:1])
        coarse_mask = torch.sigmoid(logits[:, 1:2])
        uncertainty = torch.exp(0.5 * logits[:, 2:3].clamp(-5.0, 5.0))
        if t1n is None or t2w is None or t2f is None:
            intensity = coarse_intensity
            mask_logit = logits[:, 1:2]
        else:
            intensity, mask_logit = self.refiner(
                t1n,
                t2w,
                t2f,
                coarse_intensity,
                coarse_mask,
            )
        mask = torch.sigmoid(mask_logit)
        if self.output_mode == "intensity":
            enhancement = intensity
        elif self.output_mode == "soft_mask":
            enhancement = intensity * mask
        elif self.output_mode == "smooth_mask":
            enhancement = intensity * F.avg_pool3d(
                mask,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            enhancement = intensity * (0.2 + 0.8 * mask)
        return {
            "enhancement": enhancement.clamp(0.0, 1.0),
            "intensity": intensity,
            "mask_logit": mask_logit,
            "mask": mask,
            "coarse_intensity": coarse_intensity,
            "coarse_mask": coarse_mask,
            "uncertainty": uncertainty,
            "base_logits": logits,
        }


def select_enhancement(
    outputs: dict[str, torch.Tensor],
    mode: str,
) -> torch.Tensor:
    intensity = outputs["intensity"]
    mask = outputs["mask"]
    if mode == "intensity":
        return intensity
    if mode in ("soft_mask", "product"):
        return intensity * mask
    if mode == "smooth_mask":
        return intensity * F.avg_pool3d(
            mask,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    if mode == "offset_mask":
        return intensity * (0.2 + 0.8 * mask)
    if mode == "mask":
        return mask
    raise ValueError(f"Unknown enhancement mode: {mode}")
