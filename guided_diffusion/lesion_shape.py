"""Dedicated target-free 3D lesion-shape predictor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import normalization
from .script_util import create_model


class ShapeRefiner(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(4, width, 3, padding=1),
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
        self.out = nn.Conv3d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        t1n: torch.Tensor,
        t2w: torch.Tensor,
        t2f: torch.Tensor,
        coarse_logit: torch.Tensor,
    ) -> torch.Tensor:
        coarse_up = F.interpolate(
            coarse_logit,
            size=t1n.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        coarse_prob = torch.sigmoid(coarse_up)
        features = self.stem(
            torch.cat([t1n, t2w, t2f, coarse_prob], dim=1)
        )
        features = features + self.block1(features)
        features = features + self.block2(features)
        features = self.activation(features)
        return coarse_up + 0.5 * torch.tanh(self.out(features))


class LesionShapePredictor(nn.Module):
    """Coarse wavelet U-Net plus full-resolution boundary refinement."""

    def __init__(self):
        super().__init__()
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
            out_channels=1,
            bottleneck_attention=False,
            resample_2d=False,
            additive_skips=False,
            use_freq=False,
        )
        self.refiner = ShapeRefiner(width=16)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = next(self.base.parameters()).device
        self.base.devices = [device, device]
        return result

    def forward(
        self,
        cond_dwt: torch.Tensor,
        timesteps: torch.Tensor,
        t1n: torch.Tensor,
        t2w: torch.Tensor,
        t2f: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        coarse_logit = self.base(cond_dwt, timesteps)
        mask_logit = self.refiner(t1n, t2w, t2f, coarse_logit)
        return {
            "coarse_logit": coarse_logit,
            "mask_logit": mask_logit,
            "mask": torch.sigmoid(mask_logit),
        }
