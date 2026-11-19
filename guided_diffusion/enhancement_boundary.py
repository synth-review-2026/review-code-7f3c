"""Boundary-aware enhancement predictor built on the V8 architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .enhancement_predictor import EnhancementPredictor, FullResolutionRefiner


class BoundaryRefiner(FullResolutionRefiner):
    """Full-resolution refiner with a learned boundary-transition gate."""

    def __init__(self, width: int = 8):
        super().__init__(in_channels=6, width=width)
        self.boundary_head = nn.Conv3d(width, 1, 1)
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.constant_(self.boundary_head.bias, -2.0)

    def forward(
        self,
        t1n: torch.Tensor,
        t2w: torch.Tensor,
        t2f: torch.Tensor,
        coarse_intensity: torch.Tensor,
        coarse_mask: torch.Tensor,
    ):
        size = t1n.shape[2:]
        intensity_up = F.interpolate(
            coarse_intensity,
            size=size,
            mode="trilinear",
            align_corners=False,
        )
        mask_up = F.interpolate(
            coarse_mask,
            size=size,
            mode="trilinear",
            align_corners=False,
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
        refined_intensity = torch.clamp(
            intensity_up + intensity_delta,
            0.0,
            1.0,
        )
        refined_mask_logit = mask_logit_up + self.mask_head(features)
        boundary_gate = torch.sigmoid(self.boundary_head(features))
        return refined_intensity, refined_mask_logit, boundary_gate


class BoundaryAwareEnhancementPredictor(EnhancementPredictor):
    """V8-compatible predictor with a smooth learned boundary transition."""

    def __init__(self):
        super().__init__(output_mode="soft_mask")
        self.refiner = BoundaryRefiner(width=8)

    def load_v8_state_dict(self, state_dict):
        return self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        cond_dwt: torch.Tensor,
        timesteps: torch.Tensor,
        t1n: torch.Tensor | None = None,
        t2w: torch.Tensor | None = None,
        t2f: torch.Tensor | None = None,
    ):
        logits = self.base(cond_dwt, timesteps)
        coarse_intensity = torch.sigmoid(logits[:, 0:1])
        coarse_mask = torch.sigmoid(logits[:, 1:2])
        uncertainty = torch.exp(0.5 * logits[:, 2:3].clamp(-5.0, 5.0))
        if t1n is None or t2w is None or t2f is None:
            intensity = coarse_intensity
            mask_logit = logits[:, 1:2]
            boundary_gate = torch.zeros_like(coarse_mask)
        else:
            intensity, mask_logit, boundary_gate = self.refiner(
                t1n,
                t2w,
                t2f,
                coarse_intensity,
                coarse_mask,
            )
        mask = torch.sigmoid(mask_logit)
        smooth_mask = F.avg_pool3d(
            mask,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        transition = (
            (1.0 - boundary_gate) * mask
            + boundary_gate * smooth_mask
        ).clamp(0.0, 1.0)
        enhancement = (intensity * transition).clamp(0.0, 1.0)
        return {
            "enhancement": enhancement,
            "transition": transition,
            "boundary_gate": boundary_gate,
            "intensity": intensity,
            "mask_logit": mask_logit,
            "mask": mask,
            "coarse_intensity": coarse_intensity,
            "coarse_mask": coarse_mask,
            "uncertainty": uncertainty,
            "base_logits": logits,
        }
