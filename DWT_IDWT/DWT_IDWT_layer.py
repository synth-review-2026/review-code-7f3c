"""Orthogonal 3D Haar DWT/IDWT layers for NCDHW tensors.

The project's original environment provided this small helper package outside
the archived source tree. This implementation keeps the same channel ordering:
LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _split_and_transform(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[dim] % 2:
        raise ValueError(f"Input dimension {dim} must be even, got {x.shape[dim]}")
    even = torch.index_select(
        x,
        dim,
        torch.arange(0, x.shape[dim], 2, device=x.device),
    )
    odd = torch.index_select(
        x,
        dim,
        torch.arange(1, x.shape[dim], 2, device=x.device),
    )
    scale = x.new_tensor(2.0).sqrt()
    return (even + odd) / scale, (even - odd) / scale


def _merge_and_inverse(
    low: torch.Tensor, high: torch.Tensor, dim: int
) -> torch.Tensor:
    scale = low.new_tensor(2.0).sqrt()
    even = (low + high) / scale
    odd = (low - high) / scale
    output = torch.empty(
        (*low.shape[:dim], low.shape[dim] * 2, *low.shape[dim + 1 :]),
        dtype=low.dtype,
        device=low.device,
    )
    output.index_copy_(
        dim,
        torch.arange(0, output.shape[dim], 2, device=low.device),
        even,
    )
    output.index_copy_(
        dim,
        torch.arange(1, output.shape[dim], 2, device=low.device),
        odd,
    )
    return output


class DWT_3D(nn.Module):
    def __init__(self, wavename: str = "haar"):
        super().__init__()
        if wavename.lower() != "haar":
            raise ValueError("Only the Haar wavelet is supported")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.ndim != 5:
            raise ValueError(f"Expected NCDHW input, got shape {tuple(x.shape)}")
        bands = [x]
        for dim in (2, 3, 4):
            next_bands: list[torch.Tensor] = []
            for band in bands:
                low, high = _split_and_transform(band, dim)
                next_bands.extend((low, high))
            bands = next_bands
        return tuple(bands)


class IDWT_3D(nn.Module):
    def __init__(self, wavename: str = "haar"):
        super().__init__()
        if wavename.lower() != "haar":
            raise ValueError("Only the Haar wavelet is supported")

    def forward(self, *coefficients: torch.Tensor) -> torch.Tensor:
        if len(coefficients) == 1 and coefficients[0].shape[1] % 8 == 0:
            packed = coefficients[0]
            group_count = packed.shape[1] // 8
            outputs = [
                self(
                    *[
                        packed[
                            :,
                            group_index * 8 + index : group_index * 8 + index + 1,
                        ]
                        for index in range(8)
                    ]
                )
                for group_index in range(group_count)
            ]
            return torch.cat(outputs, dim=1)
        if len(coefficients) != 8:
            raise ValueError("IDWT_3D expects eight coefficient tensors")
        bands = list(coefficients)
        for dim in (4, 3, 2):
            merged: list[torch.Tensor] = []
            for index in range(0, len(bands), 2):
                merged.append(_merge_and_inverse(bands[index], bands[index + 1], dim))
            bands = merged
        return bands[0]
