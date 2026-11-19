"""Synthetic smoke test for the vendored wavelet transform."""

from __future__ import annotations

import torch

from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D


def main() -> int:
    generator = torch.Generator().manual_seed(20261119)
    sample = torch.rand(
        (1, 1, 8, 8, 8),
        dtype=torch.float32,
        generator=generator,
    )
    coefficients = DWT_3D()(sample)
    reconstructed = IDWT_3D()(*coefficients)
    max_error = float((sample - reconstructed).abs().max())
    print(f"wavelet_roundtrip_max_abs_error={max_error:.3e}")
    if max_error > 1e-6:
        raise SystemExit("Wavelet round-trip check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
