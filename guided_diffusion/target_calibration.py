"""Training-label calibration for T1c enhancement targets.

Only the target sequence is transformed. T1n remains normalized using its own
statistics, so no target information enters the inference input. This module
must not be used to construct model input channels.
"""

from __future__ import annotations

import numpy as np


def _robust_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Robustly fit y ~= slope*x + intercept on pre-filtered tissue samples."""
    maximum_samples = 200_000
    if len(x) > maximum_samples:
        stride = int(np.ceil(len(x) / maximum_samples))
        x = x[::stride]
        y = y[::stride]

    keep = np.ones(len(x), dtype=bool)
    slope, intercept = 1.0, 0.0
    for _ in range(3):
        if keep.sum() < 1000:
            break
        slope, intercept = np.polyfit(x[keep], y[keep], deg=1)
        residual = y - (slope * x + intercept)
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center))
        if not np.isfinite(mad) or mad <= 1e-8:
            break
        threshold = 3.0 * 1.4826 * mad
        new_keep = np.abs(residual - center) <= threshold
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0:
        slope, intercept = 1.0, float(np.median(y - x))
    return float(slope), float(intercept), int(keep.sum())


def calibrate_t1c_target(
    t1n: np.ndarray,
    t1c: np.ndarray,
    nonenhancing_margin: float = 0.10,
    min_reliable_slope: float = 0.5,
    max_reliable_slope: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Map a normalized T1c label into the independently normalized T1n space.

    The robust fit uses tissue-like voxels that are not strongly brighter in
    T1c than T1n. T1c is used only to construct the supervised target.
    """
    t1n = np.asarray(t1n, dtype=np.float32)
    t1c = np.asarray(t1c, dtype=np.float32)
    if t1n.shape != t1c.shape:
        raise ValueError(f"T1/T1CE shape mismatch: {t1n.shape} vs {t1c.shape}")

    finite = np.isfinite(t1n) & np.isfinite(t1c)
    tissue = finite & (t1n > 0.02) & (t1c > 0.0)
    n_values = t1n[tissue]
    if len(n_values) >= 1000:
        n_low, n_high = np.percentile(n_values, (2.0, 98.0))
        tissue &= (t1n >= n_low) & (t1n <= n_high)

    initial_enhancement = t1c - t1n
    fit_mask = tissue & (initial_enhancement <= nonenhancing_margin)
    if fit_mask.sum() < 1000:
        fit_mask = tissue
    if fit_mask.sum() < 100:
        calibrated = np.clip(t1c, 0.0, 1.0)
        raw_slope, slope, intercept, fit_count = 1.0, 1.0, 0.0, int(fit_mask.sum())
        calibration_mode = "identity_fallback"
    else:
        raw_slope, raw_intercept, fit_count = _robust_affine(
            t1c[fit_mask], t1n[fit_mask]
        )
        if min_reliable_slope <= raw_slope <= max_reliable_slope:
            slope, intercept = raw_slope, raw_intercept
            calibration_mode = "robust_affine"
        else:
            # An unconstrained affine fit can collapse the T1CE label to an
            # almost constant volume. Bound the tissue contrast ratio and
            # re-anchor the normal-tissue median.
            slope = float(np.clip(raw_slope, min_reliable_slope, max_reliable_slope))
            intercept = float(
                np.median(t1n[fit_mask]) - slope * np.median(t1c[fit_mask])
            )
            calibration_mode = "bounded_slope_median_offset"
        calibrated = np.clip(slope * t1c + intercept, 0.0, 1.0)

    brain_support = (t1n > 0.0) | (t1c > 0.0)
    calibrated = np.where(finite & brain_support, calibrated, 0.0).astype(
        np.float32
    )
    enhancement = np.clip(calibrated - t1n, 0.0, 1.0).astype(np.float32)
    enhancement[~(tissue & brain_support)] = 0.0
    stats = {
        "slope": float(slope),
        "raw_slope": float(raw_slope),
        "intercept": float(intercept),
        "calibration_mode": calibration_mode,
        "slope_was_reliable": bool(
            min_reliable_slope <= raw_slope <= max_reliable_slope
        ),
        "fit_voxels": fit_count,
        "target_min": float(calibrated.min()),
        "target_max": float(calibrated.max()),
        "target_std": float(calibrated[brain_support].std()),
        "background_max": float(calibrated[~brain_support].max())
        if (~brain_support).any()
        else 0.0,
        "enhancement_mean": float(enhancement[tissue].mean()) if tissue.any() else 0.0,
        "enhancement_positive_ratio": (
            float((enhancement[tissue] > 0.0).mean()) if tissue.any() else 0.0
        ),
    }
    return calibrated, stats


def apply_t1c_calibration(
    t1c: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    t1c = np.asarray(t1c, dtype=np.float32)
    calibrated = np.clip(slope * t1c + intercept, 0.0, 1.0).astype(np.float32)
    calibrated[t1c <= 0.0] = 0.0
    return calibrated
