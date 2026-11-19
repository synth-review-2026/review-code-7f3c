from __future__ import annotations

import os

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from evaluate_segmentation_against_expert_masks import segmentation_metrics


PROJECT = Path(
    os.environ.get("EPG_SYNTH_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
COMBINED = (
    PROJECT / "data/external_validation_combined118_20261110"
)
EXPERT_MASKS = COMBINED / "expert_masks_model_192x224x192"
RUNS = (
    PROJECT
    / "runs/external_validation_118_formal_20261111"
    / "segmentation"
)
OUTPUT = (
    PROJECT
    / "runs/external_validation_118_formal_20261111"
    / "downstream_evaluation"
)
METHODS = {
    "EPG-Synth": "V12-Balanced",
    "WFM-Official": "WFM-Official",
    "WFM-Adapted": "WFM-Adapted (V12 enhancer)",
    "3D Pix2Pix cGAN": "3D Pix2Pix cGAN",
    "Conditional WGAN-GP": "Conditional WGAN-GP",
    "T1C-RFlow": "T1C-RFlow",
}


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj) > 0.5


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int = 42,
    draws: int = 10000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, (draws, values.size))
    means = values[indices].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def bh_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 1.0
    count = len(values)
    for rank in range(count - 1, -1, -1):
        index = int(order[rank])
        running = min(running, values[index] * count / (rank + 1))
        adjusted[index] = running
    return [float(value) for value in adjusted]


def paired_test(
    frame: pd.DataFrame,
    metric: str,
    candidate: str,
    lower_is_better: bool,
) -> dict[str, object]:
    baseline = frame.loc[
        frame["method"] == "EPG-Synth"
    ].set_index("subject")[metric]
    comparison = frame.loc[
        frame["method"] == candidate
    ].set_index("subject")[metric]
    difference = (baseline - comparison).dropna()
    nonzero = difference[difference.abs() > 1e-12]
    pvalue = (
        float(
            wilcoxon(
                nonzero,
                alternative="two-sided",
            ).pvalue
        )
        if len(nonzero) >= 2
        else float("nan")
    )
    low, high = bootstrap_mean_ci(difference.to_numpy())
    if lower_is_better:
        better = int((difference < -1e-12).sum())
        worse = int((difference > 1e-12).sum())
    else:
        better = int((difference > 1e-12).sum())
        worse = int((difference < -1e-12).sum())
    return {
        "candidate": candidate,
        "metric": metric,
        "mean_difference_epg_minus_candidate": float(
            difference.mean()
        ),
        "ci95_low": low,
        "ci95_high": high,
        "median_difference": float(difference.median()),
        "epg_better": better,
        "epg_worse": worse,
        "ties": int((difference.abs() <= 1e-12).sum()),
        "wilcoxon_p": pvalue,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (COMBINED / "manifests/case_registry.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        registry = {
            row["case"]: row for row in csv.DictReader(handle)
        }
    subjects = sorted(registry)
    if len(subjects) != 118:
        raise ValueError(f"Expected 118 cases, found {len(subjects)}")
    rows = []
    for subject_index, subject in enumerate(subjects, start=1):
        expert_image = nib.load(
            str(EXPERT_MASKS / f"{subject}.nii.gz")
        )
        expert = load(EXPERT_MASKS / f"{subject}.nii.gz")
        if not expert.any():
            raise ValueError(f"Empty expert mask for {subject}")
        spacing = np.asarray(
            expert_image.header.get_zooms()[:3], dtype=np.float64
        )
        diagonal_mm = float(
            np.sqrt(
                np.sum(
                    (
                        (np.asarray(expert.shape) - 1) * spacing
                    )
                    ** 2
                )
            )
        )
        for display_name, directory in METHODS.items():
            synthetic = load(
                RUNS
                / directory
                / subject
                / "brats25_1_whole_tumor.nii.gz"
            )
            if synthetic.shape != expert.shape:
                raise ValueError(
                    f"{display_name}/{subject}: shape mismatch"
                )
            raw = segmentation_metrics(expert, synthetic, spacing)
            overlap = bool(np.logical_and(expert, synthetic).any())
            row = {
                "method": display_name,
                "method_directory": directory,
                "subject": subject,
                "center": registry[subject]["center"],
                "site_code": registry[subject]["site_code"],
                "cohort": registry[subject]["cohort"],
                "expert_volume_ml": float(
                    expert.sum() * np.prod(spacing) / 1000.0
                ),
                "expert_slices": int(
                    expert.any(axis=(1, 2)).sum()
                ),
                "detected": bool(synthetic.any()),
                "overlap_detected": overlap,
                **raw,
            }
            row["Dice_zero_filled"] = (
                float(raw["Dice"]) if overlap else 0.0
            )
            row["Dice_overlap"] = (
                float(raw["Dice"]) if overlap else float("nan")
            )
            for metric in (
                "HD95_mm",
                "ASSD_mm",
                "centroid_distance_mm",
            ):
                value = raw.get(metric)
                row[f"{metric}_overlap"] = (
                    float(value)
                    if overlap
                    and value is not None
                    and np.isfinite(value)
                    else float("nan")
                )
                row[f"{metric}_censored"] = (
                    float(value)
                    if value is not None
                    and np.isfinite(value)
                    else diagonal_mm
                )
            rows.append(row)
        if subject_index % 10 == 0 or subject_index == len(subjects):
            print(
                f"compared {subject_index}/{len(subjects)} {subject}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    quartile_source = frame[
        ["subject", "expert_volume_ml"]
    ].drop_duplicates("subject")
    quartile_source["size_quartile"] = pd.qcut(
        quartile_source["expert_volume_ml"].rank(method="first"),
        4,
        labels=["Q1-small", "Q2", "Q3", "Q4-large"],
    )
    frame = frame.merge(
        quartile_source[["subject", "size_quartile"]],
        on="subject",
        how="left",
    )
    frame.to_csv(
        OUTPUT / "per_case_metrics.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )
    aggregation = {
        "cases": ("subject", "nunique"),
        "detection_rate": ("detected", "mean"),
        "overlap_detection_rate": ("overlap_detected", "mean"),
        "Dice": ("Dice_zero_filled", "mean"),
        "Dice_median": ("Dice_zero_filled", "median"),
        "Dice_overlap": ("Dice_overlap", "mean"),
        "precision": ("precision", "mean"),
        "recall": ("recall", "mean"),
        "volume_ratio_median": ("volume_ratio", "median"),
        "HD95_censored": ("HD95_mm_censored", "mean"),
        "HD95_overlap": ("HD95_mm_overlap", "mean"),
        "ASSD_censored": ("ASSD_mm_censored", "mean"),
        "centroid_distance_censored": (
            "centroid_distance_mm_censored",
            "mean",
        ),
    }
    overall = (
        frame.groupby("method")
        .agg(**aggregation)
        .reindex(METHODS)
        .reset_index()
    )
    overall.to_csv(
        OUTPUT / "summary.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )
    ci_rows = []
    for method in METHODS:
        subset = frame.loc[frame["method"] == method]
        for metric in aggregation:
            if metric == "cases":
                continue
            values = subset[aggregation[metric][0]].to_numpy(
                dtype=np.float64
            )
            low, high = bootstrap_mean_ci(values)
            ci_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(np.nanmean(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n": int(np.isfinite(values).sum()),
                }
            )
    pd.DataFrame(ci_rows).to_csv(
        OUTPUT / "summary_with_ci.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )
    by_center = (
        frame.groupby(["method", "center"])
        .agg(**aggregation)
        .reset_index()
    )
    by_center.to_csv(
        OUTPUT / "by_center.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )
    by_quartile = (
        frame.groupby(
            ["method", "size_quartile"], observed=True
        )
        .agg(
            cases=("subject", "nunique"),
            detection_rate=("detected", "mean"),
            overlap_detection_rate=("overlap_detected", "mean"),
            Dice=("Dice_zero_filled", "mean"),
            HD95_censored=("HD95_mm_censored", "mean"),
        )
        .reset_index()
    )
    by_quartile.to_csv(
        OUTPUT / "by_size_quartile.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )
    definitions = {
        "Dice_zero_filled": False,
        "HD95_mm_censored": True,
        "ASSD_mm_censored": True,
        "centroid_distance_mm_censored": True,
    }
    paired = pd.DataFrame(
        [
            paired_test(frame, metric, candidate, lower)
            for candidate in list(METHODS)[1:]
            for metric, lower in definitions.items()
        ]
    )
    paired["p_bh_within_metric"] = np.nan
    for metric, group in paired.groupby("metric"):
        finite = group["wilcoxon_p"].notna()
        if finite.any():
            paired.loc[
                group.index[finite], "p_bh_within_metric"
            ] = bh_adjust(
                group.loc[finite, "wilcoxon_p"].tolist()
            )
    paired.to_csv(
        OUTPUT / "paired_vs_epg_synth.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )
    detection_rows = []
    baseline_detected = frame.loc[
        frame["method"] == "EPG-Synth"
    ].set_index("subject")["detected"]
    for candidate in list(METHODS)[1:]:
        other = frame.loc[
            frame["method"] == candidate
        ].set_index("subject")["detected"]
        epg_only = int((baseline_detected & ~other).sum())
        other_only = int((~baseline_detected & other).sum())
        discordant = epg_only + other_only
        detection_rows.append(
            {
                "candidate": candidate,
                "both_detected": int((baseline_detected & other).sum()),
                "epg_only": epg_only,
                "candidate_only": other_only,
                "neither": int((~baseline_detected & ~other).sum()),
                "mcnemar_exact_p": (
                    float(
                        binomtest(
                            min(epg_only, other_only),
                            discordant,
                            0.5,
                        ).pvalue
                    )
                    if discordant
                    else 1.0
                ),
            }
        )
    pd.DataFrame(detection_rows).to_csv(
        OUTPUT / "paired_detection.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )
    failure_rows = []
    for method in METHODS:
        subset = frame.loc[frame["method"] == method]
        failure_rows.append(
            {
                "method": method,
                "cases": len(subset),
                "absent": int((~subset["detected"]).sum()),
                "no_spatial_overlap": int(
                    (~subset["overlap_detected"]).sum()
                ),
                "Dice_below_0_30": int(
                    (subset["Dice_zero_filled"] < 0.30).sum()
                ),
                "HD95_above_50_mm": int(
                    (subset["HD95_mm_censored"] > 50).sum()
                ),
            }
        )
    failures = pd.DataFrame(failure_rows)
    failures.to_csv(
        OUTPUT / "failure_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "cases": len(subjects),
        "methods": list(METHODS),
        "centres": frame.groupby("center")["subject"].nunique().to_dict(),
        "expert_mask_definition": (
            "model-initialized whole-tumour mask followed by "
            "expert correction; not independent from-scratch annotation"
        ),
        "external_tuning": False,
        "test_set_used": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(overall.to_string(index=False))
    print()
    print(failures.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
