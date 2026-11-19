from __future__ import annotations

import os

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity

from guided_diffusion.bratsloader import BRATSVolumes


PROJECT = Path(
    os.environ.get("EPG_SYNTH_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
DATA = (
    PROJECT
    / "data/external_validation_combined118_20261110"
    / "model_192x224x192"
)
EXPERT_MASKS = (
    PROJECT
    / "data/external_validation_combined118_20261110"
    / "expert_masks_model_192x224x192"
)
PREDICTIONS = (
    PROJECT
    / "runs/external_validation_118_formal_20261111"
    / "predictions"
)
OUTPUT = (
    PROJECT
    / "runs/external_validation_118_formal_20261111"
    / "image_roi_evaluation"
)
METHODS = {
    "EPG-Synth": "V12-Balanced",
    "WFM-Official": "WFM-Official",
    "WFM-Adapted": "WFM-Adapted (V12 enhancer)",
    "3D Pix2Pix cGAN": "3D Pix2Pix cGAN",
    "Conditional WGAN-GP": "Conditional WGAN-GP",
    "T1C-RFlow": "T1C-RFlow",
}
METRICS = [
    "MAE_full",
    "RMSE_full",
    "NMSE_full",
    "NCC_full",
    "PSNR_full",
    "SSIM_3D",
    "ROI_MAE",
    "ROI_RMSE",
    "ROI_PSNR",
    "ROI_NMSE",
    "ROI_NCC",
    "ROI_ratio",
    "enhancement_ratio",
    "enhancement_dice",
    "enhancement_volume_ratio",
    "r_native",
]


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def psnr(error: np.ndarray) -> float:
    mse = float(np.mean(error**2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def nmse(error: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.sum(error**2) / max(float(np.sum(target**2)), 1e-12)
    )


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 20:
        return float("nan")
    if first.std() <= 1e-8 or second.std() <= 1e-8:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


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


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (PROJECT / "data/external_validation_combined118_20261110/manifests/case_registry.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        import csv

        registry = {
            row["case"]: row for row in csv.DictReader(handle)
        }
    dataset = BRATSVolumes(
        str(DATA),
        mode="eval_target",
        normalization="preprocessed_dhw",
        target_calibration="t1_robust",
        target_depth=192,
        target_h=224,
        target_w=192,
    )
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        subject = str(sample["subj"])
        target = sample["t1c"][0].numpy()
        t1n = sample["t1n"][0].numpy()
        brain = t1n > 0.05
        expert = load(
            EXPERT_MASKS / f"{subject}.nii.gz"
        ) > 0.5
        if expert.shape != target.shape:
            raise ValueError(f"{subject}: expert mask shape mismatch")
        if not expert.any():
            raise ValueError(f"{subject}: empty expert mask")
        target_enhancement = np.clip(target - t1n, 0.0, 1.0)
        target_enhancement_roi = (
            target_enhancement > 0.15
        ) & expert
        for display_name, directory in METHODS.items():
            prediction = load(
                PREDICTIONS
                / directory
                / subject
                / "sample_t1c.nii.gz"
            )
            if prediction.shape != target.shape:
                raise ValueError(
                    f"{display_name}/{subject}: prediction shape mismatch"
                )
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"{display_name}/{subject}: non-finite prediction"
                )
            prediction_enhancement = np.clip(
                prediction - t1n, 0.0, 1.0
            )
            predicted_enhancement_roi = (
                prediction_enhancement > 0.15
            ) & expert
            intersection = int(
                np.logical_and(
                    predicted_enhancement_roi,
                    target_enhancement_roi,
                ).sum()
            )
            enhancement_dice = float(
                2.0
                * intersection
                / max(
                    int(predicted_enhancement_roi.sum())
                    + int(target_enhancement_roi.sum()),
                    1,
                )
            )
            error_full = prediction[brain] - target[brain]
            error_expert = prediction[expert] - target[expert]
            rows.append(
                {
                    "method": display_name,
                    "method_directory": directory,
                    "subject": subject,
                    "center": registry[subject]["center"],
                    "site_code": registry[subject]["site_code"],
                    "cohort": registry[subject]["cohort"],
                    "expert_volume_ml": float(expert.sum()),
                    "expert_slices": int(expert.any(axis=(1, 2)).sum()),
                    "MAE_full": float(np.mean(np.abs(error_full))),
                    "RMSE_full": float(np.sqrt(np.mean(error_full**2))),
                    "NMSE_full": nmse(error_full, target[brain]),
                    "NCC_full": correlation(
                        prediction[brain], target[brain]
                    ),
                    "PSNR_full": psnr(error_full),
                    "SSIM_3D": float(
                        structural_similarity(
                            target,
                            prediction,
                            data_range=1.0,
                        )
                    ),
                    "ROI_MAE": float(np.mean(np.abs(error_expert))),
                    "ROI_RMSE": float(
                        np.sqrt(np.mean(error_expert**2))
                    ),
                    "ROI_PSNR": psnr(error_expert),
                    "ROI_NMSE": nmse(error_expert, target[expert]),
                    "ROI_NCC": correlation(
                        prediction[expert], target[expert]
                    ),
                    "ROI_ratio": float(
                        prediction[expert].mean()
                        / max(float(target[expert].mean()), 1e-8)
                    ),
                    "enhancement_ratio": float(
                        prediction_enhancement[expert].mean()
                        / max(
                            float(
                                target_enhancement[expert].mean()
                            ),
                            1e-8,
                        )
                    ),
                    "enhancement_dice": enhancement_dice,
                    "enhancement_volume_ratio": float(
                        predicted_enhancement_roi.sum()
                        / max(
                            int(target_enhancement_roi.sum()), 1
                        )
                    ),
                    "r_native": correlation(
                        prediction_enhancement[expert],
                        target_enhancement[expert],
                    ),
                }
            )
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(
                f"evaluated {index + 1}/{len(dataset)} {subject}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    volumes = (
        frame.groupby("subject")["expert_volume_ml"]
        .first()
        .drop_duplicates()
    )
    quartile = pd.qcut(
        volumes.rank(method="first"),
        4,
        labels=["Q1-small", "Q2", "Q3", "Q4-large"],
    )
    frame["size_quartile"] = frame["subject"].map(quartile)
    frame.to_csv(
        OUTPUT / "per_case_metrics.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )
    overall = (
        frame.groupby("method")
        .agg(
            cases=("subject", "nunique"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
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
        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=np.float64)
            low, high = bootstrap_mean_ci(values)
            ci_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(
                        np.nanmean(values)
                    ),
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
        .agg(
            cases=("subject", "nunique"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
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
            PSNR_full=("PSNR_full", "mean"),
            SSIM_3D=("SSIM_3D", "mean"),
            ROI_PSNR=("ROI_PSNR", "mean"),
            enhancement_dice=("enhancement_dice", "mean"),
            r_native=("r_native", "mean"),
        )
        .reset_index()
    )
    by_quartile.to_csv(
        OUTPUT / "by_size_quartile.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )
    baseline = frame.loc[
        frame["method"] == "EPG-Synth"
    ].set_index("subject")
    paired_rows = []
    for candidate in list(METHODS)[1:]:
        comparison = frame.loc[
            frame["method"] == candidate
        ].set_index("subject")
        for metric in METRICS:
            difference = (
                baseline[metric] - comparison[metric]
            ).dropna()
            nonzero = difference[
                np.abs(difference) > 1e-12
            ]
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
            low, high = bootstrap_mean_ci(
                difference.to_numpy()
            )
            paired_rows.append(
                {
                    "candidate": candidate,
                    "metric": metric,
                    "mean_difference_epg_minus_candidate": float(
                        difference.mean()
                    ),
                    "ci95_low": low,
                    "ci95_high": high,
                    "median_difference": float(difference.median()),
                    "epg_better": int(
                        (
                            difference > 1e-12
                        ).sum()
                    ),
                    "epg_worse": int(
                        (
                            difference < -1e-12
                        ).sum()
                    ),
                    "ties": int(
                        (
                            difference.abs() <= 1e-12
                        ).sum()
                    ),
                    "wilcoxon_p": pvalue,
                }
            )
    paired = pd.DataFrame(paired_rows)
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
    failure_rows = []
    for method in METHODS:
        subset = frame.loc[frame["method"] == method]
        failure_rows.append(
            {
                "method": method,
                "cases": len(subset),
                "r_native_lt_0_30": int(
                    (subset["r_native"] < 0.30).sum()
                ),
                "enhancement_dice_lt_0_30": int(
                    (
                        subset["enhancement_dice"] < 0.30
                    ).sum()
                ),
                "ROI_ratio_outside_0_8_1_2": int(
                    (
                        (subset["ROI_ratio"] < 0.8)
                        | (subset["ROI_ratio"] > 1.2)
                    ).sum()
                ),
                "ROI_PSNR_below_10": int(
                    (subset["ROI_PSNR"] < 10).sum()
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
        "cases": int(frame["subject"].nunique()),
        "methods": list(METHODS),
        "centres": frame.groupby("center")["subject"].nunique().to_dict(),
        "expert_mask_type": (
            "model-initialized whole-tumour mask followed by expert correction"
        ),
        "primary_image_metrics": [
            "PSNR_full",
            "SSIM_3D",
            "ROI_PSNR",
            "r_native",
        ],
        "primary_downstream_metric": "whole-tumour Dice",
        "external_tuning": False,
        "test_set_used": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
