"""Evaluate generated-image segmentations against validation expert ROIs."""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_erosion,
    center_of_mass,
    distance_transform_edt,
)
from scipy.stats import wilcoxon


PROJECT = Path(
    os.environ.get("EPG_SYNTH_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
EXPERT_ROOT = PROJECT / "data/validation_expert_masks_20261030"
AUDIT_CSV = EXPERT_ROOT / "candidate_geometry_audit.csv"
DEFAULT_OUTPUT = PROJECT / "runs/expert_mask_evaluation_20261030"

MODEL_ROOTS = {
    "Raidionics real T1c": (
        PROJECT / "runs/downstream_seg_raidionics_20261030/masks/real"
    ),
    "V12-HF Core": (
        PROJECT / "runs/downstream_seg_raidionics_20261030/masks/synthetic"
    ),
    "WFM-Adapted": (
        PROJECT
        / "runs/downstream_seg_raidionics_wfm_adapted_20261030/"
        "masks/wfm_adapted"
    ),
    "V16-A 250": (
        PROJECT
        / "runs/downstream_seg_v16a_all87_20261030/masks/v16a_all87"
    ),
    "V16-B 250": (
        PROJECT
        / "runs/downstream_seg_v16b250_all87_20261030/masks/v16b250_all87"
    ),
    "V16-B 500": (
        PROJECT
        / "runs/downstream_seg_v16b500_all87_20261030/masks/v16b500_all87"
    ),
}


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(
        mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=1,
    )


def segmentation_metrics(
    expert: np.ndarray,
    prediction: np.ndarray,
    spacing: np.ndarray,
) -> dict[str, float | bool | int]:
    expert = expert > 0.5
    prediction = prediction > 0.5
    expert_count = int(expert.sum())
    prediction_count = int(prediction.sum())
    intersection = int((expert & prediction).sum())
    union = int((expert | prediction).sum())
    both_empty = expert_count == 0 and prediction_count == 0
    dice = (
        1.0
        if both_empty
        else 2.0 * intersection / max(expert_count + prediction_count, 1)
    )
    iou = (
        1.0
        if both_empty
        else intersection / max(union, 1)
    )
    if expert_count and prediction_count:
        union_points = np.argwhere(expert | prediction)
        margin = 8
        lower = np.maximum(union_points.min(axis=0) - margin, 0)
        upper = np.minimum(
            union_points.max(axis=0) + margin + 1,
            expert.shape,
        )
        crop = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(lower, upper)
        )
        expert_surface = surface(expert[crop])
        prediction_surface = surface(prediction[crop])
        distance_to_expert = distance_transform_edt(
            ~expert_surface,
            sampling=spacing,
        )
        distance_to_prediction = distance_transform_edt(
            ~prediction_surface,
            sampling=spacing,
        )
        directed = [
            distance_to_expert[prediction_surface],
            distance_to_prediction[expert_surface],
        ]
        assd = float(
            (directed[0].mean() + directed[1].mean()) / 2.0
        )
        hd95 = float(
            max(
                np.percentile(directed[0], 95),
                np.percentile(directed[1], 95),
            )
        )
        expert_center = np.asarray(center_of_mass(expert)) * spacing
        prediction_center = (
            np.asarray(center_of_mass(prediction)) * spacing
        )
        centroid_distance = float(
            np.linalg.norm(expert_center - prediction_center)
        )
    else:
        assd = float("nan")
        hd95 = float("nan")
        centroid_distance = float("nan")
    return {
        "expert_voxels": expert_count,
        "prediction_voxels": prediction_count,
        "expert_volume_ml": float(expert_count / 1000.0),
        "prediction_volume_ml": float(prediction_count / 1000.0),
        "detected": bool(prediction_count > 0),
        "Dice": float(dice),
        "IoU": float(iou),
        "precision": float(
            intersection / max(prediction_count, 1)
        ),
        "recall": float(intersection / max(expert_count, 1)),
        "volume_ratio": float(
            prediction_count / max(expert_count, 1)
            if expert_count
            else float("nan")
        ),
        "abs_log_volume_error": float(
            abs(
                np.log(
                    (prediction_count + 1.0) / (expert_count + 1.0)
                )
            )
        ),
        "HD95_mm": hd95,
        "ASSD_mm": assd,
        "centroid_distance_mm": centroid_distance,
    }


def summary_row(
    model: str,
    frame: pd.DataFrame,
) -> dict[str, float | str]:
    return {
        "model": model,
        "cases": len(frame),
        "detection_rate": float(frame["detected"].mean()),
        "Dice_mean": float(frame["Dice"].mean()),
        "Dice_median": float(frame["Dice"].median()),
        "volume_ratio_median": float(frame["volume_ratio"].median()),
        "abs_log_volume_error_median": float(
            frame["abs_log_volume_error"].median()
        ),
        "HD95_mean_mm": float(frame["HD95_mm"].mean()),
        "HD95_median_mm": float(frame["HD95_mm"].median()),
        "ASSD_mean_mm": float(frame["ASSD_mm"].mean()),
        "centroid_distance_mean_mm": float(
            frame["centroid_distance_mm"].mean()
        ),
    }


def paired_comparison(
    frame: pd.DataFrame,
    baseline_model: str,
    candidate_model: str,
) -> tuple[dict, pd.DataFrame]:
    baseline = frame.loc[
        frame["model"] == baseline_model
    ].set_index("subject")
    candidate = frame.loc[
        frame["model"] == candidate_model
    ].set_index("subject")
    paired = pd.DataFrame(
        {
            "baseline_Dice": baseline["Dice"],
            "candidate_Dice": candidate["Dice"],
            "delta_Dice": candidate["Dice"] - baseline["Dice"],
            "baseline_HD95": baseline["HD95_mm"],
            "candidate_HD95": candidate["HD95_mm"],
            "delta_HD95": candidate["HD95_mm"]
            - baseline["HD95_mm"],
        }
    )
    differences = paired["delta_Dice"].dropna()
    nonzero = differences[np.abs(differences) > 1e-12]
    if len(nonzero) >= 2:
        statistic = float(wilcoxon(nonzero).statistic)
        pvalue = float(wilcoxon(nonzero).pvalue)
    else:
        statistic = float("nan")
        pvalue = float("nan")
    result = {
        "baseline": baseline_model,
        "candidate": candidate_model,
        "mean_delta_Dice": float(differences.mean()),
        "median_delta_Dice": float(differences.median()),
        "wins": int((differences > 1e-12).sum()),
        "losses": int((differences < -1e-12).sum()),
        "ties": int((np.abs(differences) <= 1e-12).sum()),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": pvalue,
    }
    return result, paired.reset_index()


def write_report(
    path: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    cases: int,
) -> None:
    table = summary[
        [
            "model",
            "detection_rate",
            "Dice_mean",
            "volume_ratio_median",
            "HD95_mean_mm",
            "ASSD_mean_mm",
            "centroid_distance_mean_mm",
        ]
    ]
    lines = [
        "# 验证集专家 ROI 分割评价",
        "",
        f"- 病例数：{cases}",
        "- 专家 ROI：原始厚层空间勾画，经世界坐标最近邻映射到 1 mm 验证网格。",
        "- 合成 mask：冻结的 Raidionics `MRI_TumorCore/t1c v1.3.0-rc`。",
        "- 89 例测试集未读取。",
        "",
        "## 汇总",
        "",
        "| 模型 | 检出率 | Dice | 体积比中位数 | HD95 (mm) | ASSD (mm) | 中心偏移 (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['model']} | {row['detection_rate']:.3f} | "
            f"{row['Dice_mean']:.4f} | {row['volume_ratio_median']:.4f} | "
            f"{row['HD95_mean_mm']:.2f} | {row['ASSD_mean_mm']:.2f} | "
            f"{row['centroid_distance_mean_mm']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 相对 V12-HF Core 的配对比较",
            "",
            "| 候选 | ΔDice | 胜/负/平 | Wilcoxon p |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in paired.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['mean_delta_Dice']:+.4f} | "
            f"{int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])} | "
            f"{row['wilcoxon_p']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- 厚层专家 ROI 经最近邻映射后存在层间阶梯，HD95 和 ASSD 会受离散化影响。",
            "- 该评价衡量的是“Raidionics 对合成图像的肿瘤分割与专家 ROI 的一致性”，",
            "  不等同于临床诊断准确性。",
            "- 当前仍是固定验证集分析，模型选择和测试集规则不变。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=tuple(MODEL_ROOTS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT_CSV)
    if len(audit) != audit["subject"].nunique():
        raise ValueError("Expected exactly one expert mask per subject")
    expert_paths = {
        row.subject: (
            EXPERT_ROOT
            / "candidates"
            / row.subject
            / f"{Path(row.candidate).stem}_1mm.nii.gz"
        )
        for row in audit.itertuples()
    }
    subjects = sorted(expert_paths)
    selected_models = {
        model: MODEL_ROOTS[model]
        for model in (args.models or list(MODEL_ROOTS))
    }
    rows = []
    for model_index, (model, root) in enumerate(
        selected_models.items(),
        start=1,
    ):
        if not root.exists():
            raise FileNotFoundError(root)
        for index, subject in enumerate(subjects, start=1):
            expert_path = expert_paths[subject]
            prediction_path = root / f"{subject}.nii.gz"
            if not prediction_path.exists():
                raise FileNotFoundError(prediction_path)
            expert_image = nib.load(str(expert_path))
            prediction_image = nib.load(str(prediction_path))
            if expert_image.shape != prediction_image.shape:
                raise ValueError(
                    f"Shape mismatch for {subject}: "
                    f"{expert_image.shape} vs {prediction_image.shape}"
                )
            if np.max(
                np.abs(expert_image.affine - prediction_image.affine)
            ) > 1e-3:
                raise ValueError(
                    f"Affine mismatch for {subject}"
                )
            spacing = np.asarray(
                expert_image.header.get_zooms()[:3],
                dtype=np.float64,
            )
            rows.append(
                {
                    "model": model,
                    "subject": subject,
                    **segmentation_metrics(
                        np.asanyarray(expert_image.dataobj),
                        np.asanyarray(prediction_image.dataobj),
                        spacing,
                    ),
                }
            )
        print(
            f"evaluated {model}: {len(subjects)} cases",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(
        args.output / "per_case_metrics.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )
    summary = pd.DataFrame(
        [
            summary_row(
                model,
                frame.loc[frame["model"] == model],
            )
            for model in selected_models
        ]
    )
    summary.to_csv(
        args.output / "summary.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )

    pair_rows = []
    for candidate in [
        model for model in selected_models if model != "V12-HF Core"
    ]:
        if "V12-HF Core" not in selected_models:
            continue
        result, paired = paired_comparison(
            frame,
            "V12-HF Core",
            candidate,
        )
        pair_rows.append(result)
        safe_name = candidate.lower().replace(" ", "_").replace("-", "")
        paired.to_csv(
            args.output / f"paired_v12_vs_{safe_name}.csv",
            index=False,
            float_format="%.8f",
            encoding="utf-8-sig",
        )
    paired_frame = pd.DataFrame(
        pair_rows,
        columns=(
            "baseline",
            "candidate",
            "mean_delta_Dice",
            "median_delta_Dice",
            "wins",
            "losses",
            "ties",
            "wilcoxon_statistic",
            "wilcoxon_p",
        ),
    )
    paired_frame.to_csv(
        args.output / "paired_vs_v12.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )

    expert_sizes = (
        frame[["subject", "expert_volume_ml"]]
        .drop_duplicates("subject")
        .reset_index(drop=True)
    )
    expert_sizes["size_quartile"] = pd.qcut(
        expert_sizes["expert_volume_ml"].rank(method="first"),
        4,
        labels=["Q1 small", "Q2", "Q3", "Q4 large"],
    )
    stratified = frame.merge(
        expert_sizes[["subject", "size_quartile"]],
        on="subject",
        how="left",
    )
    stratification = (
        stratified.groupby(["model", "size_quartile"], observed=True)
        .agg(
            cases=("subject", "size"),
            Dice_mean=("Dice", "mean"),
            volume_ratio_median=("volume_ratio", "median"),
            HD95_mean_mm=("HD95_mm", "mean"),
        )
        .reset_index()
    )
    stratification.to_csv(
        args.output / "stratified_by_expert_volume.csv",
        index=False,
        float_format="%.8f",
        encoding="utf-8-sig",
    )

    payload = {
        "status": "PASS",
        "cases": len(subjects),
        "expert_mask_source": (
            str(Path(
        os.environ.get(
            "EPG_SYNTH_DATA_ROOT",
            Path(__file__).resolve().parents[1] / "data",
        )
    ).expanduser().resolve())
        ),
        "expert_mask_mapping": "world-coordinate nearest-neighbor to 1 mm",
        "models": list(selected_models),
        "summary": summary.to_dict(orient="records"),
        "paired_vs_V12_HF_Core": paired_frame.to_dict(
            orient="records"
        ),
        "test_set_used": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        args.output / "EXPERT_MASK_EVALUATION_20261030.md",
        summary,
        paired_frame,
        len(subjects),
    )
    print(summary.to_string(index=False))
    print(paired_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
