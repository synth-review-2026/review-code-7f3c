"""Validate every generated 1 mm NIfTI volume."""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(
    r"data"
    r"\deidentified_subject_1mm_192x224x192_20261027"
)
CASE_ROOT = ROOT / "model_192x224x192"
EXPECTED_SHAPE = (192, 224, 192)
EXPECTED_AXCODES = ("S", "A", "R")
MODALITIES = ("T1", "T1CE", "T2", "FLAIR")


def validate_case(case: str) -> dict:
    case_dir = CASE_ROOT / case
    problems = []
    files = sorted(case_dir.glob("*.nii.gz"))
    expected_names = {
        *(f"{case}_{modality}.nii.gz" for modality in MODALITIES),
        "brain_mask.nii.gz",
    }
    actual_names = {path.name for path in files}
    if actual_names != expected_names:
        problems.append(
            f"file set differs: missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )

    affines = []
    ranges = {}
    mask_ratio = np.nan
    for path in files:
        image = nib.load(str(path))
        affine = np.asarray(image.affine, dtype=np.float64)
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        axcodes = nib.aff2axcodes(affine)
        affines.append(affine)
        if tuple(image.shape) != EXPECTED_SHAPE:
            problems.append(f"{path.name}: shape={image.shape}")
        if not np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-5):
            problems.append(f"{path.name}: spacing={spacing.tolist()}")
        if axcodes != EXPECTED_AXCODES:
            problems.append(f"{path.name}: axcodes={axcodes}")
        if image.header["qform_code"] != 0 or image.header["sform_code"] != 1:
            problems.append(
                f"{path.name}: qform={image.header['qform_code']} "
                f"sform={image.header['sform_code']}"
            )
        data = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(data).all():
            problems.append(f"{path.name}: non-finite values")
        if data.min() < -1e-6 or data.max() > 1.0 + 1e-6:
            problems.append(f"{path.name}: range={data.min()}..{data.max()}")
        if path.name == "brain_mask.nii.gz":
            mask_ratio = float((data > 0.5).mean())
            if mask_ratio <= 0:
                problems.append("brain mask is empty")
        else:
            ranges[path.name] = [float(data.min()), float(data.max())]

    if affines and not all(np.allclose(affines[0], affine, atol=1e-5) for affine in affines[1:]):
        problems.append("modality affines differ within case")

    return {
        "case": case,
        "status": "OK" if not problems else "FAIL",
        "problems": "; ".join(problems),
        "mask_ratio": mask_ratio,
        "modality_ranges": ranges,
    }


def main() -> int:
    started = time.time()
    cases = sorted(path.name for path in CASE_ROOT.iterdir() if path.is_dir())
    if "deidentified_subject" in cases:
        raise ValueError("deidentified_subject must not be present")

    records = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(validate_case, case): case for case in cases}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if completed == 1 or completed % 25 == 0 or completed == len(cases):
                print(f"validated={completed}/{len(cases)}", flush=True)

    records.sort(key=lambda item: item["case"])
    failures = [record for record in records if record["status"] != "OK"]
    mask_ratios = [record["mask_ratio"] for record in records]
    summary = {
        "cases": len(records),
        "nifti_files": len(records) * 5,
        "status": "ok" if not failures else "failed",
        "failed_cases": len(failures),
        "shape_dhw": list(EXPECTED_SHAPE),
        "spacing_mm": [1.0, 1.0, 1.0],
        "axcodes": list(EXPECTED_AXCODES),
        "mask_ratio": {
            "min": float(np.min(mask_ratios)),
            "median": float(np.median(mask_ratios)),
            "max": float(np.max(mask_ratios)),
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "problems": [
            {"case": record["case"], "problems": record["problems"]}
            for record in failures
        ],
    }
    (ROOT / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (ROOT / "validation_cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "status",
                "mask_ratio",
                "modality_ranges",
                "problems",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                **record,
                "modality_ranges": json.dumps(
                    record["modality_ranges"], ensure_ascii=False
                ),
            }
            for record in records
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
