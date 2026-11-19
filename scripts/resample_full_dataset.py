"""Resample all 653 cases to a shared 1 mm physical grid.

Output axis order is (D,H,W) = (S,A,R), matching the project's model loader:
192 x 224 x 192 voxels at 1 x 1 x 1 mm. The target is centered on each case's
brain-mask physical bounding box. Source data is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


SOURCE_ROOT = Path(
        os.environ.get(
            "EPG_SYNTH_DATA_ROOT",
            Path(__file__).resolve().parents[1] / "data",
        )
    ).expanduser().resolve()
OUTPUT_ROOT = Path(
    r"data"
    r"\deidentified_subject_1mm_192x224x192_20261027"
)
MODEL_ROOT = OUTPUT_ROOT / "model_192x224x192"
MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
TARGET_SHAPE_DHW = (192, 224, 192)
TARGET_SPACING = (1.0, 1.0, 1.0)
NORMALIZATION_PERCENTILES = (0.1, 99.9)
MODALITIES = ("T1", "T1CE", "T2", "FLAIR")


def target_affine_for_center(center_xyz: np.ndarray) -> np.ndarray:
    """Map model (D,H,W)=(S,A,R) indices into RAS+ world coordinates."""
    depth, height, width = TARGET_SHAPE_DHW
    affine = np.array(
        [
            [0.0, 0.0, 1.0, center_xyz[0] - (width - 1) / 2.0],
            [0.0, 1.0, 0.0, center_xyz[1] - (height - 1) / 2.0],
            [1.0, 0.0, 0.0, center_xyz[2] - (depth - 1) / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return affine


def qform_disabled_sform_affine(image: nib.Nifti1Image) -> None:
    image.set_qform(None)
    image.set_sform(image.affine, code=1)


def save_normalized_volume(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    image = nib.Nifti1Image(np.clip(data, 0.0, 1.0).astype(np.float32), affine)
    qform_disabled_sform_affine(image)
    image.set_data_dtype(np.uint16)
    nib.save(image, str(path))


def process_case(case: str) -> dict:
    source_dir = SOURCE_ROOT / case
    output_dir = MODEL_ROOT / case
    done_path = output_dir / "DONE.json"
    expected_files = [
        output_dir / f"{case}_{modality}.nii.gz" for modality in MODALITIES
    ] + [output_dir / "brain_mask.nii.gz"]

    if done_path.exists() and all(path.exists() for path in expected_files):
        existing = json.loads(done_path.read_text(encoding="utf-8"))
        existing["resumed"] = True
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_image = nib.as_closest_canonical(
        nib.load(str(source_dir / "brain_mask.nii.gz"))
    )
    mask_source = np.asarray(mask_image.dataobj) > 0
    mask_voxels = np.argwhere(mask_source)
    if len(mask_voxels) == 0:
        raise ValueError(f"Empty source brain mask: {case}")
    world = nib.affines.apply_affine(mask_image.affine, mask_voxels)
    lower = world.min(axis=0)
    upper = world.max(axis=0)
    extent = upper - lower
    center = (lower + upper) / 2.0
    margins = np.asarray(TARGET_SHAPE_DHW, dtype=np.float64) - extent
    if (margins < 0).any():
        raise ValueError(f"{case} exceeds target FOV: extent={extent.tolist()}")

    target_affine = target_affine_for_center(center)
    target_template = nib.Nifti1Image(
        np.zeros(TARGET_SHAPE_DHW, dtype=np.float32), target_affine
    )
    target_mask_image = resample_from_to(
        mask_image,
        target_template,
        order=0,
        mode="constant",
        cval=0,
    )
    target_mask = np.asarray(target_mask_image.dataobj) > 0.5
    if not target_mask.any():
        raise ValueError(f"Empty target brain mask: {case}")

    normalization = {}
    for modality in MODALITIES:
        source_path = source_dir / f"{case}_{modality}.nii.gz"
        source_image = nib.as_closest_canonical(nib.load(str(source_path)))
        resampled = resample_from_to(
            source_image,
            target_template,
            order=1,
            mode="constant",
            cval=0.0,
        )
        data = np.asarray(resampled.dataobj, dtype=np.float32)
        values = data[target_mask]
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"No finite {modality} values for {case}")
        low, high = np.percentile(values, NORMALIZATION_PERCENTILES)
        if not np.isfinite([low, high]).all() or high <= low + 1e-8:
            raise ValueError(f"Invalid normalization range for {case} {modality}")
        normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
        normalized[~target_mask] = 0.0
        save_normalized_volume(
            normalized,
            target_affine,
            output_dir / f"{case}_{modality}.nii.gz",
        )
        normalization[modality] = {
            "low": float(low),
            "high": float(high),
            "min": float(normalized.min()),
            "max": float(normalized.max()),
            "mask_voxels": int(target_mask.sum()),
        }

    output_mask = nib.Nifti1Image(target_mask.astype(np.uint8), target_affine)
    qform_disabled_sform_affine(output_mask)
    output_mask.set_data_dtype(np.uint8)
    nib.save(output_mask, str(output_dir / "brain_mask.nii.gz"))

    record = {
        "case": case,
        "status": "OK",
        "source_shape_xyz": list(mask_image.shape),
        "source_spacing_mm": [
            float(value) for value in np.sqrt((mask_image.affine[:3, :3] ** 2).sum(axis=0))
        ],
        "target_shape_dhw": list(TARGET_SHAPE_DHW),
        "target_spacing_mm": list(TARGET_SPACING),
        "target_affine": target_affine.tolist(),
        "brain_lower_mm": lower.tolist(),
        "brain_upper_mm": upper.tolist(),
        "brain_extent_mm": extent.tolist(),
        "brain_center_mm": center.tolist(),
        "margin_mm_total": margins.tolist(),
        "margin_mm_per_side": (margins / 2.0).tolist(),
        "target_mask_voxels": int(target_mask.sum()),
        "normalization": normalization,
        "resumed": False,
    }
    done_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    excluded = {"deidentified_subject"}
    cases = sorted(
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and path.name not in excluded
    )
    print(f"cases={len(cases)} workers={args.workers}", flush=True)

    records = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_case, case): case for case in cases}
        for completed, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            try:
                record = future.result()
                records.append(record)
            except Exception as exc:
                print(f"FAILED {case}: {exc}", flush=True)
                raise
            if completed == 1 or completed % 10 == 0 or completed == len(cases):
                elapsed = time.time() - started
                print(
                    f"progress={completed}/{len(cases)} "
                    f"elapsed={elapsed:.1f}s rate={completed / elapsed:.2f} case/s",
                    flush=True,
                )

    records.sort(key=lambda item: item["case"])
    manifest_path = MANIFEST_ROOT / "case_geometry_1mm.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "status",
                "source_shape_xyz",
                "source_spacing_mm",
                "target_shape_dhw",
                "target_spacing_mm",
                "brain_extent_mm",
                "margin_mm_per_side",
                "target_mask_voxels",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(record[key])
                        if isinstance(record[key], list)
                        else record[key]
                    )
                    for key in writer.fieldnames
                }
            )

    summary = {
        "cases": len(records),
        "all_ok": all(record["status"] == "OK" for record in records),
        "target_shape_dhw": list(TARGET_SHAPE_DHW),
        "target_spacing_mm": list(TARGET_SPACING),
        "normalization_percentiles": list(NORMALIZATION_PERCENTILES),
        "minimum_margin_mm_per_side": [
            min(record["margin_mm_per_side"][axis] for record in records)
            for axis in range(3)
        ],
        "elapsed_seconds": round(time.time() - started, 2),
        "workers": args.workers,
        "excluded_cases": sorted(excluded),
        "output_root": str(OUTPUT_ROOT),
    }
    (OUTPUT_ROOT / "resampling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "# Meningioma 653 physical-scale dataset\n\n"
        "- Target shape (D,H,W): 192 x 224 x 192\n"
        "- Target spacing: 1 x 1 x 1 mm\n"
        "- Orientation: (D,H,W) = (S,A,R)\n"
        "- Construction: center on each brain-mask physical bounding box; "
        "crop or zero pad to the target FOV\n"
        "- Normalization: independent 0.1%-99.9% scaling per case and sequence "
        "inside the brain mask\n"
        "- Storage: uint16 NIfTI with automatic scaling to [0,1]\n"
        "- deidentified_subject is absent because its T1CE acquisition is coronal\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
