"""Audit source and model-grid orientations for the external 118 cohort."""

from __future__ import annotations

import os

import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import pandas as pd


PROJECT = Path(
    os.environ.get("EPG_SYNTH_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
DATA = (
    PROJECT
    / "data"
    / "external_validation_combined118_20261110"
)
MODEL = DATA / "model_192x224x192"
PROVENANCE = DATA / "manifests" / "source_provenance.csv"
NEW_RAW = Path(
        os.environ.get(
            "EPG_SYNTH_DATA_ROOT",
            Path(__file__).resolve().parents[1] / "data",
        )
    ).expanduser().resolve()
OUTPUT = Path(
    r"data"
    r"\summary_20261111"
)


def codes(path: Path) -> str:
    return "".join(nib.aff2axcodes(nib.load(str(path)).affine))


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = pd.read_csv(PROVENANCE)
    rows = []
    for record in provenance.itertuples(index=False):
        case = str(record.case)
        source_dir = Path(record.source_root)
        source_path = next(
            path
            for path in source_dir.iterdir()
            if "t1ce" in path.name.lower()
            and path.name.lower().endswith(".nii.gz")
        )
        model_path = MODEL / case / f"{case}_T1CE.nii.gz"
        rows.append(
            {
                "case": case,
                "cohort": record.cohort,
                "center": record.center,
                "source_t1ce": str(source_path),
                "source_axcodes": codes(source_path),
                "model_t1ce": str(model_path),
                "model_axcodes": codes(model_path),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUTPUT / "orientation_audit_118.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "source_orientation_counts": {
            cohort: dict(
                Counter(
                    frame.loc[
                        frame["cohort"] == cohort,
                        "source_axcodes",
                    ]
                )
            )
            for cohort in ("legacy30", "new88")
        },
        "model_orientation_counts": dict(
            Counter(frame["model_axcodes"])
        ),
    }
    (OUTPUT / "orientation_audit_118.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
