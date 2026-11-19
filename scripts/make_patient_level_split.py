"""Create a deterministic patient-level train/validation/test split.

The split is stratified by source dataset and MRI field strength. Since the
de-duplicated dataset contains one retained case per patient group, each
patient group appears in exactly one split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


SOURCE = Path(
    r"data"
    r"\deidentified_subject_deduplicated_20261026\manifests\retained_cases.csv"
)
OUTPUT_DIR = Path(
    r"data"
    r"\deidentified_subject_deduplicated_20261026\splits"
)
SEED = 20261026
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


def deterministic_order(case_id: str, patient_group: str) -> str:
    key = f"{SEED}|{patient_group}|{case_id}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def allocate_counts(size: int) -> dict[str, int]:
    raw = [size * fraction for fraction in SPLIT_FRACTIONS]
    counts = [math.floor(value) for value in raw]
    remainder = size - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - counts[index]), -len(SPLIT_NAMES[index])),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return dict(zip(SPLIT_NAMES, counts))


def main() -> int:
    if OUTPUT_DIR.exists() and (OUTPUT_DIR / "split_manifest.csv").exists():
        raise FileExistsError(
            f"Split already exists; refusing to overwrite: {OUTPUT_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        field = row["field_strength_t"].strip() or "unknown"
        key = f"{row['dataset']}|{field}"
        row["_stratum"] = key
        strata[key].append(row)

    assignments: dict[str, str] = {}
    for key in sorted(strata):
        items = strata[key]
        items.sort(
            key=lambda row: deterministic_order(
                row["case_id"], row["patient_group"]
            )
        )
        counts = allocate_counts(len(items))
        cursor = 0
        for split in SPLIT_NAMES:
            count = counts[split]
            for row in items[cursor : cursor + count]:
                assignments[row["case_id"]] = split
            cursor += count

    if len(assignments) != len(rows):
        raise RuntimeError("Not every case received a split")

    output_rows: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: int(item["case_id"].split("_")[-1])):
        output_rows.append(
            {
                "case_id": row["case_id"],
                "patient_group": row["patient_group"],
                "split": assignments[row["case_id"]],
                "dataset": row["dataset"],
                "sex": row["sex"],
                "age": row["age"],
                "exam_date": row["exam_date"],
                "field_strength_t": row["field_strength_t"],
                "manufacturer": row["manufacturer"],
                "diagnosis_confidence": row["diagnosis_confidence"],
            }
        )

    headers = [
        "case_id",
        "patient_group",
        "split",
        "dataset",
        "sex",
        "age",
        "exam_date",
        "field_strength_t",
        "manufacturer",
        "diagnosis_confidence",
    ]
    with (OUTPUT_DIR / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)

    for split in SPLIT_NAMES:
        case_ids = [
            row["case_id"] for row in output_rows if row["split"] == split
        ]
        (OUTPUT_DIR / f"{split}_cases.txt").write_text(
            "\n".join(case_ids) + "\n", encoding="utf-8"
        )

    summary = {
        "seed": SEED,
        "split_rule": (
            "patient-level, stratified by source dataset and field strength; "
            "deterministic SHA-256 ordering"
        ),
        "total_cases": len(output_rows),
        "counts": {
            split: sum(row["split"] == split for row in output_rows)
            for split in SPLIT_NAMES
        },
        "by_dataset_and_field": {},
        "by_sex": {},
        "diagnosis_confidence": {},
        "external_test_note": (
            "The existing 30-case BL/YL cohort is not part of this split and "
            "must remain an untouched external test set."
        ),
    }
    for split in SPLIT_NAMES:
        split_rows = [row for row in output_rows if row["split"] == split]
        summary["by_dataset_and_field"][split] = {
            key: count
            for key, count in sorted(
                (
                    (
                        f"{row['dataset']}|{row['field_strength_t'] or 'unknown'}",
                        sum(
                            other["dataset"] == row["dataset"]
                            and other["field_strength_t"]
                            == row["field_strength_t"]
                            for other in split_rows
                        ),
                    )
                    for row in split_rows
                )
            )
        }
        summary["by_sex"][split] = {
            sex: sum(row["sex"] == sex for row in split_rows)
            for sex in sorted({row["sex"] for row in split_rows})
        }
        summary["diagnosis_confidence"][split] = {
            confidence: sum(
                row["diagnosis_confidence"] == confidence for row in split_rows
            )
            for confidence in ("0", "1", "2")
        }

    (OUTPUT_DIR / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
