"""Create a non-destructive, de-duplicated meningioma dataset.

The source dataset and source workbook are never modified. Selected cases are
linked (hard links when possible) into a new directory, together with
de-identified manifests that contain no patient names or examination IDs.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


SOURCE_DATA = Path(
        os.environ.get(
            "EPG_SYNTH_DATA_ROOT",
            Path(__file__).resolve().parents[1] / "data",
        )
    ).expanduser().resolve()
SOURCE_WORKBOOK = Path(
        os.environ.get(
            "EPG_SYNTH_DATA_ROOT",
            Path(__file__).resolve().parents[1] / "data",
        )
    ).expanduser().resolve()
OUTPUT_ROOT = Path(
    r"data"
    r"\deidentified_subject_deduplicated_20261026"
)

EXCLUDED_CASES = {
    "deidentified_subject": "T1CE was acquired in the coronal plane",
}

MANIFEST_HEADERS = [
    "case_id",
    "patient_group",
    "dataset",
    "sex",
    "age",
    "exam_date",
    "diagnosis_confidence",
    "manufacturer",
    "device_model",
    "field_strength_t",
    "record_count_in_group",
    "removed_record_count",
]

REMOVED_HEADERS = [
    "removed_case_id",
    "retained_case_id",
    "patient_group",
    "reason",
    "dataset",
    "exam_date",
    "sex",
    "age",
    "diagnosis_confidence",
]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def column_number(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    result = 0
    for char in letters:
        result = result * 26 + (ord(char.upper()) - 64)
    return result


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root:
        if local_name(item.tag) != "si":
            continue
        parts = [node.text or "" for node in item.iter() if local_name(node.tag) == "t"]
        values.append("".join(parts))
    return values


def read_workbook_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        shared = read_shared_strings(archive)
        root = ET.fromstring(archive.read("xl/worksheets/sheet3.xml"))

    matrix: list[dict[int, str]] = []
    for row_node in root.iter():
        if local_name(row_node.tag) != "row":
            continue
        row: dict[int, str] = {}
        for cell in row_node:
            if local_name(cell.tag) != "c":
                continue
            index = column_number(cell.attrib.get("r", ""))
            cell_type = cell.attrib.get("t", "")
            value = ""
            for child in cell:
                if local_name(child.tag) == "v" and child.text is not None:
                    if cell_type == "s":
                        value = shared[int(child.text)]
                    else:
                        value = child.text
                elif local_name(child.tag) == "is":
                    value = "".join(
                        node.text or ""
                        for node in child.iter()
                        if local_name(node.tag) == "t"
                    )
            row[index] = value
        matrix.append(row)

    header_map = {index: value for index, value in matrix[0].items()}
    records: list[dict[str, str]] = []
    for row in matrix[1:]:
        records.append(
            {
                header_map.get(index, str(index)): value
                for index, value in row.items()
            }
        )
    return records


def normalized_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return "".join(value.split()).casefold()


def normalized_age(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").strip()
    return value[:-1].strip() if value.upper().endswith("Y") else value


def parse_exam_date(value: str) -> dt.date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return dt.datetime.fromtimestamp(
            (float(value) - 25569) * 86400, tz=dt.timezone.utc
        ).date()
    except (ValueError, OverflowError):
        return None


def diagnosis_confidence(value: str) -> int:
    text = unicodedata.normalize("NFKC", value or "").strip()
    if not text or "未提供病理诊断" in text:
        return 0
    uncertain_markers = (
        "考虑",
        "可能",
        "待排",
        "待除外",
        "不除外",
        "？",
        "?",
    )
    if any(marker in text for marker in uncertain_markers):
        return 1
    return 2


def metadata_completeness(record: dict[str, str]) -> int:
    fields = [
        "检查号",
        "检查日期",
        "诊断",
        "制造商",
        "设备型号",
        "场强_T",
    ]
    return sum(bool((record.get(field) or "").strip()) for field in fields)


def canonical_sort_key(record: dict[str, str]) -> tuple:
    exam_date = parse_exam_date(record.get("检查日期", ""))
    case_number = int(record["新序号"])
    return (
        -diagnosis_confidence(record.get("诊断", "")),
        -metadata_completeness(record),
        exam_date or dt.date.max,
        case_number,
    )


def case_id(record: dict[str, str]) -> str:
    return f"Meningioma_{int(record['新序号']):03d}"


def safe_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_tree(source: Path, destination: Path) -> None:
    for current_root, dirnames, filenames in os.walk(source):
        current = Path(current_root)
        relative = current.relative_to(source)
        target_dir = destination / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            safe_link_or_copy(current / filename, target_dir / filename)


def write_csv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not SOURCE_DATA.is_dir():
        raise FileNotFoundError(SOURCE_DATA)
    if not SOURCE_WORKBOOK.is_file():
        raise FileNotFoundError(SOURCE_WORKBOOK)
    if (OUTPUT_ROOT / "audit_summary.json").exists():
        raise FileExistsError(
            f"Completed output already exists; refusing to overwrite: {OUTPUT_ROOT}"
        )

    records = read_workbook_rows(SOURCE_WORKBOOK)
    records = [record for record in records if case_id(record) not in EXCLUDED_CASES]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        key = normalized_name(record.get("姓名", ""))
        if not key:
            raise ValueError(f"Missing normalized patient name for {case_id(record)}")
        grouped[key].append(record)

    retained: list[dict[str, str]] = []
    removed: list[dict[str, object]] = []
    retained_rows: list[dict[str, object]] = []

    for group_number, group in enumerate(
        sorted(grouped.values(), key=lambda items: min(case_id(r) for r in items)),
        start=1,
    ):
        group.sort(key=canonical_sort_key)
        canonical = group[0]
        retained.append(canonical)
        patient_group = f"patient_{group_number:04d}"
        canonical_id = case_id(canonical)
        exam_date = parse_exam_date(canonical.get("检查日期", ""))
        retained_rows.append(
            {
                "case_id": canonical_id,
                "patient_group": patient_group,
                "dataset": canonical.get("数据集", ""),
                "sex": canonical.get("性别", ""),
                "age": normalized_age(canonical.get("年龄", "")),
                "exam_date": exam_date.isoformat() if exam_date else "",
                "diagnosis_confidence": diagnosis_confidence(
                    canonical.get("诊断", "")
                ),
                "manufacturer": canonical.get("制造商", ""),
                "device_model": canonical.get("设备型号", ""),
                "field_strength_t": canonical.get("场强_T", ""),
                "record_count_in_group": len(group),
                "removed_record_count": len(group) - 1,
            }
        )
        for duplicate in group[1:]:
            duplicate_date = parse_exam_date(duplicate.get("检查日期", ""))
            removed.append(
                {
                    "removed_case_id": case_id(duplicate),
                    "retained_case_id": canonical_id,
                    "patient_group": patient_group,
                    "reason": "same normalized full name; one examination retained",
                    "dataset": duplicate.get("数据集", ""),
                    "exam_date": duplicate_date.isoformat() if duplicate_date else "",
                    "sex": duplicate.get("性别", ""),
                    "age": normalized_age(duplicate.get("年龄", "")),
                    "diagnosis_confidence": diagnosis_confidence(
                        duplicate.get("诊断", "")
                    ),
                }
            )

    retained.sort(key=lambda record: int(record["新序号"]))

    for subset in ("model_128x256x256", "native"):
        for record in retained:
            selected_case = case_id(record)
            source = SOURCE_DATA / subset / selected_case
            destination = OUTPUT_ROOT / subset / selected_case
            if not source.exists():
                raise FileNotFoundError(source)
            link_tree(source, destination)
    for record in retained:
        selected_case = case_id(record)
        source = SOURCE_DATA / "qc" / f"{selected_case}.png"
        destination = OUTPUT_ROOT / "qc" / source.name
        if not source.exists():
            raise FileNotFoundError(source)
        safe_link_or_copy(source, destination)

    manifests = OUTPUT_ROOT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    write_csv(manifests / "retained_cases.csv", MANIFEST_HEADERS, retained_rows)
    write_csv(manifests / "removed_duplicates.csv", REMOVED_HEADERS, removed)

    duplicate_group_count = sum(
        1 for group in grouped.values() if len(group) > 1
    )
    summary = {
        "source_case_count_after_exclusion": len(records),
        "source_workbook_row_count": len(records) + len(EXCLUDED_CASES),
        "excluded_cases": EXCLUDED_CASES,
        "candidate_patient_groups": len(grouped),
        "duplicate_groups_removed": duplicate_group_count,
        "removed_case_count": len(removed),
        "retained_case_count": len(retained),
        "grouping_rule": "normalized exact full name",
        "canonical_rule": (
            "prefer definite diagnosis; then metadata completeness; "
            "then earliest exam date; then lowest case number"
        ),
        "source_data_modified": False,
        "contains_patient_names": False,
        "contains_examination_ids": False,
        "output_tree": {
            "model_128x256x256": len(
                [p for p in (OUTPUT_ROOT / "model_128x256x256").rglob("*") if p.is_file()]
            ),
            "native": len(
                [p for p in (OUTPUT_ROOT / "native").rglob("*") if p.is_file()]
            ),
            "qc": len(
                [p for p in (OUTPUT_ROOT / "qc").rglob("*") if p.is_file()]
            ),
        },
    }
    summary["output_manifest_sha256"] = {
        name: hashlib.sha256((manifests / name).read_bytes()).hexdigest()
        for name in ("retained_cases.csv", "removed_duplicates.csv")
    }
    (OUTPUT_ROOT / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "# Meningioma 653 de-duplicated dataset\n\n"
        f"- Retained cases: {len(retained)}\n"
        f"- Removed duplicate cases: {len(removed)}\n"
        f"- Candidate patient groups: {len(grouped)}\n"
        "- Source data was not modified.\n"
        "- Patient names and examination IDs are deliberately absent from the "
        "generated manifests.\n\n"
        "Selection order: definite diagnosis, metadata completeness, earliest "
        "examination date, then lowest case number.\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
