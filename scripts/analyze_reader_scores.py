from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


ROOT = Path(
    os.environ.get(
        "EPG_SYNTH_READER_ROOT",
        Path(__file__).resolve().parents[1]
        / "data"
        / "EPG-Synth_Formal_Reader_Study_OrientationCorrected_20261112",
    )
).expanduser().resolve()
BLIND = ROOT / "BLIND_REVIEW"
MAPPING_PATH = ROOT / "COORDINATOR_ONLY" / "candidate_mapping_coordinator_only.csv"
OUTPUT = Path(
    os.environ.get(
        "EPG_SYNTH_READER_OUTPUT",
        Path(__file__).resolve().parents[1] / "runs" / "reader_study_analysis_20261112",
    )
).expanduser().resolve()

READERS = ("R01", "R02", "R03")
ARMS = ("Real T1c", "EPG-Synth", "WFM-Adapted")
SCORE_FIELDS = {
    "Global technical quality": "global_quality",
    "Lesion enhancement conspicuity": "enhancement_conspicuity",
    "Enhancement fidelity": "enhancement_fidelity",
    "Boundary and dural relation": "boundary_dural",
    "Diagnostic confidence": "diagnostic_confidence",
}
PRIMARY = "Lesion enhancement conspicuity"
KEY_SECONDARY = "Enhancement fidelity"
SEED = 20261112
BOOTSTRAP_DRAWS = 20000
ICC_BOOTSTRAP_DRAWS = 20000
PERMUTATION_DRAWS = 50000


def validate_score(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value).is_integer()
        and 1 <= float(value) <= 5
    )


def fmt_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_DRAWS, values.size))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def permutation_mean_p(values: np.ndarray, rng: np.random.Generator) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    if observed <= 1e-15:
        return 1.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=(PERMUTATION_DRAWS, values.size))
    permuted = np.abs((signs * values).mean(axis=1))
    return float((1 + np.sum(permuted >= observed)) / (PERMUTATION_DRAWS + 1))


def bh_adjust(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for rank in range(count - 1, -1, -1):
        index = int(order[rank])
        running = min(running, values[index] * count / (rank + 1))
        adjusted[index] = running
    return [float(value) for value in adjusted]


def icc_from_matrix(matrix: np.ndarray) -> tuple[float | None, float | None]:
    n, k = matrix.shape
    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ss_rows = k * float(np.sum((row_means - grand) ** 2))
    ss_cols = n * float(np.sum((col_means - grand) ** 2))
    residual = matrix - row_means[:, None] - col_means[None, :] + grand
    ss_error = float(np.sum(residual**2))
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator_single = (
        ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    )
    denominator_average = ms_rows + (ms_cols - ms_error) / n
    if abs(denominator_single) <= 1e-15 or abs(denominator_average) <= 1e-15:
        return None, None
    icc_single = (ms_rows - ms_error) / denominator_single
    icc_average = (ms_rows - ms_error) / denominator_average
    return float(icc_single), float(icc_average)


def bootstrap_icc_ci(
    matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float | None, float | None, float | None, float | None]:
    single_values = []
    average_values = []
    for _ in range(ICC_BOOTSTRAP_DRAWS):
        indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        single, average = icc_from_matrix(matrix[indices, :])
        if single is not None:
            single_values.append(single)
            average_values.append(average)
    if not single_values:
        return None, None, None, None
    return (
        float(np.percentile(single_values, 2.5)),
        float(np.percentile(single_values, 97.5)),
        float(np.percentile(average_values, 2.5)),
        float(np.percentile(average_values, 97.5)),
    )


def quadratic_weighted_kappa(
    first: np.ndarray,
    second: np.ndarray,
    categories: int = 5,
) -> float | None:
    observed = np.zeros((categories, categories), dtype=float)
    for left, right in zip(first, second, strict=True):
        observed[int(left) - 1, int(right) - 1] += 1
    observed /= observed.sum()
    row = observed.sum(axis=1)
    column = observed.sum(axis=0)
    expected = np.outer(row, column)
    values = np.arange(categories)
    weights = (values[:, None] - values[None, :]) ** 2 / (categories - 1) ** 2
    observed_disagreement = float(np.sum(weights * observed))
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement <= 1e-15:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(MAPPING_PATH, encoding="utf-8-sig")
    mapping = mapping[
        [
            "subject",
            "cohort",
            "center",
            "size_quartile",
            "candidate_arm",
            "display_id",
        ]
    ].copy()

    records: list[dict[str, object]] = []
    qc: dict[str, object] = {}
    for reader in READERS:
        workbook_path = (
            BLIND / f"Reader_{reader}" / f"formal_scoring_workbook_{reader}.xlsx"
        )
        order_path = (
            BLIND / f"Reader_{reader}" / f"Reader_{reader}_image_order.csv"
        )
        workbook = load_workbook(workbook_path, data_only=False, read_only=True)
        sheet = workbook["病例评分"]
        order = pd.read_csv(order_path, encoding="utf-8-sig")
        order_ids = order["display_id"].tolist()
        reader_records: list[dict[str, object]] = []
        for row_number in range(8, 188):
            values = {
                "display_id": sheet.cell(row_number, 1).value,
                "reader_id": sheet.cell(row_number, 2).value,
                "session_type": sheet.cell(row_number, 3).value,
                "session_id": sheet.cell(row_number, 4).value,
                "review_date": sheet.cell(row_number, 5).value,
                "global_quality": sheet.cell(row_number, 6).value,
                "enhancement_conspicuity": sheet.cell(row_number, 7).value,
                "enhancement_fidelity": sheet.cell(row_number, 8).value,
                "boundary_dural": sheet.cell(row_number, 9).value,
                "diagnostic_confidence": sheet.cell(row_number, 10).value,
                "comments": sheet.cell(row_number, 11).value,
            }
            rec = {"reader": reader, "row_number": row_number, **values}
            reader_records.append(rec)
            records.append(rec)

        display_ids = [item["display_id"] for item in reader_records]
        score_values = [
            item[field]
            for item in reader_records
            for field in SCORE_FIELDS.values()
        ]
        qc[reader] = {
            "workbook": str(workbook_path),
            "rows": len(reader_records),
            "nonblank_display_ids": sum(bool(value) for value in display_ids),
            "unique_display_ids": len(set(display_ids)),
            "duplicate_display_ids": len(display_ids) - len(set(display_ids)),
            "all_scores_integer_1_to_5": all(validate_score(value) for value in score_values),
            "complete_five_score_rows": sum(
                all(validate_score(item[field]) for field in SCORE_FIELDS.values())
                for item in reader_records
            ),
            "review_date_nonblank": sum(bool(item["review_date"]) for item in reader_records),
            "reader_id_values": sorted(set(item["reader_id"] for item in reader_records)),
            "session_type_values": sorted(
                set(str(item["session_type"]) for item in reader_records)
            ),
            "workbook_order_matches_csv": display_ids == order_ids,
            "workbook_set_matches_csv": set(display_ids) == set(order_ids),
        }

    scores = pd.DataFrame(records).merge(
        mapping, on="display_id", how="left", validate="many_to_one"
    )
    if scores["subject"].isna().any():
        raise ValueError("Some Display_IDs could not be mapped to a subject.")
    for metric in SCORE_FIELDS.values():
        scores[metric] = pd.to_numeric(scores[metric], errors="coerce")
    scores["complete"] = scores[list(SCORE_FIELDS.values())].notna().all(axis=1)
    scores.to_csv(
        OUTPUT / "reader_scores_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coverage = []
    for (reader, subject), group in scores.groupby(["reader", "subject"]):
        coverage.append(
            {
                "reader": reader,
                "subject": subject,
                "unique_candidates": int(group["display_id"].nunique()),
                "unique_arms": int(group["candidate_arm"].nunique()),
                "arms": ";".join(sorted(group["candidate_arm"].dropna().unique())),
            }
        )
    coverage_frame = pd.DataFrame(coverage)
    qc["coverage"] = {
        "reader_case_pairs": len(coverage_frame),
        "all_have_three_unique_candidates": bool(
            (coverage_frame["unique_candidates"] == 3).all()
        ),
        "all_have_three_unique_arms": bool(
            (coverage_frame["unique_arms"] == 3).all()
        ),
        "unmapped_display_ids": int(scores["subject"].isna().sum()),
        "arm_counts_by_reader": {
            reader: {
                arm: int(count)
                for arm, count in group["candidate_arm"].value_counts().items()
            }
            for reader, group in scores.groupby("reader")
        },
    }

    # Descriptive summaries use all reader-case observations.
    descriptive_rows = []
    for arm in ARMS:
        for label, metric in SCORE_FIELDS.items():
            values = scores.loc[
                scores["candidate_arm"] == arm, metric
            ].dropna().to_numpy(dtype=float)
            descriptive_rows.append(
                {
                    "candidate_arm": arm,
                    "endpoint": label,
                    "n_observations": len(values),
                    "n_cases": int(
                        scores.loc[scores["candidate_arm"] == arm, "subject"].nunique()
                    ),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    descriptive = pd.DataFrame(descriptive_rows)
    descriptive.to_csv(
        OUTPUT / "reader_descriptives.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    by_reader_rows = []
    for reader in READERS:
        for arm in ARMS:
            for label, metric in SCORE_FIELDS.items():
                values = scores.loc[
                    (scores["reader"] == reader)
                    & (scores["candidate_arm"] == arm),
                    metric,
                ].dropna().to_numpy(dtype=float)
                by_reader_rows.append(
                    {
                        "reader": reader,
                        "candidate_arm": arm,
                        "endpoint": label,
                        "n_cases": len(values),
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "median": float(np.median(values)),
                        "q1": float(np.quantile(values, 0.25)),
                        "q3": float(np.quantile(values, 0.75)),
                    }
                )
    by_reader = pd.DataFrame(by_reader_rows)
    by_reader.to_csv(
        OUTPUT / "reader_descriptives_by_reader.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    # Case means average the three readers, respecting the paired design.
    case_means = (
        scores.groupby(["subject", "cohort", "center", "size_quartile", "candidate_arm"])[
            list(SCORE_FIELDS.values())
        ]
        .mean()
        .reset_index()
    )
    case_means.to_csv(
        OUTPUT / "reader_case_means.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    rng = np.random.default_rng(SEED)
    comparisons = []
    for candidate in ("WFM-Adapted", "Real T1c"):
        for label, metric in SCORE_FIELDS.items():
            wide = case_means.pivot(index="subject", columns="candidate_arm", values=metric)
            diff = (wide["EPG-Synth"] - wide[candidate]).dropna().to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(diff, rng)
            comparisons.append(
                {
                    "comparison": f"EPG-Synth minus {candidate}",
                    "candidate": candidate,
                    "endpoint": label,
                    "n_cases": len(diff),
                    "mean_difference": float(diff.mean()),
                    "median_difference": float(np.median(diff)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "permutation_p": permutation_mean_p(diff, rng),
                    "epg_better_cases": int((diff > 1e-12).sum()),
                    "epg_worse_cases": int((diff < -1e-12).sum()),
                    "tied_cases": int((np.abs(diff) <= 1e-12).sum()),
                    "improved_by_ge_1_point": int((diff >= 1.0).sum()),
                    "worsened_by_ge_1_point": int((diff <= -1.0).sum()),
                    "sd_of_paired_differences": float(diff.std(ddof=1)),
                    "standardized_paired_effect": float(
                        diff.mean() / diff.std(ddof=1)
                    ),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    primary_mask = comparison_frame["candidate"] == "WFM-Adapted"
    comparison_frame.loc[primary_mask, "bh_adjusted_p_within_5_endpoints"] = bh_adjust(
        comparison_frame.loc[primary_mask, "permutation_p"].tolist()
    )
    comparison_frame.to_csv(
        OUTPUT / "reader_paired_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    # Frequency table by candidate and endpoint.
    frequency_rows = []
    for arm in ARMS:
        for label, metric in SCORE_FIELDS.items():
            values = scores.loc[
                scores["candidate_arm"] == arm, metric
            ].dropna().astype(int)
            row = {
                "candidate_arm": arm,
                "endpoint": label,
                "n_observations": len(values),
            }
            for score_value in range(1, 6):
                row[f"score_{score_value}"] = int((values == score_value).sum())
            frequency_rows.append(row)
    pd.DataFrame(frequency_rows).to_csv(
        OUTPUT / "reader_score_frequencies.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Absolute-agreement ICC across the three readers, with bootstrap CIs.
    icc_rows = []
    for arm in ARMS:
        for label, metric in SCORE_FIELDS.items():
            matrix = (
                scores.loc[scores["candidate_arm"] == arm]
                .pivot_table(
                    index="subject",
                    columns="reader",
                    values=metric,
                    aggfunc="mean",
                )
                .reindex(columns=list(READERS))
                .to_numpy(dtype=float)
            )
            single, average = icc_from_matrix(matrix)
            single_low, single_high, average_low, average_high = bootstrap_icc_ci(
                matrix, rng
            )
            pairwise_kappas = []
            exact_agreement = 0
            for left_index in range(len(READERS)):
                for right_index in range(left_index + 1, len(READERS)):
                    kappa = quadratic_weighted_kappa(
                        matrix[:, left_index],
                        matrix[:, right_index],
                    )
                    if kappa is not None:
                        pairwise_kappas.append(kappa)
            exact_agreement = float(
                np.mean(np.all(matrix == matrix[:, [0]], axis=1))
            )
            icc_rows.append(
                {
                    "candidate_arm": arm,
                    "endpoint": label,
                    "icc_2_1_single_reader": single,
                    "icc_2_1_ci95_low": single_low,
                    "icc_2_1_ci95_high": single_high,
                    "icc_2_k_three_reader_average": average,
                    "icc_2_k_ci95_low": average_low,
                    "icc_2_k_ci95_high": average_high,
                    "mean_pairwise_quadratic_weighted_kappa": (
                        float(np.mean(pairwise_kappas))
                        if pairwise_kappas
                        else None
                    ),
                    "triple_reader_exact_agreement": exact_agreement,
                }
            )
    pd.DataFrame(icc_rows).to_csv(
        OUTPUT / "reader_interrater_icc.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    # Reader-specific paired differences for the primary endpoint.
    reader_primary_rows = []
    for reader in READERS:
        subset = scores.loc[scores["reader"] == reader]
        wide = subset.pivot(
            index="subject",
            columns="candidate_arm",
            values="enhancement_conspicuity",
        )
        for candidate in ("WFM-Adapted", "Real T1c"):
            diff = (wide["EPG-Synth"] - wide[candidate]).dropna().to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(diff, rng)
            reader_primary_rows.append(
                {
                    "reader": reader,
                    "candidate": candidate,
                    "endpoint": PRIMARY,
                    "n_cases": len(diff),
                    "mean_difference": float(diff.mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "permutation_p": permutation_mean_p(diff, rng),
                }
            )
    pd.DataFrame(reader_primary_rows).to_csv(
        OUTPUT / "reader_primary_by_reader.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    # Descriptive stratification for the primary endpoint using case means.
    strata_rows = []
    for grouping_name, columns in (
        ("cohort", ["cohort"]),
        ("center", ["center"]),
        ("size_quartile", ["size_quartile"]),
    ):
        for keys, group in case_means.groupby(columns, dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            label = " / ".join(str(value) for value in key_values)
            wide = group.pivot(
                index="subject",
                columns="candidate_arm",
                values="enhancement_conspicuity",
            )
            if not {"EPG-Synth", "WFM-Adapted"}.issubset(wide.columns):
                continue
            diff = (wide["EPG-Synth"] - wide["WFM-Adapted"]).dropna().to_numpy()
            if diff.size < 2:
                continue
            ci_low, ci_high = bootstrap_mean_ci(diff, rng)
            strata_rows.append(
                {
                    "stratum_type": grouping_name,
                    "stratum": label,
                    "n_cases": len(diff),
                    "epg_mean": float(wide["EPG-Synth"].dropna().mean()),
                    "wfm_mean": float(wide["WFM-Adapted"].dropna().mean()),
                    "mean_difference": float(diff.mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    pd.DataFrame(strata_rows).to_csv(
        OUTPUT / "reader_primary_stratified.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    qc["all_outputs_ready"] = True
    (OUTPUT / "reader_scoring_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(qc, ensure_ascii=False, indent=2))
    print("\nDESCRIPTIVES")
    print(descriptive.to_string(index=False))
    print("\nPAIRWISE")
    print(comparison_frame.to_string(index=False))
    print("\nPRIMARY BY READER")
    print(pd.DataFrame(reader_primary_rows).to_string(index=False))
    print("\nICC")
    print(pd.DataFrame(icc_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
