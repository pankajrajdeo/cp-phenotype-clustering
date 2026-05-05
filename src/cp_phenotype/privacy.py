"""Subgroup-size checks for controlled phenotype data sharing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir, safe_to_csv, write_json


GMFCS_LEVELS = ["I", "II", "III", "IV", "V"]


def _normalize_cluster(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    if len(text) == 1 and text.upper() in {"A", "B", "C", "D", "E"}:
        return text.upper()
    return text


def _normalize_gmfcs(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    text = text.replace("LEVEL", "").replace("GMFCS", "").strip()
    roman = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V"}
    if text in GMFCS_LEVELS:
        return text
    if text in roman:
        return roman[text]
    return None


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _choose_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    lower_to_original = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise ValueError(f"Could not find {label} column. Tried: {candidates}")


def load_assignments_and_cohort(
    assignments_path: str | Path,
    cohort_path: str | Path,
    cluster_col: str | None = None,
    gmfcs_col: str | None = None,
    person_col: str = "person_id",
    assignment_person_col: str | None = None,
    cohort_person_col: str | None = None,
) -> pd.DataFrame:
    assignments = _read_table(assignments_path)
    cohort = _read_table(cohort_path)
    person_candidates = [person_col, "person_id", "PERSON_ID", "h5ad_id", "numeric_person_id"]
    assignment_person_col = assignment_person_col or _choose_column(
        assignments,
        person_candidates,
        "assignment person ID",
    )
    cohort_person_col = cohort_person_col or _choose_column(
        cohort,
        person_candidates,
        "cohort person ID",
    )
    cluster_col = cluster_col or _choose_column(
        assignments,
        [
            "stored_graph_published",
            "reference_subtype",
            "stored_graph_letter",
            "subtype",
            "cluster",
            "Cluster",
        ],
        "cluster",
    )
    gmfcs_col = gmfcs_col or _choose_column(cohort, ["GMFCS", "GMFCS_M", "gmfcs_level"], "GMFCS")

    left = assignments[[assignment_person_col, cluster_col]].rename(
        columns={assignment_person_col: "person_id", cluster_col: "cluster"}
    )
    right = cohort[[cohort_person_col, gmfcs_col]].rename(
        columns={cohort_person_col: "person_id", gmfcs_col: "gmfcs"}
    )
    left["person_id"] = left["person_id"].astype(str)
    right["person_id"] = right["person_id"].astype(str)
    merged = left.merge(right, on="person_id", how="inner")
    return merged


def compute_subgroup_sizes(
    data: pd.DataFrame,
    cluster_col: str = "cluster",
    gmfcs_col: str = "gmfcs",
) -> pd.DataFrame:
    if cluster_col not in data.columns or gmfcs_col not in data.columns:
        raise ValueError(f"Input must contain {cluster_col!r} and {gmfcs_col!r}")
    work = data[[cluster_col, gmfcs_col]].copy()
    work["cluster"] = work[cluster_col].map(_normalize_cluster)
    work["gmfcs"] = work[gmfcs_col].map(_normalize_gmfcs)
    work = work.dropna(subset=["cluster", "gmfcs"])
    counts = (
        work.groupby(["cluster", "gmfcs"], dropna=False)
        .size()
        .rename("n_patients")
        .reset_index()
        .sort_values(["cluster", "gmfcs"])
    )
    return counts


def check_minimum_cell_size(counts: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    result = counts.copy()
    result["threshold"] = int(threshold)
    result["below_threshold"] = result["n_patients"] < int(threshold)
    return result


def generate_privacy_report(counts: pd.DataFrame, threshold: int = 10) -> str:
    checked = check_minimum_cell_size(counts, threshold)
    minimum = int(checked["n_patients"].min()) if not checked.empty else 0
    below = checked[checked["below_threshold"]]
    lines = [
        "# Subgroup Size Audit",
        "",
        f"- Threshold: `{int(threshold)}` patients",
        f"- Subgroups evaluated: `{len(checked)}`",
        f"- Minimum subgroup size: `{minimum}`",
        f"- Subgroups below threshold: `{len(below)}`",
        "",
    ]
    if below.empty:
        lines.append("All evaluated Cluster x GMFCS subgroups meet the minimum cell-size threshold.")
    else:
        lines.extend(
            [
                "Subgroups below threshold require masking, collapsing, or controlled-access handling.",
                "",
                "| Cluster | GMFCS | n_patients |",
                "|---|---:|---:|",
            ]
        )
        for row in below.itertuples(index=False):
            lines.append(f"| {row.cluster} | {row.gmfcs} | {row.n_patients} |")
    lines.append("")
    return "\n".join(lines)


def run_privacy_check(
    assignments_path: str | Path,
    cohort_path: str | Path,
    out_dir: str | Path,
    threshold: int = 10,
    cluster_col: str | None = None,
    gmfcs_col: str | None = None,
    person_col: str = "person_id",
    assignment_person_col: str | None = None,
    cohort_person_col: str | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    merged = load_assignments_and_cohort(
        assignments_path,
        cohort_path,
        cluster_col=cluster_col,
        gmfcs_col=gmfcs_col,
        person_col=person_col,
        assignment_person_col=assignment_person_col,
        cohort_person_col=cohort_person_col,
    )
    counts = compute_subgroup_sizes(merged)
    checked = check_minimum_cell_size(counts, threshold)
    report = generate_privacy_report(counts, threshold)

    safe_to_csv(checked, out_dir / "subgroup_counts.csv")
    (out_dir / "subgroup_size_audit.md").write_text(report, encoding="utf-8")
    summary = {
        "assignments": str(assignments_path),
        "cohort": str(cohort_path),
        "threshold": int(threshold),
        "n_joined_patients": int(merged["person_id"].nunique()),
        "n_subgroups": int(len(checked)),
        "minimum_subgroup_size": int(checked["n_patients"].min()) if not checked.empty else 0,
        "n_below_threshold": int(checked["below_threshold"].sum()) if not checked.empty else 0,
        "counts": str(out_dir / "subgroup_counts.csv"),
        "report": str(out_dir / "subgroup_size_audit.md"),
    }
    write_json(summary, out_dir / "subgroup_size_audit.json")
    return summary
