"""Reference REHAB/Clarity artifact matrix reconstruction.

This module rebuilds the final patient-by-Phecode matrix from the recovered
reference artifacts. It is intentionally separate from the generic OMOP
extraction path because the reference workflow used a frozen, already
harmonized REHAB/Clarity extract with an ICD-to-ICD10-to-Phecode step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ArtifactPaths, _as_dense, _norm_id
from .utils import ensure_dir, safe_to_csv, safe_to_parquet, write_json


REFERENCE_DEMOGRAPHIC_COLUMNS = [
    "PERSON_ID",
    "GENDER_CONCEPT_ID",
    "GENDER_SOURCE_VALUE",
    "NUM_VISITS_TOTAL",
    "RACE_CONCEPT_ID",
    "RACE_SOURCE_VALUE",
    "ETHNICITY_CONCEPT_ID",
    "DEATH_DATE",
    "GMFCS_M",
    "CP_DX_SS",
    "CP_DX_AGE_INT",
    "NUM_VISITS_TOTAL_FILTERED",
    "VISIT_LENGTH",
    "INPERSON_VISIT_NUM",
    "gmfm_score",
    "qol_score",
]


DIRECT_DATE_COLUMNS = {
    "BIRTH_DATETIME",
    "YEAR_OF_BIRTH",
    "MONTH_OF_BIRTH",
    "DAY_OF_BIRTH",
    "FIRST_VISIT_DATETIME",
    "DEATH_DATE",
    "CP_DX_DATE",
}


def _clean_phecode(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _clean_reference_person_id(value: object) -> str:
    text = _norm_id(value)
    if text.endswith("-ALL"):
        text = text[: -len("-ALL")]
    return text


def _load_reference_pivot(paths: ArtifactPaths) -> pd.DataFrame:
    pivot = pd.read_csv(paths.pivot, low_memory=False)
    if "PERSON_ID" not in pivot.columns:
        raise ValueError(f"Reference pivot is missing PERSON_ID: {paths.pivot}")
    pivot.index = pivot["PERSON_ID"].map(_norm_id)
    pivot = pivot.drop(columns=["PERSON_ID"])
    pivot.columns = [_clean_phecode(col) for col in pivot.columns]
    pivot = pivot.T.groupby(level=0).max().T
    pivot.index.name = "person_id"
    return (pivot > 0).astype("uint8")


def _load_reference_patient_metadata(paths: ArtifactPaths) -> pd.DataFrame:
    events = pd.read_parquet(paths.filtered_events)
    columns = [col for col in REFERENCE_DEMOGRAPHIC_COLUMNS if col in events.columns]
    metadata = (
        events.sort_values(["PERSON_ID", "CONDITION_START_DATETIME"], na_position="last")
        .drop_duplicates("PERSON_ID")
        .loc[:, columns]
        .copy()
    )
    metadata["person_id"] = metadata["PERSON_ID"].map(_norm_id)
    return metadata


def strip_direct_date_fields(metadata: pd.DataFrame) -> pd.DataFrame:
    """Remove direct date/year fields from shareable reference metadata.

    The paper-facing cohort metadata keeps derived fields such as
    CP_DX_AGE_INT, VISIT_LENGTH, and visit counts, but removes dates,
    birth dates, and year/month/day-of-birth values. Death is retained only as
    a binary indicator when a source death date is available.
    """
    result = metadata.copy()
    if "DEATH_DATE" in result.columns:
        result["DEATH"] = result["DEATH_DATE"].notna().astype("uint8")
    drop_columns = [col for col in result.columns if col.upper() in DIRECT_DATE_COLUMNS]
    return result.drop(columns=drop_columns, errors="ignore")


def _load_h5ad_binary(paths: ArtifactPaths) -> pd.DataFrame | None:
    if not paths.final_h5ad.exists():
        return None
    import anndata as ad

    reference = ad.read_h5ad(paths.final_h5ad)
    x = _as_dense(reference.X)
    ids = [_clean_reference_person_id(value) for value in reference.obs_names]
    columns = [_clean_phecode(value) for value in reference.var_names]
    matrix = pd.DataFrame((x > 0).astype("uint8"), index=ids, columns=columns)
    matrix = matrix.T.groupby(level=0).max().T
    matrix.index.name = "person_id"
    return matrix


def build_reference_matrix(
    root: str | Path,
    out_dir: str | Path,
    require_gmfcs: bool = True,
) -> dict[str, Any]:
    """Rebuild the final reference feature matrix from recovered artifacts.

    The original recovered funnel is:
    1. `cp_demo_ori.parquet`: 10,104 CP patients.
    2. `cpphe_pivot_s.csv`: 6,806 patients with mapped Phecode features.
    3. final clustering matrix: 3,618 patients with non-null `GMFCS_M`.

    The final matrix is a row subset of `cpphe_pivot_s.csv`; this function
    reproduces that row subset and verifies it against the stored h5ad when
    present.
    """
    paths = ArtifactPaths(Path(root))
    out_dir = ensure_dir(out_dir)

    pivot = _load_reference_pivot(paths)
    metadata = _load_reference_patient_metadata(paths)
    metadata = metadata[metadata["person_id"].isin(pivot.index)].copy()

    if require_gmfcs:
        selected_ids = metadata.loc[metadata["GMFCS_M"].notna(), "person_id"]
    else:
        selected_ids = metadata["person_id"]
    selected_ids = list(dict.fromkeys(selected_ids.astype(str)))

    matrix = pivot.loc[selected_ids].copy()
    metadata = metadata.set_index("person_id").loc[selected_ids].reset_index()
    metadata = strip_direct_date_fields(metadata)

    h5ad_matrix = _load_h5ad_binary(paths)
    verification: dict[str, Any] = {
        "h5ad_available": h5ad_matrix is not None,
        "matches_h5ad_common_cells": None,
        "h5ad_common_patients": None,
        "h5ad_common_features": None,
        "h5ad_cell_differences": None,
    }
    if h5ad_matrix is not None:
        common_ids = [pid for pid in h5ad_matrix.index if pid in matrix.index]
        common_cols = [col for col in h5ad_matrix.columns if col in matrix.columns]
        left = matrix.loc[common_ids, common_cols].to_numpy(dtype="uint8")
        right = h5ad_matrix.loc[common_ids, common_cols].to_numpy(dtype="uint8")
        verification.update(
            {
                "h5ad_common_patients": int(len(common_ids)),
                "h5ad_common_features": int(len(common_cols)),
                "matches_h5ad_common_cells": bool(np.array_equal(left, right)),
                "h5ad_cell_differences": int(np.not_equal(left, right).sum()),
            }
        )

    feature_metadata = pd.DataFrame(
        {
            "feature_id": matrix.columns.astype(str),
            "phecode": matrix.columns.astype(str),
        }
    )

    safe_to_parquet(matrix, out_dir / "feature_matrix.parquet")
    safe_to_csv(metadata, out_dir / "cohort_reference.csv")
    safe_to_csv(feature_metadata, out_dir / "feature_metadata.csv")

    summary = {
        "root": str(Path(root)),
        "require_gmfcs": bool(require_gmfcs),
        "pivot_patients": int(pivot.shape[0]),
        "pivot_features": int(pivot.shape[1]),
        "selected_patients": int(matrix.shape[0]),
        "selected_features": int(matrix.shape[1]),
        "selected_positive_entries": int(matrix.to_numpy(dtype="uint8").sum()),
        **verification,
        "feature_matrix": str(out_dir / "feature_matrix.parquet"),
        "cohort_reference": str(out_dir / "cohort_reference.csv"),
        "feature_metadata": str(out_dir / "feature_metadata.csv"),
    }
    write_json(summary, out_dir / "reference_matrix_summary.json")
    return summary
