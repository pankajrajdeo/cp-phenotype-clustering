"""Feature matrix construction from raw diagnosis extracts.

Maps ICD-9/ICD-10 codes to Phecodes, builds a binary patient-by-Phecode
matrix, and applies cohort filters (GMFCS, minimum visits, excluded codes).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .phecodes import load_phecode_definitions, load_phecode_map, map_diagnoses_to_phecodes
from .utils import ensure_dir, make_feature_id, safe_to_csv, safe_to_parquet, write_json


def _phecode_keys(value: object, feature_prefix: str = "phe_") -> set[str]:
    raw = str(value).strip()
    keys = {raw, make_feature_id(raw, feature_prefix)}
    if raw.endswith(".0"):
        trimmed = raw[:-2]
        keys.update({trimmed, make_feature_id(trimmed, feature_prefix)})
    return keys


def _exclude_phecodes(events: pd.DataFrame, exclude_phecodes: list[str] | tuple[str, ...], feature_prefix: str) -> pd.DataFrame:
    if not exclude_phecodes:
        return events
    excluded: set[str] = set()
    for code in exclude_phecodes:
        excluded.update(_phecode_keys(code, feature_prefix))
    keep = ~events["phecode"].map(lambda value: bool(_phecode_keys(value, feature_prefix) & excluded))
    return events.loc[keep].copy()


def eligible_patients_from_cohort(
    cohort_path: str | Path | None,
    min_visits: int | None = None,
    require_gmfcs: bool = False,
) -> tuple[set[str] | None, dict[str, int | bool | None]]:
    if cohort_path is None:
        return None, {"cohort_filter_applied": False, "min_visits": min_visits, "require_gmfcs": require_gmfcs}
    cohort = pd.read_parquet(cohort_path)
    cohort["person_id"] = cohort["person_id"].astype(str)
    before = int(cohort["person_id"].nunique())
    work = cohort.copy()
    if min_visits is not None:
        work = work[pd.to_numeric(work["num_visits"], errors="coerce").fillna(0) >= int(min_visits)]
    if require_gmfcs:
        work = work[work["gmfcs_level"].notna()]
    eligible = set(work["person_id"].astype(str))
    return eligible, {
        "cohort_filter_applied": True,
        "cohort_patients_before_filter": before,
        "cohort_patients_after_filter": int(len(eligible)),
        "min_visits": min_visits,
        "require_gmfcs": require_gmfcs,
    }


def _has_premapped_phecodes(diagnoses: pd.DataFrame) -> bool:
    return "phecode" in diagnoses.columns and diagnoses["phecode"].notna().any()


def _events_from_premapped_phecodes(diagnoses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = diagnoses[diagnoses["phecode"].notna()].copy()
    events["phecode"] = events["phecode"].astype(str).str.strip()
    events = events[events["phecode"] != ""].copy()
    if "phecode_str" not in events.columns:
        events["phecode_str"] = events["phecode"]
    else:
        events["phecode_str"] = events["phecode_str"].fillna(events["phecode"]).astype(str)

    audit = (
        pd.DataFrame(
            {
                "mapping_source": ["premapped_phecode"],
                "diagnosis_rows": [int(len(diagnoses))],
                "mapped_rows": [int(len(events))],
                "unmapped_rows": [int(len(diagnoses) - len(events))],
                "mapped_patients": [int(events["person_id"].nunique())],
                "mapped_phecodes": [int(events["phecode"].nunique())],
            }
        )
    )
    unmapped = diagnoses[diagnoses["phecode"].isna()].copy()
    return events, audit, unmapped


def build_feature_matrix(
    diagnoses_path: str | Path,
    map_path: str | Path,
    out_dir: str | Path,
    definitions_path: str | Path | None = None,
    cohort_path: str | Path | None = None,
    min_visits: int | None = None,
    require_gmfcs: bool = False,
    min_patients: int = 3,
    feature_prefix: str = "phe_",
    exclude_phecodes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str | int]:
    out_dir = ensure_dir(out_dir)
    diagnoses = pd.read_parquet(diagnoses_path)
    diagnoses["person_id"] = diagnoses["person_id"].astype(str)
    eligible_patients, filter_summary = eligible_patients_from_cohort(cohort_path, min_visits, require_gmfcs)
    if eligible_patients is not None:
        diagnoses = diagnoses[diagnoses["person_id"].isin(eligible_patients)].copy()

    mapping_source = "premapped_phecode" if _has_premapped_phecodes(diagnoses) else "icd_phecode_map"
    if mapping_source == "premapped_phecode":
        events, audit, unmapped = _events_from_premapped_phecodes(diagnoses)
    else:
        empty_premapped_columns = [
            column
            for column in ["phecode", "phecode_str", "phecode_category"]
            if column in diagnoses.columns and diagnoses[column].notna().sum() == 0
        ]
        diagnoses = diagnoses.drop(columns=empty_premapped_columns)
        phecode_map = load_phecode_map(map_path, definitions_path)
        events, audit, unmapped = map_diagnoses_to_phecodes(diagnoses, phecode_map)

    events = _exclude_phecodes(events, list(exclude_phecodes or []), feature_prefix)
    events["feature_id"] = events["phecode"].map(lambda value: make_feature_id(value, feature_prefix))

    feature_metadata = (
        events[["feature_id", "phecode", "phecode_str"] + [col for col in ["phecode_category"] if col in events.columns]]
        .drop_duplicates("feature_id")
        .sort_values("feature_id")
    )

    matrix = pd.crosstab(events["person_id"], events["feature_id"])
    matrix = (matrix > 0).astype("uint8")
    keep = matrix.columns[matrix.sum(axis=0) >= min_patients]
    matrix = matrix.loc[:, keep].sort_index(axis=1)
    matrix.index.name = "person_id"
    feature_metadata = feature_metadata[feature_metadata["feature_id"].isin(matrix.columns)]

    safe_to_parquet(events, out_dir / "phecode_events.parquet")
    safe_to_parquet(matrix, out_dir / "feature_matrix.parquet")
    safe_to_csv(feature_metadata, out_dir / "feature_metadata.csv")
    safe_to_csv(audit, out_dir / "mapping_audit.csv")
    safe_to_csv(unmapped, out_dir / "unmapped_codes.csv")
    write_json(filter_summary, out_dir / "cohort_filter_summary.json")

    return {
        "n_diagnosis_rows": int(len(diagnoses)),
        "n_phecode_event_rows": int(len(events)),
        "n_patients": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "mapping_source": mapping_source,
        "excluded_phecodes": list(exclude_phecodes or []),
        **filter_summary,
        "feature_matrix": str(out_dir / "feature_matrix.parquet"),
        "feature_metadata": str(out_dir / "feature_metadata.csv"),
    }


def find_phecode_resource_paths(map_dir: str | Path) -> tuple[Path, Path | None]:
    map_dir = Path(map_dir)
    map_candidates = sorted(map_dir.glob("*icd9*icd10cm*.csv")) + sorted(
        map_dir.glob("Phecode_map_v1_2_icd9_icd10cm.csv")
    )
    if not map_candidates:
        raise FileNotFoundError(f"No combined ICD-9/ICD-10 Phecode map found in {map_dir}")

    definition_candidates = sorted(map_dir.glob("*definitions*1.2*.csv"))
    return map_candidates[0], definition_candidates[0] if definition_candidates else None


def load_feature_metadata(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)
