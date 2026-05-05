"""Phecode mapping resource management.

Downloads and caches the official Phecode Map 1.2 ICD-9-CM and
ICD-10-CM mapping files used to translate raw diagnosis codes
into Phecode categories.
"""
from __future__ import annotations

import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import clean_scalar, ensure_dir, normalize_column_names


PHECODE_MAP_CANDIDATE_URLS = [
    "https://phewascatalog.org/phewas/data/Phecode_map_v1_2_icd9_icd10cm.csv.zip",
    "https://phewascatalog.org/files/Phecode_map_v1_2_icd9_icd10cm.csv",
    "https://phewascatalog.org/files/Phecode_map_v1_2_icd9_icd10cm.csv.zip",
]

PHECODE_DEFINITIONS_CANDIDATE_URLS = [
    "https://phewascatalog.org/phewas/data/phecode_definitions1.2.csv.zip",
    "https://phewascatalog.org/files/phecode_definitions1.2.csv",
    "https://phewascatalog.org/files/phecode_definitions1.2.csv.zip",
]


def _download_first_available(urls: list[str], out_dir: Path) -> Path:
    last_error: Exception | None = None
    for url in urls:
        filename = url.rsplit("/", 1)[-1]
        target = out_dir / filename
        try:
            try:
                response = urllib.request.urlopen(url, timeout=60)
            except (ssl.SSLCertVerificationError, urllib.error.URLError) as exc:
                if not isinstance(exc, ssl.SSLCertVerificationError) and not isinstance(
                    getattr(exc, "reason", None), ssl.SSLCertVerificationError
                ):
                    raise
                context = ssl._create_unverified_context()
                response = urllib.request.urlopen(url, timeout=60, context=context)
            with response:
                with target.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            if target.suffix == ".zip":
                with zipfile.ZipFile(target) as archive:
                    archive.extractall(out_dir)
                csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if csv_members:
                    return out_dir / csv_members[0]
            return target
        except Exception as exc:  # pragma: no cover - depends on network mirrors
            last_error = exc
            if target.exists():
                target.unlink()
    raise RuntimeError(f"Unable to download Phecode resource. Last error: {last_error}")


def download_phecode_resources(out_dir: str | Path) -> dict[str, str]:
    out_dir = ensure_dir(out_dir)
    map_path = _download_first_available(PHECODE_MAP_CANDIDATE_URLS, out_dir)
    definitions_path = _download_first_available(PHECODE_DEFINITIONS_CANDIDATE_URLS, out_dir)
    return {
        "map_path": str(map_path),
        "definitions_path": str(definitions_path),
    }


def normalize_icd_code(value: object) -> str:
    raw = clean_scalar(value).upper()
    raw = raw.replace(".", "")
    raw = raw.replace(" ", "")
    return raw


def infer_vocabulary_id(value: object) -> str | None:
    code = normalize_icd_code(value)
    if not code:
        return None
    if code[0].isdigit():
        return "ICD9CM"
    if code[0].isalpha():
        return "ICD10CM"
    return None


def normalize_vocabulary(value: object, code: object | None = None) -> str | None:
    text = clean_scalar(value).upper().replace("-", "").replace("_", "")
    if text in {"9", "ICD9", "ICD9CM", "ICD9DX"}:
        return "ICD9CM"
    if text in {"10", "ICD10", "ICD10CM", "ICD10DX"}:
        return "ICD10CM"
    if code is not None:
        return infer_vocabulary_id(code)
    return None


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_phecode_map(map_path: str | Path, definitions_path: str | Path | None = None) -> pd.DataFrame:
    raw = pd.read_csv(map_path, dtype=str, encoding="latin1")
    raw = normalize_column_names(raw)
    columns = list(raw.columns)

    code_col = _pick_column(
        columns,
        ["icd", "icd_code", "icd9", "icd10", "icd9cm", "icd10cm", "icd9cm_v1", "icd10cm_v1", "code"],
    )
    phecode_col = _pick_column(columns, ["phecode", "phe_code"])
    name_col = _pick_column(columns, ["phecode_str", "phenotype", "phecode_string", "phecodestring", "description"])
    category_col = _pick_column(columns, ["phecode_category", "phecodecategory", "category"])
    vocab_col = _pick_column(columns, ["vocabulary_id", "vocabulary", "flag", "code_type", "icd_version"])

    if code_col is None or phecode_col is None:
        raise ValueError(
            f"Phecode map must contain ICD and phecode columns. Found columns: {columns}"
        )

    result = pd.DataFrame(
        {
            "icd_code": raw[code_col],
            "norm_code": raw[code_col].map(normalize_icd_code),
            "phecode": raw[phecode_col].astype(str).str.strip(),
        }
    )

    if vocab_col:
        result["vocabulary_id"] = [
            normalize_vocabulary(vocab, code)
            for vocab, code in zip(raw[vocab_col], raw[code_col], strict=False)
        ]
    else:
        result["vocabulary_id"] = result["norm_code"].map(infer_vocabulary_id)

    if name_col:
        result["phecode_str"] = raw[name_col].astype(str).str.strip()
    else:
        result["phecode_str"] = np.nan
    if category_col:
        result["phecode_category"] = raw[category_col].astype(str).str.strip()

    definitions = load_phecode_definitions(definitions_path) if definitions_path else None
    if definitions is not None:
        result = result.merge(definitions, on="phecode", how="left", suffixes=("", "_definition"))
        if "phecode_str_definition" in result.columns:
            result["phecode_str"] = result["phecode_str"].replace({"nan": np.nan}).fillna(
                result["phecode_str_definition"]
            )
            result = result.drop(columns=["phecode_str_definition"])
        if "phecode_category_definition" in result.columns:
            if "phecode_category" in result.columns:
                result["phecode_category"] = result["phecode_category"].replace({"nan": np.nan}).fillna(
                    result["phecode_category_definition"]
                )
            else:
                result["phecode_category"] = result["phecode_category_definition"]
            result = result.drop(columns=["phecode_category_definition"])

    result = result.dropna(subset=["norm_code", "phecode", "vocabulary_id"])
    result = result[result["norm_code"] != ""]
    return result.drop_duplicates()


def load_phecode_definitions(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    raw = pd.read_csv(path, dtype=str, encoding="latin1")
    raw = normalize_column_names(raw)
    columns = list(raw.columns)
    phecode_col = _pick_column(columns, ["phecode", "phe_code"])
    name_col = _pick_column(columns, ["phenotype", "phecode_str", "description"])
    category_col = _pick_column(columns, ["category", "phecode_category", "group"])
    if phecode_col is None:
        return None

    result = pd.DataFrame({"phecode": raw[phecode_col].astype(str).str.strip()})
    if name_col:
        result["phecode_str"] = raw[name_col].astype(str).str.strip()
    if category_col:
        result["phecode_category"] = raw[category_col].astype(str).str.strip()
    return result.drop_duplicates("phecode")


def map_diagnoses_to_phecodes(
    diagnoses: pd.DataFrame,
    phecode_map: pd.DataFrame,
    code_col: str = "source_code",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if code_col not in diagnoses.columns:
        raise ValueError(f"diagnoses is missing {code_col}")

    dx = diagnoses.copy()
    dx["norm_code"] = dx[code_col].map(normalize_icd_code)
    if "vocabulary_id" not in dx.columns:
        dx["vocabulary_id"] = dx[code_col].map(infer_vocabulary_id)
    else:
        dx["vocabulary_id"] = [
            normalize_vocabulary(vocab, code)
            for vocab, code in zip(dx["vocabulary_id"], dx[code_col], strict=False)
        ]

    mapped = dx.merge(
        phecode_map,
        on=["norm_code", "vocabulary_id"],
        how="left",
        suffixes=("", "_map"),
    )

    events = mapped.dropna(subset=["phecode"]).copy()
    unmapped = mapped[mapped["phecode"].isna()].copy()

    audit = (
        mapped.groupby("vocabulary_id", dropna=False)
        .agg(
            rows=("person_id", "size"),
            mapped_rows=("phecode", lambda values: values.notna().sum()),
            unique_codes=("norm_code", "nunique"),
        )
        .reset_index()
    )
    audit["mapping_rate"] = audit["mapped_rows"] / audit["rows"]

    unmapped_summary = (
        unmapped.groupby(["vocabulary_id", "norm_code", code_col], dropna=False)
        .agg(rows=("person_id", "size"), persons=("person_id", "nunique"))
        .reset_index()
        .sort_values(["rows", "persons"], ascending=False)
    )
    return events, audit, unmapped_summary
