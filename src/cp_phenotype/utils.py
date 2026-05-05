"""Shared utility functions for the Cerebral Palsy sub-phenotyping pipeline.

Provides file I/O helpers (JSON, CSV, YAML), directory management,
and optional patient ID hashing for de-identification.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=True)


def safe_to_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def make_feature_id(phecode: object, prefix: str = "phe_") -> str:
    value = str(phecode).strip()
    value = re.sub(r"\.0$", "", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return f"{prefix}{value}"


def make_study_id(person_id: object, salt: str | None = None) -> str | None:
    if not salt:
        return None
    digest = hmac.new(
        salt.encode("utf-8"),
        str(person_id).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest[:24]


def add_study_id(df: pd.DataFrame, person_col: str = "person_id") -> pd.DataFrame:
    salt = os.environ.get("CP_PHENOTYPE_ID_SALT")
    if not salt or person_col not in df.columns:
        return df
    result = df.copy()
    result["study_id"] = result[person_col].map(lambda value: make_study_id(value, salt))
    return result


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def clean_scalar(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()
