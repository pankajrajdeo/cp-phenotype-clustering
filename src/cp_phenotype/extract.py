"""Data extraction from Oracle clinical data schemas.

Queries patient demographics, diagnoses, visits, procedures,
drugs, death records, and GMFCS scores from the configured
database and writes raw parquet files for downstream processing.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .db import read_sql
from .utils import ensure_dir, safe_to_parquet, write_json


ROMAN_TO_INT = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
}


def parse_gmfcs(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    text = text.replace("LEVEL", " ")
    text = re.sub(r"[^A-Z0-9]+", " ", text).strip()
    tokens = text.split()
    for token in tokens:
        if token in ROMAN_TO_INT:
            return ROMAN_TO_INT[token]
        if token.isdigit() and 1 <= int(token) <= 5:
            return int(token)
    if text in ROMAN_TO_INT:
        return ROMAN_TO_INT[text]
    return None


def _limit_clause(limit_rows: int | None) -> str:
    return f" FETCH FIRST {int(limit_rows)} ROWS ONLY" if limit_rows else ""


def _order_clause(order_by: str, limit_rows: int | None) -> str:
    return f" ORDER BY {order_by}" if limit_rows else ""


def _table_owner_name(table: str) -> tuple[str, str]:
    owner, name = table.split(".", 1)
    return owner.upper(), name.upper()


def _table_columns(table: str, env_path: str | Path | None = None) -> set[str]:
    owner, name = _table_owner_name(table)
    sql = f"""
        SELECT column_name
        FROM all_tab_columns
        WHERE owner = '{owner}' AND table_name = '{name}'
    """
    return set(read_sql(sql, env_path)["COLUMN_NAME"].str.lower())


def _nullable_expr(column: str, data_type: str) -> str:
    return f"CAST(NULL AS {data_type}) AS {column}"


def _qualified_or_null(alias: str, column: str, data_type: str, available: set[str]) -> str:
    return f"{alias}.{column}" if column in available else _nullable_expr(column, data_type)


def _qualified_min_or_null(alias: str, column: str, data_type: str, available: set[str]) -> str:
    return f"MIN({alias}.{column}) AS {column}" if column in available else _nullable_expr(column, data_type)


def extract_schema_snapshot(
    tables: list[str],
    out_path: str | Path,
    env_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = []
    for table in tables:
        owner, name = _table_owner_name(table)
        sql = f"""
            SELECT owner, table_name, column_name, data_type, column_id
            FROM all_tab_columns
            WHERE owner = '{owner}' AND table_name = '{name}'
            ORDER BY column_id
        """
        columns = read_sql(sql, env_path)
        rows.append(
            {
                "table": table,
                "columns": columns[["COLUMN_NAME", "DATA_TYPE"]].to_dict(orient="records"),
            }
        )
    snapshot = {"tables": rows}
    write_json(snapshot, out_path)
    return snapshot


def extract_cohort(
    table: str,
    env_path: str | Path | None = None,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    available = _table_columns(table, env_path)
    desired = [
        ("person_id", "NUMBER"),
        ("person_source_value", "VARCHAR2(4000)"),
        ("gender_source_value", "VARCHAR2(4000)"),
        ("race_source_value", "VARCHAR2(4000)"),
        ("ethnicity_source_value", "VARCHAR2(4000)"),
        ("year_of_birth", "NUMBER"),
        ("month_of_birth", "NUMBER"),
        ("day_of_birth", "NUMBER"),
        ("birth_date", "DATE"),
        ("birth_datetime", "TIMESTAMP"),
    ]
    select_exprs = [column if column in available else _nullable_expr(column, data_type) for column, data_type in desired]
    sql = f"""
        SELECT
            {", ".join(select_exprs)}
        FROM {table}
        {_order_clause("person_id", limit_rows)}
        {_limit_clause(limit_rows)}
    """
    return read_sql(sql, env_path).rename(columns=str.lower)


def extract_diagnoses(
    diagnosis_table: str,
    cohort_table: str,
    env_path: str | Path | None = None,
    mode: str = "cp_person_conditions",
    distinct_patient_code: bool = False,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    available = _table_columns(diagnosis_table, env_path)
    premapped_selects = [
        _qualified_or_null("d", "phecode", "VARCHAR2(4000)", available),
        _qualified_or_null("d", "phecode_str", "VARCHAR2(4000)", available),
        _qualified_or_null("d", "phecode_category", "VARCHAR2(4000)", available),
        _qualified_or_null("d", "icd_10", "VARCHAR2(4000)", available),
        _qualified_or_null("d", "icd", "VARCHAR2(4000)", available),
    ]
    premapped_grouped_selects = [
        _qualified_min_or_null("d", "phecode", "VARCHAR2(4000)", available),
        _qualified_min_or_null("d", "phecode_str", "VARCHAR2(4000)", available),
        _qualified_min_or_null("d", "phecode_category", "VARCHAR2(4000)", available),
        _qualified_min_or_null("d", "icd_10", "VARCHAR2(4000)", available),
        _qualified_min_or_null("d", "icd", "VARCHAR2(4000)", available),
    ]
    if mode == "cp_person_conditions" and distinct_patient_code:
        sql = f"""
            SELECT
                d.person_id,
                CAST(NULL AS NUMBER) AS visit_occurrence_id,
                CAST(NULL AS NUMBER) AS condition_occurrence_id,
                d.dx_code_char AS source_code,
                MIN(d.dx_name) AS source_name,
                MIN(d.dx_category) AS diagnosis_category,
                MIN(d.condition_start_datetime) AS diagnosis_datetime,
                MIN(CAST(d.condition_start_datetime AS DATE)) AS diagnosis_date,
                {", ".join(premapped_grouped_selects)},
                COUNT(*) AS event_count,
                'cp_person_conditions_patient_code' AS source_table
            FROM {diagnosis_table} d
            JOIN {cohort_table} c ON c.person_id = d.person_id
            WHERE d.dx_code_char IS NOT NULL
            GROUP BY d.person_id, d.dx_code_char
            {_order_clause("d.person_id, d.dx_code_char", limit_rows)}
            {_limit_clause(limit_rows)}
        """
    elif mode == "condition_occurrence" and distinct_patient_code:
        sql = f"""
            SELECT
                d.person_id,
                CAST(NULL AS NUMBER) AS visit_occurrence_id,
                CAST(NULL AS NUMBER) AS condition_occurrence_id,
                d.condition_source_value AS source_code,
                CAST(NULL AS VARCHAR2(4000)) AS source_name,
                CAST(NULL AS VARCHAR2(4000)) AS diagnosis_category,
                MIN(d.condition_start_datetime) AS diagnosis_datetime,
                MIN(d.condition_start_date) AS diagnosis_date,
                COUNT(*) AS event_count,
                'condition_occurrence_patient_code' AS source_table
            FROM {diagnosis_table} d
            JOIN {cohort_table} c ON c.person_id = d.person_id
            WHERE d.condition_source_value IS NOT NULL
            GROUP BY d.person_id, d.condition_source_value
            {_order_clause("d.person_id, d.condition_source_value", limit_rows)}
            {_limit_clause(limit_rows)}
        """
    elif mode == "condition_occurrence":
        sql = f"""
            SELECT
                d.person_id,
                d.visit_occurrence_id,
                d.condition_occurrence_id,
                d.condition_source_value AS source_code,
                CAST(NULL AS VARCHAR2(4000)) AS source_name,
                CAST(NULL AS VARCHAR2(4000)) AS diagnosis_category,
                d.condition_start_datetime AS diagnosis_datetime,
                d.condition_start_date AS diagnosis_date,
                1 AS event_count,
                'condition_occurrence' AS source_table
            FROM {diagnosis_table} d
            JOIN {cohort_table} c ON c.person_id = d.person_id
            WHERE d.condition_source_value IS NOT NULL
            {_order_clause("d.person_id, d.condition_start_datetime", limit_rows)}
            {_limit_clause(limit_rows)}
        """
    else:
        sql = f"""
            SELECT
                d.person_id,
                d.visit_occurrence_id,
                d.condition_occurrence_id,
                d.dx_code_char AS source_code,
                d.dx_name AS source_name,
                d.dx_category AS diagnosis_category,
                d.condition_start_datetime AS diagnosis_datetime,
                CAST(d.condition_start_datetime AS DATE) AS diagnosis_date,
                {", ".join(premapped_selects)},
                1 AS event_count,
                'cp_person_conditions' AS source_table
            FROM {diagnosis_table} d
            JOIN {cohort_table} c ON c.person_id = d.person_id
            WHERE d.dx_code_char IS NOT NULL
            {_order_clause("d.person_id, d.condition_start_datetime", limit_rows)}
            {_limit_clause(limit_rows)}
        """
    return read_sql(sql, env_path).rename(columns=str.lower)


def extract_visits(
    visits_table: str,
    cohort_table: str,
    env_path: str | Path | None = None,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    sql = f"""
        SELECT
            v.person_id,
            v.visit_occurrence_id,
            v.visit_start_date,
            v.visit_start_datetime,
            v.visit_end_date,
            v.visit_end_datetime,
            v.visit_source_value,
            v.visit_concept_id
        FROM {visits_table} v
        JOIN {cohort_table} c ON c.person_id = v.person_id
        {_order_clause("v.person_id, v.visit_start_date", limit_rows)}
        {_limit_clause(limit_rows)}
    """
    return read_sql(sql, env_path).rename(columns=str.lower)


def extract_visit_summary(
    visits_table: str,
    cohort_table: str,
    env_path: str | Path | None = None,
) -> pd.DataFrame:
    sql = f"""
        SELECT
            v.person_id,
            COUNT(DISTINCT v.visit_occurrence_id) AS num_visits,
            MIN(v.visit_start_date) AS first_visit_date,
            MAX(v.visit_start_date) AS last_visit_date,
            MAX(v.visit_start_date) - MIN(v.visit_start_date) AS visit_duration_days
        FROM {visits_table} v
        JOIN {cohort_table} c ON c.person_id = v.person_id
        GROUP BY v.person_id
    """
    return read_sql(sql, env_path).rename(columns=str.lower)


def extract_deaths(
    deaths_table: str,
    fallback_table: str,
    cohort_table: str,
    env_path: str | Path | None = None,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    for table in [deaths_table, fallback_table]:
        sql = f"""
            SELECT
                d.person_id,
                d.death_date,
                d.death_datetime,
                '{table}' AS source_table
            FROM {table} d
            JOIN {cohort_table} c ON c.person_id = d.person_id
            {_order_clause("d.person_id", limit_rows)}
            {_limit_clause(limit_rows)}
        """
        try:
            return read_sql(sql, env_path).rename(columns=str.lower)
        except Exception:
            continue
    return pd.DataFrame(columns=["person_id", "death_date", "death_datetime", "source_table"])


def extract_gmfcs_observations(
    observations_table: str,
    cohort_table: str,
    env_path: str | Path | None = None,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    available = _table_columns(observations_table, env_path)
    select_exprs = [
        _qualified_or_null("o", "person_id", "NUMBER", available),
        _qualified_or_null("o", "visit_occurrence_id", "NUMBER", available),
        _qualified_or_null("o", "observation_id", "NUMBER", available),
        _qualified_or_null("o", "observation_source_value", "VARCHAR2(4000)", available),
        _qualified_or_null("o", "observation_date", "DATE", available),
        _qualified_or_null("o", "observation_datetime", "TIMESTAMP", available),
        _qualified_or_null("o", "value_as_string", "VARCHAR2(4000)", available),
        _qualified_or_null("o", "value_source_value", "VARCHAR2(4000)", available),
        _qualified_or_null("o", "value_as_number", "NUMBER", available),
        _qualified_or_null("o", "value_as_concept_id", "NUMBER", available),
    ]
    sql = f"""
        SELECT
            {", ".join(select_exprs)}
        FROM {observations_table} o
        JOIN {cohort_table} c ON c.person_id = o.person_id
        WHERE UPPER(o.observation_source_value) LIKE '%GMFCS%'
        {_order_clause("o.person_id, o.observation_datetime", limit_rows)}
        {_limit_clause(limit_rows)}
    """
    df = read_sql(sql, env_path).rename(columns=str.lower)
    if df.empty:
        df["gmfcs_level"] = pd.Series(dtype="float")
        return df
    values = df["value_source_value"].fillna(df["value_as_string"]).fillna(df["value_as_number"])
    df["gmfcs_level"] = values.map(parse_gmfcs)
    return df


def latest_gmfcs(gmfcs: pd.DataFrame) -> pd.DataFrame:
    if gmfcs.empty or "gmfcs_level" not in gmfcs.columns:
        return pd.DataFrame(columns=["person_id", "gmfcs_level", "gmfcs_observation_datetime", "gmfcs_source"])
    valid = gmfcs.dropna(subset=["gmfcs_level"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["person_id", "gmfcs_level", "gmfcs_observation_datetime", "gmfcs_source"])
    valid["sort_datetime"] = pd.to_datetime(valid["observation_datetime"]).fillna(
        pd.to_datetime(valid["observation_date"])
    )
    valid = valid.sort_values(["person_id", "sort_datetime", "observation_id"])
    latest = valid.groupby("person_id", as_index=False).tail(1)
    return latest.rename(
        columns={
            "observation_datetime": "gmfcs_observation_datetime",
            "observation_source_value": "gmfcs_source",
        }
    )[["person_id", "gmfcs_level", "gmfcs_observation_datetime", "gmfcs_source"]]


def summarize_visits(visits: pd.DataFrame) -> pd.DataFrame:
    if visits.empty:
        return pd.DataFrame(columns=["person_id", "num_visits", "first_visit_date", "last_visit_date", "visit_duration_days"])
    summary_columns = {"person_id", "num_visits", "first_visit_date", "last_visit_date", "visit_duration_days"}
    if summary_columns.issubset(visits.columns):
        return visits.loc[:, ["person_id", "num_visits", "first_visit_date", "last_visit_date", "visit_duration_days"]].copy()
    work = visits.copy()
    work["visit_start_date"] = pd.to_datetime(work["visit_start_date"])
    summary = (
        work.groupby("person_id")
        .agg(
            num_visits=("visit_occurrence_id", "nunique"),
            first_visit_date=("visit_start_date", "min"),
            last_visit_date=("visit_start_date", "max"),
        )
        .reset_index()
    )
    summary["visit_duration_days"] = (
        summary["last_visit_date"] - summary["first_visit_date"]
    ).dt.days
    return summary


def build_cohort_table(cohort: pd.DataFrame, visits: pd.DataFrame, deaths: pd.DataFrame, gmfcs: pd.DataFrame) -> pd.DataFrame:
    result = cohort.copy()
    result = result.merge(summarize_visits(visits), on="person_id", how="left")
    latest = latest_gmfcs(gmfcs)
    result = result.merge(latest, on="person_id", how="left")
    death_flags = deaths[["person_id"]].drop_duplicates().assign(death=1) if not deaths.empty else pd.DataFrame(columns=["person_id", "death"])
    result = result.merge(death_flags, on="person_id", how="left")
    result["death"] = result["death"].fillna(0).astype(int)
    result["num_visits"] = result["num_visits"].fillna(0).astype(int)
    return result


def extract_all(
    config: dict[str, Any],
    out_dir: str | Path,
    env_path: str | Path | None = None,
    limit_rows: int | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    primary_sources = config["sources"]
    extraction = config.get("extraction", {})
    visits_mode = extraction.get("visits_mode", "summary")

    sources = primary_sources
    cohort = extract_cohort(sources["cohort_table"], env_path, limit_rows)
    diagnoses = extract_diagnoses(
        sources["diagnosis_table"],
        sources["cohort_table"],
        env_path,
        mode=sources.get("diagnosis_mode", "cp_person_conditions"),
        distinct_patient_code=bool(sources.get("diagnosis_distinct_patient_code", False)),
        limit_rows=limit_rows,
    )
    primary_diagnosis_rows = int(len(diagnoses))
    primary_diagnosis_patients = int(diagnoses["person_id"].nunique()) if not diagnoses.empty else 0

    minimum_diagnosis_patients = int(primary_sources.get("minimum_diagnosis_patients", 0))
    fallback_used = False
    if (
        not limit_rows
        and primary_sources.get("fallback")
        and primary_diagnosis_patients < minimum_diagnosis_patients
    ):
        sources = primary_sources["fallback"]
        cohort = extract_cohort(sources["cohort_table"], env_path, limit_rows)
        diagnoses = extract_diagnoses(
            sources["diagnosis_table"],
            sources["cohort_table"],
            env_path,
            mode=sources.get("diagnosis_mode", "cp_person_conditions"),
            distinct_patient_code=bool(sources.get("diagnosis_distinct_patient_code", False)),
            limit_rows=limit_rows,
        )
        fallback_used = True

    if visits_mode == "raw" or limit_rows:
        visits = extract_visits(sources["visits_table"], sources["cohort_table"], env_path, limit_rows)
    else:
        visits = extract_visit_summary(sources["visits_table"], sources["cohort_table"], env_path)
    safe_to_parquet(visits, out_dir / "visits.parquet")
    death_fallback_table = sources.get("death_fallback_table", sources["deaths_table"])
    deaths = extract_deaths(
        sources["deaths_table"],
        death_fallback_table,
        sources["cohort_table"],
        env_path,
        limit_rows,
    )
    safe_to_parquet(deaths, out_dir / "deaths.parquet")
    gmfcs = extract_gmfcs_observations(sources["observations_table"], sources["cohort_table"], env_path, limit_rows)
    safe_to_parquet(gmfcs, out_dir / "gmfcs_observations.parquet")
    safe_to_parquet(diagnoses, out_dir / "diagnoses.parquet")
    cohort_enriched = build_cohort_table(cohort, visits, deaths, gmfcs)

    safe_to_parquet(cohort_enriched, out_dir / "cohort.parquet")

    tables = [
        sources["cohort_table"],
        sources["visits_table"],
        sources["observations_table"],
        sources["deaths_table"],
        death_fallback_table,
        sources["diagnosis_table"],
    ]
    if fallback_used:
        tables.extend([primary_sources["cohort_table"], primary_sources["diagnosis_table"]])
    tables = list(dict.fromkeys(tables))
    extract_schema_snapshot(tables, out_dir / "schema_snapshot.json", env_path)

    summary = {
        "selected_cohort_table": sources["cohort_table"],
        "selected_diagnosis_table": sources["diagnosis_table"],
        "diagnosis_fallback_used": fallback_used,
        "primary_cohort_table": primary_sources["cohort_table"],
        "primary_diagnosis_rows": primary_diagnosis_rows,
        "primary_diagnosis_patients": primary_diagnosis_patients,
        "cohort_rows": int(len(cohort_enriched)),
        "diagnosis_rows": int(len(diagnoses)),
        "diagnosis_patients": int(diagnoses["person_id"].nunique()) if not diagnoses.empty else 0,
        "diagnosis_unique_source_codes": int(diagnoses["source_code"].nunique()) if not diagnoses.empty else 0,
        "visit_rows": int(len(visits)),
        "death_rows": int(len(deaths)),
        "gmfcs_observation_rows": int(len(gmfcs)),
        "gmfcs_patients": int(cohort_enriched["gmfcs_level"].notna().sum()),
        "limit_rows": limit_rows,
    }
    write_json(summary, out_dir / "extract_summary.json")
    return summary
