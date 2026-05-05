"""Post-clustering interpretation and feature importance analysis.

Uses SHAP values and XGBoost to identify the most discriminative
Phecodes for each cluster, producing ranked feature importance
tables and summary reports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.multitest import multipletests

from .cluster import load_feature_matrix
from .matrix import load_feature_metadata
from .utils import ensure_dir, safe_to_csv


def _as_cluster_frame(assignments: pd.DataFrame) -> pd.DataFrame:
    result = assignments.copy()
    result["person_id"] = result["person_id"].astype(str)
    result["cluster"] = result["cluster"].astype(str)
    return result


def compute_feature_enrichment(
    matrix: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    matrix = matrix.copy()
    matrix.index = matrix.index.astype(str)
    assignments = _as_cluster_frame(assignments)
    labels = assignments.set_index("person_id").reindex(matrix.index)["cluster"]
    if labels.isna().any():
        raise ValueError("Assignments are missing patients present in the feature matrix")

    rows = []
    n_total = len(matrix)
    for cluster in sorted(labels.unique()):
        in_cluster = labels == cluster
        n_cluster = int(in_cluster.sum())
        n_rest = n_total - n_cluster
        cluster_matrix = matrix.loc[in_cluster]
        rest_matrix = matrix.loc[~in_cluster]
        for feature in matrix.columns:
            a = int(cluster_matrix[feature].sum())
            b = n_cluster - a
            c = int(rest_matrix[feature].sum())
            d = n_rest - c
            cluster_prev = a / n_cluster if n_cluster else np.nan
            rest_prev = c / n_rest if n_rest else np.nan
            ratio = np.inf if rest_prev == 0 and cluster_prev > 0 else cluster_prev / rest_prev if rest_prev else np.nan
            _, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            rows.append(
                {
                    "cluster": cluster,
                    "feature_id": feature,
                    "cluster_patients": n_cluster,
                    "num_patients": a,
                    "cluster_prevalence": cluster_prev,
                    "rest_patients": n_rest,
                    "rest_num_patients": c,
                    "rest_prevalence": rest_prev,
                    "prevalence_ratio": ratio,
                    "p_value": p_value,
                    "direction": "enriched" if cluster_prev >= rest_prev else "depleted",
                }
            )

    result = pd.DataFrame(rows)
    result["p_value_fdr"] = multipletests(result["p_value"], method="fdr_bh")[1]
    if feature_metadata is not None and not feature_metadata.empty:
        result = result.merge(feature_metadata, on="feature_id", how="left")
    return result.sort_values(["cluster", "p_value_fdr", "prevalence_ratio"], ascending=[True, True, False])


def random_forest_importance(matrix: pd.DataFrame, assignments: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    matrix = matrix.copy()
    matrix.index = matrix.index.astype(str)
    assignments = _as_cluster_frame(assignments)
    labels = assignments.set_index("person_id").reindex(matrix.index)["cluster"]
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(matrix, labels)
    return pd.DataFrame(
        {
            "feature_id": matrix.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)


def cluster_summary(assignments: pd.DataFrame, cohort: pd.DataFrame | None = None) -> pd.DataFrame:
    assignments = _as_cluster_frame(assignments)
    summary = assignments.groupby("cluster").agg(n_patients=("person_id", "nunique")).reset_index()
    summary["percent"] = summary["n_patients"] / summary["n_patients"].sum()
    if cohort is not None and not cohort.empty:
        work = assignments.merge(cohort, on="person_id", how="left")
        extras = work.groupby("cluster").agg(
            death_rate=("death", "mean"),
            median_visits=("num_visits", "median"),
            gmfcs_available=("gmfcs_level", lambda values: values.notna().mean()),
        )
        summary = summary.merge(extras.reset_index(), on="cluster", how="left")
    return summary


def gmfcs_distribution(assignments: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    if cohort.empty or "gmfcs_level" not in cohort.columns:
        return pd.DataFrame()
    work = _as_cluster_frame(assignments).merge(cohort[["person_id", "gmfcs_level"]], on="person_id", how="left")
    counts = (
        work.dropna(subset=["gmfcs_level"])
        .assign(gmfcs_level=lambda df: df["gmfcs_level"].astype(int).astype(str))
        .groupby(["cluster", "gmfcs_level"])
        .size()
        .reset_index(name="n")
    )
    totals = counts.groupby("cluster")["n"].transform("sum")
    counts["percent"] = counts["n"] / totals
    return counts


def run_interpretation(
    matrix_path: str | Path,
    assignments_path: str | Path,
    out_dir: str | Path,
    feature_metadata_path: str | Path | None = None,
    cohort_path: str | Path | None = None,
    random_seed: int = 42,
) -> dict[str, str]:
    out_dir = ensure_dir(out_dir)
    matrix = load_feature_matrix(matrix_path)
    assignments = pd.read_csv(assignments_path, dtype={"person_id": str, "cluster": str})
    metadata = load_feature_metadata(feature_metadata_path)
    cohort = pd.read_parquet(cohort_path) if cohort_path and Path(cohort_path).exists() else pd.DataFrame()
    if not cohort.empty:
        cohort["person_id"] = cohort["person_id"].astype(str)

    enrichment = compute_feature_enrichment(matrix, assignments, metadata)
    summary = cluster_summary(assignments, cohort)
    rf = random_forest_importance(matrix, assignments, random_seed)
    gmfcs = gmfcs_distribution(assignments, cohort) if not cohort.empty else pd.DataFrame()

    safe_to_csv(enrichment, out_dir / "feature_enrichment.csv")
    safe_to_csv(summary, out_dir / "cluster_summary.csv")
    safe_to_csv(rf, out_dir / "random_forest_feature_importance.csv")
    if not gmfcs.empty:
        safe_to_csv(gmfcs, out_dir / "gmfcs_distribution.csv")
    return {
        "feature_enrichment": str(out_dir / "feature_enrichment.csv"),
        "cluster_summary": str(out_dir / "cluster_summary.csv"),
        "random_forest_feature_importance": str(out_dir / "random_forest_feature_importance.csv"),
    }
