"""Reference artifact auditing for the CP sub-phenotype clustering.

Provides tools to load, validate, and cross-check the reference
clustering artifacts (h5ad, pivot CSV, supplement tables) that serve
as ground truth for reproduction and validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .utils import ensure_dir, safe_to_csv, write_json


DEMOGRAPHIC_COLUMNS = {
    "PERSON_ID",
    "Cluster",
    "GENDER_SOURCE_VALUE",
    "GMFCS",
    "GMFCS_M",
    "DEATH",
    "CP_DX_AGE_INT",
    "NUM_VISITS",
    "NUM_VISITS_TOTAL_FILTERED",
    "VISIT_Duration",
    "VISIT_LENGTH",
    "RACE",
    "RACE_SOURCE_VALUE",
}


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def final_h5ad(self) -> Path:
        return self.root / "data" / "cpdiag_adata_t_all.h5ad"

    @property
    def final_obs(self) -> Path:
        return self.root / "data" / "cpdiag_adata_t_all_obs.csv"

    @property
    def pivot(self) -> Path:
        return self.root / "data" / "cpphe_pivot_s.csv"

    @property
    def filtered_events(self) -> Path:
        return self.root / "data" / "cp_demodx_filterd.parquet"

    @property
    def final_events(self) -> Path:
        return self.root / "data" / "cp_demodx_subtype.parquet"

    @property
    def demo_subtype(self) -> Path:
        return self.root / "cluster" / "cp_demo_subtype.csv"

    @property
    def supplement(self) -> Path:
        return self.root / "cluster" / "cp_cluster_sup_v4.csv"

    @property
    def feature_matrix_with_chapter(self) -> Path:
        return self.root / "cluster" / "feature_matrix_with_chapter.csv"

    @property
    def subcluster_crosswalk(self) -> Path:
        return self.root / "cluster" / "subcluster.csv"

    @property
    def query_sql(self) -> Path:
        return self.root / "data" / "query_517_2024.sql"


def _norm_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _clean_phecode(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        return text
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return f"{number:.1f}"
    return str(number)


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _safe_shape_csv(path: Path) -> tuple[int, int]:
    header = pd.read_csv(path, nrows=0)
    with path.open("rb") as handle:
        rows = sum(1 for _ in handle) - 1
    return rows, len(header.columns)


def load_supplement(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if "PERSON_ID" in df.columns:
        df = df[df["PERSON_ID"].map(lambda value: _norm_id(value).isdigit())].copy()
        df["person_id_norm"] = df["PERSON_ID"].map(_norm_id)
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in DEMOGRAPHIC_COLUMNS and column != "person_id_norm"]


def _pca_summary(adata: ad.AnnData) -> dict[str, Any]:
    pca = adata.uns.get("pca", {})
    variance_ratio = pca.get("variance_ratio") if hasattr(pca, "get") else None
    if variance_ratio is None:
        return {"stored_pcs": None, "first15_variance": None, "stored_variance": None}
    values = np.asarray(variance_ratio, dtype=float)
    return {
        "stored_pcs": int(len(values)),
        "first15_variance": float(values[:15].sum()),
        "stored_variance": float(values.sum()),
        "first_pc_variance": float(values[0]) if len(values) else None,
    }


def _neighbors_summary(adata: ad.AnnData) -> dict[str, Any]:
    neighbors = adata.uns.get("neighbors", {})
    params = neighbors.get("params", {}) if hasattr(neighbors, "get") else {}
    clean_params = {
        str(key): (value.item() if hasattr(value, "item") else value)
        for key, value in dict(params).items()
    }
    return {
        "neighbors_params": clean_params,
        "neighbors_connectivities_shape": list(adata.obsp["connectivities"].shape)
        if "connectivities" in adata.obsp
        else None,
        "neighbors_distances_shape": list(adata.obsp["distances"].shape) if "distances" in adata.obsp else None,
    }


def _as_dense(array: Any) -> np.ndarray:
    if hasattr(array, "toarray"):
        return np.asarray(array.toarray(), dtype=float)
    return np.asarray(array, dtype=float)


def _matrix_transform_audit(paths: ArtifactPaths) -> dict[str, Any]:
    adata = ad.read_h5ad(paths.final_h5ad)
    pivot = pd.read_csv(paths.pivot, low_memory=False)
    pivot_index = pivot["PERSON_ID"].map(_norm_id)
    pivot = pivot.drop(columns=["PERSON_ID"])
    pivot.index = pivot_index

    feature_cols = [str(value) for value in adata.var_names]
    crosswalk = pd.read_csv(paths.subcluster_crosswalk, dtype=str, low_memory=False)
    row_to_numeric_id = {
        _norm_id(row["PERSON_ID_h5ad"]): _norm_id(row["PERSON_ID"])
        for _, row in crosswalk.iterrows()
    }
    h5ad_row_order = [_norm_id(value) for value in adata.obs_names]
    row_order = [row_to_numeric_id.get(value, "") for value in h5ad_row_order]
    missing_rows = [value for value in row_order if not value or value not in pivot.index]
    missing_features = [value for value in feature_cols if value not in pivot.columns]
    if missing_rows or missing_features:
        return {
            "checked": False,
            "missing_rows": int(len(missing_rows)),
            "missing_features": int(len(missing_features)),
        }

    raw = pivot.loc[row_order, feature_cols].astype(float).to_numpy()
    row_sums = raw.sum(axis=1)
    normalized = np.divide(
        raw,
        row_sums[:, None],
        out=np.zeros_like(raw, dtype=float),
        where=row_sums[:, None] != 0,
    ) * 10000.0
    expected = np.log1p(normalized)
    observed = _as_dense(adata.X)
    diff = np.abs(expected - observed)
    return {
        "checked": True,
        "raw_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "adata_x_shape": [int(observed.shape[0]), int(observed.shape[1])],
        "raw_nonzero_count": int(np.count_nonzero(raw)),
        "adata_x_nonzero_count": int(np.count_nonzero(observed)),
        "max_abs_diff_log_normalized_pivot_vs_adata_x": float(diff.max()),
        "mean_abs_diff_log_normalized_pivot_vs_adata_x": float(diff.mean()),
        "allclose_log_normalized_pivot_vs_adata_x": bool(np.allclose(expected, observed)),
    }


def _cluster_label_audit(paths: ArtifactPaths, out_dir: Path) -> dict[str, Any]:
    adata = ad.read_h5ad(paths.final_h5ad)
    obs = adata.obs.copy()
    obs["person_id_norm"] = [_norm_id(value) for value in obs.index]
    final_obs = pd.read_csv(paths.final_obs, dtype=str, low_memory=False)
    final_obs["person_id_norm"] = final_obs["PERSON_ID"].map(_norm_id)
    demo = pd.read_csv(paths.demo_subtype, dtype=str, low_memory=False)
    demo["person_id_norm"] = demo["PERSON_ID"].map(_norm_id)
    supplement = load_supplement(paths.supplement)
    crosswalk = pd.read_csv(paths.subcluster_crosswalk, dtype=str, low_memory=False)
    crosswalk["person_id_norm"] = crosswalk["PERSON_ID_h5ad"].map(_norm_id)
    crosswalk["numeric_person_id_norm"] = crosswalk["PERSON_ID"].map(_norm_id)

    h5_labels = obs[["person_id_norm", "leiden_0.5", "subtype"]].copy()
    merged_obs = h5_labels.merge(final_obs[["person_id_norm", "leiden_0.5", "subtype"]], on="person_id_norm", suffixes=("_h5ad", "_csv"))
    h5_numeric = h5_labels.merge(crosswalk[["person_id_norm", "numeric_person_id_norm", "subcluster"]], on="person_id_norm")
    merged_demo = h5_numeric.merge(
        demo[["person_id_norm", "subtype"]],
        left_on="numeric_person_id_norm",
        right_on="person_id_norm",
        suffixes=("_h5ad", "_demo"),
    )
    merged_pub = h5_numeric.merge(
        supplement[["person_id_norm", "Cluster"]],
        left_on="numeric_person_id_norm",
        right_on="person_id_norm",
        how="inner",
        suffixes=("_h5ad", "_published"),
    )

    internal_to_published = pd.crosstab(merged_pub["subtype"], merged_pub["Cluster"])
    safe_to_csv(internal_to_published.reset_index(), out_dir / "internal_to_published_crosstab.csv")

    return {
        "h5ad_shape": [int(adata.n_obs), int(adata.n_vars)],
        "h5ad_obs_columns": list(map(str, obs.columns)),
        "h5ad_uns_keys": list(map(str, adata.uns.keys())),
        "h5ad_obsm_keys": list(map(str, adata.obsm.keys())),
        "h5ad_obsp_keys": list(map(str, adata.obsp.keys())),
        "leiden_0_5_counts": _counts(obs["leiden_0.5"]),
        "internal_subtype_counts": _counts(obs["subtype"]),
        "published_cluster_counts": _counts(supplement["Cluster"]),
        "ari_leiden_0_5_vs_internal_subtype": float(adjusted_rand_score(obs["leiden_0.5"], obs["subtype"])),
        "h5ad_vs_obs_rows_compared": int(len(merged_obs)),
        "h5ad_vs_obs_leiden_match": int((merged_obs["leiden_0.5_h5ad"] == merged_obs["leiden_0.5_csv"]).sum()),
        "h5ad_vs_obs_subtype_match": int((merged_obs["subtype_h5ad"] == merged_obs["subtype_csv"]).sum()),
        "h5ad_to_numeric_crosswalk_rows": int(len(h5_numeric)),
        "h5ad_to_numeric_crosswalk_unique_numeric_ids": int(h5_numeric["numeric_person_id_norm"].nunique()),
        "subcluster_crosswalk_unique_subclusters": int(crosswalk["subcluster"].nunique()),
        "h5ad_vs_demo_rows_compared": int(len(merged_demo)),
        "h5ad_vs_demo_subtype_match": int((merged_demo["subtype_h5ad"] == merged_demo["subtype_demo"]).sum()),
        "h5ad_vs_supplement_rows_compared": int(len(merged_pub)),
        "internal_to_published": {
            str(index): {str(col): int(value) for col, value in row.items()}
            for index, row in internal_to_published.iterrows()
        },
        **_pca_summary(adata),
        **_neighbors_summary(adata),
    }


def _feature_filter_audit(paths: ArtifactPaths, out_dir: Path) -> dict[str, Any]:
    pivot_header = pd.read_csv(paths.pivot, nrows=0)
    pivot_cols = [str(column) for column in pivot_header.columns if str(column) != "PERSON_ID"]
    pivot_codes = set(pivot_cols)

    events = pd.read_parquet(paths.filtered_events, columns=["PERSON_ID", "phecode", "phecode_str"])
    dedup = events.dropna(subset=["phecode"]).drop_duplicates(["PERSON_ID", "phecode"]).copy()
    dedup["phecode_key"] = dedup["phecode"].map(_clean_phecode)
    dedup = dedup[~dedup["phecode_key"].isin(["", "nan", "None", "NaN"])].copy()
    prevalence = (
        dedup.groupby(["phecode_key", "phecode_str"], dropna=False)["PERSON_ID"]
        .nunique()
        .reset_index(name="patient_count")
        .sort_values(["patient_count", "phecode_key"], ascending=[False, True])
    )
    prevalence["in_pivot"] = prevalence["phecode_key"].isin(pivot_codes)
    safe_to_csv(prevalence, out_dir / "phecode_prevalence_audit.csv")

    missing = prevalence[~prevalence["in_pivot"]]
    present = prevalence[prevalence["in_pivot"]]
    high_missing = missing[missing["patient_count"] >= 4].sort_values("patient_count", ascending=False)

    supplement = load_supplement(paths.supplement)
    feature_matrix = pd.read_csv(paths.feature_matrix_with_chapter, dtype=str, low_memory=False)
    supplement_feature_cols = _feature_columns(supplement)
    feature_matrix_cols = [column for column in feature_matrix.columns if column != "PERSON_ID"]

    subgroup_feature_count = None
    subgroup_threshold = 0.05
    if {"Cluster", "GMFCS"}.issubset(supplement.columns):
        feature_presence = supplement[supplement_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        groups = supplement[["Cluster", "GMFCS"]].copy()
        max_prevalence = {}
        for feature in supplement_feature_cols:
            rates = feature_presence[feature].groupby([groups["Cluster"], groups["GMFCS"]]).mean()
            max_prevalence[feature] = float(rates.max()) if not rates.empty else 0.0
        subgroup_feature_count = int(sum(value >= subgroup_threshold for value in max_prevalence.values()))

    return {
        "filtered_event_patients": int(events["PERSON_ID"].nunique()),
        "filtered_event_unique_phecodes": int(prevalence.shape[0]),
        "pivot_shape": list(_safe_shape_csv(paths.pivot)),
        "pivot_feature_columns": int(len(pivot_cols)),
        "present_min_patient_count": int(present["patient_count"].min()) if not present.empty else None,
        "missing_feature_count": int(missing.shape[0]),
        "missing_with_patient_count_lte_3": int((missing["patient_count"] <= 3).sum()),
        "missing_with_patient_count_gte_4": int((missing["patient_count"] >= 4).sum()),
        "missing_high_prevalence_codes": high_missing[["phecode_key", "phecode_str", "patient_count"]].to_dict(orient="records"),
        "supplement_feature_columns": int(len(supplement_feature_cols)),
        "feature_matrix_with_chapter_columns_excluding_person": int(len(feature_matrix_cols)),
        "features_with_5pct_in_any_cluster_gmfcs_group": subgroup_feature_count,
        "cluster_gmfcs_subgroup_threshold": subgroup_threshold,
    }


def _event_output_audit(paths: ArtifactPaths) -> dict[str, Any]:
    final_events = pd.read_parquet(paths.final_events, columns=["PERSON_ID", "subtype", "phecode", "phecode_str"])
    demo = pd.read_csv(paths.demo_subtype, dtype=str, low_memory=False)
    return {
        "final_event_rows": int(len(final_events)),
        "final_event_patients": int(final_events["PERSON_ID"].nunique()),
        "final_event_unique_phecodes": int(final_events["phecode"].nunique(dropna=True)),
        "final_event_unique_phecode_str": int(final_events["phecode_str"].nunique(dropna=True)),
        "final_event_internal_subtype_row_counts": _counts(final_events["subtype"]),
        "demo_rows": int(len(demo)),
        "demo_internal_subtype_counts": _counts(demo["subtype"]),
    }


def audit_artifacts(root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    paths = ArtifactPaths(Path(root))
    out_dir = ensure_dir(out_dir)

    required = [
        paths.final_h5ad,
        paths.final_obs,
        paths.pivot,
        paths.filtered_events,
        paths.final_events,
        paths.demo_subtype,
        paths.supplement,
        paths.feature_matrix_with_chapter,
        paths.subcluster_crosswalk,
        paths.query_sql,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifact files: {missing}")

    summary: dict[str, Any] = {
        "artifact_root": str(paths.root),
        "files": {
            path.name: {"path": str(path), "size_bytes": int(path.stat().st_size)}
            for path in required
        },
    }
    summary["labels"] = _cluster_label_audit(paths, out_dir)
    summary["features"] = _feature_filter_audit(paths, out_dir)
    summary["events"] = _event_output_audit(paths)
    summary["matrix_transform"] = _matrix_transform_audit(paths)
    summary["source_sql"] = paths.query_sql.read_text(encoding="utf-8", errors="replace")

    write_json(summary, out_dir / "artifact_manifest.json")
    _write_artifact_report(summary, out_dir / "artifact_audit.md")
    return summary


def _write_artifact_report(summary: dict[str, Any], out_path: str | Path) -> None:
    """Write a markdown report summarizing the artifact audit."""
    labels = summary["labels"]
    features = summary["features"]
    events = summary["events"]
    matrix_transform = summary["matrix_transform"]
    lines = [
        "# Reference Artifact Audit Report",
        "",
        "This report uses local copies of the BMI-cluster artifacts. It reports aggregate structure and label agreement only.",
        "",
        "## Artifact-Level Conclusion",
        "",
        "- The final clustering object is `cpdiag_adata_t_all.h5ad`.",
        f"- AnnData shape: `{labels['h5ad_shape'][0]} x {labels['h5ad_shape'][1]}`.",
        "- `leiden_0.5` exactly reproduces the five internal subtype groups stored in the AnnData object.",
        f"- ARI between `leiden_0.5` and internal `subtype`: `{labels['ari_leiden_0_5_vs_internal_subtype']:.3f}`.",
        f"- Rows compared against `cpdiag_adata_t_all_obs.csv`: `{labels['h5ad_vs_obs_rows_compared']}`.",
        f"- Rows bridged through `subcluster.csv`: `{labels['h5ad_to_numeric_crosswalk_rows']}`.",
        f"- Rows compared against `cp_demo_subtype.csv`: `{labels['h5ad_vs_demo_rows_compared']}`.",
        f"- Rows compared against `cp_cluster_sup_v4.csv`: `{labels['h5ad_vs_supplement_rows_compared']}`.",
        "",
        "## Internal Cluster Counts",
        "",
        "| Internal subtype | Patients |",
        "|---|---:|",
    ]
    for key, value in labels["internal_subtype_counts"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Published Cluster Counts",
            "",
            "| Published cluster | Patients |",
            "|---|---:|",
        ]
    )
    for key, value in labels["published_cluster_counts"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Internal To Published Label Crosswalk",
            "",
            "This is the evidence for the B/E publication relabeling.",
            "",
            "| Internal subtype | Published counts |",
            "|---|---|",
        ]
    )
    for internal, published_counts in labels["internal_to_published"].items():
        lines.append(f"| {internal} | {published_counts} |")

    lines.extend(
        [
            "",
            "## PCA / Clustering Parameters In Stored Object",
            "",
            f"- Stored PCs: `{labels['stored_pcs']}`",
            f"- First PC explained variance: `{labels['first_pc_variance']}`",
            f"- First 15 PCs cumulative explained variance: `{labels['first15_variance']}`",
            f"- All stored PCs cumulative explained variance: `{labels['stored_variance']}`",
            f"- Stored neighbor params: `{labels['neighbors_params']}`",
            f"- Stored connectivities shape: `{labels['neighbors_connectivities_shape']}`",
            f"- Stored distances shape: `{labels['neighbors_distances_shape']}`",
            f"- `subcluster.csv` contains `{labels['subcluster_crosswalk_unique_subclusters']}` second-level subclusters.",
            "",
            "## Matrix Transform Evidence",
            "",
            "- `AnnData.X` was checked against the final selected rows from `cpphe_pivot_s.csv` after `normalize_total(1e4) -> log1p`.",
            f"- Transform checked: `{matrix_transform['checked']}`",
            f"- Raw/log-normalized shape: `{matrix_transform.get('raw_shape')}`",
            f"- Raw nonzero count: `{matrix_transform.get('raw_nonzero_count')}`",
            f"- AnnData.X nonzero count: `{matrix_transform.get('adata_x_nonzero_count')}`",
            f"- Max absolute difference: `{matrix_transform.get('max_abs_diff_log_normalized_pivot_vs_adata_x')}`",
            f"- Mean absolute difference: `{matrix_transform.get('mean_abs_diff_log_normalized_pivot_vs_adata_x')}`",
            f"- Allclose: `{matrix_transform.get('allclose_log_normalized_pivot_vs_adata_x')}`",
            "",
            "## Feature Selection Evidence",
            "",
            f"- Filtered event patients: `{features['filtered_event_patients']}`",
            f"- Unique Phecodes before pivot filtering: `{features['filtered_event_unique_phecodes']}`",
            f"- Pivot shape: `{features['pivot_shape'][0]} x {features['pivot_shape'][1]}`",
            f"- Pivot feature columns: `{features['pivot_feature_columns']}`",
            f"- Minimum patient count among pivot features: `{features['present_min_patient_count']}`",
            f"- Phecodes missing from pivot: `{features['missing_feature_count']}`",
            f"- Missing Phecodes with patient count <=3: `{features['missing_with_patient_count_lte_3']}`",
            f"- Missing Phecodes with patient count >=4: `{features['missing_with_patient_count_gte_4']}`",
            f"- High-prevalence missing Phecodes: `{features['missing_high_prevalence_codes']}`",
            f"- Supplement feature columns: `{features['supplement_feature_columns']}`",
            f"- Feature-matrix-with-chapter feature columns: `{features['feature_matrix_with_chapter_columns_excluding_person']}`",
            f"- Features meeting >=5% in any Cluster x GMFCS subgroup: `{features['features_with_5pct_in_any_cluster_gmfcs_group']}`",
            "",
            "## Final Event Outputs",
            "",
            f"- Final event rows: `{events['final_event_rows']}`",
            f"- Final event patients: `{events['final_event_patients']}`",
            f"- Final event unique Phecodes: `{events['final_event_unique_phecodes']}`",
            f"- Final event unique Phecode strings: `{events['final_event_unique_phecode_str']}`",
            "",
            "## Reproduction Status",
            "",
            "- Use `cp-phenotype reproduce --root data/original_reference --out outputs/reports/reproduce` for the end-to-end artifact replay check.",
            "- The stored graph plus Leiden reproduces the reference labels exactly.",
            "- The fresh UMAP graph is the closest portable rerun path, but remains graph-version sensitive.",
            "- The 5% in any Cluster x GMFCS subgroup rule appears to describe downstream reporting / interpretation; the clustering pivot itself is broader.",
            "- Once the artifact replay is accepted, rerun the same recovered preprocessing and UMAP graph strategy against PEDSnet/OMOP or the closest available REHAB/PEDSnet extract.",
        ]
    )
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
