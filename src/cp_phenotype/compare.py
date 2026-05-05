"""Comparison of clustering results against a baseline.

Computes adjusted Rand index, cluster-size overlap, and GMFCS
distribution statistics between a new clustering run and a
reference baseline.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from .utils import ensure_dir, read_json, safe_to_csv


def load_baseline_signatures(baseline_dir: str | Path) -> pd.DataFrame:
    baseline_dir = Path(baseline_dir)
    path = baseline_dir / "big_cluster_filter_features_category.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline workbook: {path}")
    rows = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
        workbook = pd.ExcelFile(path)
        sheet_names = workbook.sheet_names
    for sheet in sheet_names:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
            df = pd.read_excel(path, sheet_name=sheet, dtype=str)
        cluster = sheet.replace("cluster_", "").replace("Cluster_", "")
        for _, row in df.iterrows():
            rows.append(
                {
                    "baseline_cluster": cluster,
                    "phecode": str(row.get("phecode", "")).strip(),
                    "node_name": str(row.get("node_name", "")).strip(),
                    "com_category": str(row.get("Com_Category", "")).strip(),
                    "phe_chapter": str(row.get("phe_chapter", "")).strip(),
                }
            )
    return pd.DataFrame(rows)


def load_cluster_names(baseline_dir: str | Path) -> dict[str, str]:
    path = Path(baseline_dir) / "cluster_name.xlsx"
    if not path.exists():
        return {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
        df = pd.read_excel(path, dtype=str)
    names = {}
    for _, row in df.iterrows():
        cluster = row.iloc[0] if pd.notna(row.iloc[0]) else None
        final = row.get("Final") if "Final" in df.columns else None
        if cluster and final and str(cluster).strip() in {"A", "B", "C", "D", "E"}:
            names[str(cluster).strip()] = str(final).strip()
    return names


def score_cluster_matches(enrichment: pd.DataFrame, baseline: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    empty_columns = ["cluster", "baseline_cluster", "phecode_overlap", "category_overlap", "score"]
    if baseline.empty:
        return pd.DataFrame(columns=empty_columns)
    discovered = (
        enrichment[enrichment["direction"] == "enriched"]
        .sort_values(["cluster", "p_value_fdr", "prevalence_ratio"], ascending=[True, True, False])
        .groupby("cluster")
        .head(top_n)
    )
    rows = []
    for cluster, group in discovered.groupby("cluster"):
        phecodes = set(group["phecode"].dropna().astype(str))
        if "com_category" in group.columns:
            category_col = "com_category"
        elif "phecode_category" in group.columns:
            category_col = "phecode_category"
        else:
            category_col = None
        categories = set(group[category_col].dropna().astype(str)) if category_col else set()
        for baseline_cluster, base_group in baseline.groupby("baseline_cluster"):
            base_phecodes = set(base_group["phecode"].dropna().astype(str))
            base_categories = set(base_group["com_category"].dropna().astype(str))
            phe_overlap = len(phecodes & base_phecodes)
            cat_overlap = len(categories & base_categories)
            score = phe_overlap * 2 + cat_overlap
            rows.append(
                {
                    "cluster": str(cluster),
                    "baseline_cluster": str(baseline_cluster),
                    "phecode_overlap": phe_overlap,
                    "category_overlap": cat_overlap,
                    "score": score,
                }
            )
    if not rows:
        return pd.DataFrame(columns=empty_columns)
    return pd.DataFrame(rows).sort_values(["cluster", "score"], ascending=[True, False])


def greedy_mapping(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the globally best one-to-one discovered-to-baseline mapping."""
    if scores.empty:
        return scores.copy()

    from scipy.optimize import linear_sum_assignment

    clusters = sorted(scores["cluster"].astype(str).unique())
    baselines = sorted(scores["baseline_cluster"].astype(str).unique())
    score_lookup = {
        (str(row["cluster"]), str(row["baseline_cluster"])): float(row["score"])
        for _, row in scores.iterrows()
    }
    cost = [
        [-score_lookup.get((cluster, baseline), 0.0) for baseline in baselines]
        for cluster in clusters
    ]
    row_idx, col_idx = linear_sum_assignment(cost)

    rows = []
    for row_pos, col_pos in zip(row_idx, col_idx, strict=False):
        cluster = clusters[int(row_pos)]
        baseline = baselines[int(col_pos)]
        match = scores[
            (scores["cluster"].astype(str) == cluster)
            & (scores["baseline_cluster"].astype(str) == baseline)
        ]
        if not match.empty:
            rows.append(match.iloc[0])
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def write_validation_report(
    out_path: str | Path,
    manifest: dict,
    extract_summary: dict | None,
    summary: pd.DataFrame,
    audit: pd.DataFrame | None,
    mapping: pd.DataFrame,
    cluster_names: dict[str, str],
    enrichment: pd.DataFrame | None = None,
    baseline_note: str | None = None,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    lines = [
        "# CP Phenotype Validation Summary",
        "",
        "## Clustering Run",
        f"- Patients: {manifest.get('n_patients')}",
        f"- Features: {manifest.get('n_features')}",
        f"- PCA components: {manifest.get('n_pcs')}",
        f"- PCA explained variance: {manifest.get('pca_explained_variance_sum')}",
        f"- Preprocessing: {manifest.get('preprocess_method', 'zscore')}",
        f"- Graph method: {manifest.get('graph_method', 'sklearn')}",
        f"- Graph PCs: {manifest.get('n_graph_pcs')}",
        f"- Neighbors: {manifest.get('n_neighbors')}",
    ]
    if manifest.get("pca_n_components_requested"):
        lines.append(f"- Fixed PCA components requested: {manifest.get('pca_n_components_requested')}")
    target_mode = manifest.get("modes", {}).get("target5", {}) if isinstance(manifest.get("modes"), dict) else {}
    if target_mode:
        lines.append(f"- Target mode clusters: {target_mode.get('n_clusters')}")
        lines.append(f"- Target mode resolution: {target_mode.get('resolution')}")
        if target_mode.get("postprocess"):
            lines.append(
                f"- Target mode post-processing: {target_mode.get('postprocess')} "
                f"(original clusters: {target_mode.get('original_n_clusters')})"
            )

    lines.extend(["", "## Data Sources"])
    if extract_summary:
        lines.extend(
            [
                f"- Selected cohort table: {extract_summary.get('selected_cohort_table')}",
                f"- Selected diagnosis table: {extract_summary.get('selected_diagnosis_table')}",
                f"- Diagnosis fallback used: {extract_summary.get('diagnosis_fallback_used')}",
                f"- Primary diagnosis overlap: {extract_summary.get('primary_diagnosis_patients')} patients, "
                f"{extract_summary.get('primary_diagnosis_rows')} rows",
                f"- Diagnosis rows used: {extract_summary.get('diagnosis_rows')} across "
                f"{extract_summary.get('diagnosis_patients')} patients and "
                f"{extract_summary.get('diagnosis_unique_source_codes')} source codes",
                f"- GMFCS available: {extract_summary.get('gmfcs_patients')}/{extract_summary.get('cohort_rows')} patients",
            ]
        )
    else:
        lines.append("- Extraction summary not available.")

    lines.extend(["", "## Cluster Counts"])
    if not summary.empty:
        for _, row in summary.iterrows():
            percent = float(row.get("percent", 0)) * 100
            cluster = str(row["cluster"]).removesuffix(".0")
            lines.append(f"- Cluster {cluster}: {int(row['n_patients'])} patients ({percent:.1f}%)")

    lines.extend(["", "## Phecode Mapping Coverage"])
    if audit is not None and not audit.empty:
        for _, row in audit.iterrows():
            lines.append(
                f"- {row['vocabulary_id']}: {int(row['mapped_rows'])}/{int(row['rows'])} rows mapped "
                f"({float(row['mapping_rate']) * 100:.1f}%)"
            )
    else:
        lines.append("- Mapping audit not available.")

    lines.extend(["", "## Likely A-E Correspondence"])
    if baseline_note:
        lines.append(f"- {baseline_note}")
    elif not mapping.empty:
        for _, row in mapping.iterrows():
            baseline = str(row["baseline_cluster"])
            name = cluster_names.get(baseline, "")
            label = f"{baseline} ({name})" if name else baseline
            lines.append(
                f"- Discovered cluster {row['cluster']} -> baseline {label}; "
                f"score={row['score']}, phecode_overlap={row['phecode_overlap']}, "
                f"category_overlap={row['category_overlap']}"
            )
    else:
        lines.append("- No A-E mapping could be computed.")

    lines.extend(["", "## Top Enriched Phecodes"])
    if enrichment is not None and not enrichment.empty:
        enriched = enrichment[enrichment["direction"] == "enriched"].copy()
        enriched = enriched.sort_values(["cluster", "p_value_fdr", "prevalence_ratio"], ascending=[True, True, False])
        for cluster, group in enriched.groupby("cluster"):
            cluster = str(cluster).removesuffix(".0")
            values = []
            for _, row in group.head(5).iterrows():
                phecode = str(row.get("phecode", row.get("feature_id", ""))).removesuffix(".0")
                name = str(row.get("phecode_str", "")).strip()
                label = f"{phecode} {name}".strip()
                values.append(label)
            lines.append(f"- Cluster {cluster}: " + "; ".join(values))
        lines.append("- Full enriched and depleted feature statistics are in `feature_enrichment.csv` for each interpretation directory.")
    else:
        lines.append("- Feature enrichment output not available.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "- This report compares structured diagnosis-code clusters only.",
            "- Differences should be interpreted against cohort definition, diagnosis source, mapping coverage, GMFCS availability, and OMOP/vocabulary versions.",
            "- Temporal, treatment/outcome, note, imaging, and gait analyses are intentionally out of scope for this first validation baseline.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_comparison(
    run_dir: str | Path,
    baseline_dir: str | Path,
    out_dir: str | Path,
    processed_dir: str | Path | None = None,
    extract_summary_path: str | Path | None = None,
    mode: str = "target5",
) -> dict[str, str]:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)
    interpretation_dir = run_dir / f"interpretation_{mode}"
    enrichment_path = interpretation_dir / "feature_enrichment.csv"
    summary_path = interpretation_dir / "cluster_summary.csv"
    manifest_path = run_dir / "clustering_manifest.json"

    if not enrichment_path.exists():
        raise FileNotFoundError(f"Missing interpretation output: {enrichment_path}")
    enrichment = pd.read_csv(enrichment_path, dtype=str)
    for col in ["p_value_fdr", "prevalence_ratio"]:
        if col in enrichment.columns:
            enrichment[col] = pd.to_numeric(enrichment[col], errors="coerce")

    baseline_note = None
    baseline_path = Path(baseline_dir) / "big_cluster_filter_features_category.xlsx"
    if baseline_path.exists():
        baseline = load_baseline_signatures(baseline_dir)
        cluster_names = load_cluster_names(baseline_dir)
    else:
        baseline = pd.DataFrame()
        cluster_names = {}
        baseline_note = (
            "Baseline workbook was not found, so reference-cluster correspondence "
            f"was skipped: {baseline_path}"
        )
    scores = score_cluster_matches(enrichment, baseline)
    mapping = greedy_mapping(scores) if not scores.empty else pd.DataFrame()
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    extract_summary = (
        read_json(extract_summary_path)
        if extract_summary_path and Path(extract_summary_path).exists()
        else None
    )

    audit = None
    if processed_dir:
        audit_path = Path(processed_dir) / "mapping_audit.csv"
        if audit_path.exists():
            audit = pd.read_csv(audit_path)

    safe_to_csv(scores, out_dir / "cluster_match_scores.csv")
    safe_to_csv(mapping, out_dir / "cluster_mapping.csv")
    write_validation_report(
        out_dir / "validation_summary.md",
        manifest,
        extract_summary,
        summary,
        audit,
        mapping,
        cluster_names,
        enrichment,
        baseline_note=baseline_note,
    )
    return {
        "scores": str(out_dir / "cluster_match_scores.csv"),
        "mapping": str(out_dir / "cluster_mapping.csv"),
        "report": str(out_dir / "validation_summary.md"),
    }
