#!/usr/bin/env python
"""Experimental clustering sensitivity grid for the CP phenotype paper.

This script intentionally lives outside the main reproducibility CLI. It is an
analysis/audit runner for testing alternate feature subsets, preprocessing
choices, PCA rules, neighbor counts, and Leiden resolutions against the
recovered A-E reference labels.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from cp_phenotype.cluster import (
    build_graph,
    leiden_labels,
    merge_small_clusters_to_target,
    score_labels,
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns: pd.Index
    label_informed: bool = False


def _load_reference_matrix(root: Path) -> tuple[pd.DataFrame, pd.Series]:
    adata = ad.read_h5ad(root / "data/cpdiag_adata_t_all.h5ad")
    pivot = pd.read_csv(root / "data/cpphe_pivot_s.csv", dtype={0: str})
    id_col = pivot.columns[0]
    pivot[id_col] = pivot[id_col].astype(str)
    pivot = pivot.set_index(id_col)

    person_ids = pd.Index([str(value).replace("-ALL", "") for value in adata.obs_names], name="person_id")
    feature_ids = [str(value) for value in adata.var_names]
    matrix = pivot.loc[person_ids, feature_ids].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = (matrix > 0).astype(float)
    labels = pd.Series(adata.obs["subtype"].astype(str).to_numpy(), index=person_ids, name="subtype")
    return matrix, labels


def _feature_sets(matrix: pd.DataFrame, labels: pd.Series) -> list[FeatureSet]:
    prevalence = matrix.sum(axis=0).sort_values(ascending=False)
    by_cluster = matrix.groupby(labels).mean()
    ae_range = (by_cluster.max(axis=0) - by_cluster.min(axis=0)).sort_values(ascending=False)

    sets: list[FeatureSet] = []
    for cutoff in [3, 50, 100, 150, 300, 500]:
        cols = prevalence[prevalence > cutoff].index
        if len(cols) >= 5:
            sets.append(FeatureSet(f"prevalence_gt_{cutoff}", pd.Index(cols), False))
    for n in [100, 300, 600]:
        sets.append(FeatureSet(f"top_{n}_overall_prevalence", pd.Index(prevalence.head(n).index), False))

    # Label-informed sets are included only as a diagnostic upper-bound check.
    # They should not be used as the primary discovery method.
    for n in [25, 100, 300]:
        sets.append(FeatureSet(f"top_{n}_AE_prevalence_range", pd.Index(ae_range.head(n).index), True))
    return sets


def _transform(matrix: pd.DataFrame, method: str) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    if method == "raw_binary":
        return values
    if method == "lognorm":
        row_sums = values.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = np.nan
        out = values / row_sums * 10000.0
        out = np.nan_to_num(out, nan=0.0)
        return np.log1p(out)
    if method == "zscore_binary":
        return StandardScaler().fit_transform(values)
    raise ValueError(f"Unsupported transform: {method}")


def _pca_scores(values: np.ndarray, strategy: str) -> tuple[np.ndarray, int, float, int | None]:
    max_rank = min(values.shape[0] - 1, values.shape[1])
    if strategy == "fixed15":
        n_components = min(15, max_rank)
        pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    elif strategy == "fixed50":
        n_components = min(50, max_rank)
        pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    elif strategy == "variance80":
        pca = PCA(n_components=0.80, svd_solver="full", random_state=0)
    else:
        raise ValueError(f"Unsupported pca strategy: {strategy}")

    scores = pca.fit_transform(values)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pcs_to_80 = int(np.searchsorted(cumulative, 0.80) + 1) if np.any(cumulative >= 0.80) else None
    return scores, int(scores.shape[1]), float(cumulative[-1]), pcs_to_80


def _target5_labels(scores: np.ndarray, graph_method: str, n_neighbors: int, resolution: float) -> tuple[np.ndarray, int, str | None]:
    graph = build_graph(scores, n_neighbors=n_neighbors, graph_method=graph_method, random_seed=0)
    labels = leiden_labels(graph, resolution=resolution, random_seed=0, n_iterations=-1)
    original_n = int(len(np.unique(labels)))
    postprocess = None
    if original_n > 5:
        labels = merge_small_clusters_to_target(scores, labels, 5)
        postprocess = "merged_to_5"
    return labels, original_n, postprocess


def run(root: Path, out_dir: Path, include_umap: bool = False) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix, reference_labels = _load_reference_matrix(root)
    feature_sets = _feature_sets(matrix, reference_labels)

    transforms = ["raw_binary", "lognorm", "zscore_binary"]
    pca_strategies = ["fixed15", "fixed50", "variance80"]
    neighbor_grid = [15, 30, 50]
    resolution_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    graph_methods = ["sklearn"]
    if include_umap:
        graph_methods.append("umap")

    rows: list[dict[str, object]] = []
    for feature_set in feature_sets:
        subset = matrix.loc[:, feature_set.columns]
        for transform in transforms:
            transformed = _transform(subset, transform)
            for pca_strategy in pca_strategies:
                scores, n_pcs, pca_variance, pcs_to_80 = _pca_scores(transformed, pca_strategy)
                for graph_method in graph_methods:
                    for n_neighbors in neighbor_grid:
                        graph = build_graph(scores, n_neighbors=n_neighbors, graph_method=graph_method, random_seed=0)
                        for resolution in resolution_grid:
                            labels = leiden_labels(graph, resolution=resolution, random_seed=0, n_iterations=-1)
                            original_n = int(len(np.unique(labels)))
                            postprocess = None
                            if original_n > 5:
                                labels = merge_small_clusters_to_target(scores, labels, 5)
                                postprocess = "merged_to_5"
                            n_clusters = int(len(np.unique(labels)))
                            silhouette = float("nan") if n_clusters < 2 else score_labels(scores, labels)
                            rows.append(
                                {
                                    "feature_set": feature_set.name,
                                    "label_informed_feature_set": feature_set.label_informed,
                                    "n_features": int(subset.shape[1]),
                                    "transform": transform,
                                    "pca_strategy": pca_strategy,
                                    "n_pcs": n_pcs,
                                    "pca_variance": pca_variance,
                                    "pcs_to_80": pcs_to_80,
                                    "graph_method": graph_method,
                                    "n_neighbors": n_neighbors,
                                    "resolution": resolution,
                                    "original_n_clusters": original_n,
                                    "n_clusters": n_clusters,
                                    "postprocess": postprocess,
                                    "silhouette": silhouette,
                                    "ari_to_reference_AE": adjusted_rand_score(reference_labels, labels),
                                    "cluster_sizes": json.dumps(
                                        {
                                            str(k): int(v)
                                            for k, v in pd.Series(labels).value_counts().sort_index().items()
                                        }
                                    ),
                                }
                            )
                print(
                    f"done {feature_set.name} {transform} {pca_strategy} "
                    f"features={subset.shape[1]} pcs={n_pcs} variance={pca_variance:.3f}",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "experimental_cluster_sensitivity.csv", index=False)

    primary = results[~results["label_informed_feature_set"]].copy()
    best_primary = primary.sort_values(
        ["ari_to_reference_AE", "silhouette"], ascending=[False, False], na_position="last"
    ).head(30)
    best_true80 = primary[primary["pca_strategy"].eq("variance80")].sort_values(
        ["ari_to_reference_AE", "silhouette"], ascending=[False, False], na_position="last"
    ).head(30)
    best_clean = primary[
        primary["transform"].isin(["raw_binary", "lognorm"])
        & primary["feature_set"].isin(
            ["top_300_overall_prevalence", "top_600_overall_prevalence", "prevalence_gt_100", "prevalence_gt_150"]
        )
    ].sort_values(["ari_to_reference_AE", "silhouette"], ascending=[False, False], na_position="last").head(30)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def table(frame: pd.DataFrame) -> str:
        cols = [
            "feature_set",
            "n_features",
            "transform",
            "pca_strategy",
            "n_pcs",
            "pca_variance",
            "n_neighbors",
            "resolution",
            "original_n_clusters",
            "n_clusters",
            "postprocess",
            "silhouette",
            "ari_to_reference_AE",
        ]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in frame[cols].iterrows():
            lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# Experimental Cluster Sensitivity",
        "",
        "This is an exploratory audit and does not modify the main reproducibility pipeline.",
        "",
        "## Best Primary Unsupervised Feature Sets",
        table(best_primary),
        "",
        "## Best Primary Feature Sets Using True 80% PCA",
        table(best_true80),
        "",
        "## Best Clinically Plausible Reduced Feature Sets",
        table(best_clean),
        "",
        "## Files",
        f"- Full CSV: `{out_dir / 'experimental_cluster_sensitivity.csv'}`",
    ]
    (out_dir / "experimental_cluster_sensitivity.md").write_text("\n".join(lines))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/original_reference"))
    parser.add_argument("--out", type=Path, default=Path("outputs/reports/experimental_cluster_sensitivity"))
    parser.add_argument("--include-umap", action="store_true", help="Also test UMAP fuzzy graphs. Slower.")
    args = parser.parse_args()
    run(args.root, args.out, include_umap=args.include_umap)


if __name__ == "__main__":
    main()
