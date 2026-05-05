"""Graph-based clustering with Leiden community detection.

Provides PCA dimensionality reduction, kNN graph construction,
Leiden clustering, silhouette scoring, and resolution scanning
for the CP sub-phenotype discovery pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dir, safe_to_csv, safe_to_parquet, write_json


@dataclass(frozen=True)
class ClusterSettings:
    pca_variance: float = 0.80
    pca_n_components: int | None = None
    pca_svd_solver: str = "full"
    preprocess_method: str = "zscore"
    normalize_total_target: float = 10000.0
    scale_max_value: float | None = None
    neighbors_n_pcs: int | None = None
    n_neighbors: int = 15
    graph_method: str = "sklearn"
    target_clusters: int = 5
    discovery_min_clusters: int = 3
    discovery_max_clusters: int = 10
    resolution_grid: tuple[float, ...] = (
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.80,
        1.00,
        1.25,
        1.50,
        2.00,
    )
    random_seed: int = 42
    leiden_n_iterations: int = -1


def load_feature_matrix(path: str | Path) -> pd.DataFrame:
    matrix = pd.read_parquet(path)
    if "person_id" in matrix.columns:
        matrix = matrix.set_index("person_id")
    matrix.index.name = "person_id"
    return matrix.sort_index().sort_index(axis=1)


def preprocess_matrix(
    matrix: pd.DataFrame,
    pca_variance: float,
    random_seed: int,
    pca_n_components: int | None = None,
    pca_svd_solver: str = "full",
    preprocess_method: str = "zscore",
    normalize_total_target: float = 10000.0,
    scale_max_value: float | None = None,
) -> tuple[np.ndarray, PCA, StandardScaler | None]:
    numeric = matrix.astype(float)
    non_constant = numeric.columns[numeric.var(axis=0) > 0]
    if len(non_constant) == 0:
        raise ValueError("Feature matrix has no non-constant features after filtering")
    numeric = numeric.loc[:, non_constant]

    if preprocess_method in {"scanpy_log1p", "scanpy_log1p_scale"}:
        row_sums = numeric.sum(axis=1).replace(0, np.nan)
        values = numeric.div(row_sums, axis=0).fillna(0.0) * float(normalize_total_target)
        values = np.log1p(values)
    elif preprocess_method == "zscore":
        values = numeric
    else:
        raise ValueError(f"Unsupported preprocess_method: {preprocess_method}")

    scaler: StandardScaler | None = None
    if preprocess_method == "scanpy_log1p":
        transformed = values.to_numpy(dtype=float) if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=float)
    else:
        scaler = StandardScaler()
        transformed = scaler.fit_transform(values)
        if scale_max_value is not None:
            transformed = np.minimum(transformed, float(scale_max_value))
    n_components: int | float = int(pca_n_components) if pca_n_components else float(pca_variance)
    pca = PCA(n_components=n_components, svd_solver=pca_svd_solver, random_state=random_seed)
    pcs = pca.fit_transform(transformed)
    return pcs, pca, scaler


def build_knn_graph(pcs: np.ndarray, n_neighbors: int) -> ig.Graph:
    n_samples = pcs.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least three samples for graph clustering")
    k = min(n_neighbors + 1, n_samples)
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors.fit(pcs)
    distances, indices = neighbors.kneighbors(pcs)

    edge_weights: dict[tuple[int, int], float] = {}
    for source in range(n_samples):
        for distance, target in zip(distances[source], indices[source], strict=False):
            target = int(target)
            if source == target:
                continue
            left, right = sorted((source, target))
            weight = 1.0 / (1.0 + float(distance))
            edge_weights[(left, right)] = max(edge_weights.get((left, right), 0.0), weight)

    graph = ig.Graph(n=n_samples, edges=list(edge_weights.keys()), directed=False)
    graph.es["weight"] = list(edge_weights.values())
    return graph


def build_umap_graph(pcs: np.ndarray, n_neighbors: int, random_seed: int) -> ig.Graph:
    from umap.umap_ import fuzzy_simplicial_set

    connectivities, _, _ = fuzzy_simplicial_set(
        pcs,
        n_neighbors=int(n_neighbors),
        random_state=int(random_seed),
        metric="euclidean",
    )
    upper = sparse.triu(connectivities + connectivities.T, k=1, format="coo")
    graph = ig.Graph(
        n=pcs.shape[0],
        edges=list(zip(upper.row.tolist(), upper.col.tolist(), strict=False)),
        directed=False,
    )
    graph.es["weight"] = upper.data.tolist()
    return graph


def build_graph(pcs: np.ndarray, n_neighbors: int, graph_method: str, random_seed: int) -> ig.Graph:
    if graph_method == "sklearn":
        return build_knn_graph(pcs, n_neighbors)
    if graph_method == "umap":
        return build_umap_graph(pcs, n_neighbors, random_seed)
    raise ValueError(f"Unsupported graph_method: {graph_method}")


def leiden_labels(graph: ig.Graph, resolution: float, random_seed: int, n_iterations: int = -1) -> np.ndarray:
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=float(resolution),
        seed=int(random_seed),
        n_iterations=int(n_iterations),
    )
    return np.asarray(partition.membership, dtype=int)


def score_labels(pcs: np.ndarray, labels: np.ndarray) -> float:
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or n_clusters >= len(labels):
        return float("nan")
    return float(silhouette_score(pcs, labels))


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    mapping = {old: new for new, old in enumerate(sorted(np.unique(labels)))}
    return np.asarray([mapping[label] for label in labels], dtype=int)


def merge_small_clusters_to_target(pcs: np.ndarray, labels: np.ndarray, target_clusters: int) -> np.ndarray:
    labels = relabel_sequential(labels)
    while len(np.unique(labels)) > target_clusters:
        unique, counts = np.unique(labels, return_counts=True)
        smallest = int(unique[np.argmin(counts)])
        centroids = {cluster: pcs[labels == cluster].mean(axis=0) for cluster in unique}
        candidates = [cluster for cluster in unique if int(cluster) != smallest]
        nearest = min(
            candidates,
            key=lambda cluster: float(np.linalg.norm(centroids[smallest] - centroids[int(cluster)])),
        )
        labels[labels == smallest] = int(nearest)
        labels = relabel_sequential(labels)
    return labels


def evaluate_resolutions(
    pcs: np.ndarray,
    graph: ig.Graph,
    resolutions: Iterable[float],
    random_seed: int,
    leiden_n_iterations: int = -1,
) -> pd.DataFrame:
    rows = []
    for resolution in resolutions:
        labels = leiden_labels(graph, resolution, random_seed, leiden_n_iterations)
        rows.append(
            {
                "resolution": float(resolution),
                "n_clusters": int(len(np.unique(labels))),
                "silhouette": score_labels(pcs, labels),
                "labels_json": json.dumps(labels.tolist()),
            }
        )
    return pd.DataFrame(rows)


def _select_target5(scores: pd.DataFrame, target_clusters: int) -> pd.Series:
    target = scores[scores["n_clusters"] == target_clusters].copy()
    if not target.empty:
        return target.sort_values("silhouette", ascending=False, na_position="last").iloc[0]
    closest = scores.copy()
    closest["cluster_distance"] = (closest["n_clusters"] - target_clusters).abs()
    return closest.sort_values(["cluster_distance", "silhouette"], ascending=[True, False], na_position="last").iloc[0]


def _select_discovery(scores: pd.DataFrame, min_clusters: int, max_clusters: int) -> pd.Series:
    eligible = scores[(scores["n_clusters"] >= min_clusters) & (scores["n_clusters"] <= max_clusters)]
    if eligible.empty:
        eligible = scores
    return eligible.sort_values("silhouette", ascending=False, na_position="last").iloc[0]


def run_clustering(
    matrix_path: str | Path,
    out_dir: str | Path,
    settings: ClusterSettings | None = None,
) -> dict[str, str | int | float]:
    settings = settings or ClusterSettings()
    out_dir = ensure_dir(out_dir)
    matrix = load_feature_matrix(matrix_path)
    pcs, pca, _ = preprocess_matrix(
        matrix,
        settings.pca_variance,
        settings.random_seed,
        settings.pca_n_components,
        settings.pca_svd_solver,
        preprocess_method=settings.preprocess_method,
        normalize_total_target=settings.normalize_total_target,
        scale_max_value=settings.scale_max_value,
    )
    graph_pcs = pcs
    if settings.neighbors_n_pcs is not None:
        n_graph_pcs = min(int(settings.neighbors_n_pcs), pcs.shape[1])
        graph_pcs = pcs[:, :n_graph_pcs]
    graph = build_graph(graph_pcs, settings.n_neighbors, settings.graph_method, settings.random_seed)
    scores = evaluate_resolutions(
        graph_pcs,
        graph,
        settings.resolution_grid,
        settings.random_seed,
        settings.leiden_n_iterations,
    )

    safe_to_csv(scores.drop(columns=["labels_json"]), out_dir / "resolution_scores.csv")
    pcs_df = pd.DataFrame(
        pcs,
        index=matrix.index,
        columns=[f"PC{i + 1}" for i in range(pcs.shape[1])],
    )
    pcs_df.index.name = "person_id"
    safe_to_parquet(pcs_df, out_dir / "pca_embeddings.parquet")

    selections = {
        "target5": _select_target5(scores, settings.target_clusters),
        "discovery": _select_discovery(scores, settings.discovery_min_clusters, settings.discovery_max_clusters),
    }

    manifest: dict[str, object] = {
        "matrix_path": str(Path(matrix_path)),
        "n_patients": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "n_pcs": int(pcs.shape[1]),
        "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "pca_variance_target": float(settings.pca_variance),
        "pca_n_components_requested": settings.pca_n_components,
        "pca_svd_solver": settings.pca_svd_solver,
        "preprocess_method": settings.preprocess_method,
        "normalize_total_target": float(settings.normalize_total_target),
        "scale_max_value": settings.scale_max_value,
        "neighbors_n_pcs_requested": settings.neighbors_n_pcs,
        "n_graph_pcs": int(graph_pcs.shape[1]),
        "n_neighbors": int(settings.n_neighbors),
        "graph_method": settings.graph_method,
        "graph_edges": int(graph.ecount()),
        "random_seed": int(settings.random_seed),
        "leiden_n_iterations": int(settings.leiden_n_iterations),
        "modes": {},
    }

    for mode, selected in selections.items():
        labels = np.asarray(json.loads(selected["labels_json"]), dtype=int)
        original_n_clusters = int(len(np.unique(labels)))
        postprocess = None
        if mode == "target5" and original_n_clusters > settings.target_clusters:
            labels = merge_small_clusters_to_target(graph_pcs, labels, settings.target_clusters)
            postprocess = f"merged_smallest_clusters_to_{settings.target_clusters}"
        n_clusters = int(len(np.unique(labels)))
        silhouette = score_labels(graph_pcs, labels)
        assignments = pd.DataFrame(
            {
                "person_id": matrix.index,
                "cluster": labels.astype(str),
                "mode": mode,
                "resolution": float(selected["resolution"]),
                "silhouette": silhouette,
            }
        )
        safe_to_csv(assignments, out_dir / f"cluster_assignments_{mode}.csv")
        safe_to_parquet(assignments, out_dir / f"cluster_assignments_{mode}.parquet")
        manifest["modes"][mode] = {
            "resolution": float(selected["resolution"]),
            "n_clusters": n_clusters,
            "original_n_clusters": original_n_clusters,
            "postprocess": postprocess,
            "silhouette": silhouette if pd.notna(silhouette) else None,
            "assignments": str(out_dir / f"cluster_assignments_{mode}.csv"),
        }

    write_json(manifest, out_dir / "clustering_manifest.json")
    return {
        "n_patients": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "n_pcs": int(pcs.shape[1]),
        "manifest": str(out_dir / "clustering_manifest.json"),
    }
