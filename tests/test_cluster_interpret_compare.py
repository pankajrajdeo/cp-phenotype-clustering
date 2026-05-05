from __future__ import annotations

import pandas as pd

from cp_phenotype.cluster import ClusterSettings, run_clustering
from cp_phenotype.compare import greedy_mapping, score_cluster_matches
from cp_phenotype.interpret import compute_feature_enrichment


def test_clustering_and_enrichment(tmp_path) -> None:
    matrix = pd.DataFrame(
        {
            "phe_100": [1, 1, 1, 0, 0, 0],
            "phe_200": [0, 0, 0, 1, 1, 1],
            "phe_300": [1, 1, 0, 0, 0, 1],
        },
        index=pd.Index([1, 2, 3, 4, 5, 6], name="person_id"),
    )
    matrix_path = tmp_path / "feature_matrix.parquet"
    matrix.to_parquet(matrix_path)

    result = run_clustering(
        matrix_path,
        tmp_path / "run",
        ClusterSettings(
            pca_variance=0.8,
            n_neighbors=2,
            target_clusters=2,
            discovery_min_clusters=2,
            discovery_max_clusters=3,
            resolution_grid=(0.1, 0.5, 1.0),
            random_seed=42,
        ),
    )
    assert result["n_patients"] == 6
    assignments = pd.read_csv(tmp_path / "run" / "cluster_assignments_target5.csv", dtype={"person_id": str})
    enrichment = compute_feature_enrichment(matrix, assignments)
    assert {"cluster", "feature_id", "p_value_fdr"}.issubset(enrichment.columns)


def test_fixed_pca_components(tmp_path) -> None:
    matrix = pd.DataFrame(
        {
            "phe_100": [1, 1, 1, 0, 0, 0],
            "phe_200": [0, 0, 0, 1, 1, 1],
            "phe_300": [1, 1, 0, 0, 0, 1],
        },
        index=pd.Index([1, 2, 3, 4, 5, 6], name="person_id"),
    )
    matrix_path = tmp_path / "feature_matrix.parquet"
    matrix.to_parquet(matrix_path)

    result = run_clustering(
        matrix_path,
        tmp_path / "run_fixed",
        ClusterSettings(
            pca_n_components=2,
            n_neighbors=2,
            target_clusters=2,
            discovery_min_clusters=2,
            discovery_max_clusters=3,
            resolution_grid=(0.1, 0.5, 1.0),
            random_seed=42,
        ),
    )

    assert result["n_pcs"] == 2


def test_scanpy_like_preprocessing_manifest(tmp_path) -> None:
    matrix = pd.DataFrame(
        {
            "phe_100": [1, 1, 1, 0, 0, 0],
            "phe_200": [0, 0, 0, 1, 1, 1],
            "phe_300": [1, 0, 1, 0, 1, 0],
        },
        index=pd.Index([1, 2, 3, 4, 5, 6], name="person_id"),
    )
    matrix_path = tmp_path / "feature_matrix.parquet"
    matrix.to_parquet(matrix_path)

    result = run_clustering(
        matrix_path,
        tmp_path / "run_scanpy_like",
        ClusterSettings(
            preprocess_method="scanpy_log1p",
            pca_n_components=2,
            pca_svd_solver="arpack",
            neighbors_n_pcs=1,
            n_neighbors=2,
            graph_method="umap",
            target_clusters=2,
            resolution_grid=(0.5,),
            random_seed=42,
        ),
    )

    assert result["n_pcs"] == 2
    manifest = pd.read_json(tmp_path / "run_scanpy_like" / "clustering_manifest.json", typ="series")
    assert manifest["preprocess_method"] == "scanpy_log1p"
    assert manifest["n_graph_pcs"] == 1
    assert manifest["pca_svd_solver"] == "arpack"
    assert manifest["graph_method"] == "umap"


def test_signature_matching() -> None:
    enrichment = pd.DataFrame(
        {
            "cluster": ["0", "0", "1", "1"],
            "feature_id": ["phe_345", "phe_480", "phe_741", "phe_745"],
            "phecode": ["345", "480", "741", "745"],
            "com_category": ["Epilepsy", "respiratory", "Joint", "Pain_wide"],
            "direction": ["enriched", "enriched", "enriched", "enriched"],
            "p_value_fdr": [0.001, 0.002, 0.001, 0.002],
            "prevalence_ratio": [3.0, 2.0, 4.0, 3.0],
        }
    )
    baseline = pd.DataFrame(
        {
            "baseline_cluster": ["D", "D", "B", "B"],
            "phecode": ["345", "345.12", "741", "741.2"],
            "com_category": ["Epilepsy", "Epilepsy", "Joint", "Joint"],
        }
    )
    scores = score_cluster_matches(enrichment, baseline)
    mapping = greedy_mapping(scores)
    mapped = dict(zip(mapping["cluster"], mapping["baseline_cluster"], strict=False))
    assert mapped["0"] == "D"
    assert mapped["1"] == "B"
