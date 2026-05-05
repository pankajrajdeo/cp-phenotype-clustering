"""Command-line interface for the Cerebral Palsy sub-phenotyping pipeline.

Provides subcommands for data extraction, feature matrix construction,
clustering, comparison, artifact auditing, and reproduction of the
original CP sub-phenotype analysis.

All heavy native-library imports (oracledb, sklearn, scipy, igraph, etc.)
are deferred to the command functions that actually need them, preventing
segfaults caused by conflicting OpenMP runtimes on macOS.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# OpenMP / threading guards.
# Must be set BEFORE any native library (numpy, scipy, sklearn) is imported.
# Prevents macOS segfaults from conflicting OpenMP runtimes (libgomp vs libomp).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

# Lightweight imports only (no native libraries).
from .utils import load_yaml, read_json, write_json


def resolve_path(path: str | Path | None, base: Path | None = None) -> Path | None:
    if path is None:
        return None
    result = Path(path)
    if result.is_absolute():
        return result
    return (base or Path.cwd()) / result


def load_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path)
    config = load_yaml(config_path)
    return config, config_path


def env_path_from_config(config: dict) -> Path | None:
    env_path = config.get("env_path")
    return resolve_path(env_path, Path.cwd()) if env_path else None


def settings_from_config(config: dict):
    """Build ClusterSettings from config dict. Lazy-imports cluster module."""
    from .cluster import ClusterSettings

    clustering = config.get("clustering", {})
    explicit_grid = clustering.get("resolution_grid", [])
    if explicit_grid:
        grid = tuple(float(v) for v in explicit_grid)
    else:
        grid = ClusterSettings().resolution_grid

    return ClusterSettings(
        pca_variance=float(clustering.get("pca_variance", 0.80)),
        pca_n_components=int(clustering["pca_n_components"]) if clustering.get("pca_n_components") else None,
        pca_svd_solver=str(clustering.get("pca_svd_solver", "full")),
        preprocess_method=str(clustering.get("preprocess_method", "zscore")),
        normalize_total_target=float(clustering.get("normalize_total_target", clustering.get("target_sum", 1e4))),
        scale_max_value=float(clustering["scale_max_value"]) if clustering.get("scale_max_value") is not None else None,
        neighbors_n_pcs=int(clustering["neighbors_n_pcs"]) if clustering.get("neighbors_n_pcs") else None,
        n_neighbors=int(clustering.get("n_neighbors", 15)),
        graph_method=str(clustering.get("graph_method", "sklearn")),
        target_clusters=int(clustering.get("target_clusters", 5)),
        discovery_min_clusters=int(clustering.get("discovery_min_clusters", 3)),
        discovery_max_clusters=int(clustering.get("discovery_max_clusters", 10)),
        resolution_grid=grid,
        random_seed=int(config.get("random_seed", 42)),
        leiden_n_iterations=int(clustering.get("leiden_n_iterations", -1)),
    )


# Command functions (lazy imports inside each).

def command_download_maps(args: argparse.Namespace) -> int:
    from .phecodes import download_phecode_resources

    resources = download_phecode_resources(args.out)
    write_json(resources, Path(args.out) / "download_manifest.json")
    print(f"Downloaded Phecode resources to {args.out}")
    return 0


def command_db_smoke(args: argparse.Namespace) -> int:
    from .db import smoke_test

    config = load_yaml(args.config) if args.config else {}
    result = smoke_test(env_path_from_config(config))
    print("CONNECTED")
    print(f"SERVER_VERSION={result['server_version']}")
    print(f"SMOKE_QUERY={result['smoke_query']}")
    return 0


def command_extract(args: argparse.Namespace) -> int:
    from .extract import extract_all

    config, _ = load_config(args.config)
    summary = extract_all(
        config,
        args.out,
        env_path=env_path_from_config(config),
        limit_rows=args.limit_rows,
    )
    print(summary)
    return 0


def command_build_matrix(args: argparse.Namespace) -> int:
    from .matrix import build_feature_matrix, find_phecode_resource_paths

    map_path, definitions_path = find_phecode_resource_paths(args.maps)
    summary = build_feature_matrix(
        Path(args.input) / "diagnoses.parquet",
        map_path,
        args.out,
        definitions_path=definitions_path,
        cohort_path=args.cohort,
        min_visits=args.min_visits,
        require_gmfcs=args.require_gmfcs,
        min_patients=args.min_patients,
        exclude_phecodes=args.exclude_phecode,
    )
    print(summary)
    return 0


def _run_interpretations_for_run(
    matrix_path: Path,
    run_dir: Path,
    processed_dir: Path | None,
    raw_dir: Path | None,
    random_seed: int,
) -> None:
    from .interpret import run_interpretation

    metadata_path = processed_dir / "feature_metadata.csv" if processed_dir else matrix_path.parent / "feature_metadata.csv"
    cohort_path = raw_dir / "cohort.parquet" if raw_dir else None
    for mode in ["target5", "discovery"]:
        assignments_path = run_dir / f"cluster_assignments_{mode}.csv"
        if assignments_path.exists():
            run_interpretation(
                matrix_path,
                assignments_path,
                run_dir / f"interpretation_{mode}",
                feature_metadata_path=metadata_path if metadata_path.exists() else None,
                cohort_path=cohort_path if cohort_path and cohort_path.exists() else None,
                random_seed=random_seed,
            )


def command_cluster(args: argparse.Namespace) -> int:
    from .cluster import run_clustering

    config = load_yaml(args.config) if args.config else {}
    settings = settings_from_config(config)
    result = run_clustering(args.matrix, args.out, settings)
    matrix_path = Path(args.matrix)
    processed_dir = matrix_path.parent
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    _run_interpretations_for_run(matrix_path, Path(args.out), processed_dir, raw_dir, settings.random_seed)
    print(result)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    from .compare import run_comparison

    result = run_comparison(
        args.run,
        args.baseline,
        args.out,
        processed_dir=args.processed_dir,
        extract_summary_path=args.extract_summary,
        mode=args.mode,
    )
    print(result)
    return 0


def _build_cluster_compare(
    config: dict,
    raw_dir: Path,
    processed_dir: Path,
    run_dir: Path,
    report_dir: Path,
    cohort_filters: dict | None = None,
    clustering_overrides: dict | None = None,
) -> None:
    from .cluster import run_clustering
    from .compare import run_comparison
    from .matrix import build_feature_matrix, find_phecode_resource_paths

    paths = config["paths"]
    phecode_dir = Path(paths["phecode_dir"])
    baseline_dir = Path(paths["baseline_results_dir"])
    map_path, definitions_path = find_phecode_resource_paths(phecode_dir)
    cohort_filters = config.get("cohort_filters", {}) if cohort_filters is None else cohort_filters
    feature_config = config.get("feature_matrix", {})
    build_feature_matrix(
        raw_dir / "diagnoses.parquet",
        map_path,
        processed_dir,
        definitions_path=definitions_path,
        cohort_path=raw_dir / "cohort.parquet" if cohort_filters else None,
        min_visits=cohort_filters.get("min_visits"),
        require_gmfcs=bool(cohort_filters.get("require_gmfcs", False)),
        min_patients=int(cohort_filters.get("min_patients", feature_config.get("min_patients", 3))),
        feature_prefix=feature_config.get("feature_prefix", "phe_"),
        exclude_phecodes=feature_config.get("exclude_phecodes", []),
    )
    run_config = dict(config)
    if clustering_overrides:
        run_config["clustering"] = {**config.get("clustering", {}), **clustering_overrides}
    settings = settings_from_config(run_config)
    matrix_path = processed_dir / "feature_matrix.parquet"
    run_clustering(matrix_path, run_dir, settings)
    _run_interpretations_for_run(matrix_path, run_dir, processed_dir, raw_dir, settings.random_seed)
    run_comparison(
        run_dir,
        baseline_dir,
        report_dir,
        processed_dir=processed_dir,
        extract_summary_path=raw_dir / "extract_summary.json",
        mode="target5",
    )


def command_run_all(args: argparse.Namespace) -> int:
    from .extract import extract_all
    from .phecodes import download_phecode_resources

    config, _ = load_config(args.config)
    paths = config["paths"]
    phecode_dir = Path(paths["phecode_dir"])
    raw_dir = Path(paths["raw_dir"])
    processed_dir = Path(paths["processed_dir"])
    run_dir = Path(paths["run_dir"])
    report_dir = Path(paths["report_dir"])

    if not list(phecode_dir.glob("*icd9*icd10cm*.csv")):
        download_phecode_resources(phecode_dir)

    extract_all(config, raw_dir, env_path=env_path_from_config(config), limit_rows=args.limit_rows)
    _build_cluster_compare(config, raw_dir, processed_dir, run_dir, report_dir)
    print(f"Validation report: {report_dir / 'validation_summary.md'}")
    return 0


def command_run_paper_filtered(args: argparse.Namespace) -> int:
    from .extract import extract_all

    config, _ = load_config(args.config)
    paths = config["paths"]
    raw_dir = Path(paths["raw_dir"])
    if not (raw_dir / "diagnoses.parquet").exists() or not (raw_dir / "cohort.parquet").exists():
        extract_all(config, raw_dir, env_path=env_path_from_config(config), limit_rows=args.limit_rows)
    profile = config.get("paper_filtered", {})
    processed_dir = Path(profile.get("processed_dir", "data/processed/paper_filtered"))
    run_dir = Path(profile.get("run_dir", "outputs/runs/paper_filtered"))
    report_dir = Path(profile.get("report_dir", "outputs/reports/paper_filtered"))
    cohort_filters = profile.get("cohort_filters", {"require_gmfcs": True, "min_visits": 3})
    clustering_overrides = profile.get("clustering", {"pca_n_components": 15})
    _build_cluster_compare(
        config,
        raw_dir,
        processed_dir,
        run_dir,
        report_dir,
        cohort_filters=cohort_filters,
        clustering_overrides=clustering_overrides,
    )
    print(f"Paper-filtered validation report: {report_dir / 'validation_summary.md'}")
    return 0


def command_audit_artifacts(args: argparse.Namespace) -> int:
    """Run the reference artifact audit and print a summary."""
    from .artifacts import audit_artifacts

    result = audit_artifacts(args.root, args.out)
    print(
        {
            "report": str(Path(args.out) / "artifact_audit.md"),
            "h5ad_shape": result["labels"]["h5ad_shape"],
            "ari_leiden_vs_subtype": result["labels"]["ari_leiden_0_5_vs_internal_subtype"],
            "pivot_feature_columns": result["features"]["pivot_feature_columns"],
        }
    )
    return 0


def command_reproduce(args: argparse.Namespace) -> int:
    from .reproduce import ReproduceConfig, reproduce

    config = ReproduceConfig(
        pca_n_components=args.pca_n_components,
        neighbors_n_pcs=args.neighbors_n_pcs,
        n_neighbors=args.n_neighbors,
        resolution=args.resolution,
        random_seed=args.random_seed,
    )
    summary = reproduce(args.root, args.out, config=config)
    sg = summary["stored_graph"]
    ug = summary.get("umap_fresh_graph", {})
    sk = summary["sklearn_fresh_graph"]
    print(f"Mode A (stored graph): ARI={sg['ari_vs_reference']:.4f}  clusters={sg['cluster_counts']}")
    if "error" not in ug:
        print(f"Mode B (UMAP fresh):   ARI={ug['ari_vs_reference']:.4f}  clusters={ug['cluster_counts']}")
    else:
        print(f"Mode B (UMAP fresh):   SKIPPED - {ug['error']}")
    print(f"Mode C (sklearn kNN):  ARI={sk['ari_vs_reference']:.4f}  clusters={sk['cluster_counts']}")
    print(f"Report: {args.out}/reproduction_report.md")
    return 0


def command_build_reference_matrix(args: argparse.Namespace) -> int:
    from .reference_pipeline import build_reference_matrix

    summary = build_reference_matrix(
        args.root,
        args.out,
        require_gmfcs=not args.include_missing_gmfcs,
    )
    print(summary)
    return 0


def command_privacy_check(args: argparse.Namespace) -> int:
    from .privacy import run_privacy_check

    summary = run_privacy_check(
        args.assignments,
        args.cohort,
        args.out,
        threshold=args.threshold,
        cluster_col=args.cluster_col,
        gmfcs_col=args.gmfcs_col,
        person_col=args.person_col,
        assignment_person_col=args.assignment_person_col,
        cohort_person_col=args.cohort_person_col,
    )
    print(summary)
    return 0


# Parser and entry point.

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cp-phenotype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-maps")
    download.add_argument("--out", required=True)
    download.set_defaults(func=command_download_maps)

    db_smoke = subparsers.add_parser("db-smoke")
    db_smoke.add_argument("--config", default="configs/cchmc.yaml")
    db_smoke.set_defaults(func=command_db_smoke)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--config", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--limit-rows", type=int)
    extract.set_defaults(func=command_extract)

    build_matrix = subparsers.add_parser("build-matrix")
    build_matrix.add_argument("--input", required=True)
    build_matrix.add_argument("--maps", required=True)
    build_matrix.add_argument("--out", required=True)
    build_matrix.add_argument("--min-patients", type=int, default=3)
    build_matrix.add_argument("--cohort")
    build_matrix.add_argument("--min-visits", type=int)
    build_matrix.add_argument("--require-gmfcs", action="store_true")
    build_matrix.add_argument("--exclude-phecode", action="append", default=[])
    build_matrix.set_defaults(func=command_build_matrix)

    cluster = subparsers.add_parser("cluster")
    cluster.add_argument("--matrix", required=True)
    cluster.add_argument("--out", required=True)
    cluster.add_argument("--config", default="configs/cchmc.yaml")
    cluster.add_argument("--raw-dir")
    cluster.set_defaults(func=command_cluster)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--run", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--processed-dir")
    compare.add_argument("--extract-summary")
    compare.add_argument("--mode", default="target5", choices=["target5", "discovery"])
    compare.set_defaults(func=command_compare)

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--config", required=True)
    run_all.add_argument("--limit-rows", type=int)
    run_all.set_defaults(func=command_run_all)

    run_paper_filtered = subparsers.add_parser("run-paper-filtered")
    run_paper_filtered.add_argument("--config", required=True)
    run_paper_filtered.add_argument("--limit-rows", type=int)
    run_paper_filtered.set_defaults(func=command_run_paper_filtered)

    audit = subparsers.add_parser("audit-artifacts", help="Cross-check reference clustering artifacts")
    audit.add_argument("--root", default="data/original_reference", help="Root of reference artifacts")
    audit.add_argument("--out", default="outputs/reports/artifact_audit", help="Output directory")
    audit.set_defaults(func=command_audit_artifacts)

    repro = subparsers.add_parser("reproduce", help="Reproduce A-E clusters from reference artifacts")
    repro.add_argument("--root", default="data/original_reference", help="Root of reference artifacts")
    repro.add_argument("--out", default="outputs/reports/reproduce", help="Output directory")
    repro.add_argument("--pca-n-components", type=int, default=50)
    repro.add_argument("--neighbors-n-pcs", type=int, default=15)
    repro.add_argument("--n-neighbors", type=int, default=30)
    repro.add_argument("--resolution", type=float, default=0.5)
    repro.add_argument("--random-seed", type=int, default=0)
    repro.set_defaults(func=command_reproduce)

    ref_matrix = subparsers.add_parser(
        "build-reference-matrix",
        help="Rebuild the final matrix from recovered frozen reference artifacts",
    )
    ref_matrix.add_argument("--root", default="data/original_reference", help="Root of reference artifacts")
    ref_matrix.add_argument("--out", default="data/processed/reference_rebuild", help="Output directory")
    ref_matrix.add_argument(
        "--include-missing-gmfcs",
        action="store_true",
        help="Keep all patients in the recovered pivot instead of applying the final GMFCS-present filter",
    )
    ref_matrix.set_defaults(func=command_build_reference_matrix)

    privacy = subparsers.add_parser(
        "privacy-check",
        help="Audit Cluster x GMFCS subgroup sizes for controlled data sharing",
    )
    privacy.add_argument("--assignments", required=True, help="Cluster assignment CSV/parquet")
    privacy.add_argument("--cohort", required=True, help="Cohort or reference metadata CSV/parquet")
    privacy.add_argument("--out", default="outputs/reports/privacy_check", help="Output directory")
    privacy.add_argument("--threshold", type=int, default=10, help="Minimum subgroup size")
    privacy.add_argument("--cluster-col", help="Cluster column override")
    privacy.add_argument("--gmfcs-col", help="GMFCS column override")
    privacy.add_argument("--person-col", default="person_id", help="Person ID column name")
    privacy.add_argument("--assignment-person-col", help="Assignment person ID column override")
    privacy.add_argument("--cohort-person-col", help="Cohort person ID column override")
    privacy.set_defaults(func=command_privacy_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
