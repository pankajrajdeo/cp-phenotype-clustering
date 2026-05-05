# CP Phenotype Clustering

`cp-phenotype` is a reproducible Python workflow for discovering and validating cerebral palsy multimorbidity sub-phenotypes from structured diagnosis-code data.

The pipeline converts ICD-coded clinical events into patient-level Phecode features, applies the recovered reference clustering workflow, and produces auditable cluster assignments and interpretation reports. It is intended for controlled institutional EHR environments where patient-level data cannot be committed to source control.

**License:** non-commercial research use only. Commercial use, paid service use, product integration, or production deployment requires separate written permission. See `LICENSE`.

## Objectives

- Reproduce an existing five-cluster cerebral palsy sub-phenotype reference result from controlled local artifacts.
- Apply the same recovered workflow to OMOP, PEDSnet, or REHAB-derived diagnosis extracts.
- Keep extraction, feature construction, clustering, interpretation, and comparison steps scriptable from a single CLI.
- Package the codebase so another research institution can run the same workflow against its own approved data extract.

## What This Repository Contains

- Oracle/OMOP/REHAB extraction helpers
- ICD-9/ICD-10 normalization and Phecode v1.2 mapping
- Binary patient x Phecode matrix construction
- Reference-style preprocessing: `normalize_total(1e4)` followed by `log1p`
- PCA, UMAP fuzzy graph construction, and Leiden clustering
- Reference artifact audit and replay commands
- Cluster enrichment, feature-importance, and comparison reports
- Unit tests for mapping, matrix construction, clustering, interpretation, and comparison logic

This repository does not contain patient-level data, extracted EHR tables, reference artifacts, cluster assignments, generated reports, credentials, or local working notes.

## Repository Layout

```text
.
|-- configs/
|   |-- cchmc.yaml                 # Strict CCHMC PEDSnet validation profile
|   |-- paper_reproduction.yaml    # Reference reproduction profile
|   |-- pedsnet_validation.yaml    # Future multi-site PEDSnet template
|   `-- rehab_reference_live.yaml  # Explicit live REHAB reference profile
|-- scripts/
|   `-- cp-phenotype               # Local CLI wrapper
|-- src/cp_phenotype/
|   |-- artifacts.py               # Reference artifact auditing
|   |-- cli.py                     # Command-line interface
|   |-- cluster.py                 # PCA, graph construction, Leiden clustering
|   |-- compare.py                 # Cluster comparison reports
|   |-- db.py                      # Oracle connectivity
|   |-- extract.py                 # Oracle / OMOP / REHAB extraction
|   |-- interpret.py               # Enrichment and feature-importance reports
|   |-- matrix.py                  # Patient x Phecode matrix construction
|   |-- phecodes.py                # Phecode map download and ICD mapping
|   |-- privacy.py                 # Subgroup-size audit for controlled sharing
|   |-- reference_pipeline.py      # Rebuild final matrix from reference artifacts
|   |-- reproduce.py               # Reference artifact reproduction
|   `-- utils.py                   # Shared I/O and ID helpers
|-- tests/
|-- .env.example
|-- pyproject.toml
`-- requirements.txt
```

Ignored local-only paths:

```text
data/       # raw extracts, controlled artifacts, matrices
outputs/    # generated reports, assignments, embeddings
misc/       # notes and working materials
.env        # credentials
```

## Installation

Use Python 3.11.

```bash
python3.11 -m venv .venv-cp-phenotype
source .venv-cp-phenotype/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional model-interpretation dependencies:

```bash
python -m pip install -e ".[interpret]"
```

If the environment is not activated, use the local wrapper:

```bash
scripts/cp-phenotype --help
```

## Environment Configuration

Create a local `.env` file from the template:

```bash
cp .env.example .env
```

Required for Oracle-backed extraction:

```text
ORACLE_USER=
ORACLE_PASSWORD=
ORACLE_HOST=
ORACLE_PORT=1521
ORACLE_SERVICE=
```

Optional:

```text
CP_PHENOTYPE_ID_SALT=
```

Set `CP_PHENOTYPE_ID_SALT` only when generating stable deidentified study IDs for controlled sharing. If it is not set, the pipeline avoids producing shareable patient-level IDs.

## Data Inputs

The pipeline expects controlled data to be stored locally under ignored directories.

Reference artifact replay expects:

```text
data/original_reference/
|-- data/
|   |-- cpdiag_adata_t_all.h5ad
|   |-- cpdiag_adata_t_all_obs.csv
|   |-- cpphe_pivot_s.csv
|   |-- cp_demodx_filterd.parquet
|   `-- cp_demodx_subtype.parquet
|-- cluster/
|   |-- cp_demo_subtype.csv
|   |-- cp_cluster_sup_v4.csv
|   |-- feature_matrix_with_chapter.csv
|   `-- subcluster.csv
`-- dict/
    `-- Phecode_map_v1_2_icd10cm.csv
```

The `.h5ad` file is the original Scanpy AnnData object containing the stored transformed matrix, PCA embeddings, neighbor graph, and cluster labels. It is read as a controlled reference artifact for exact reproduction; the pipeline does not write `.h5ad` as an output format. New institutional validation runs produce `parquet`, `csv`, and `json`.

Site validation expects extracted raw files such as:

```text
data/raw/<site>/
|-- cohort.parquet
|-- diagnoses.parquet
|-- visits.parquet
|-- deaths.parquet
`-- gmfcs_observations.parquet
```

## Command-Line Workflows

Download Phecode resources:

```bash
cp-phenotype download-maps --out data/external/phecode
```

Audit the controlled reference artifacts:

```bash
cp-phenotype audit-artifacts \
  --root data/original_reference \
  --out outputs/reports/artifact_audit
```

Replay the reference clustering workflow:

```bash
cp-phenotype reproduce \
  --root data/original_reference \
  --out outputs/reports/reproduce
```

Run an Oracle-backed site validation profile:

```bash
cp-phenotype run-all --config configs/cchmc.yaml
```

Run the live REHAB/Clarity-style reference validation profile:

```bash
cp-phenotype run-all --config configs/rehab_reference_live.yaml
```

`configs/rehab_reference_live.yaml` is intentionally separate from PEDSnet
validation. It should not be used as an automatic fallback for PEDSnet.

Run strict CCHMC PEDSnet validation after the PEDSnet condition table is exposed:

```bash
cp-phenotype run-all --config configs/cchmc.yaml
```

As of the May 5, 2026 project review, PEDSnet validation depends on two
infrastructure confirmations from the data team:

- the exact PEDSnet condition/diagnosis table name and available diagnosis fields
- the exact person-ID crosswalk table and join columns, if reference-cohort
  alignment is required

Do not rely on direct `person_id` overlap between PEDSnet and REHAB/Clarity
schemas; those IDs come from different ETLs.

Run the validation workflow step by step:

```bash
cp-phenotype db-smoke --config configs/cchmc.yaml

cp-phenotype extract \
  --config configs/cchmc.yaml \
  --out data/raw/cchmc_pedsnet

cp-phenotype build-matrix \
  --input data/raw/cchmc_pedsnet \
  --maps data/external/phecode \
  --out data/processed/cchmc_pedsnet \
  --cohort data/raw/cchmc_pedsnet/cohort.parquet \
  --require-gmfcs \
  --min-visits 3 \
  --min-patients 4 \
  --exclude-phecode 343.0 \
  --exclude-phecode 333.4

cp-phenotype cluster \
  --matrix data/processed/cchmc_pedsnet/feature_matrix.parquet \
  --out outputs/runs/cchmc_pedsnet \
  --config configs/cchmc.yaml \
  --raw-dir data/raw/cchmc_pedsnet
```

Audit Cluster x GMFCS subgroup sizes for controlled data sharing:

```bash
cp-phenotype privacy-check \
  --assignments outputs/reports/reproduce/reproduction_assignments.csv \
  --cohort data/original_reference/data/cpdiag_adata_t_all_obs.csv \
  --out outputs/reports/privacy_check \
  --threshold 10
```

## Clustering Method

The recovered reference-style workflow is:

1. Build a binary patient x Phecode matrix.
2. Keep features present in at least 4 patients.
3. Exclude CP-defining or otherwise non-informative high-prevalence features configured for the run.
4. Apply `normalize_total(1e4)` and `log1p`.
5. Run PCA with 50 components using the `arpack` solver.
6. Build a UMAP fuzzy graph from the first 15 PCs with 30 neighbors.
7. Run Leiden clustering at resolution `0.5` with fixed random seed `0`.

The stored reference graph exactly reproduces the reference labels. Fresh graph construction is close but may vary slightly by package versions and graph implementation.

This preprocessing was recovered from the reference implementation. It is a
single-cell-inspired transform applied to a binary clinical feature matrix, and
it normalizes total comorbidity burden. Alternative encodings such as raw
binary features, TF-IDF weighting, or categorical-data methods may emphasize
different clinical structure and should be evaluated as future validation work,
not substituted into the paper reproduction workflow.

## Outputs

Typical outputs are written under ignored `outputs/` directories:

```text
outputs/reports/artifact_audit/
outputs/reports/reproduce/
outputs/runs/<site>/
```

Common files include:

- `artifact_audit.md`
- `reproduction_report.md`
- `reproduction_manifest.json`
- `cluster_assignments_*.csv`
- `clustering_manifest.json`
- `feature_enrichment.csv`
- `validation_summary.md`

Treat generated outputs as controlled artifacts. Some files may include patient-level assignments or derived patient-level features.

## Testing

```bash
python -m pytest
python -m compileall -q src tests
python -m pip check
```

If the package has not been installed in editable mode, run tests with:

```bash
PYTHONPATH=src python -m pytest
```

## Data Safety

Do not commit or upload:

- `.env`
- `data/`
- `outputs/`
- `misc/`
- virtual environments
- patient-level extracts, matrices, assignments, or reports

Before pushing, verify ignore behavior:

```bash
git status --ignored
git check-ignore -v .env data outputs misc .venv-cp-phenotype
```

## Development Notes

`cp-phenotype` is the project CLI name.

## License

This project is released under a custom non-commercial research-use license. Commercial use is not allowed without separate written permission. See `LICENSE` for details.
