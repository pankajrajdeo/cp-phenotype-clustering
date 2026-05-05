from __future__ import annotations

import pandas as pd

from cp_phenotype.matrix import build_feature_matrix


def test_build_feature_matrix(tmp_path) -> None:
    diagnoses = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 3],
            "source_code": ["G80.9", "G40.9", "G80.9", "343.9"],
        }
    )
    diagnosis_path = tmp_path / "diagnoses.parquet"
    diagnoses.to_parquet(diagnosis_path)
    map_path = tmp_path / "map.csv"
    pd.DataFrame(
        {
            "icd": ["G80.9", "G40.9", "343.9"],
            "flag": ["10", "10", "9"],
            "phecode": ["343", "345", "343"],
            "phecode_str": ["Cerebral palsy", "Epilepsy", "Cerebral palsy"],
        }
    ).to_csv(map_path, index=False)

    summary = build_feature_matrix(diagnosis_path, map_path, tmp_path / "out", min_patients=1)
    matrix = pd.read_parquet(tmp_path / "out" / "feature_matrix.parquet")

    assert summary["n_patients"] == 3
    assert set(matrix.columns) == {"phe_343", "phe_345"}
    assert matrix.loc["1", "phe_343"] == 1
    assert matrix.loc["1", "phe_345"] == 1


def test_build_feature_matrix_filters_to_paper_comparable_cohort(tmp_path) -> None:
    diagnoses = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "source_code": ["G80.9", "G80.9", "G80.9"],
        }
    )
    diagnosis_path = tmp_path / "diagnoses.parquet"
    diagnoses.to_parquet(diagnosis_path)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "num_visits": [3, 2, 4],
            "gmfcs_level": [2, 3, None],
        }
    ).to_parquet(cohort_path)
    map_path = tmp_path / "map.csv"
    pd.DataFrame(
        {
            "icd": ["G80.9"],
            "flag": ["10"],
            "phecode": ["343"],
            "phecode_str": ["Cerebral palsy"],
        }
    ).to_csv(map_path, index=False)

    summary = build_feature_matrix(
        diagnosis_path,
        map_path,
        tmp_path / "out",
        cohort_path=cohort_path,
        min_visits=3,
        require_gmfcs=True,
        min_patients=1,
    )
    matrix = pd.read_parquet(tmp_path / "out" / "feature_matrix.parquet")

    assert summary["cohort_patients_after_filter"] == 1
    assert summary["n_patients"] == 1
    assert list(matrix.index) == ["1"]


def test_build_feature_matrix_excludes_phecodes(tmp_path) -> None:
    diagnoses = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2],
            "source_code": ["G80.9", "G40.9", "G80.9", "G40.9"],
        }
    )
    diagnosis_path = tmp_path / "diagnoses.parquet"
    diagnoses.to_parquet(diagnosis_path)
    map_path = tmp_path / "map.csv"
    pd.DataFrame(
        {
            "icd": ["G80.9", "G40.9"],
            "flag": ["10", "10"],
            "phecode": ["343.0", "345"],
            "phecode_str": ["Infantile cerebral palsy", "Epilepsy"],
        }
    ).to_csv(map_path, index=False)

    summary = build_feature_matrix(
        diagnosis_path,
        map_path,
        tmp_path / "out_excluded",
        min_patients=1,
        exclude_phecodes=["343.0"],
    )
    matrix = pd.read_parquet(tmp_path / "out_excluded" / "feature_matrix.parquet")

    assert summary["excluded_phecodes"] == ["343.0"]
    assert set(matrix.columns) == {"phe_345"}


def test_build_feature_matrix_uses_premapped_phecode_column(tmp_path) -> None:
    diagnoses = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 3],
            "source_code": ["UNMAPPED", "ALSO_UNMAPPED", "UNMAPPED", "UNMAPPED"],
            "phecode": ["100.0", "200", "100.0", None],
            "phecode_str": ["A", "B", "A", None],
        }
    )
    diagnosis_path = tmp_path / "diagnoses.parquet"
    diagnoses.to_parquet(diagnosis_path)
    map_path = tmp_path / "map.csv"
    pd.DataFrame(
        {
            "icd": ["SHOULD_NOT_BE_USED"],
            "flag": ["10"],
            "phecode": ["999"],
            "phecode_str": ["Wrong"],
        }
    ).to_csv(map_path, index=False)

    summary = build_feature_matrix(diagnosis_path, map_path, tmp_path / "out_premapped", min_patients=1)
    matrix = pd.read_parquet(tmp_path / "out_premapped" / "feature_matrix.parquet")
    audit = pd.read_csv(tmp_path / "out_premapped" / "mapping_audit.csv")

    assert summary["mapping_source"] == "premapped_phecode"
    assert audit.loc[0, "mapping_source"] == "premapped_phecode"
    assert set(matrix.columns) == {"phe_100", "phe_200"}
    assert list(matrix.index) == ["1", "2"]
