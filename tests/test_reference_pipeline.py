from __future__ import annotations

import pandas as pd

from cp_phenotype.reference_pipeline import build_reference_matrix


def test_build_reference_matrix_applies_final_gmfcs_filter(tmp_path) -> None:
    root = tmp_path / "reference"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "PERSON_ID": [1, 2, 3],
            "100.0": [1, 0, 1],
            "200": [0, 1, 1],
        }
    ).to_csv(data_dir / "cpphe_pivot_s.csv", index=False)

    pd.DataFrame(
        {
            "PERSON_ID": [1, 1, 2, 3],
            "CONDITION_START_DATETIME": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-01-01", "2020-01-01"]
            ),
            "GMFCS_M": ["I", "I", None, "III"],
            "YEAR_OF_BIRTH": [2010, 2010, 2011, 2012],
            "BIRTH_DATETIME": pd.to_datetime(["2010-01-01", "2010-01-01", "2011-01-01", "2012-01-01"]),
            "DEATH_DATE": [pd.NaT, pd.NaT, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")],
            "CP_DX_DATE": pd.to_datetime(["2020-03-01", "2020-03-01", "2020-03-01", "2020-03-01"]),
            "CP_DX_AGE_INT": [10, 10, 9, 8],
            "NUM_VISITS_TOTAL_FILTERED": [3, 3, 5, 4],
            "INPERSON_VISIT_NUM": [3, 3, 5, 4],
        }
    ).to_parquet(data_dir / "cp_demodx_filterd.parquet")

    summary = build_reference_matrix(root, tmp_path / "out")
    matrix = pd.read_parquet(tmp_path / "out" / "feature_matrix.parquet")
    cohort = pd.read_csv(tmp_path / "out" / "cohort_reference.csv", dtype={"person_id": str})

    assert summary["pivot_patients"] == 3
    assert summary["selected_patients"] == 2
    assert summary["selected_features"] == 2
    assert list(matrix.index) == ["1", "3"]
    assert list(matrix.columns) == ["100", "200"]
    assert cohort["person_id"].tolist() == ["1", "3"]
    assert "YEAR_OF_BIRTH" not in cohort.columns
    assert "BIRTH_DATETIME" not in cohort.columns
    assert "DEATH_DATE" not in cohort.columns
    assert "CP_DX_DATE" not in cohort.columns
    assert "CP_DX_AGE_INT" in cohort.columns
    assert "DEATH" in cohort.columns


def test_build_reference_matrix_can_keep_missing_gmfcs(tmp_path) -> None:
    root = tmp_path / "reference"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "PERSON_ID": [1, 2],
            "100": [1, 0],
        }
    ).to_csv(data_dir / "cpphe_pivot_s.csv", index=False)

    pd.DataFrame(
        {
            "PERSON_ID": [1, 2],
            "CONDITION_START_DATETIME": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "GMFCS_M": ["II", None],
        }
    ).to_parquet(data_dir / "cp_demodx_filterd.parquet")

    summary = build_reference_matrix(root, tmp_path / "out", require_gmfcs=False)
    matrix = pd.read_parquet(tmp_path / "out" / "feature_matrix.parquet")

    assert summary["selected_patients"] == 2
    assert list(matrix.index) == ["1", "2"]
