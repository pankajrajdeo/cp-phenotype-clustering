from __future__ import annotations

import pandas as pd

from cp_phenotype.phecodes import (
    infer_vocabulary_id,
    load_phecode_map,
    map_diagnoses_to_phecodes,
    normalize_icd_code,
)


def test_normalize_icd_code() -> None:
    assert normalize_icd_code(" G40.309 ") == "G40309"
    assert normalize_icd_code("343.9") == "3439"


def test_infer_vocabulary_id() -> None:
    assert infer_vocabulary_id("343.9") == "ICD9CM"
    assert infer_vocabulary_id("G80.9") == "ICD10CM"


def test_load_and_map_phecodes(tmp_path) -> None:
    map_path = tmp_path / "map.csv"
    pd.DataFrame(
        {
            "icd": ["343.9", "G80.9"],
            "flag": ["9", "10"],
            "phecode": ["343", "343"],
            "phecode_str": ["Cerebral palsy", "Cerebral palsy"],
        }
    ).to_csv(map_path, index=False)

    mapping = load_phecode_map(map_path)
    diagnoses = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "source_code": ["343.9", "G80.9", "ZZZ"],
        }
    )
    events, audit, unmapped = map_diagnoses_to_phecodes(diagnoses, mapping)

    assert len(events) == 2
    assert set(events["phecode"]) == {"343"}
    assert int(audit["mapped_rows"].sum()) == 2
    assert unmapped.iloc[0]["norm_code"] == "ZZZ"
