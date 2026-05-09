from __future__ import annotations

from cp_phenotype.utils import load_yaml


def test_cchmc_config_has_no_rehab_fallback() -> None:
    config = load_yaml("configs/cchmc.yaml")
    sources = config["sources"]

    assert "fallback" not in sources
    assert sources["cohort_table"].startswith("CP_PEDSNET.")
    assert sources["diagnosis_table"].startswith("CP_PEDSNET.")
    assert sources["diagnosis_table"] == "CP_PEDSNET.OMOP_CONDITION_OCCURRENCE"
    assert sources.get("concept_table") == "REHAB.CONCEPT"
    assert "REHAB.CP_PERSON" not in str(config)
    assert "REHAB.CONDITION_OCCURRENCE" not in str(config)


def test_pedsnet_template_marks_confirmed_mapping_table() -> None:
    config = load_yaml("configs/pedsnet_validation.yaml")

    assert config["crosswalk"]["status"] == "confirmed_table_available"
    assert config["crosswalk"]["pedsnet_table"] == "CP_PEDSNET.OMOP_PATIENT_MAPPING"
    assert config["crosswalk"]["pedsnet_mrn_col"] == "MRN"
    assert config["sources"]["diagnosis_table"].startswith("CP_PEDSNET.")
