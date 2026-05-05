from __future__ import annotations

import json

from cp_phenotype.utils import load_yaml


def test_cchmc_config_has_no_rehab_fallback() -> None:
    config = load_yaml("configs/cchmc.yaml")
    sources = config["sources"]

    assert "fallback" not in sources
    assert sources["cohort_table"].startswith("CP_PEDSNET.")
    assert sources["diagnosis_table"].startswith("CP_PEDSNET.")
    assert "REHAB" not in json.dumps(config)


def test_pedsnet_template_marks_crosswalk_pending() -> None:
    config = load_yaml("configs/pedsnet_validation.yaml")

    assert config["crosswalk"]["status"] == "pending"
    assert config["crosswalk"]["confirmed_table"] is None
    assert config["sources"]["diagnosis_table"].startswith("CP_PEDSNET.")
