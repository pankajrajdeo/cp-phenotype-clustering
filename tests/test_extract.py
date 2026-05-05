from __future__ import annotations

import pandas as pd

from cp_phenotype.extract import latest_gmfcs, parse_gmfcs, summarize_visits


def test_parse_gmfcs_values() -> None:
    assert parse_gmfcs("I") == 1
    assert parse_gmfcs("Level II - Walks with limitations") == 2
    assert parse_gmfcs("GMFCS III") == 3
    assert parse_gmfcs("4") == 4
    assert parse_gmfcs("Level V - Transported") == 5
    assert parse_gmfcs("gross motor delay") is None


def test_latest_gmfcs_selects_most_recent_valid() -> None:
    observations = pd.DataFrame(
        {
            "person_id": [1, 1, 2],
            "observation_id": [10, 11, 12],
            "observation_datetime": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-02-01"]),
            "observation_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-02-01"]),
            "observation_source_value": ["GMFCS Level", "GMFCS Status", "GMFCS Level"],
            "gmfcs_level": [2, 3, None],
        }
    )
    latest = latest_gmfcs(observations)
    assert latest.shape[0] == 1
    assert latest.iloc[0]["person_id"] == 1
    assert latest.iloc[0]["gmfcs_level"] == 3


def test_summarize_visits_accepts_database_summary() -> None:
    summary = pd.DataFrame(
        {
            "person_id": [1],
            "num_visits": [4],
            "first_visit_date": pd.to_datetime(["2020-01-01"]),
            "last_visit_date": pd.to_datetime(["2022-01-01"]),
            "visit_duration_days": [731],
        }
    )
    result = summarize_visits(summary)
    assert result.equals(summary)
