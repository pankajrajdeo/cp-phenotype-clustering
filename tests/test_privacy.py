from __future__ import annotations

import pandas as pd

from cp_phenotype.privacy import compute_subgroup_sizes, run_privacy_check


def test_compute_subgroup_sizes_normalizes_gmfcs() -> None:
    data = pd.DataFrame(
        {
            "cluster": ["A", "A", "B", "B", "B"],
            "gmfcs": ["I", "1", "Level II", "2", "V"],
        }
    )

    counts = compute_subgroup_sizes(data)
    observed = {
        (row.cluster, row.gmfcs): row.n_patients
        for row in counts.itertuples(index=False)
    }

    assert observed[("A", "I")] == 2
    assert observed[("B", "II")] == 2
    assert observed[("B", "V")] == 1


def test_run_privacy_check_flags_small_cells_and_writes_outputs(tmp_path) -> None:
    assignments = pd.DataFrame(
        {
            "h5ad_id": ["1-ALL", "2-ALL", "3-ALL", "4-ALL", "5-ALL"],
            "stored_graph_published": ["A", "A", "B", "B", "B"],
        }
    )
    cohort = pd.DataFrame(
        {
            "PERSON_ID": ["1-ALL", "2-ALL", "3-ALL", "4-ALL", "5-ALL"],
            "GMFCS_M": ["I", "I", "II", "II", "V"],
        }
    )
    assignments_path = tmp_path / "assignments.csv"
    cohort_path = tmp_path / "cohort.csv"
    assignments.to_csv(assignments_path, index=False)
    cohort.to_csv(cohort_path, index=False)

    summary = run_privacy_check(
        assignments_path,
        cohort_path,
        tmp_path / "privacy",
        threshold=2,
    )
    counts = pd.read_csv(tmp_path / "privacy" / "subgroup_counts.csv")
    report = (tmp_path / "privacy" / "subgroup_size_audit.md").read_text()

    assert summary["minimum_subgroup_size"] == 1
    assert summary["n_below_threshold"] == 1
    assert counts.loc[counts["below_threshold"], "n_patients"].tolist() == [1]
    assert "Subgroups below threshold" in report
