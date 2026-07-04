from queries import (
    cohort_summary,
    length_of_stay_by_careunit,
    top_diagnoses,
    mortality_rate_by_admission_type,
    mortality_rate_by_careunit,
    mortality_significance_test,
)


def test_cohort_summary_counts(conn):
    df = cohort_summary(conn)
    assert df.loc[0, "n_patients"] == 40
    assert df.loc[0, "n_admissions"] == 47


def test_length_of_stay_has_no_negative_values(conn):
    df = length_of_stay_by_careunit(conn)
    assert (df["avg_los_days"] >= 0).all()


def test_top_diagnoses_respects_limit(conn):
    df = top_diagnoses(conn, n=3)
    assert len(df) <= 3


def test_mortality_rates_are_valid_percentages(conn):
    df = mortality_rate_by_admission_type(conn)
    assert df["mortality_rate_pct"].between(0, 100).all()

    df2 = mortality_rate_by_careunit(conn)
    assert df2["mortality_rate_pct"].between(0, 100).all()


def test_significance_test_runs_without_raising(conn, capsys):
    df = mortality_rate_by_careunit(conn)
    mortality_significance_test(df, "unit test grouping")
    captured = capsys.readouterr()
    assert "chi-square" in captured.out
