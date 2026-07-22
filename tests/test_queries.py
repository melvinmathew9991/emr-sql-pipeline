from queries import (
    cohort_summary,
    length_of_stay_by_careunit,
    top_diagnoses,
    mortality_rate_by_admission_type,
    mortality_rate_by_careunit,
    mortality_significance_test,
    readmission_intervals,
    readmission_target,
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


def test_readmission_intervals_first_admission_has_no_prior_interval(conn):
    df = readmission_intervals(conn)
    first_admissions = df[df["admission_seq"] == 1]
    assert first_admissions["days_since_last_discharge"].isna().all()


def test_readmission_intervals_sequence_increments_per_patient(conn):
    df = readmission_intervals(conn)
    repeat_patient = df["subject_id"].value_counts()
    repeat_patient = repeat_patient[repeat_patient > 1].index[0]
    seqs = df[df["subject_id"] == repeat_patient].sort_values("admittime")["admission_seq"].tolist()
    assert seqs == list(range(1, len(seqs) + 1))


def test_readmission_intervals_are_nonnegative_where_present(conn):
    df = readmission_intervals(conn)
    known = df["days_since_last_discharge"].dropna()
    assert (known >= 0).all()


def test_readmission_target_null_exactly_when_no_next_admission(conn):
    df = readmission_target(conn)
    assert df["readmit_30d"].isna().equals(df["days_to_next_admission"].isna())


def test_readmission_target_death_admissions_have_no_next_admission(conn):
    """A patient who died this admission can't have a later one — hospital_expire_flag=1
    admissions must be the last admittime row for that subject_id, so both
    days_to_next_admission and readmit_30d come out NULL without special-casing."""
    df = readmission_target(conn)
    deaths = df[df["hospital_expire_flag"] == 1]
    assert deaths["days_to_next_admission"].isna().all()
    assert deaths["readmit_30d"].isna().all()


def test_readmission_target_matches_30_day_threshold(conn):
    df = readmission_target(conn)
    known = df.dropna(subset=["readmit_30d"])
    assert ((known["days_to_next_admission"] <= 30) == (known["readmit_30d"] == 1)).all()
