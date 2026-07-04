from cohort_features import build_cohort_features


def test_row_count_matches_admissions(conn):
    df = build_cohort_features(conn)
    assert len(df) == 47


def test_no_nulls_left_by_the_left_joins(conn):
    df = build_cohort_features(conn)
    required = [
        "age_years", "hospital_los_days", "n_icu_stays",
        "n_lab_results_24h", "pct_abnormal_labs_24h", "first_icu_careunit",
    ]
    for col in required:
        assert df[col].isna().sum() == 0, f"{col} still has nulls after finalize_features"


def test_mortality_rate_is_a_valid_proportion(conn):
    df = build_cohort_features(conn)
    rate = df["hospital_expire_flag"].mean()
    assert 0 <= rate <= 1
