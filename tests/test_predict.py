from cohort_features import build_cohort_features
from mortality_model import prepare_model_matrix, train_and_evaluate
from predict import load_model, predict_mortality_risk


def test_predict_mortality_risk_returns_valid_probability(conn, tmp_path):
    df = build_cohort_features(conn)
    X, y, groups = prepare_model_matrix(df)
    train_and_evaluate(X, y, groups, save_dir=str(tmp_path))

    bundle = load_model(str(tmp_path / "mortality_model.joblib"))
    admission = {
        "age_years": 72,
        "n_lab_results_24h": 8,
        "pct_abnormal_labs_24h": 37.5,
        "gender": "M",
        "admission_type": "EMERGENCY",
        "first_icu_careunit": "MICU",
    }
    risk = predict_mortality_risk(admission, bundle)
    assert 0.0 <= risk <= 1.0


def test_predict_handles_unseen_category_without_raising(conn, tmp_path):
    """A category not present in training (e.g. a care unit not in the
    synthetic fixture) should reindex to all-zero dummies, not raise."""
    df = build_cohort_features(conn)
    X, y, groups = prepare_model_matrix(df)
    train_and_evaluate(X, y, groups, save_dir=str(tmp_path))

    bundle = load_model(str(tmp_path / "mortality_model.joblib"))
    admission = {
        "age_years": 55,
        "n_lab_results_24h": 2,
        "pct_abnormal_labs_24h": 0.0,
        "gender": "F",
        "admission_type": "ELECTIVE",
        "first_icu_careunit": "No_ICU_Stay",
    }
    risk = predict_mortality_risk(admission, bundle)
    assert 0.0 <= risk <= 1.0
