from cohort_features import build_cohort_features
from mortality_model import (
    FEATURE_COLS_NUMERIC,
    FEATURE_COLS_CATEGORICAL,
    prepare_model_matrix,
    train_and_evaluate,
)

# Regression guard: these columns are only known at/after discharge (LOS ends
# when a patient dies; ICD-9 codes are assigned at discharge for the whole
# encounter) and must never be reintroduced as predictive features — see
# README's "Avoiding leakage" section and CHANGELOG.md for the full story.
LEAKY_COLUMNS = {"hospital_los_days", "total_icu_los_days", "n_diagnoses", "n_icu_stays"}


def test_feature_set_excludes_known_leakage_columns():
    assert LEAKY_COLUMNS.isdisjoint(FEATURE_COLS_NUMERIC)
    assert LEAKY_COLUMNS.isdisjoint(FEATURE_COLS_CATEGORICAL)


def test_prepare_model_matrix_shapes_align(conn):
    df = build_cohort_features(conn)
    X, y, groups = prepare_model_matrix(df)
    assert len(X) == len(y) == len(groups)
    assert set(y.unique()) <= {0, 1}


def test_train_and_evaluate_runs_on_small_imbalanced_data(conn, tmp_path):
    """
    Regression test: the synthetic fixture has only 2 deaths in 47
    admissions. This previously crashed — StratifiedGroupKFold(n_splits=5)
    is invalid with only 2 minority-class members, and the confusion matrix
    display then raised when a held-out fold contained only one class. Both
    are fixed (adaptive n_splits, labels=[0, 1] in confusion_matrix); this
    must complete without raising and produce all three plots plus a saved
    model.
    """
    df = build_cohort_features(conn)
    X, y, groups = prepare_model_matrix(df)

    model, importances = train_and_evaluate(X, y, groups, save_dir=str(tmp_path))

    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "feature_importance.png").exists()
    assert (tmp_path / "mortality_model.joblib").exists()
    assert len(importances) == X.shape[1]
