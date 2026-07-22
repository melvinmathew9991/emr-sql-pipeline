import numpy as np
import pandas as pd

from cohort_features import build_cohort_features
from queries import readmission_target
from readmission_model import (
    TARGET,
    merge_target,
    prepare_model_matrix,
    train_and_evaluate,
)


def _build_df(conn):
    features = build_cohort_features(conn)
    target = readmission_target(conn)
    return merge_target(features, target)


def test_prepare_model_matrix_drops_death_and_censored_rows(conn):
    """Death admissions (readmission was never possible) and right-censored
    rows (no later admission to check against) must both be excluded from
    the modeling cohort, not survive as an inferred 0."""
    df = _build_df(conn)
    X, y, groups = prepare_model_matrix(df)
    expected = df[(df["hospital_expire_flag"] == 0) & df[TARGET].notna()]
    assert len(X) == len(expected)
    assert not y.isna().any()
    assert set(y.unique()) <= {0, 1}


def test_prepare_model_matrix_shapes_align(conn):
    df = _build_df(conn)
    X, y, groups = prepare_model_matrix(df)
    assert len(X) == len(y) == len(groups)


def test_train_and_evaluate_skips_gracefully_on_single_class_target(conn, tmp_path):
    """On the small synthetic fixture, dropping deaths and censored rows leaves
    zero admissions readmitted within 30 days — a single-class target that no
    classifier below can fit. Must not raise; must skip without writing outputs."""
    df = _build_df(conn)
    X, y, groups = prepare_model_matrix(df)
    assert y.nunique() < 2  # documents *why* this fixture exercises the guard

    model, importances = train_and_evaluate(X, y, groups, save_dir=str(tmp_path))

    assert model is None
    assert importances is None
    assert not (tmp_path / "readmission_model.joblib").exists()


def test_train_and_evaluate_runs_on_synthetic_two_class_data(tmp_path):
    """The conn fixture's readmission target has no positive class (see the
    guard test above), so the real CV/fit/plot/save path is exercised here
    against a handcrafted two-class sample instead."""
    rng = np.random.RandomState(0)
    n = 40
    X = pd.DataFrame({
        "age_years": rng.uniform(20, 90, n),
        "n_lab_results_24h": rng.randint(0, 20, n),
    })
    y = pd.Series(rng.randint(0, 2, n))
    groups = pd.Series(range(n))

    model, importances = train_and_evaluate(X, y, groups, save_dir=str(tmp_path))

    assert (tmp_path / "readmission_confusion_matrix.png").exists()
    assert (tmp_path / "readmission_feature_importance.png").exists()
    assert (tmp_path / "readmission_model.joblib").exists()
    assert len(importances) == X.shape[1]
