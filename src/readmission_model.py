"""
readmission_model.py

Trains a classification model to predict 30-day readmission from the same
admission-time clinical features as mortality_model.py, evaluated with the
same patient-grouped, stratified cross-validation scheme, logistic-regression
baseline, and held-out fold for ROC-AUC/precision/recall/confusion matrix.

Target: readmit_30d (sql/readmission_target.sql), a forward-looking LEAD
label — the opposite direction from readmission_intervals.sql's backward-
looking LAG. Two cohort/target restrictions follow directly from how that
label is built:
  - admissions with hospital_expire_flag=1 are dropped from the modeling
    cohort entirely — readmission was never possible for them, not merely
    unobserved, so keeping them and treating a NULL label as 0 would teach
    the model that death lowers readmission risk.
  - admissions with a NULL readmit_30d among survivors (a patient's most
    recent admission, right-censored — no later admission exists in this
    dataset to check against) are dropped from the target only.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

from mortality_model import (
    FEATURE_COLS_NUMERIC,
    FEATURE_COLS_CATEGORICAL,
    RF_PARAM_GRID,
    N_CV_SPLITS,
)

TARGET = "readmit_30d"


def merge_target(features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the readmit_30d label to the admission-time feature table on (subject_id, hadm_id)."""
    return features_df.merge(
        target_df[["subject_id", "hadm_id", "readmit_30d"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )


def prepare_model_matrix(df: pd.DataFrame):
    """
    Build the design matrix: same numeric/categorical feature set as
    mortality_model.py, one-hot encoded, with subject_id kept as a grouping
    key for patient-safe CV.

    Cohort restriction specific to this target: rows where the admission
    ended in death are dropped first (readmission was never possible), then
    rows with a NULL readmit_30d (right-censored — no later admission to
    check against) are dropped from what remains.
    """
    df = df[df["hospital_expire_flag"] == 0].copy()

    cols = ["subject_id"] + FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL + [TARGET]
    model_df = df[[c for c in cols if c in df.columns]].copy()
    model_df = model_df.dropna(subset=[TARGET])

    categorical_present = [c for c in FEATURE_COLS_CATEGORICAL if c in model_df.columns]
    model_df = pd.get_dummies(model_df, columns=categorical_present, drop_first=True)

    y = model_df[TARGET].astype(int)
    groups = model_df["subject_id"]
    X = model_df.drop(columns=[TARGET, "subject_id"])
    return X, y, groups


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, groups: pd.Series, save_dir="outputs"):
    """
    Same patient-grouped StratifiedGroupKFold scheme as mortality_model.py,
    with n_splits adaptively capped at the minority class count — readmission
    is rarer and this cohort smaller (deaths already excluded), so this
    matters even on the real MIMIC-III data, not just the synthetic fixture.

    Guards against a single-class target outright: on the small synthetic
    fixture, excluding deaths and censored rows can leave zero admissions
    with a within-30-day readmission at all, and every classifier below
    requires two classes to fit.
    """
    if y.nunique() < 2:
        print("[readmission_model] only one class present in the target after "
              "dropping death/censored rows — can't train or cross-validate a "
              "classifier (expected on the small synthetic fixture). Skipping.")
        return None, None

    n_splits = min(N_CV_SPLITS, int(y.value_counts().min()))
    if n_splits < N_CV_SPLITS:
        print(f"[readmission_model] reducing to {n_splits}-fold CV — the minority "
              f"class only has {int(y.value_counts().min())} members")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    baseline_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    baseline_aucs = cross_val_score(baseline_pipe, X, y, cv=cv, groups=groups, scoring="roc_auc")
    print(f"[readmission_model] Baseline (Logistic Regression) ROC-AUC: "
          f"{baseline_aucs.mean():.3f} +/- {baseline_aucs.std():.3f} "
          f"({n_splits}-fold CV, grouped by patient)")

    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ])
    search = GridSearchCV(rf_pipe, RF_PARAM_GRID, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X, y, groups=groups)
    print(f"[readmission_model] Random Forest best params: {search.best_params_}")

    rf_cv_aucs = cross_val_score(search.best_estimator_, X, y, cv=cv, groups=groups, scoring="roc_auc")
    print(f"[readmission_model] Random Forest ROC-AUC: "
          f"{rf_cv_aucs.mean():.3f} +/- {rf_cv_aucs.std():.3f} "
          f"({n_splits}-fold CV, grouped by patient)")

    train_idx, test_idx = next(cv.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = search.best_estimator_
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n[readmission_model] Held-out fold classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if y_test.nunique() > 1:
        auc = roc_auc_score(y_test, y_proba)
        print(f"[readmission_model] Held-out fold ROC-AUC: {auc:.3f}")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.3f})", color="steelblue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("30-Day Readmission Prediction — ROC Curve")
        ax.legend()
        fig.savefig(f"{save_dir}/readmission_roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[readmission_model] saved ROC curve to {save_dir}/readmission_roc_curve.png")
    else:
        print("[readmission_model] only one class present in held-out fold — "
              "AUC undefined for this fold (expected on very small samples)")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Not Readmitted", "Readmitted"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix — 30-Day Readmission")
    fig.savefig(f"{save_dir}/readmission_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[readmission_model] saved confusion matrix to {save_dir}/readmission_confusion_matrix.png")

    importances = pd.Series(
        model.named_steps["clf"].feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, len(importances) * 0.3)))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title("Feature Importance — 30-Day Readmission Model")
    ax.set_xlabel("Importance")
    fig.savefig(f"{save_dir}/readmission_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[readmission_model] saved feature importance plot to {save_dir}/readmission_feature_importance.png")

    model_path = f"{save_dir}/readmission_model.joblib"
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, model_path)
    print(f"[readmission_model] saved trained model to {model_path}")

    return model, importances


if __name__ == "__main__":
    import sys
    import sqlite3
    sys.path.insert(0, "src")
    from cohort_features import build_cohort_features
    from queries import readmission_target

    db_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/mimic_demo.db"
    conn = sqlite3.connect(db_path)
    features = build_cohort_features(conn)
    target = readmission_target(conn)
    conn.close()

    df = merge_target(features, target)
    X, y, groups = prepare_model_matrix(df)
    train_and_evaluate(X, y, groups)
