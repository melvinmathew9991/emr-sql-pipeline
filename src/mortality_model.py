"""
mortality_model.py

Trains a classification model to predict in-hospital mortality from
admission-time clinical features, evaluated with a patient-grouped,
stratified cross-validation scheme, a logistic-regression baseline, and a
held-out fold for ROC-AUC, precision/recall, and a confusion matrix.
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

TARGET = "hospital_expire_flag"

# Deliberately admission-time-only features. cohort_features.py also computes
# hospital_los_days, total_icu_los_days, n_diagnoses, and n_icu_stays, but
# they're excluded here: length of stay is only known once a stay ends (and
# for a patient who dies, discharge time = death time), MIMIC's ICD-9 codes
# are assigned at discharge for the whole encounter, and repeat ICU stays are
# only known in hindsight. Including them would leak outcome-adjacent
# information into a model framed as "predict at admission." Lab features are
# windowed to the first 24h in sql/cohort_features.sql for the same reason.
FEATURE_COLS_NUMERIC = [
    "age_years",
    "n_lab_results_24h",
    "pct_abnormal_labs_24h",
]

FEATURE_COLS_CATEGORICAL = [
    "gender",
    "admission_type",
    "first_icu_careunit",
]

RF_PARAM_GRID = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [3, 5, 7, None],
    "clf__min_samples_leaf": [1, 3, 5],
}

N_CV_SPLITS = 5


def prepare_model_matrix(df: pd.DataFrame):
    """
    Build the design matrix: numeric features pass through, categoricals
    one-hot encoded. Also returns subject_id as a grouping key, since some
    patients in this cohort have multiple admissions — without grouping,
    the same patient's admissions could land in both train and test folds.
    """
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
    Evaluate with StratifiedGroupKFold so admissions from the same patient
    never split across train/test, and so the reported ROC-AUC reflects
    variance across folds rather than one lucky (or unlucky) split. A
    logistic-regression baseline is reported alongside the tuned Random
    Forest so the model's added complexity has something to beat. The final
    held-out fold below (same CV scheme) is used only for the plots.

    n_splits is capped at the minority class count: stratified k-fold is
    undefined if any fold can't contain at least one minority-class sample,
    which happens on the small synthetic test fixture (only 2 deaths) even
    though N_CV_SPLITS=5 is appropriate for the real MIMIC-III data (40 deaths).
    """
    n_splits = min(N_CV_SPLITS, int(y.value_counts().min()))
    if n_splits < N_CV_SPLITS:
        print(f"[mortality_model] reducing to {n_splits}-fold CV — the minority "
              f"class only has {int(y.value_counts().min())} members (expected on "
              f"the small synthetic test fixture)")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    baseline_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    baseline_aucs = cross_val_score(baseline_pipe, X, y, cv=cv, groups=groups, scoring="roc_auc")
    print(f"[mortality_model] Baseline (Logistic Regression) ROC-AUC: "
          f"{baseline_aucs.mean():.3f} +/- {baseline_aucs.std():.3f} "
          f"({n_splits}-fold CV, grouped by patient)")

    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ])
    search = GridSearchCV(rf_pipe, RF_PARAM_GRID, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X, y, groups=groups)
    print(f"[mortality_model] Random Forest best params: {search.best_params_}")

    rf_cv_aucs = cross_val_score(search.best_estimator_, X, y, cv=cv, groups=groups, scoring="roc_auc")
    print(f"[mortality_model] Random Forest ROC-AUC: "
          f"{rf_cv_aucs.mean():.3f} +/- {rf_cv_aucs.std():.3f} "
          f"({n_splits}-fold CV, grouped by patient)")

    # Held-out fold for the plots and classification report below — same
    # grouped, stratified scheme as the CV above, so this split is also
    # patient-safe, not just admission-safe.
    train_idx, test_idx = next(cv.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = search.best_estimator_
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n[mortality_model] Held-out fold classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if y_test.nunique() > 1:
        auc = roc_auc_score(y_test, y_proba)
        print(f"[mortality_model] Held-out fold ROC-AUC: {auc:.3f}")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.3f})", color="steelblue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("In-Hospital Mortality Prediction — ROC Curve")
        ax.legend()
        fig.savefig(f"{save_dir}/roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[mortality_model] saved ROC curve to {save_dir}/roc_curve.png")
    else:
        print("[mortality_model] only one class present in held-out fold — "
              "AUC undefined for this fold (expected on very small samples)")

    # labels=[0, 1] forces a consistent 2x2 matrix even if the held-out fold
    # happens to contain only one class (possible on the small synthetic
    # fixture), rather than crashing the display with a mismatched shape.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Survived", "Died"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix — In-Hospital Mortality")
    fig.savefig(f"{save_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[mortality_model] saved confusion matrix to {save_dir}/confusion_matrix.png")

    # Feature importance
    importances = pd.Series(
        model.named_steps["clf"].feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, len(importances) * 0.3)))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title("Feature Importance — In-Hospital Mortality Model")
    ax.set_xlabel("Importance")
    fig.savefig(f"{save_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[mortality_model] saved feature importance plot to {save_dir}/feature_importance.png")

    # Persist the fitted pipeline (scaler + classifier) alongside the exact
    # training column schema, so predict.py can one-hot encode a new
    # admission consistently and score it without retraining.
    model_path = f"{save_dir}/mortality_model.joblib"
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, model_path)
    print(f"[mortality_model] saved trained model to {model_path}")

    return model, importances


if __name__ == "__main__":
    import sys
    import sqlite3
    sys.path.insert(0, "src")
    from cohort_features import build_cohort_features

    db_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/mimic_demo.db"
    conn = sqlite3.connect(db_path)
    df = build_cohort_features(conn)
    conn.close()

    X, y, groups = prepare_model_matrix(df)
    train_and_evaluate(X, y, groups)
