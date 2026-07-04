"""
predict.py

Loads the model saved by mortality_model.py and scores a new admission's
in-hospital mortality risk from the same admission-time features used in
training. This is the inference path the pipeline was otherwise missing —
train_and_evaluate() only reported metrics; nothing could score a new patient.

Usage:
    python predict.py
"""

import os

import joblib
import pandas as pd

from mortality_model import FEATURE_COLS_CATEGORICAL


def load_model(model_path: str = "outputs/mortality_model.joblib") -> dict:
    """Load the {model, feature_columns} bundle saved by train_and_evaluate()."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found — run `python main.py` first to train and save a model."
        )
    return joblib.load(model_path)


def predict_mortality_risk(admission: dict, bundle: dict) -> float:
    """
    Score a single new admission's in-hospital mortality risk (0-1).

    `admission` must supply the same raw admission-time fields used in
    training: age_years, n_lab_results_24h, pct_abnormal_labs_24h, gender,
    admission_type, first_icu_careunit. One-hot encoding is aligned to the
    exact training columns via reindex, so a category unseen at training
    time (or missing here) is safely treated as all-zero dummy columns
    rather than raising or silently misaligning.
    """
    row = pd.DataFrame([admission])
    categorical_present = [c for c in FEATURE_COLS_CATEGORICAL if c in row.columns]
    row = pd.get_dummies(row, columns=categorical_present)
    row = row.reindex(columns=bundle["feature_columns"], fill_value=0)
    return float(bundle["model"].predict_proba(row)[:, 1][0])


if __name__ == "__main__":
    bundle = load_model()

    example_admission = {
        "age_years": 72,
        "n_lab_results_24h": 8,
        "pct_abnormal_labs_24h": 37.5,
        "gender": "M",
        "admission_type": "EMERGENCY",
        "first_icu_careunit": "MICU",
    }
    risk = predict_mortality_risk(example_admission, bundle)
    print(f"Example admission: {example_admission}")
    print(f"Predicted in-hospital mortality risk: {risk:.1%}")
