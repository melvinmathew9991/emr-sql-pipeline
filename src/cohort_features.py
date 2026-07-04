"""
cohort_features.py

Builds a patient-admission-level, ML-ready feature table by joining across the
relational EMR schema (admissions, ICU stays, diagnoses, labs) in SQL, then
finishing feature engineering in pandas.

Target: in-hospital mortality (admissions.hospital_expire_flag)
"""

import os
import sqlite3
import pandas as pd
import numpy as np

SQL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql")


def query_cohort_base(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    One row per hospital admission, joined entirely in SQL (sql/cohort_features.sql):
    admissions and patients joined directly; ICU stays, diagnoses, and labs
    first aggregated to admission grain in CTEs (to avoid fan-out from their
    one-to-many relationship to hadm_id), then LEFT JOINed onto the base in
    the same query. No table combination happens in pandas — only the feature
    engineering below (age capping, percentage calc, missing-value fill) does.
    """
    with open(os.path.join(SQL_DIR, "cohort_features.sql")) as f:
        query = f.read()
    return pd.read_sql_query(query, conn)


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering on the SQL-joined table: age capping, the abnormal-lab
    percentage, and filling values left NULL by the LEFT JOINs (e.g., no ICU
    stay, no labs recorded). None of this combines separate tables — that
    already happened in SQL.

    Note: this table retains whole-encounter columns (hospital_los_days,
    total_icu_los_days, n_diagnoses, n_icu_stays) for descriptive analysis,
    even though mortality_model.py deliberately excludes them from the
    predictive feature set — they're only fully known at or after discharge,
    so using them to predict an admission-time outcome would leak information
    the model wouldn't actually have at prediction time.
    """
    df = df.copy()

    # MIMIC shifts dates for de-identification; ages over ~200 indicate the
    # "90+ years old" masking convention used in the real database. Cap at 90.
    df["age_years"] = (df["age_days"] / 365.25).clip(upper=90)
    df = df.drop(columns=["age_days"])

    df["pct_abnormal_labs_24h"] = (
        100.0 * df["n_abnormal_labs_24h"] / df["n_lab_results_24h"].replace(0, np.nan)
    )
    df = df.drop(columns=["n_abnormal_labs_24h"])

    numeric_fill_zero = ["n_icu_stays", "total_icu_los_days", "n_diagnoses",
                          "n_lab_results_24h", "pct_abnormal_labs_24h"]
    for col in numeric_fill_zero:
        df[col] = df[col].fillna(0)

    df["first_icu_careunit"] = df["first_icu_careunit"].fillna("No_ICU_Stay")
    df["hospital_los_days"] = df["hospital_los_days"].fillna(df["hospital_los_days"].median())
    df["age_years"] = df["age_years"].fillna(df["age_years"].median())

    return df


def build_cohort_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Full pipeline: SQL-join admissions + ICU stays + diagnoses + labs, then engineer features."""
    df = query_cohort_base(conn)
    df = finalize_features(df)
    print(f"[cohort_features] built feature table: {len(df)} admissions, {len(df.columns)} columns")
    print(f"[cohort_features] in-hospital mortality rate: {df['hospital_expire_flag'].mean():.1%}")
    return df


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/mimic_demo.db"
    conn = sqlite3.connect(db_path)
    features = build_cohort_features(conn)
    print(features.head())
    print(features.dtypes)
    conn.close()
