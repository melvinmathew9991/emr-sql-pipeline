"""
queries.py

Descriptive analytics queries run directly against the relational EMR schema.
Each function loads its query from sql/ and returns a pandas DataFrame so
results can be inspected, plotted, or exported.
"""

import os
import pandas as pd
import sqlite3
from scipy.stats import chi2_contingency

SQL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql")


def _load_sql(filename: str) -> str:
    """Read a query from the sql/ directory."""
    with open(os.path.join(SQL_DIR, filename)) as f:
        return f.read()


def cohort_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Basic cohort characterization: patient count, admission count, gender split."""
    return pd.read_sql_query(_load_sql("cohort_summary.sql"), conn)


def length_of_stay_by_careunit(conn: sqlite3.Connection) -> pd.DataFrame:
    """Average and median ICU length of stay (days), grouped by care unit."""
    return pd.read_sql_query(_load_sql("length_of_stay_by_careunit.sql"), conn)


def top_diagnoses(conn: sqlite3.Connection, n: int = 10) -> pd.DataFrame:
    """Most frequently occurring diagnoses in the cohort, with human-readable titles."""
    return pd.read_sql_query(_load_sql("top_diagnoses.sql"), conn, params=(n,))


def mortality_rate_by_admission_type(conn: sqlite3.Connection) -> pd.DataFrame:
    """In-hospital mortality rate, grouped by admission type (EMERGENCY, ELECTIVE, URGENT)."""
    return pd.read_sql_query(_load_sql("mortality_by_admission_type.sql"), conn)


def mortality_rate_by_careunit(conn: sqlite3.Connection) -> pd.DataFrame:
    """In-hospital mortality rate, grouped by first ICU care unit."""
    return pd.read_sql_query(_load_sql("mortality_by_careunit.sql"), conn)


def readmission_intervals(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per-patient admission sequence number and days since that patient's last discharge."""
    return pd.read_sql_query(_load_sql("readmission_intervals.sql"), conn)


def readmission_target(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per-admission 30-day readmission label via a forward-looking LEAD; NULL for in-hospital deaths and right-censored final admissions."""
    return pd.read_sql_query(_load_sql("readmission_target.sql"), conn)


def mortality_significance_test(df: pd.DataFrame, label: str) -> None:
    """
    Chi-square test of independence on a mortality-by-group table: is the
    difference in death rate across groups distinguishable from chance, or is
    it consistent with noise given how small some groups are? Flags when any
    expected cell count falls below 5 — the standard rule of thumb below
    which a chi-square test isn't considered reliable — since several groups
    in this cohort (e.g., a 6- or 11-admission care unit) are that small.
    """
    survived = df["n_admissions"] - df["n_deaths"]
    contingency = pd.DataFrame({"died": df["n_deaths"], "survived": survived})
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"[queries] {label}: chi-square={chi2:.2f}, dof={dof}, p-value={p:.3f}")
    if (expected < 5).any():
        print(f"[queries] caveat: at least one expected cell count < 5 for "
              f"{label} — group sizes are small enough that this test may not "
              f"be reliable; treat the mortality differences as descriptive, "
              f"not statistically confirmed.")


def run_all_descriptive_queries(conn: sqlite3.Connection) -> dict:
    """Run every descriptive query and return results as a dict of DataFrames."""
    results = {
        "cohort_summary": cohort_summary(conn),
        "length_of_stay_by_careunit": length_of_stay_by_careunit(conn),
        "top_diagnoses": top_diagnoses(conn),
        "mortality_by_admission_type": mortality_rate_by_admission_type(conn),
        "mortality_by_careunit": mortality_rate_by_careunit(conn),
    }
    for name, df in results.items():
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))

    print()
    mortality_significance_test(results["mortality_by_admission_type"], "mortality by admission type")
    mortality_significance_test(results["mortality_by_careunit"], "mortality by care unit")

    return results


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/mimic_demo.db"
    conn = sqlite3.connect(db_path)
    run_all_descriptive_queries(conn)
    conn.close()
