"""
main.py

Runs the full EMR analytics pipeline end to end:
  1. Load raw MIMIC-III CSVs into a SQLite relational database
  2. Run descriptive analytics SQL queries
  3. Build ML-ready cohort features via SQL joins + pandas
  4. Train and evaluate an in-hospital mortality prediction model

Usage:
    python main.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from build_database import build_database
from queries import run_all_descriptive_queries
from cohort_features import build_cohort_features
from mortality_model import prepare_model_matrix, train_and_evaluate

# -----------------------------------------------------------------------
# CONFIG — set USE_SYNTHETIC = False once you've downloaded the real data
# from https://physionet.org/content/mimiciii-demo/1.4/
# -----------------------------------------------------------------------
USE_SYNTHETIC = False
REAL_DATA_DIR = "data/mimic_demo"
SYNTHETIC_DATA_DIR = "data/synthetic_test"
DB_PATH = "outputs/mimic_demo.db"
OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_dir = SYNTHETIC_DATA_DIR if USE_SYNTHETIC else REAL_DATA_DIR
    if USE_SYNTHETIC:
        print("=" * 70)
        print("RUNNING ON SYNTHETIC TEST DATA — for pipeline verification only.")
        print("Set USE_SYNTHETIC = False and point to the real MIMIC-III demo")
        print("files before drawing any conclusions or using results anywhere.")
        print("=" * 70 + "\n")

    print("=" * 70)
    print("STEP 1: Building relational database from EMR CSVs")
    print("=" * 70)
    conn = build_database(data_dir, DB_PATH)

    print("\n" + "=" * 70)
    print("STEP 2: Descriptive Analytics (SQL)")
    print("=" * 70)
    run_all_descriptive_queries(conn)

    print("\n" + "=" * 70)
    print("STEP 3: Building Cohort Feature Table (SQL joins + pandas)")
    print("=" * 70)
    features = build_cohort_features(conn)
    conn.close()

    print("\n" + "=" * 70)
    print("STEP 4: In-Hospital Mortality Prediction Model")
    print("=" * 70)
    X, y, groups = prepare_model_matrix(features)
    train_and_evaluate(X, y, groups, save_dir=OUTPUT_DIR)

    print("\n" + "=" * 70)
    print(f"Done. Database and plots saved to {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
