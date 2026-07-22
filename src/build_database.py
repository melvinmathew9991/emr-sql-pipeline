"""
build_database.py

Loads the raw MIMIC-III demo CSV tables into a local SQLite database,
replicating the real EMR relational schema.
"""

import os
import sqlite3
import pandas as pd

TABLES = [
    "PATIENTS",
    "ADMISSIONS",
    "ICUSTAYS",
    "DIAGNOSES_ICD",
    "D_ICD_DIAGNOSES",
    "LABEVENTS",
]

# Column(s) actually used as join keys or filter predicates by sql/*.sql.
# Without these, every join in cohort_features.sql (and the descriptive
# queries) forces SQLite into a full table scan on the inner side of each
# join. labevents is the largest table and is both joined on hadm_id and
# filtered by charttime in the same CTE (lab_agg), hence the composite index.
INDEXES = {
    "patients": [("subject_id",)],
    "admissions": [("subject_id",), ("hadm_id",)],
    "icustays": [("hadm_id",)],
    "diagnoses_icd": [("hadm_id",), ("icd9_code",)],
    "d_icd_diagnoses": [("icd9_code",)],
    "labevents": [("hadm_id", "charttime")],
}


def _create_indexes(conn: sqlite3.Connection, loaded_tables: set) -> None:
    """Create indexes on join/filter columns for whichever tables loaded."""
    cur = conn.cursor()
    for table, column_sets in INDEXES.items():
        if table not in loaded_tables:
            continue
        for columns in column_sets:
            index_name = f"idx_{table}_{'_'.join(columns)}"
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({', '.join(columns)})"
            )
            print(f"[build_database] created index {index_name}")


def build_database(data_dir: str, db_path: str) -> sqlite3.Connection:
    """
    Load each MIMIC-III CSV table into a SQLite database.
    Table and column names are lowercased for consistent SQL querying.
    """
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    loaded_tables = set()

    for table in TABLES:
        csv_path = os.path.join(data_dir, f"{table}.csv")
        if not os.path.exists(csv_path):
            print(f"[build_database] warning: {csv_path} not found, skipping {table}")
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df.to_sql(table.lower(), conn, if_exists="replace", index=False)
        loaded_tables.add(table.lower())
        print(f"[build_database] loaded {table.lower()}: {len(df)} rows, {len(df.columns)} columns")

    _create_indexes(conn, loaded_tables)
    conn.commit()
    return conn


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/mimic_demo"
    db_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/mimic_demo.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = build_database(data_dir, db_path)

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print("\nTables in database:", [r[0] for r in cur.fetchall()])
    conn.close()
