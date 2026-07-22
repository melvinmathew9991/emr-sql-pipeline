from build_database import TABLES, INDEXES


def test_loads_all_expected_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert tables == {t.lower() for t in TABLES}


def test_column_names_are_lowercased(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(admissions)")
    columns = [row[1] for row in cur.fetchall()]
    assert columns == [c.lower() for c in columns]


def test_row_counts_match_synthetic_fixture(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM patients")
    assert cur.fetchone()[0] == 40
    cur.execute("SELECT COUNT(*) FROM admissions")
    assert cur.fetchone()[0] == 47


def test_all_declared_indexes_are_created(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
    created = {row[0] for row in cur.fetchall()}
    expected = {
        f"idx_{table}_{'_'.join(columns)}"
        for table, column_sets in INDEXES.items()
        for columns in column_sets
    }
    assert expected <= created


def test_admissions_patients_join_uses_index(conn):
    """cohort_features.sql's admissions-to-patients join should seek one side
    via the subject_id index rather than a plain nested-loop scan of both."""
    cur = conn.cursor()
    cur.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT * FROM admissions a JOIN patients p ON a.subject_id = p.subject_id"
    )
    plan = " ".join(row[-1] for row in cur.fetchall())
    assert "USING INDEX idx_" in plan


def test_labevents_admissions_join_uses_index(conn):
    """lab_agg's labevents-to-admissions join (the largest table in the
    schema) should seek one side via the hadm_id index rather than a plain
    nested-loop scan of both."""
    cur = conn.cursor()
    cur.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT * FROM labevents l JOIN admissions adm ON l.hadm_id = adm.hadm_id"
    )
    plan = " ".join(row[-1] for row in cur.fetchall())
    assert "USING INDEX idx_" in plan
