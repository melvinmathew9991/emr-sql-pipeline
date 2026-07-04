from build_database import TABLES


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
