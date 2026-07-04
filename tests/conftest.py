import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SYNTHETIC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_test")


@pytest.fixture
def conn():
    """A fresh in-memory SQLite connection built from the synthetic test fixture."""
    from build_database import build_database

    connection = build_database(SYNTHETIC_DIR, ":memory:")
    yield connection
    connection.close()
