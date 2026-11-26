"""Central configuration."""
from pathlib import Path

SEED = 42
TEST_SIZE = 0.20
TARGET = "is_canceled"
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hotel_bookings.csv"
TEST_PATH = ROOT / "data" / "test_data.csv"
RESULTS_DIR = ROOT / "results"
# Columns dropped: outcome leakage (status + its date) and ~94% missing 'company'
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]
DROP_COLS = ["company"]
