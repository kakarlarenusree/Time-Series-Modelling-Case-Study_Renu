import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "electricity_demand.csv"
)

# Column names
DATE_COLUMN = "date"
TARGET_COLUMN = "demand"

# Forecast settings
TEST_SIZE = 0.2
RANDOM_STATE = 42