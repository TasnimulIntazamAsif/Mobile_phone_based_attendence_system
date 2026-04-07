import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "data/dataset")
ENCODING_PATH = os.path.join(BASE_DIR, "data/encodings.pkl")
CSV_PATH = os.path.join(BASE_DIR, "data/attendance.csv")

# Ensure folders exist
os.makedirs(DATASET_PATH, exist_ok=True)