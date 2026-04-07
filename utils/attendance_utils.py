import pandas as pd
from datetime import datetime
from config import CSV_PATH
import os

EXPECTED_COLUMNS = ["ID", "Name", "Date", "Arrival", "Exit", "Status"]


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Migrate older CSV format to the new arrival/exit schema safely.
    """
    if "Arrival" not in df.columns:
        if "Time" in df.columns:
            df["Arrival"] = df["Time"]
        else:
            df["Arrival"] = ""

    if "Exit" not in df.columns:
        df["Exit"] = ""

    if "Status" not in df.columns:
        if "Status" in df.columns:
            pass
        else:
            df["Status"] = "Arrived"

    if "ID" not in df.columns:
        df["ID"] = ""
    if "Name" not in df.columns:
        df["Name"] = ""
    if "Date" not in df.columns:
        df["Date"] = ""

    # Keep only the canonical column order
    return df[EXPECTED_COLUMNS]


def mark_attendance(user_id, name):
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = _ensure_schema(df)
    else:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    # Match current user for today's row
    mask = (df["ID"].astype(str) == str(user_id)) & (df["Date"].astype(str) == today)

    if mask.any():
        idx = df[mask].index[0]
        existing_exit = str(df.at[idx, "Exit"]).strip()

        if existing_exit and existing_exit.lower() != "nan":
            return "Already Exited"

        df.at[idx, "Exit"] = current_time
        df.at[idx, "Status"] = "Exited"
        df.to_csv(CSV_PATH, index=False)
        return f"Exit Marked ({current_time})"

    new_row = {
        "ID": user_id,
        "Name": name,
        "Date": today,
        "Arrival": current_time,
        "Exit": "",
        "Status": "Arrived",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    return f"Arrival Marked ({current_time})"


def get_attendance_records():
    if not os.path.exists(CSV_PATH):
        return []

    df = pd.read_csv(CSV_PATH)
    df = _ensure_schema(df)
    # Replace NaN with empty strings for clean JSON output
    df = df.fillna("")
    return df.to_dict(orient="records")