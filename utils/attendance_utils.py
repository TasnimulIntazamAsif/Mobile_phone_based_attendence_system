import pandas as pd
from datetime import datetime
from config import CSV_PATH
import os

def mark_attendance(user_id, name, status):
    today = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)

        if ((df["ID"] == user_id) & (df["Date"] == today)).any():
            return "Already Marked"

    now = datetime.now()

    new_row = {
        "ID": user_id,
        "Name": name,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Status": status
    }

    df = pd.DataFrame([new_row])

    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_PATH, index=False)

    return "Marked"