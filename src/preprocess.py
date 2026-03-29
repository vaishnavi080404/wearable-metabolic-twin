# src/preprocess.py
# Run with: python -m src.preprocess

import os
import pandas as pd
import numpy as np
from glob import glob

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, OUTPUTS_DIR

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# -------------------- COLUMN NAMES --------------------
# PAMAP2 has exactly 54 columns per row (no header in .dat files)
COLUMNS = [
    "timestamp", "activityID", "heartRate",

    # Hand IMU
    "handTemp",
    "handAccX",  "handAccY",  "handAccZ",    # ±16g accelerometer
    "handAcc6X", "handAcc6Y", "handAcc6Z",   # ±6g  accelerometer
    "handGyroX", "handGyroY", "handGyroZ",
    "handMagX",  "handMagY",  "handMagZ",
    "handOrientation1", "handOrientation2", "handOrientation3", "handOrientation4",

    # Chest IMU
    "chestTemp",
    "chestAccX",  "chestAccY",  "chestAccZ",
    "chestAcc6X", "chestAcc6Y", "chestAcc6Z",
    "chestGyroX", "chestGyroY", "chestGyroZ",
    "chestMagX",  "chestMagY",  "chestMagZ",
    "chestOrientation1", "chestOrientation2", "chestOrientation3", "chestOrientation4",

    # Ankle IMU
    "ankleTemp",
    "ankleAccX",  "ankleAccY",  "ankleAccZ",
    "ankleAcc6X", "ankleAcc6Y", "ankleAcc6Z",
    "ankleGyroX", "ankleGyroY", "ankleGyroZ",
    "ankleMagX",  "ankleMagY",  "ankleMagZ",
    "ankleOrientation1", "ankleOrientation2", "ankleOrientation3", "ankleOrientation4",
]

# Columns to drop before saving (orientation = invalid in dataset)
ORIENTATION_COLS = [c for c in COLUMNS if "Orientation" in c]
# ±6g accel is redundant with ±16g for most activities — keep both for completeness
# but drop orientation as it has documented NaN issues in PAMAP2


# -------------------- DATA INSPECTION --------------------
def inspect_data(df):
    print("\n--- DATA INSPECTION ---")
    print(f"Shape: {df.shape}")
    print(f"\nUnique Activities: {sorted(df['activityID'].unique())}")
    print(f"\nActivity counts:\n{df['activityID'].value_counts().sort_index()}")
    print(f"\nMissing values (top 10):\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
    print(f"\nBasic stats:\n{df.describe().round(3)}")


# -------------------- LOAD DATA --------------------
def load_data():
    files = sorted(glob(os.path.join(RAW_DATA_PATH, "subject*.dat")))
    if not files:
        raise FileNotFoundError(
            f"No .dat files found in: {RAW_DATA_PATH}\n"
            "Make sure the PAMAP2 dataset is placed at data/raw/PAMAP2_Dataset/Protocol/"
        )

    dataframes = []
    for file in files:
        print(f"  Reading {os.path.basename(file)} ...")
        df = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            names=COLUMNS,
            na_values=["NaN"],
        )
        subject_id = os.path.basename(file).replace(".dat", "")
        df["subject"] = subject_id
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)
    print(f"\nLoaded {len(files)} subject files — total rows: {len(combined):,}")
    return combined


# -------------------- CLEAN DATA --------------------
def clean_data(df):
    print("\nCleaning data...")

    # 1. Remove transient activity (activityID == 0 means transition)
    before = len(df)
    df = df[df["activityID"] != 0].copy()
    print(f"  Removed {before - len(df):,} transient rows (activityID=0)")

    # 2. Drop orientation columns (invalid/NaN in PAMAP2)
    df = df.drop(columns=ORIENTATION_COLS, errors="ignore")

    # 3. Resample each subject to 10 Hz (100 ms intervals)
    print("  Resampling to 10 Hz per subject...")
    resampled = []
    for subject, group in df.groupby("subject"):
        group = group.copy().sort_values("timestamp")
        group["timestamp"] = pd.to_datetime(group["timestamp"], unit="s")
        group = group.set_index("timestamp")
        group_resampled = group.resample("100ms").mean(numeric_only=True)
        # Restore categorical columns after mean (they become float)
        group_resampled["activityID"] = (
            group_resampled["activityID"].round().ffill().bfill().astype(int)
        )
        group_resampled["subject"] = subject
        resampled.append(group_resampled)

    df = pd.concat(resampled).reset_index()
    df = df.rename(columns={"timestamp": "timestamp"})

    # 4. Forward-fill + backward-fill missing sensor values per subject+activity
    print("  Filling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = (
        df.groupby(["subject", "activityID"])[numeric_cols]
        .transform(lambda x: x.ffill().bfill())
    )

    # 5. Drop any remaining NaN rows (can happen at boundaries)
    before = len(df)
    df = df.dropna(subset=["heartRate"]).reset_index(drop=True)
    print(f"  Dropped {before - len(df):,} rows with no heart rate after fill")

    print(f"  Final shape: {df.shape}")
    return df


# -------------------- SAVE DATA --------------------
def save_data(df):
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    print(f"\nSaved cleaned data -> {PROCESSED_DATA_PATH}")
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print(f"  Subjects: {sorted(df['subject'].unique())}")
    print(f"  Activities: {sorted(df['activityID'].unique())}")


# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("=" * 50)
    print("  PAMAP2 Preprocessing Pipeline")
    print("=" * 50)

    print("\n[1/3] Loading raw data...")
    df = load_data()

    print("\n[2/3] Inspecting raw data...")
    inspect_data(df)

    print("\n[3/3] Cleaning and saving...")
    df = clean_data(df)
    save_data(df)

    print("\nDone! Next step: python -m src.features")