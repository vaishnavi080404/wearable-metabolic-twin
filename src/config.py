# src/config.py

# -------------------- PATHS --------------------
RAW_DATA_PATH       = "data/raw/PAMAP2_Dataset/Protocol"
PROCESSED_DATA_PATH = "data/processed/clean_data.parquet"
FEATURE_DATA_PATH   = "data/processed/features.parquet"

ARTIFACTS_DIR = "artifacts"
OUTPUTS_DIR   = "outputs"

MODEL_PATH          = "artifacts/activity_model.pkl"
REGRESSOR_PATH      = "artifacts/met_regressor.pkl"
FEATURE_COLS_PATH   = "artifacts/feature_columns.json"
LABEL_MAP_PATH      = "artifacts/label_map.json"
MODEL_META_PATH     = "artifacts/model_meta.json"
EXERTION_RULES_PATH = "artifacts/exertion_rules.json"
SCALER_PATH         = "artifacts/scaler.pkl"

# -------------------- WINDOW SETTINGS --------------------
WINDOW_SIZE   = 50    # 5 sec * 10 Hz
STEP_SIZE     = 25    # 50% overlap
SAMPLING_RATE = 10    # Hz

# -------------------- ACTIVITY LABELS --------------------
ACTIVITY_LABELS = {
    1:  "Lying",
    2:  "Sitting",
    3:  "Standing",
    4:  "Walking",
    5:  "Running",
    6:  "Cycling",
    7:  "Nordic Walking",
    9:  "Watching TV",
    10: "Computer Work",
    11: "Car Driving",
    12: "Ascending Stairs",
    13: "Descending Stairs",
    16: "Vacuum Cleaning",
    17: "Ironing",
    18: "Folding Laundry",
    19: "House Cleaning",
    20: "Playing Soccer",
    24: "Rope Jumping",
}

# -------------------- MET VALUES (published compendium) --------------------
ACTIVITY_METS = {
    1:  1.0,   # Lying
    2:  1.3,   # Sitting
    3:  1.8,   # Standing
    4:  3.5,   # Walking
    5:  8.3,   # Running
    6:  6.8,   # Cycling
    7:  6.0,   # Nordic Walking
    9:  1.2,   # Watching TV
    10: 1.5,   # Computer Work
    11: 2.0,   # Car Driving
    12: 8.8,   # Ascending Stairs
    13: 4.0,   # Descending Stairs
    16: 3.3,   # Vacuum Cleaning
    17: 2.3,   # Ironing
    18: 2.0,   # Folding Laundry
    19: 3.0,   # House Cleaning
    20: 7.0,   # Playing Soccer
    24: 11.0,  # Rope Jumping
}

# -------------------- EXERTION RULES --------------------
EXERTION_RULES = {
    "bands": {
        "low":       {"hr_max": 90,  "motion_max": 0.3},
        "moderate":  {"hr_max": 120, "motion_max": 0.7},
        "high":      {"hr_max": 150, "motion_max": 1.2},
        "very_high": {"hr_max": 999, "motion_max": 999},
    },
    "description": "Rule-based exertion proxy using HR mean + motion SMA",
}

# -------------------- SENSOR GROUPS (for feature extraction) --------------------
# Each entry: (name, x_col, y_col, z_col)
SENSOR_GROUPS = [
    ("hand_acc",   "handAccX",   "handAccY",   "handAccZ"),
    ("chest_acc",  "chestAccX",  "chestAccY",  "chestAccZ"),
    ("ankle_acc",  "ankleAccX",  "ankleAccY",  "ankleAccZ"),
    ("hand_gyro",  "handGyroX",  "handGyroY",  "handGyroZ"),
    ("chest_gyro", "chestGyroX", "chestGyroY", "chestGyroZ"),
    ("ankle_gyro", "ankleGyroX", "ankleGyroY", "ankleGyroZ"),
    ("hand_mag",   "handMagX",   "handMagY",   "handMagZ"),
    ("chest_mag",  "chestMagX",  "chestMagY",  "chestMagZ"),
    ("ankle_mag",  "ankleMagX",  "ankleMagY",  "ankleMagZ"),
]