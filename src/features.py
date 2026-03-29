# src/features.py
# Run with: python -m src.features

import os
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from src.config import (
    PROCESSED_DATA_PATH,
    FEATURE_DATA_PATH,
    ACTIVITY_METS,
    SENSOR_GROUPS,
    WINDOW_SIZE,
    STEP_SIZE,
    ARTIFACTS_DIR,
)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ================================================================
#  HELPER: vector magnitude
# ================================================================
def magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


# ================================================================
#  CORE: extract all statistical features from one 3-axis sensor
#  Returns ~14 features per sensor group
# ================================================================
def extract_sensor_features(window, name, x_col, y_col, z_col):
    """
    For each 3-axis sensor, compute:
      - magnitude stats: mean, std, min, max, range, median, IQR
      - energy of magnitude
      - SMA (signal magnitude area) across axes
      - jerk mean + jerk std  (first-order difference of magnitude)
      - correlation between x and y axes (captures coordinated motion)
      - spectral entropy of magnitude (captures regularity)
    Total: 13 features per sensor group
    """
    feat = {}

    # Check all columns exist (they should after preprocessing)
    for col in [x_col, y_col, z_col]:
        if col not in window.columns:
            # Fill zeros if column missing (shouldn't happen with clean data)
            window = window.copy()
            window[col] = 0.0

    x = window[x_col].values.astype(float)
    y = window[y_col].values.astype(float)
    z = window[z_col].values.astype(float)
    mag = magnitude(x, y, z)

    # -- Basic stats on magnitude --
    feat[f"{name}_mean"]   = float(np.mean(mag))
    feat[f"{name}_std"]    = float(np.std(mag, ddof=0))
    feat[f"{name}_min"]    = float(np.min(mag))
    feat[f"{name}_max"]    = float(np.max(mag))
    feat[f"{name}_range"]  = float(np.max(mag) - np.min(mag))
    feat[f"{name}_median"] = float(np.median(mag))
    feat[f"{name}_iqr"]    = float(np.percentile(mag, 75) - np.percentile(mag, 25))

    # -- Energy --
    feat[f"{name}_energy"] = float(np.sum(mag ** 2))

    # -- Signal Magnitude Area (sum of absolute values per axis) --
    feat[f"{name}_sma"] = float(np.sum(np.abs(x) + np.abs(y) + np.abs(z)))

    # -- Jerk (rate of change of magnitude) --
    jerk = np.diff(mag)
    feat[f"{name}_jerk_mean"] = float(np.mean(jerk)) if len(jerk) > 0 else 0.0
    feat[f"{name}_jerk_std"]  = float(np.std(jerk, ddof=0)) if len(jerk) > 0 else 0.0

    # -- Cross-axis correlation (x–y) —— helps distinguish activities by coordination --
    if np.std(x) > 1e-9 and np.std(y) > 1e-9:
        feat[f"{name}_xy_corr"] = float(np.corrcoef(x, y)[0, 1])
    else:
        feat[f"{name}_xy_corr"] = 0.0

    # -- Spectral entropy of magnitude (low = periodic like walking, high = random) --
    feat[f"{name}_spectral_entropy"] = _spectral_entropy(mag)

    return feat  # 14 features per sensor group


def _spectral_entropy(signal):
    """Normalized spectral entropy using FFT power spectrum."""
    if len(signal) < 4:
        return 0.0
    fft_mag = np.abs(np.fft.rfft(signal - np.mean(signal)))
    power = fft_mag ** 2
    total = power.sum()
    if total < 1e-12:
        return 0.0
    prob = power / total
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    max_entropy = np.log2(len(prob)) if len(prob) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# ================================================================
#  HEART RATE FEATURES
# ================================================================
def extract_hr_features(window):
    """7 heart rate features."""
    feat = {}
    hr = window["heartRate"].values.astype(float)
    feat["hr_mean"]    = float(np.mean(hr))
    feat["hr_std"]     = float(np.std(hr, ddof=0))
    feat["hr_min"]     = float(np.min(hr))
    feat["hr_max"]     = float(np.max(hr))
    feat["hr_range"]   = float(np.max(hr) - np.min(hr))
    feat["hr_delta"]   = float(hr[-1] - hr[0])
    # Rolling mean of last 15 samples relative to first 15
    half = max(1, len(hr) // 2)
    feat["hr_trend"]   = float(np.mean(hr[half:]) - np.mean(hr[:half]))
    return feat  # 7 features


# ================================================================
#  CROSS-BODY FEATURES (relationships between sensor locations)
# ================================================================
def extract_cross_body_features(window):
    """
    4 features capturing coordination between body parts.
    These are key for distinguishing walking vs nordic walking, etc.
    """
    feat = {}

    def safe_mag(prefix, x, y, z):
        if all(c in window.columns for c in [x, y, z]):
            return magnitude(
                window[x].values.astype(float),
                window[y].values.astype(float),
                window[z].values.astype(float),
            )
        return np.zeros(len(window))

    hand_mag  = safe_mag("hand",  "handAccX",  "handAccY",  "handAccZ")
    chest_mag = safe_mag("chest", "chestAccX", "chestAccY", "chestAccZ")
    ankle_mag = safe_mag("ankle", "ankleAccX", "ankleAccY", "ankleAccZ")

    # Ratio: hand motion vs ankle motion (high = arm-heavy activity like nordic walking)
    ankle_mean = float(np.mean(ankle_mag))
    feat["hand_ankle_ratio"] = (
        float(np.mean(hand_mag)) / ankle_mean if ankle_mean > 1e-6 else 0.0
    )

    # Correlation: hand and ankle acceleration (sync = walking, async = running)
    min_len = min(len(hand_mag), len(ankle_mag))
    if min_len > 2 and np.std(hand_mag[:min_len]) > 1e-9 and np.std(ankle_mag[:min_len]) > 1e-9:
        feat["hand_ankle_corr"] = float(np.corrcoef(hand_mag[:min_len], ankle_mag[:min_len])[0, 1])
    else:
        feat["hand_ankle_corr"] = 0.0

    # Overall motion intensity (mean of all three)
    feat["motion_intensity"] = float(
        (np.mean(hand_mag) + np.mean(chest_mag) + np.mean(ankle_mag)) / 3.0
    )

    # Gyroscope intensity — captures rotation across body (key for sitting vs standing)
    gyro_cols = [
        ("handGyroX",  "handGyroY",  "handGyroZ"),
        ("chestGyroX", "chestGyroY", "chestGyroZ"),
        ("ankleGyroX", "ankleGyroY", "ankleGyroZ"),
    ]
    gyro_means = []
    for x, y, z in gyro_cols:
        if all(c in window.columns for c in [x, y, z]):
            gyro_means.append(float(np.mean(magnitude(
                window[x].values.astype(float),
                window[y].values.astype(float),
                window[z].values.astype(float),
            ))))
    feat["gyro_intensity"] = float(np.mean(gyro_means)) if gyro_means else 0.0

    return feat  # 4 features


# ================================================================
#  REFERENCE MET (target for regressor)
# ================================================================
def compute_reference_met(window, age=25, resting_hr=60):
    """
    Compute a continuous MET estimate for the window.
    Base MET from activity type, adjusted by HR reserve intensity.
    """
    activity_id = int(window["activityID"].mode().iloc[0])
    base_met = ACTIVITY_METS.get(activity_id, 2.0)
    hr_mean = float(window["heartRate"].mean())
    max_hr = max(160, 220 - age)
    hr_reserve = max(max_hr - resting_hr, 1)
    hr_ratio = float(np.clip((hr_mean - resting_hr) / hr_reserve, 0.0, 1.0))
    # Adjust base MET by HR ratio (max adjustment ±30%)
    adjusted_met = float(np.clip(base_met * (0.7 + 0.6 * hr_ratio), 1.0, 13.0))
    return round(adjusted_met, 3)


# ================================================================
#  MAIN FEATURE EXTRACTION LOOP
# ================================================================
def extract_features(df):
    """
    Sliding window feature extraction over all subjects.

    Feature count breakdown:
      9 sensor groups × 14 features = 126
      7 heart rate features           =   7
      4 cross-body features           =   4
      ─────────────────────────────────────
      Total                           = 137 features
    """
    features = []
    total_subjects = df["subject"].nunique()

    for subj_idx, (subject, group) in enumerate(df.groupby("subject"), 1):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n_windows = 0

        for i in range(0, len(group) - WINDOW_SIZE + 1, STEP_SIZE):
            window = group.iloc[i : i + WINDOW_SIZE]

            # Skip windows with too many missing values
            if window["heartRate"].isna().sum() > WINDOW_SIZE * 0.3:
                continue

            feat = {}

            # Metadata (dropped before training)
            feat["activityID"]    = int(window["activityID"].mode().iloc[0])
            feat["subject"]       = subject
            feat["reference_met"] = compute_reference_met(window)

            # -- 9 sensor groups --
            for name, x_col, y_col, z_col in SENSOR_GROUPS:
                feat.update(extract_sensor_features(window, name, x_col, y_col, z_col))

            # -- Heart rate --
            feat.update(extract_hr_features(window))

            # -- Cross-body --
            feat.update(extract_cross_body_features(window))

            features.append(feat)
            n_windows += 1

        print(f"  [{subj_idx}/{total_subjects}] {subject}: {n_windows} windows")

    df_feat = pd.DataFrame(features)
    print(f"\nTotal windows: {len(df_feat):,}")

    # Count actual feature columns (exclude metadata)
    meta_cols = {"activityID", "subject", "reference_met"}
    feature_cols = [c for c in df_feat.columns if c not in meta_cols]
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Unique activities: {sorted(df_feat['activityID'].unique())}")

    return df_feat


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  Feature Extraction Pipeline")
    print("=" * 50)

    print(f"\nLoading processed data from {PROCESSED_DATA_PATH}...")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print(f"  Shape: {df.shape}")

    print("\nExtracting features (sliding window)...")
    features_df = extract_features(df)

    os.makedirs(os.path.dirname(FEATURE_DATA_PATH), exist_ok=True)
    features_df.to_parquet(FEATURE_DATA_PATH, index=False)
    print(f"\nSaved features -> {FEATURE_DATA_PATH}")
    print("\nDone! Next step: python -m src.train")