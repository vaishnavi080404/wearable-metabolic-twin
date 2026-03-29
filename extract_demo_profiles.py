
import pandas as pd
import numpy as np
 
FEATURE_DATA_PATH = "data/processed/features.parquet"
 
ACTIVITY_LABELS = {
    1:"Lying", 2:"Sitting", 3:"Standing", 4:"Walking", 5:"Running",
    6:"Cycling", 7:"Nordic Walking", 9:"Watching TV", 10:"Computer Work",
    11:"Car Driving", 12:"Ascending Stairs", 13:"Descending Stairs",
    16:"Vacuum Cleaning", 17:"Ironing", 18:"Folding Laundry",
    19:"House Cleaning", 20:"Playing Soccer", 24:"Rope Jumping",
}
 
# These are the key features the demo profiles use
KEY_FEATURES = [
    "hand_acc_mean",
    "ankle_acc_mean",      # Note: this is NOT in MODEL_FEATURES directly,
                           # but we compute it here just for understanding
    "ankle_acc_median",    # This IS in model features
    "hand_gyro_median",
    "ankle_gyro_median",
    "motion_intensity",
    "hr_mean",
    "chest_acc_mean",
    "hand_ankle_ratio",
]
 
print("=" * 65)
print("  Real Feature Averages per Activity (from your training data)")
print("=" * 65)
 
try:
    df = pd.read_parquet(FEATURE_DATA_PATH)
except FileNotFoundError:
    print(f"\nERROR: Could not find {FEATURE_DATA_PATH}")
    print("Make sure you have run:")
    print("  python -m src.preprocess")
    print("  python -m src.features")
    print("  python -m src.train")
    exit()
 
print(f"\nLoaded features: {df.shape[0]:,} windows, {df.shape[1]} columns")
print(f"Activities present: {sorted(df['activityID'].unique())}")
 
# Compute per-activity averages for key features
available = [f for f in KEY_FEATURES if f in df.columns]
missing   = [f for f in KEY_FEATURES if f not in df.columns]
 
if missing:
    print(f"\nNote: These features not found (may use different names): {missing}")
 
summary = df.groupby("activityID")[available].mean().round(3)
summary.index = summary.index.map(lambda x: ACTIVITY_LABELS.get(int(x), str(x)))
summary.index.name = "Activity"
 
print("\n--- Key Feature Averages by Activity ---")
print(summary.to_string())
 
# Now print the 6 most important activities as copy-paste demo profile values
print("\n" + "=" * 65)
print("  Demo Profile Values (copy these into app.py DEMO_PROFILES)")
print("=" * 65)
 
important_activities = [
    "Running", "Walking", "Cycling", "Lying",
    "Ascending Stairs", "Rope Jumping"
]
 
for activity in important_activities:
    if activity not in summary.index:
        continue
    row = summary.loc[activity]
    print(f"\n# {activity}")
 
    # hand_acc_mean
    ham = row.get("hand_acc_mean", 1.0)
    # ankle_acc_median as proxy for ankle motion
    aam = row.get("ankle_acc_median", row.get("ankle_acc_mean", 1.0))
    # hr_mean
    hr  = row.get("hr_mean", 90.0)
    # motion_intensity
    mi  = row.get("motion_intensity", 1.0)
    # gyros
    hgm = row.get("hand_gyro_median", 0.2)
    agm = row.get("ankle_gyro_median", 0.3)
    # chest
    cam = row.get("chest_acc_mean", 0.5)
 
    print(f"  hand_acc_mean  = {ham:.3f}")
    print(f"  ankle_acc_median = {aam:.3f}   ← use this for ankle_acc_mean slider")
    print(f"  hr_mean        = {hr:.1f}")
    print(f"  motion_intensity = {mi:.3f}")
    print(f"  hand_gyro_median = {hgm:.3f}  ← use this for hand_gyro_mean slider")
    print(f"  ankle_gyro_median = {agm:.3f} ← use this for ankle_gyro_mean slider")
    print(f"  chest_acc_mean  = {cam:.3f}")
 
print("\n" + "=" * 65)
print("  How to use this output:")
print("=" * 65)
print("""
1. Look at the values printed above for each activity.
2. Open app.py and find the DEMO_PROFILES dictionary.
3. For each profile (e.g. 'Intense Running'), update:
      hand_acc_mean  → use the value from Running above
      ankle_acc_mean → use ankle_acc_median value from above
      hr_mean        → use the hr_mean value from above
4. The model will then see realistic values and predict correctly.
 
The most important features to get right are:
  - ankle_acc_mean/median   (biggest difference between activities)
  - hr_mean                 (separates rest from exercise)
  - motion_intensity        (overall body movement)
  - hand_ankle_ratio        (arms vs legs: high for nordic walking)
""")
print("Done!")
