# src/evaluate.py
# Run with: python -m src.evaluate

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, confusion_matrix, accuracy_score,
    mean_absolute_error, r2_score,
)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
FEATURE_DATA_PATH = "data/processed/features.parquet"
CLEAN_PATH = "data/processed/clean_data.parquet"
ARTIFACTS_DIR = "artifacts"
OUTPUTS_DIR = "outputs"
MODEL_PATH = "artifacts/activity_model.pkl"
LABEL_MAP_PATH = "artifacts/label_map.json"
FEATURE_COLS_PATH = "artifacts/feature_columns.json"

# ADD THESE TWO LINES HERE:
REGRESSOR_PATH = "artifacts/met_regressor.pkl"
MODEL_META_PATH = "artifacts/model_meta.json"

# Import only what is confirmed in config.py
from src.config import ACTIVITY_LABELS 

os.makedirs(OUTPUTS_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────
#  PLOT STYLE
# ────────────────────────────────────────────────────────
PALETTE   = "Blues"
FIG_DPI   = 150
COLOR_POS = "#2ecc71"
COLOR_NEG = "#e74c3c"
COLOR_MID = "#3498db"
COLOR_ACC = "#f39c12"


def _save(fig, name):
    path = os.path.join(OUTPUTS_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ================================================================
#  EDA CHARTS (Step 6 — from clean_data.parquet)
# ================================================================

def plot_activity_distribution(df):
    print("Plotting activity distribution...")
    counts = df["activityID"].value_counts().sort_index()
    labels = [ACTIVITY_LABELS.get(i, str(i)) for i in counts.index]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, counts.values, color=COLOR_MID, edgecolor="white")
    ax.set_title("Activity Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Row Count")
    plt.xticks(rotation=45, ha="right")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{val:,}", ha="center", va="bottom", fontsize=7,
        )
    _save(fig, "activity_distribution.png")


def plot_hr_by_activity(df):
    print("Plotting heart rate by activity...")
    df2 = df[df["heartRate"].notna()].copy()
    df2["activity_name"] = df2["activityID"].map(ACTIVITY_LABELS).fillna(df2["activityID"].astype(str))
    order = df2.groupby("activity_name")["heartRate"].median().sort_values().index.tolist()
    df2["activity_name"] = pd.Categorical(df2["activity_name"], categories=order, ordered=True)
    df2 = df2.sort_values("activity_name")

    fig, ax = plt.subplots(figsize=(14, 5))
    groups_data = [
        df2[df2["activity_name"] == act]["heartRate"].dropna().values
        for act in order
    ]
    ax.boxplot(groups_data, labels=order, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.set_title("Heart Rate by Activity", fontsize=14, fontweight="bold")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Heart Rate (bpm)")
    plt.xticks(rotation=45, ha="right")
    _save(fig, "hr_by_activity.png")


def plot_missing_values(df):
    print("Plotting missing values heatmap...")
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(sample.isnull(), cbar=False, yticklabels=False, cmap="Blues", ax=ax)
    ax.set_title("Missing Values Heatmap (sample)", fontsize=14, fontweight="bold")
    _save(fig, "missing_values.png")


def plot_sensor_signal(df):
    print("Plotting sensor signals (walking)...")
    sample = df[df["activityID"] == 4].head(500)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    sensor_sets = [
        ("Chest Accelerometer", ["chestAccX", "chestAccY", "chestAccZ"]),
        ("Ankle Gyroscope",     ["ankleGyroX", "ankleGyroY", "ankleGyroZ"]),
        ("Hand Magnetometer",   ["handMagX", "handMagY", "handMagZ"]),
    ]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for ax, (title, cols) in zip(axes, sensor_sets):
        for col, color in zip(cols, colors):
            if col in sample.columns:
                ax.plot(sample[col].values, label=col, color=color, linewidth=0.8)
        ax.set_title(f"{title} — Walking (Activity 4)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Signal Value")
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("Sample Index")
    fig.suptitle("Multi-Sensor Signals During Walking", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "sensor_signal.png")


def plot_sensor_comparison_by_activity(df):
    """New: compare motion intensity across activities for each sensor location."""
    print("Plotting sensor comparison by activity...")
    df2 = df.copy()
    df2["activity_name"] = df2["activityID"].map(ACTIVITY_LABELS).fillna("Unknown")

    def mag(x, y, z):
        return np.sqrt(x**2 + y**2 + z**2)

    if all(c in df2.columns for c in ["handAccX", "handAccY", "handAccZ"]):
        df2["hand_mag"]  = mag(df2["handAccX"], df2["handAccY"], df2["handAccZ"])
    if all(c in df2.columns for c in ["chestAccX", "chestAccY", "chestAccZ"]):
        df2["chest_mag"] = mag(df2["chestAccX"], df2["chestAccY"], df2["chestAccZ"])
    if all(c in df2.columns for c in ["ankleAccX", "ankleAccY", "ankleAccZ"]):
        df2["ankle_mag"] = mag(df2["ankleAccX"], df2["ankleAccY"], df2["ankleAccZ"])

    mag_cols = [c for c in ["hand_mag", "chest_mag", "ankle_mag"] if c in df2.columns]
    if not mag_cols:
        return

    agg = df2.groupby("activity_name")[mag_cols].mean().sort_values(mag_cols[0], ascending=False)
    agg = agg.head(12)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(agg))
    width = 0.25
    colors_bar = [COLOR_POS, COLOR_MID, COLOR_ACC]
    for i, (col, color) in enumerate(zip(mag_cols, colors_bar)):
        ax.bar(x + i * width, agg[col].values, width, label=col.replace("_mag", ""), color=color, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(agg.index, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Mean Acceleration Magnitude")
    ax.set_title("Acceleration Magnitude per Body Location by Activity", fontsize=13, fontweight="bold")
    ax.legend()
    _save(fig, "sensor_comparison_by_activity.png")


# ================================================================
#  EVALUATION CHARTS (Step 8 — from trained model)
# ================================================================

def plot_confusion_matrix(all_y_true, all_y_pred, reverse_map):
    print("Plotting confusion matrix...")
    unique_ids  = sorted(set(all_y_true))
    label_names = [ACTIVITY_LABELS.get(reverse_map.get(i, i), str(i)) for i in unique_ids]

    cm      = confusion_matrix(all_y_true, all_y_pred, labels=unique_ids)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
    )
    ax.set_title("Confusion Matrix (Normalized) — GroupKFold CV", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Activity", fontsize=11)
    ax.set_ylabel("True Activity", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    _save(fig, "confusion_matrix.png")


def plot_per_activity_f1(all_y_true, all_y_pred, reverse_map):
    print("Plotting per-activity F1 scores...")
    unique_ids  = sorted(set(all_y_true))
    label_names = [ACTIVITY_LABELS.get(reverse_map.get(i, i), str(i)) for i in unique_ids]

    f1_scores = f1_score(
        all_y_true, all_y_pred,
        labels=unique_ids,
        average=None,
        zero_division=0,
    )
    pairs = sorted(zip(label_names, f1_scores), key=lambda x: x[1])
    names, scores = zip(*pairs)
    colors = [COLOR_POS if s >= 0.6 else COLOR_ACC if s >= 0.4 else COLOR_NEG for s in scores]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names, scores, color=colors, edgecolor="white")
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, label="0.5 threshold")
    ax.set_title("Per-Activity Macro F1 Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Macro F1")
    ax.set_xlim(0, 1.05)
    for i, (name, score) in enumerate(zip(names, scores)):
        ax.text(score + 0.01, i, f"{score:.2f}", va="center", fontsize=9)
    ax.legend()
    _save(fig, "per_activity_f1.png")


def plot_per_subject_scores(X, y, y_met, groups, label_map, reverse_map):
    print("Plotting per-subject F1 scores (GroupKFold)...")
    y_encoded = y.map(label_map)
    gkf       = GroupKFold(n_splits=4)
    subject_acc   = {}
    subject_f1    = {}

    unique_labels = sorted(y_encoded.unique())
    params = {**{
        "objective": "multiclass",
        "num_class": len(unique_labels),
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }}

    for train_idx, val_idx in gkf.split(X, y_encoded, groups):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[train_idx], y_encoded.iloc[train_idx],
              callbacks=[lgb.log_evaluation(False)])
        y_pred = m.predict(X.iloc[val_idx])
        val_subjects = groups.iloc[val_idx]

        for subj in val_subjects.unique():
            mask  = val_subjects == subj
            true  = y_encoded.iloc[val_idx][mask]
            pred  = y_pred[mask.values]
            subject_f1[subj]  = round(float(f1_score(true, pred, average="macro", zero_division=0)), 4)
            subject_acc[subj] = round(float(accuracy_score(true, pred)), 4)

    subjects = sorted(subject_f1.keys())
    f1s      = [subject_f1[s] for s in subjects]
    accs     = [subject_acc[s] for s in subjects]
    colors   = [COLOR_POS if s >= 0.5 else COLOR_NEG for s in f1s]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, values, ylabel, title in [
        (axes[0], f1s,  "Macro F1",  "Per-Subject Macro F1 Score"),
        (axes[1], accs, "Accuracy",  "Per-Subject Accuracy"),
    ]:
        bars = ax.bar(subjects, values, color=colors, edgecolor="white")
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, "per_subject_scores.png")


def plot_loso_results(loso_rows):
    """New: LOSO per-subject results bar chart."""
    if not loso_rows:
        return
    print("Plotting LOSO results...")
    df_loso = pd.DataFrame(loso_rows).sort_values("macro_f1")
    colors  = [COLOR_POS if v >= 0.5 else COLOR_NEG for v in df_loso["macro_f1"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df_loso["subject"], df_loso["macro_f1"], color=colors, edgecolor="white")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="0.5 threshold")
    ax.set_title("LOSO — Per-Subject Macro F1 (Leave-One-Subject-Out)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Held-Out Subject")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, df_loso["macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    _save(fig, "loso_results.png")


def plot_feature_importance(model, feature_cols, top_n=30):
    print(f"Plotting top-{top_n} feature importances...")
    feat_imp = pd.Series(
        model.feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=True).tail(top_n)

    colors = [COLOR_POS if v >= feat_imp.max() * 0.6
              else COLOR_MID if v >= feat_imp.max() * 0.3
              else "#95a5a6" for v in feat_imp.values]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor="white")
    ax.set_title(f"Feature Importance — Top {top_n} (LightGBM)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    _save(fig, "feature_importance.png")


def plot_met_scatter(reference_mets, predicted_mets):
    """New: Predicted vs Reference MET scatter plot."""
    if not reference_mets or not predicted_mets:
        return
    print("Plotting MET scatter...")
    ref  = np.array(reference_mets)
    pred = np.array(predicted_mets)

    mae = mean_absolute_error(ref, pred)
    r2  = r2_score(ref, pred)
    low  = min(ref.min(), pred.min()) - 0.2
    high = max(ref.max(), pred.max()) + 0.2

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ref, pred, alpha=0.4, color=COLOR_MID, s=10)
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("Reference MET")
    ax.set_ylabel("Predicted MET")
    ax.set_title(f"MET Regressor — MAE={mae:.3f}  R²={r2:.3f}", fontsize=12, fontweight="bold")
    ax.legend()
    _save(fig, "met_scatter.png")


def plot_sensor_feature_heatmap(df_feat):
    """New: correlation heatmap of all numeric features."""
    print("Plotting feature correlation heatmap...")
    numeric = df_feat.select_dtypes(include=[np.number])
    # Pick a subset — full 137×137 is unreadable
    important = [c for c in numeric.columns
                 if any(k in c for k in ["mean", "std", "energy", "hr_"])][:25]
    if len(important) < 2:
        return

    corr = numeric[important].dropna().corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, annot=False, cmap="RdBu_r",
        vmin=-1, vmax=1, linewidths=0.3,
        xticklabels=True, yticklabels=True, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap (selected features)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    _save(fig, "feature_correlation_heatmap.png")


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  PAMAP2 Evaluation Pipeline")
    print("=" * 55)

    # ── EDA Charts ─────────────────────────────────────────────
    print("\n[1/3] EDA Charts (from clean data)...")
    if os.path.exists(CLEAN_PATH):
        df_clean = pd.read_parquet(CLEAN_PATH)
        plot_activity_distribution(df_clean)
        plot_hr_by_activity(df_clean)
        plot_missing_values(df_clean)
        plot_sensor_signal(df_clean)
        plot_sensor_comparison_by_activity(df_clean)
    else:
        print(f"  Skipping EDA — {CLEAN_PATH} not found (run preprocess first)")

    # ── Load model artifacts ────────────────────────────────────
    print("\n[2/3] Loading model artifacts...")
    model_ok = os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH)
    if not model_ok:
        print("  No trained model found. Run: python -m src.train")
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(LABEL_MAP_PATH, "r") as f:
            raw       = json.load(f)
            label_map = {int(k): int(v) for k, v in raw.items()}
            reverse_map = {v: k for k, v in label_map.items()}

        with open(FEATURE_COLS_PATH, "r") as f:
            feature_cols = json.load(f)

        reg = None
        if os.path.exists(REGRESSOR_PATH):
            with open(REGRESSOR_PATH, "rb") as f:
                reg = pickle.load(f)

        meta = {}
        if os.path.exists(MODEL_META_PATH):
            with open(MODEL_META_PATH, "r") as f:
                meta = json.load(f)

        # ── Evaluation Charts ───────────────────────────────────
        print("\n[3/3] Evaluation Charts (from features + model)...")
        df_feat = pd.read_parquet(FEATURE_DATA_PATH)

        drop_meta = [c for c in ["activityID", "subject", "reference_met"] if c in df_feat.columns]
        available_features = [c for c in feature_cols if c in df_feat.columns]

        X      = df_feat[available_features].select_dtypes(include=[np.number]).fillna(0)
        y      = df_feat["activityID"]
        groups = df_feat["subject"]
        y_met  = df_feat["reference_met"] if "reference_met" in df_feat.columns else None

        # Re-run CV to collect fold predictions
        print("  Re-running CV to collect predictions...")
        y_encoded     = y.map(label_map)
        gkf           = GroupKFold(n_splits=4)
        all_true, all_pred = [], []
        all_met_true, all_met_pred = [], []
        unique_labels = sorted(y_encoded.unique())

        for train_idx, val_idx in gkf.split(X, y_encoded, groups):
            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=len(unique_labels),
                num_leaves=63, learning_rate=0.05, n_estimators=300,
                random_state=42, n_jobs=-1, verbose=-1,
            )
            m.fit(X.iloc[train_idx], y_encoded.iloc[train_idx],
                  callbacks=[lgb.log_evaluation(False)])
            preds = m.predict(X.iloc[val_idx])
            all_true.extend(y_encoded.iloc[val_idx].tolist())
            all_pred.extend(preds.tolist())

            if reg is not None and y_met is not None:
                met_preds = reg.predict(X.iloc[val_idx])
                all_met_true.extend(y_met.iloc[val_idx].tolist())
                all_met_pred.extend(met_preds.tolist())

        # Print summary
        overall_f1  = f1_score(all_true, all_pred, average="macro", zero_division=0)
        overall_acc = accuracy_score(all_true, all_pred)
        print(f"\n  GroupKFold Macro F1 : {overall_f1:.4f}")
        print(f"  GroupKFold Accuracy : {overall_acc:.4f}")

        # Plot everything
        plot_confusion_matrix(all_true, all_pred, reverse_map)
        plot_per_activity_f1(all_true, all_pred, reverse_map)
        plot_per_subject_scores(X, y, y_met, groups, label_map, reverse_map)
        plot_feature_importance(model, available_features, top_n=30)
        plot_sensor_feature_heatmap(df_feat)

        if all_met_true:
            plot_met_scatter(all_met_true, all_met_pred)

        if meta.get("loso_subject_rows"):
            plot_loso_results(meta["loso_subject_rows"])

    print(f"\nAll charts saved in: {OUTPUTS_DIR}/")
    charts = [
        "activity_distribution", "hr_by_activity", "missing_values",
        "sensor_signal", "sensor_comparison_by_activity",
        "confusion_matrix", "per_activity_f1", "per_subject_scores",
        "loso_results", "feature_importance", "feature_correlation_heatmap",
        "met_scatter",
    ]
    print("  " + "  |  ".join(charts))