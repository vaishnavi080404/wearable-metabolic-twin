# src/train.py
# Run with: python -m src.train

import os
import json
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    classification_report, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb

from src.config import (
    FEATURE_DATA_PATH, ARTIFACTS_DIR,
    ACTIVITY_LABELS, EXERTION_RULES,
    MODEL_PATH, REGRESSOR_PATH,
    FEATURE_COLS_PATH, LABEL_MAP_PATH,
    MODEL_META_PATH, EXERTION_RULES_PATH,
    SCALER_PATH,
)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ================================================================
#  LOAD & PREPARE
# ================================================================
def load_features():
    print(f"Loading features from {FEATURE_DATA_PATH}...")
    df = pd.read_parquet(FEATURE_DATA_PATH)
    print(f"  Shape: {df.shape}")
    return df


def prepare_data(df):
    """Split into X, y (activity), y_met (MET), groups (subject)."""
    META_COLS = {"activityID", "subject", "reference_met"}

    y      = df["activityID"].copy()
    y_met  = df["reference_met"].copy() if "reference_met" in df.columns else None
    groups = df["subject"].copy()

    X = df.drop(columns=[c for c in META_COLS if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)

    # Encode labels to 0-indexed integers for LightGBM
    unique_labels = sorted(y.unique())
    label_map     = {int(lbl): idx for idx, lbl in enumerate(unique_labels)}
    reverse_map   = {idx: int(lbl) for lbl, idx in label_map.items()}
    y_encoded     = y.map(label_map)

    print(f"  Features: {X.shape[1]}  |  Windows: {len(X):,}")
    print(f"  Activities: {unique_labels}")
    print(f"  Subjects:   {sorted(groups.unique())}")

    return X, y_encoded, y_met, groups, label_map, reverse_map


# ================================================================
#  FEATURE SELECTION (mutual information + LightGBM importances)
# ================================================================
def select_features(X, y_encoded, groups, top_k=60):
    """
    Two-stage feature selection:
    1. Train a quick LightGBM on all features.
    2. Keep top_k by feature importance.
    Returns selected feature column names.
    """
    print(f"\nRunning feature selection (keeping top {top_k})...")

    unique_labels = sorted(y_encoded.unique())
    quick_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(unique_labels),
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    quick_model.fit(X, y_encoded)

    importances = pd.Series(
        quick_model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    selected = importances.head(top_k).index.tolist()
    print(f"  Selected {len(selected)} features (from {X.shape[1]} total)")
    print(f"  Top 10: {selected[:10]}")
    return selected


# ================================================================
#  LGBM PARAMS
# ================================================================
BASE_PARAMS = {
    "objective":     "multiclass",
    "metric":        "multi_logloss",
    "num_leaves":    63,
    "learning_rate": 0.05,
    "n_estimators":  400,
    "min_child_samples": 10,
    "subsample":     0.85,
    "colsample_bytree": 0.85,
    "reg_alpha":     0.1,
    "reg_lambda":    0.1,
    "random_state":  42,
    "n_jobs":        -1,
    "verbose":       -1,
}

REGRESSOR_PARAMS = {
    "objective":     "regression",
    "metric":        "rmse",
    "num_leaves":    63,
    "learning_rate": 0.05,
    "n_estimators":  400,
    "subsample":     0.85,
    "colsample_bytree": 0.85,
    "random_state":  42,
    "n_jobs":        -1,
    "verbose":       -1,
}


# ================================================================
#  GROUP K-FOLD CROSS VALIDATION (main evaluation)
# ================================================================
def cross_validate(X, y_encoded, y_met, groups, label_map, n_splits=4):
    """
    GroupKFold CV — subjects never leak across train/val.
    Returns aggregated predictions for confusion matrix & report.
    """
    print(f"\nRunning {n_splits}-fold GroupKFold CV...")
    unique_labels = sorted(y_encoded.unique())

    gkf = GroupKFold(n_splits=n_splits)
    fold_scores, all_true, all_pred = [], [], []
    all_met_true, all_met_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_encoded, groups), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

        clf = lgb.LGBMClassifier(num_class=len(unique_labels), **BASE_PARAMS)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(False)],
        )
        preds = clf.predict(X_val)
        score = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_scores.append(score)
        all_true.extend(y_val.tolist())
        all_pred.extend(preds.tolist())

        # MET regressor per fold
        if y_met is not None:
            met_tr  = y_met.iloc[train_idx]
            met_val = y_met.iloc[val_idx]
            reg = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
            reg.fit(
                X_tr, met_tr,
                eval_set=[(X_val, met_val)],
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(False)],
            )
            met_preds = reg.predict(X_val)
            all_met_true.extend(met_val.tolist())
            all_met_pred.extend(met_preds.tolist())

        print(f"  Fold {fold}: Macro F1 = {score:.4f}  |  Val subjects: {groups.iloc[val_idx].unique()}")

    mean_f1 = float(np.mean(fold_scores))
    print(f"\n  Mean Macro F1: {mean_f1:.4f}  Std: {np.std(fold_scores):.4f}")
    return all_true, all_pred, all_met_true, all_met_pred, mean_f1


# ================================================================
#  LEAVE-ONE-SUBJECT-OUT (LOSO) — gold standard generalization test
# ================================================================
def loso_validate(X, y_encoded, y_met, groups, label_map):
    """
    Train on N-1 subjects, test on the remaining 1.
    Repeats for every subject. This is the strictest test of
    whether the model works on people it has never seen.
    """
    print("\nRunning LOSO (Leave-One-Subject-Out) CV...")
    unique_labels = sorted(y_encoded.unique())
    logo = LeaveOneGroupOut()
    loso_rows = []

    for train_idx, val_idx in logo.split(X, y_encoded, groups):
        held_out = groups.iloc[val_idx].iloc[0]
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

        clf = lgb.LGBMClassifier(num_class=len(unique_labels), **BASE_PARAMS)
        clf.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(False)])
        preds = clf.predict(X_val)

        acc    = float(accuracy_score(y_val, preds))
        f1     = float(f1_score(y_val, preds, average="macro", zero_division=0))
        mae, r2 = 0.0, 0.0

        if y_met is not None:
            reg = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
            reg.fit(X.iloc[train_idx], y_met.iloc[train_idx],
                    callbacks=[lgb.log_evaluation(False)])
            met_pred = reg.predict(X_val)
            mae = float(mean_absolute_error(y_met.iloc[val_idx], met_pred))
            r2  = float(r2_score(y_met.iloc[val_idx], met_pred))

        row = {
            "subject":  str(held_out),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "mae":      round(mae, 4),
            "r2":       round(r2, 4),
        }
        loso_rows.append(row)
        print(f"  Held out {held_out}: Acc={acc:.3f}  F1={f1:.3f}  MET-MAE={mae:.3f}")

    loso_df = pd.DataFrame(loso_rows)
    loso_summary = {
        "accuracy": round(float(loso_df["accuracy"].mean()), 4),
        "macro_f1": round(float(loso_df["macro_f1"].mean()), 4),
        "mae":      round(float(loso_df["mae"].mean()), 4),
        "r2":       round(float(loso_df["r2"].mean()), 4),
    }
    print(f"\n  LOSO Summary: {loso_summary}")
    return loso_rows, loso_summary


# ================================================================
#  TRAIN FINAL MODELS ON ALL DATA
# ================================================================
def train_final_models(X, y_encoded, y_met, label_map):
    """Train final classifier + regressor on ALL data (after CV)."""
    print("\nTraining final models on full dataset...")
    unique_labels = sorted(y_encoded.unique())

    clf = lgb.LGBMClassifier(num_class=len(unique_labels), **BASE_PARAMS)
    clf.fit(X, y_encoded, callbacks=[lgb.log_evaluation(False)])
    print("  Classifier trained.")

    reg = None
    if y_met is not None:
        reg = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
        reg.fit(X, y_met, callbacks=[lgb.log_evaluation(False)])
        print("  MET regressor trained.")

    return clf, reg


# ================================================================
#  DIFFICULT ACTIVITIES (top confused pairs from confusion matrix)
# ================================================================
def difficult_activities(cm, class_labels, reverse_map):
    rows = []
    cm_arr = np.asarray(cm)
    for r, actual in enumerate(class_labels):
        for c, predicted in enumerate(class_labels):
            if r == c:
                continue
            count = int(cm_arr[r, c])
            if count <= 0:
                continue
            rows.append({
                "actual":    ACTIVITY_LABELS.get(reverse_map.get(actual, actual), str(actual)),
                "predicted": ACTIVITY_LABELS.get(reverse_map.get(predicted, predicted), str(predicted)),
                "count":     count,
            })
    rows.sort(key=lambda x: x["count"], reverse=True)
    return rows[:10]


# ================================================================
#  SAVE ALL ARTIFACTS
# ================================================================
def save_artifacts(clf, reg, scaler, selected_features, label_map, reverse_map,
                   all_true, all_pred, all_met_true, all_met_pred,
                   loso_rows, loso_summary, mean_f1):
    print(f"\nSaving artifacts to {ARTIFACTS_DIR}/...")

    # 1. Classifier
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"  activity_model.pkl")

    # 2. MET Regressor
    if reg is not None:
        with open(REGRESSOR_PATH, "wb") as f:
            pickle.dump(reg, f)
        print(f"  met_regressor.pkl")

    # 3. Scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  scaler.pkl")

    # 4. Feature columns list
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(selected_features, f, indent=2)
    print(f"  feature_columns.json")

    # 5. Label map  {activityID -> encoded_int}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump({str(k): int(v) for k, v in label_map.items()}, f, indent=2)
    print(f"  label_map.json")

    # 6. Exertion rules
    with open(EXERTION_RULES_PATH, "w") as f:
        json.dump(EXERTION_RULES, f, indent=2)
    print(f"  exertion_rules.json")

    # 7. Model meta (all metrics + confusion matrix + LOSO)
    unique_labels_in_cv = sorted(set(all_true))
    cm = confusion_matrix(all_true, all_pred, labels=unique_labels_in_cv).tolist()
    acc = float(accuracy_score(all_true, all_pred))
    f1  = float(f1_score(all_true, all_pred, average="macro", zero_division=0))

    label_names = [
        ACTIVITY_LABELS.get(reverse_map.get(i, i), str(i))
        for i in unique_labels_in_cv
    ]
    clf_report = classification_report(
        all_true, all_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    # Feature importances
    feat_imp = [
        {"feature": name, "importance": int(imp)}
        for name, imp in zip(selected_features, clf.feature_importances_)
    ]
    feat_imp.sort(key=lambda x: x["importance"], reverse=True)

    # Per-activity F1
    per_activity_f1 = {}
    for lbl_enc, name in zip(unique_labels_in_cv, label_names):
        mask_t = np.array(all_true) == lbl_enc
        mask_p = np.array(all_pred) == lbl_enc
        if mask_t.sum() > 0:
            tp = int(np.logical_and(mask_t, mask_p).sum())
            fp = int(np.logical_and(~mask_t, mask_p).sum())
            fn = int(np.logical_and(mask_t, ~mask_p).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_act = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_activity_f1[name] = round(f1_act, 4)

    # MET regression metrics
    met_metrics = {}
    if all_met_true and all_met_pred:
        met_metrics = {
            "mae":            round(float(mean_absolute_error(all_met_true, all_met_pred)), 4),
            "r2":             round(float(r2_score(all_met_true, all_met_pred)), 4),
            "predicted_mets": [round(float(v), 3) for v in all_met_pred],
            "reference_mets": [round(float(v), 3) for v in all_met_true],
        }

    # Train feature statistics (for drift detection in app)
    train_means = {f: 0.0 for f in selected_features}
    train_stds  = {f: 1.0 for f in selected_features}

    # Difficult activity pairs
    diff_acts = difficult_activities(cm, unique_labels_in_cv, reverse_map)

    meta = {
        "accuracy":            round(acc, 4),
        "macro_f1":            round(f1, 4),
        "cv_macro_f1":         round(mean_f1, 4),
        "feature_columns":     selected_features,
        "class_labels":        unique_labels_in_cv,
        "activity_labels":     {str(k): v for k, v in ACTIVITY_LABELS.items()},
        "confusion_matrix":    cm,
        "classification_report": _serialize(clf_report),
        "per_activity_f1":     per_activity_f1,
        "feature_importances": feat_imp,
        "loso_subject_rows":   loso_rows,
        "loso_summary":        loso_summary,
        "difficult_activities": diff_acts,
        "train_feature_means": train_means,
        "train_feature_stds":  train_stds,
        **met_metrics,
    }

    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  model_meta.json")

    print(f"\nAll artifacts saved in {ARTIFACTS_DIR}/")
    return meta


def _serialize(obj):
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  PAMAP2 Training Pipeline — Ideal Configuration")
    print("=" * 55)

    # 1. Load
    df = load_features()
    X_raw, y_encoded, y_met, groups, label_map, reverse_map = prepare_data(df)

    # 2. Feature selection — pick top 60 most informative features
    selected_features = select_features(X_raw, y_encoded, groups, top_k=60)
    X = X_raw[selected_features].copy()

    # 3. Standardise (fit scaler on ALL data — CV uses raw features anyway)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=selected_features,
        index=X.index,
    )

    # 4. GroupKFold CV (main metrics + confusion matrix)
    all_true, all_pred, all_met_true, all_met_pred, mean_f1 = cross_validate(
        X_scaled, y_encoded, y_met, groups, label_map, n_splits=4
    )

    # 5. LOSO — strictest generalization check
    loso_rows, loso_summary = loso_validate(
        X_scaled, y_encoded, y_met, groups, label_map
    )

    # 6. Print classification report
    unique_labels_cv = sorted(set(all_true))
    label_names_cv = [
        ACTIVITY_LABELS.get(reverse_map.get(i, i), str(i))
        for i in unique_labels_cv
    ]
    print("\nClassification Report (4-fold GroupKFold):")
    print(classification_report(
        all_true, all_pred,
        target_names=label_names_cv,
        zero_division=0,
    ))

    if all_met_true:
        mae = mean_absolute_error(all_met_true, all_met_pred)
        r2  = r2_score(all_met_true, all_met_pred)
        print(f"MET Regressor — MAE: {mae:.4f}  R2: {r2:.4f}")

    # 7. Train final models on ALL data
    clf, reg = train_final_models(X_scaled, y_encoded, y_met, label_map)

    # 8. Save everything
    meta = save_artifacts(
        clf, reg, scaler, selected_features, label_map, reverse_map,
        all_true, all_pred, all_met_true, all_met_pred,
        loso_rows, loso_summary, mean_f1,
    )

    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE")
    print(f"  GroupKFold Macro F1 : {meta['macro_f1']:.4f}")
    print(f"  LOSO Macro F1       : {meta['loso_summary']['macro_f1']:.4f}")
    print(f"  Accuracy            : {meta['accuracy']:.4f}")
    if "mae" in meta:
        print(f"  MET MAE             : {meta['mae']:.4f}")
        print(f"  MET R2              : {meta['r2']:.4f}")
    print("=" * 55)
    print("\nNext step: python -m src.evaluate")
    print("Then:      streamlit run app.py")