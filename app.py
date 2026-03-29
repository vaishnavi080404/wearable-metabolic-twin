# app.py — Wearable Metabolic Twin
# Run with: streamlit run app.py

import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Wearable Metabolic Twin", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--accent:#00d4aa;--text:#e6edf3;--muted:#8b949e;}
html,body,[data-testid="stApp"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border);}
h1,h2,h3{font-family:'Space Mono',monospace!important;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;text-align:center;transition:border-color 0.2s;}
.metric-card:hover{border-color:var(--accent);}
.metric-value{font-size:2rem;font-weight:700;color:var(--accent);font-family:'Space Mono',monospace;}
.metric-label{font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:4px;}
.tag{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;margin:2px;}
.tag-green{background:#1a3a2a;color:#00d4aa;border:1px solid #00d4aa44;}
.tag-red{background:#3a1a1a;color:#ff6b6b;border:1px solid #ff6b6b44;}
.tag-yellow{background:#3a2f1a;color:#ffd166;border:1px solid #ffd16644;}
.tag-blue{background:#1a2a3a;color:#79c0ff;border:1px solid #79c0ff44;}
.section-header{font-family:'Space Mono',monospace;font-size:0.7rem;letter-spacing:3px;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:8px;margin-bottom:16px;}
.prediction-box{background:linear-gradient(135deg,#1a2a3a 0%,#162a22 100%);border:1px solid var(--accent);border-radius:16px;padding:28px;text-align:center;}
.stTabs [data-baseweb="tab-list"]{background-color:var(--surface)!important;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-size:0.85rem;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}
div[data-testid="stButton"] button{background:var(--accent)!important;color:#0d1117!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
ACTIVITY_LABELS = {
    1:"Lying",2:"Sitting",3:"Standing",4:"Walking",5:"Running",6:"Cycling",
    7:"Nordic Walking",9:"Watching TV",10:"Computer Work",11:"Car Driving",
    12:"Ascending Stairs",13:"Descending Stairs",16:"Vacuum Cleaning",
    17:"Ironing",18:"Folding Laundry",19:"House Cleaning",
    20:"Playing Soccer",24:"Rope Jumping",
}
ACTIVITY_EMOJI = {
    "Lying":"🛌","Sitting":"🪑","Standing":"🧍","Walking":"🚶","Running":"🏃",
    "Cycling":"🚴","Nordic Walking":"🥾","Ascending Stairs":"⬆️",
    "Descending Stairs":"⬇️","Vacuum Cleaning":"🧹","Ironing":"👔",
    "Rope Jumping":"🪢","Watching TV":"📺","Computer Work":"💻",
    "Car Driving":"🚗","Folding Laundry":"👕","House Cleaning":"🏠",
    "Playing Soccer":"⚽",
}
EXERTION_COLORS = {
    "low":"#00d4aa","moderate":"#ffd166","high":"#ff9f43","very_high":"#ff6b6b"
}
MET_VALUES = {
    "Lying":1.0,"Sitting":1.3,"Standing":1.8,"Walking":3.5,"Running":8.3,
    "Cycling":6.8,"Nordic Walking":6.0,"Watching TV":1.2,"Computer Work":1.5,
    "Car Driving":2.0,"Ascending Stairs":8.8,"Descending Stairs":4.0,
    "Vacuum Cleaning":3.3,"Ironing":2.3,"Folding Laundry":2.0,
    "House Cleaning":3.0,"Playing Soccer":7.0,"Rope Jumping":11.0,
}
DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="DM Sans"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)

# ─── EXACT 60 FEATURES THE MODEL USES ───────────────────────────────────────
# These are the exact names from feature_columns.json — do NOT rename them.
MODEL_FEATURES = [
    "hr_mean","hr_max","chest_acc_sma","chest_mag_sma","hr_min",
    "chest_acc_mean","chest_acc_median","ankle_mag_median","chest_mag_median",
    "ankle_mag_sma","hand_acc_median","ankle_gyro_min","ankle_mag_min",
    "ankle_mag_max","hand_mag_max","ankle_acc_min","hand_acc_sma",
    "hand_mag_median","ankle_acc_median","chest_mag_min","chest_mag_max",
    "hand_acc_min","ankle_acc_sma","ankle_gyro_median","ankle_gyro_xy_corr",
    "hand_acc_mean","hand_mag_sma","ankle_acc_spectral_entropy",
    "hand_ankle_ratio","hand_ankle_corr","hand_acc_xy_corr","chest_acc_min",
    "chest_acc_spectral_entropy","hand_gyro_min","chest_acc_energy",
    "chest_gyro_min","motion_intensity","chest_acc_xy_corr","hand_mag_min",
    "chest_gyro_xy_corr","chest_mag_jerk_std","ankle_acc_xy_corr",
    "chest_mag_xy_corr","hr_trend","chest_acc_max","hand_gyro_xy_corr",
    "ankle_mag_mean","hand_gyro_median","chest_gyro_median",
    "hand_acc_spectral_entropy","chest_acc_range","ankle_gyro_jerk_mean",
    "hand_gyro_iqr","hand_acc_iqr","chest_acc_jerk_std","ankle_mag_xy_corr",
    "hr_delta","chest_acc_iqr","ankle_acc_max","ankle_mag_jerk_std",
]


def build_feature_row(
    # ── Heart rate ──────────────────────────────────────────────────────────
    hr_mean=80.0, hr_std=5.0, hr_delta=2.0,
    # ── Hand accelerometer ──────────────────────────────────────────────────
    hand_acc_mean=0.5, hand_acc_std=0.2, hand_acc_energy=1.5,
    hand_acc_sma=0.6, hand_acc_jerk_mean=0.1, hand_acc_jerk_std=0.05,
    # ── Chest accelerometer ─────────────────────────────────────────────────
    chest_acc_mean=0.4, chest_acc_std=0.15, chest_acc_energy=1.2,
    # ── Ankle accelerometer ─────────────────────────────────────────────────
    ankle_acc_mean=0.6, ankle_acc_std=0.25, ankle_acc_energy=1.8,
    # ── Gyroscopes ──────────────────────────────────────────────────────────
    hand_gyro_mean=0.1, hand_gyro_std=0.05,
    chest_gyro_mean=0.1, chest_gyro_std=0.05,
    ankle_gyro_mean=0.15, ankle_gyro_std=0.06,
    # ── Magnetometers ───────────────────────────────────────────────────────
    hand_mag_mean=30.0, chest_mag_mean=30.0, ankle_mag_mean=30.0,
):
    """
    Build the exact 60-feature dict the model expects.
    Every key must match MODEL_FEATURES exactly.
    """
    # ── Helpers ─────────────────────────────────────────────────────────────
    def stats(mean, std):
        """Return min, max, median, iqr, range, sma from mean+std."""
        return {
            "min":    max(0.0, mean - 2 * std),
            "max":    mean + 2 * std,
            "median": mean * 0.97,
            "iqr":    std * 1.35,
            "range":  4 * std,
            "sma":    mean * 1.5,
        }

    def spectral_entropy(jerk_mean):
        """Approximate: high jerk → high entropy (random), low jerk → low (periodic)."""
        return float(np.clip(jerk_mean / 1.5, 0.0, 1.0))

    def xy_corr(mean, std):
        """Approximate cross-axis correlation: high motion → lower correlation."""
        return float(np.clip(0.8 - mean * 0.15, -1.0, 1.0))

    # ── Build row with ALL 60 exact feature names ────────────────────────────
    ha = stats(hand_acc_mean, hand_acc_std)
    ca = stats(chest_acc_mean, chest_acc_std)
    aa = stats(ankle_acc_mean, ankle_acc_std)
    hg = stats(hand_gyro_mean, hand_gyro_std)
    cg = stats(chest_gyro_mean, chest_gyro_std)
    ag = stats(ankle_gyro_mean, ankle_gyro_std)
    hm = stats(hand_mag_mean,  hand_mag_mean * 0.05)
    cm_ = stats(chest_mag_mean, chest_mag_mean * 0.05)
    am = stats(ankle_mag_mean, ankle_mag_mean * 0.05)

    ankle_m = max(ankle_acc_mean, 1e-6)

    row = {
        # ── Heart rate (7 direct) ─────────────────────────────────────────
        "hr_mean":   hr_mean,
        "hr_max":    hr_mean + 2 * hr_std,
        "hr_min":    max(40.0, hr_mean - 2 * hr_std),
        "hr_delta":  hr_delta,
        "hr_trend":  hr_delta * 0.5,

        # ── Hand accelerometer ────────────────────────────────────────────
        "hand_acc_mean":             hand_acc_mean,
        "hand_acc_min":              ha["min"],
        "hand_acc_median":           ha["median"],
        "hand_acc_sma":              hand_acc_sma,
        "hand_acc_iqr":              ha["iqr"],
        "hand_acc_xy_corr":          xy_corr(hand_acc_mean, hand_acc_std),
        "hand_acc_spectral_entropy": spectral_entropy(hand_acc_jerk_mean),

        # ── Chest accelerometer ───────────────────────────────────────────
        "chest_acc_mean":             chest_acc_mean,
        "chest_acc_min":              ca["min"],
        "chest_acc_max":              ca["max"],
        "chest_acc_median":           ca["median"],
        "chest_acc_sma":              chest_acc_mean * 1.5,
        "chest_acc_range":            ca["range"],
        "chest_acc_iqr":              ca["iqr"],
        "chest_acc_energy":           chest_acc_energy,
        "chest_acc_jerk_std":         hand_acc_jerk_std * 0.8,
        "chest_acc_xy_corr":          xy_corr(chest_acc_mean, chest_acc_std),
        "chest_acc_spectral_entropy": spectral_entropy(hand_acc_jerk_mean * 0.8),

        # ── Ankle accelerometer ───────────────────────────────────────────
        "ankle_acc_min":              aa["min"],
        "ankle_acc_max":              aa["max"],
        "ankle_acc_median":           aa["median"],
        "ankle_acc_sma":              ankle_acc_mean * 1.6,
        "ankle_acc_xy_corr":          xy_corr(ankle_acc_mean, ankle_acc_std),
        "ankle_acc_spectral_entropy": spectral_entropy(hand_acc_jerk_mean * 1.2),

        # ── Hand gyroscope ────────────────────────────────────────────────
        "hand_gyro_min":    hg["min"],
        "hand_gyro_median": hg["median"],
        "hand_gyro_iqr":    hg["iqr"],
        "hand_gyro_xy_corr": xy_corr(hand_gyro_mean, hand_gyro_std),

        # ── Chest gyroscope ───────────────────────────────────────────────
        "chest_gyro_min":    cg["min"],
        "chest_gyro_median": cg["median"],
        "chest_gyro_xy_corr": xy_corr(chest_gyro_mean, chest_gyro_std),

        # ── Ankle gyroscope ───────────────────────────────────────────────
        "ankle_gyro_min":       ag["min"],
        "ankle_gyro_median":    ag["median"],
        "ankle_gyro_xy_corr":   xy_corr(ankle_gyro_mean, ankle_gyro_std),
        "ankle_gyro_jerk_mean": ankle_gyro_mean * 0.35,

        # ── Hand magnetometer ─────────────────────────────────────────────
        "hand_mag_min":    hm["min"],
        "hand_mag_max":    hm["max"],
        "hand_mag_median": hm["median"],
        "hand_mag_sma":    hm["sma"],

        # ── Chest magnetometer ────────────────────────────────────────────
        "chest_mag_min":      cm_["min"],
        "chest_mag_max":      cm_["max"],
        "chest_mag_median":   cm_["median"],
        "chest_mag_sma":      cm_["sma"],
        "chest_mag_jerk_std": chest_mag_mean * 0.02,
        "chest_mag_xy_corr":  0.6,

        # ── Ankle magnetometer ────────────────────────────────────────────
        "ankle_mag_min":      am["min"],
        "ankle_mag_max":      am["max"],
        "ankle_mag_median":   am["median"],
        "ankle_mag_sma":      am["sma"],
        "ankle_mag_mean":     ankle_mag_mean,
        "ankle_mag_jerk_std": ankle_mag_mean * 0.02,
        "ankle_mag_xy_corr":  0.6,

        # ── Cross-body features ───────────────────────────────────────────
        "hand_ankle_ratio":  hand_acc_mean / ankle_m,
        "hand_ankle_corr":   float(np.clip(0.6 - abs(hand_acc_mean - ankle_acc_mean) * 0.3, -1.0, 1.0)),
        "motion_intensity":  (hand_acc_mean + chest_acc_mean + ankle_acc_mean) / 3.0,
    }

    # Verify all 60 model features are present
    for feat in MODEL_FEATURES:
        if feat not in row:
            row[feat] = 0.0  # safety fallback

    return row


# ─── LOAD ARTIFACTS ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("artifacts/activity_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("artifacts/feature_columns.json") as f:
            features = json.load(f)
        with open("artifacts/label_map.json") as f:
            raw = json.load(f)
            label_map   = {int(k): int(v) for k, v in raw.items()}
            reverse_map = {v: k for k, v in label_map.items()}
        return model, features, label_map, reverse_map
    except Exception:
        return None, None, None, None

@st.cache_resource
def load_regressor():
    try:
        with open("artifacts/met_regressor.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_resource
def load_scaler():
    try:
        with open("artifacts/scaler.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data
def load_meta():
    try:
        with open("artifacts/model_meta.json") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data
def load_features_data():
    try:
        return pd.read_parquet("data/processed/features.parquet")
    except Exception:
        return None

model, feature_cols, label_map, reverse_map = load_model()
regressor = load_regressor()
scaler    = load_scaler()
meta      = load_meta()
df_feat   = load_features_data()


# ─── CORE PREDICTION ────────────────────────────────────────────────────────
def predict_activity(feature_row_dict):
    """
    Align feature_row_dict to the exact 60 model features,
    scale, and predict. Returns (name, confidence, top5, met).
    """
    if model is None or feature_cols is None:
        return "Walking", 0.72, {"Walking":0.72,"Running":0.15,"Cycling":0.08,"Standing":0.05}, 3.5

    # Build DataFrame with exactly the 60 trained features in the right order
    row = pd.DataFrame([feature_row_dict])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0.0
    row = row[feature_cols].fillna(0.0)

    # Scale if scaler available
    if scaler is not None:
        try:
            row_s = pd.DataFrame(scaler.transform(row), columns=feature_cols)
        except Exception:
            row_s = row
    else:
        row_s = row

    proba    = model.predict_proba(row_s)[0]
    pred_idx = int(np.argmax(proba))
    act_id   = reverse_map.get(pred_idx, 4)
    act_name = ACTIVITY_LABELS.get(act_id, "Unknown")

    top5 = {}
    for i, p in enumerate(proba):
        aid  = reverse_map.get(i, i)
        name = ACTIVITY_LABELS.get(aid, str(aid))
        top5[name] = float(p)
    top5 = dict(sorted(top5.items(), key=lambda x: -x[1])[:5])

    pred_met = MET_VALUES.get(act_name, 3.0)
    if regressor is not None:
        try:
            pred_met = float(regressor.predict(row_s)[0])
        except Exception:
            pass

    return act_name, float(np.max(proba)), top5, pred_met


# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────
def get_exertion(hr, motion_intensity, age=25, resting_hr=60):
    """Karvonen-based exertion — age-adjusted."""
    max_hr      = max(160, 220 - age)
    hr_reserve  = max(max_hr - resting_hr, 1)
    hr_ratio    = float(np.clip((hr - resting_hr) / hr_reserve, 0.0, 1.0))
    mot_ratio   = float(np.clip(motion_intensity / 3.0, 0.0, 1.0))
    score       = round((0.65 * hr_ratio + 0.35 * mot_ratio) * 100, 1)
    if score < 25:   band = "low"
    elif score < 50: band = "moderate"
    elif score < 75: band = "high"
    else:            band = "very_high"
    return band, score

def get_hr_zone(hr, age=25, resting_hr=60):
    max_hr     = max(160, 220 - age)
    hr_reserve = max(max_hr - resting_hr, 1)
    intensity  = float(np.clip((hr - resting_hr) / hr_reserve, 0.0, 1.0))
    if intensity < 0.30:   return "Rest",     "#79c0ff"
    if intensity < 0.55:   return "Fat Burn", "#ffd166"
    if intensity < 0.75:   return "Cardio",   "#ff9f43"
    return "Peak", "#ff6b6b"

def calc_calories(activity, weight_kg, duration_min):
    return round(MET_VALUES.get(activity, 3.0) * weight_kg * (duration_min / 60), 1)

def calc_bmr(weight_kg, height_cm, age):
    return round((10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5)

def calc_daily_goal(weight_kg, height_cm, age):
    return round(calc_bmr(weight_kg, height_cm, age) * 1.375)

def make_ring_gauge(value, label, color="#00d4aa", max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"font": {"color": color, "size": 36, "family": "Space Mono"}},
        gauge={
            "axis": {"range": [0, max_val], "tickfont": {"color":"#8b949e","size":10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#161b22", "bordercolor": "#30363d", "borderwidth": 1,
            "steps": [
                {"range": [0, max_val*0.33], "color": "#1a2a22"},
                {"range": [max_val*0.33, max_val*0.66], "color": "#2a3a22"},
                {"range": [max_val*0.66, max_val], "color": "#3a4a22"},
            ],
            "threshold": {"line":{"color":color,"width":3},"thickness":0.8,"value":value},
        },
        title={"text": label, "font": {"color":"#8b949e","size":13}},
    ))
    fig.update_layout(height=220, margin=dict(t=40,b=10,l=20,r=20),
                      paper_bgcolor="#0d1117", font_color="#e6edf3")
    return fig


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Metabolic Twin")
    st.markdown('<p style="color:#8b949e;font-size:0.8rem;">Wearable AI · PAMAP2 Dataset</p>',
                unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-header">User Profile</div>', unsafe_allow_html=True)
    user_name   = st.text_input("Name", value="Vaishnavi")
    user_weight = st.slider("Weight (kg)", 40, 120, 60)
    user_age    = st.slider("Age", 15, 80, 25)
    user_height = st.slider("Height (cm)", 140, 200, 162)
    resting_hr  = st.slider("Resting HR (bpm)", 45, 90, 60)

    st.divider()
    st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)
    if model is not None:
        st.markdown('<span class="tag tag-green">✓ Classifier</span>', unsafe_allow_html=True)
        if regressor:
            st.markdown('<span class="tag tag-green">✓ MET Regressor</span>', unsafe_allow_html=True)
        if scaler:
            st.markdown('<span class="tag tag-green">✓ Scaler</span>', unsafe_allow_html=True)
        if feature_cols:
            st.markdown(f'<span class="tag tag-blue">{len(feature_cols)} Features</span>',
                        unsafe_allow_html=True)
        if meta:
            st.markdown(f'<span class="tag tag-yellow">F1: {meta.get("macro_f1","?")}</span>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<span class="tag tag-red">✗ No Model — run src.train</span>',
                    unsafe_allow_html=True)

    st.divider()
    bmr        = calc_bmr(user_weight, user_height, user_age)
    daily_goal = calc_daily_goal(user_weight, user_height, user_age)
    bmi        = round(user_weight / ((user_height / 100) ** 2), 1)
    bmi_cat    = ("Underweight" if bmi < 18.5 else "Normal" if bmi < 25
                  else "Overweight" if bmi < 30 else "Obese")
    bmi_color  = ("#79c0ff" if bmi < 18.5 else "#00d4aa" if bmi < 25
                  else "#ffd166" if bmi < 30 else "#ff6b6b")

    for val, color, label in [
        (bmi,        bmi_color,  f"BMI · {bmi_cat}"),
        (bmr,        "#ffd166",  "BMR (kcal/day)"),
        (daily_goal, "#ff9f43",  "Daily Goal (kcal)"),
    ]:
        st.markdown(
            f'<div class="metric-card" style="margin-top:10px">'
            f'<div class="metric-value" style="color:{color}">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


# ─── TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview", "📂 Upload/Demo", "⚡ Live Prediction",
    "💍 Metabolic Ring", "🔋 Energy Story",
    "📡 Sensor Explorer", "📊 Model Results", "ℹ️ About",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("# 🫀 Wearable Metabolic Twin")
    st.markdown(
        f'<p style="color:#8b949e;">Welcome back, <b style="color:#00d4aa">{user_name}</b>'
        f' · AI-powered activity & metabolic monitor</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    for col, (val, color, label) in zip(
        st.columns(4),
        [(daily_goal, "#ffd166", "Daily Goal (kcal)"),
         (bmr,        "#00d4aa", "BMR (kcal/day)"),
         (bmi,        bmi_color, f"BMI · {bmi_cat}"),
         (f"{user_age}y · {user_weight}kg", "#79c0ff", "Profile")],
    ):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value" style="color:{color}">'
                f'{val}</div><div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### How It Works")
    for col, (icon, title, desc) in zip(
        st.columns(5),
        [("📥","Collect","3 IMU sensors + HR at 10 Hz"),
         ("🔧","Preprocess","Clean, resample, fill gaps"),
         ("⚙️","Features","137 features, 9 sensor groups"),
         ("🧠","Predict","LightGBM + MET regressor"),
         ("📊","Insights","Calories, zones, freshness")],
    ):
        with col:
            st.markdown(
                f'<div class="metric-card" style="min-height:130px">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<div style="font-weight:700;color:#e6edf3;margin:8px 0 4px">{title}</div>'
                f'<div style="font-size:0.78rem;color:#8b949e">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Dataset Overview")
    col_a, col_b = st.columns(2)
    with col_a:
        activities = list(MET_VALUES.keys())
        mets       = list(MET_VALUES.values())
        fig = go.Figure(go.Bar(
            x=activities, y=mets,
            marker_color=[f"hsl({int(m*18)},70%,55%)" for m in mets],
            text=[f"{m}x" for m in mets], textposition="outside",
            textfont=dict(color="#e6edf3", size=10),
        ))
        fig.update_layout(title="MET Values by Activity", xaxis_tickangle=-40,
                          height=380, **DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if df_feat is not None and "activityID" in df_feat.columns:
            counts = df_feat["activityID"].value_counts().reset_index()
            counts.columns = ["activityID", "count"]
            counts["name"] = counts["activityID"].map(ACTIVITY_LABELS)
            fig2 = px.pie(counts, values="count", names="name",
                          color_discrete_sequence=px.colors.sequential.Teal, hole=0.45)
            fig2.update_layout(title="Training Window Distribution", height=380,
                               paper_bgcolor="#0d1117", font_color="#e6edf3")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run the pipeline to see training distribution.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 · UPLOAD / DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📂 Upload Data or Try a Demo")
    mode = st.radio("Mode", ["🎭 Demo Examples", "📤 Upload CSV"], horizontal=True)

    # Demo profiles — values match real PAMAP2 sensor magnitudes per activity
    # ── DEMO PROFILES: activity IDs mapped to real PAMAP2 activity codes ────
    # We sample actual median feature windows from features.parquet so the
    # model sees exactly what it was trained on — no approximation needed.
    DEMO_PROFILES = {
        "🏃 Intense Running":   5,
        "🛌 Resting / Lying":   1,
        "🚶 Casual Walking":    4,
        "🚴 Cycling":           6,
        "🪢 Rope Jumping":      24,
        "⬆️ Ascending Stairs":  12,
        "🧍 Standing":          3,
        "🪑 Sitting":           2,
        "⬇️ Descending Stairs": 13,
        "🧹 Vacuum Cleaning":   16,
        "👔 Ironing":           17,
        "🥾 Nordic Walking":    7,
    }

    def get_real_demo_row(activity_id, feat_cols, df_feat):
        """
        Sample the median feature window for a given activity from real training data.
        This guarantees the model sees a realistic input — no manual approximation.
        """
        if df_feat is None or feat_cols is None:
            return None
        rows = df_feat[df_feat["activityID"] == activity_id]
        if len(rows) == 0:
            return None
        # Use median of all windows for this activity — most representative single point
        available = [c for c in feat_cols if c in rows.columns]
        median_row = rows[available].median()
        # Fill any missing feature columns with 0
        full_row = {c: float(median_row.get(c, 0.0)) for c in feat_cols}
        return full_row

    if mode == "🎭 Demo Examples":
        selected_demo = st.selectbox("Choose a demo profile", list(DEMO_PROFILES.keys()))
        act_id        = DEMO_PROFILES[selected_demo]
        feature_row   = get_real_demo_row(act_id, feature_cols, df_feat)
        hr_for_demo   = float(df_feat[df_feat["activityID"]==act_id]["hr_mean"].median()) if df_feat is not None and feature_row is not None else 100.0

        if feature_row is None:
            st.warning("Could not load real demo data. Make sure features.parquet exists.")
        else:
            st.markdown("#### Key Sensor Values")
            display_keys = ["hand_acc_mean", "ankle_acc_median", "hand_gyro_median",
                            "ankle_gyro_median", "motion_intensity", "hr_mean"]
            dcols = st.columns(3)
            for i, k in enumerate(display_keys):
                with dcols[i % 3]:
                    val = feature_row.get(k, 0.0)
                    st.metric(k, f"{val:.2f}")

        if st.button("🔮 Predict This Profile"):
            if feature_row is None:
                st.error("No real data available for this activity.")
            else:
                activity, confidence, top5, pred_met = predict_activity(feature_row)
                mi       = feature_row.get("motion_intensity", 1.0)
                exertion, ex_score = get_exertion(hr_for_demo, mi,
                                                  age=user_age, resting_hr=resting_hr)
                zone, z_color = get_hr_zone(hr_for_demo, user_age, resting_hr)
                calories = calc_calories(activity, user_weight, 30)
                ex_color = EXERTION_COLORS.get(exertion, "#00d4aa")

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
            with c1:
                emoji = ACTIVITY_EMOJI.get(activity, "🏃")
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<div style="font-size:3rem">{emoji}</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:#00d4aa;margin:8px 0">{activity}</div>'
                    f'<div style="color:#8b949e">Confidence: <b style="color:#e6edf3">{confidence:.1%}</b></div>'
                    f'</div>', unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div style="font-size:1.8rem;font-weight:700;color:{ex_color}">{exertion.upper()}</div>'
                    f'<div class="metric-label">Exertion · {ex_score}/100</div>'
                    f'<div style="margin-top:8px;color:{z_color};font-weight:700">{zone} Zone</div>'
                    f'<div style="color:#8b949e;font-size:0.8rem">HR: {hr_for_demo:.0f} bpm</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:#ffd166">{calories}</div>'
                    f'<div class="metric-label">Calories (30 min)</div>'
                    f'<div style="margin-top:12px;color:#8b949e;font-size:0.8rem">MET: {pred_met:.2f} · {user_weight}kg</div>'
                    f'</div>', unsafe_allow_html=True,
                )

            st.markdown("#### Top 5 Predictions")
            names = list(top5.keys())
            probs = list(top5.values())
            fig = go.Figure(go.Bar(
                x=probs, y=names, orientation="h",
                marker_color=["#00d4aa" if n == activity else "#30363d" for n in names],
                text=[f"{p:.1%}" for p in probs], textposition="outside",
                textfont=dict(color="#e6edf3"),
            ))
            fig.update_layout(height=250, xaxis_range=[0, 1], **DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    else:
        uploaded = st.file_uploader("Upload CSV (must have the 60 model feature columns)", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.dataframe(df_up.head(10), use_container_width=True)
            if feature_cols:
                missing = [c for c in feature_cols if c not in df_up.columns]
                if missing:
                    st.warning(f"Missing {len(missing)} columns. First 5: {missing[:5]}")
                else:
                    if st.button("🔮 Predict All Rows"):
                        results = []
                        for _, row in df_up.iterrows():
                            act, conf, _, pm = predict_activity(row.to_dict())
                            results.append({"predicted_activity": act,
                                            "confidence": f"{conf:.1%}",
                                            "predicted_MET": round(pm, 2)})
                        df_results = pd.concat([df_up, pd.DataFrame(results)], axis=1)
                        st.dataframe(df_results, use_container_width=True)
                        st.download_button("⬇️ Download Results",
                                           df_results.to_csv(index=False),
                                           "predictions.csv", "text/csv")
            else:
                st.info("Train model first to enable predictions.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 · LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## ⚡ Live Sensor Input & Prediction")
    st.markdown('<p style="color:#8b949e;">Adjust sliders — prediction updates instantly</p>',
                unsafe_allow_html=True)

    col_sliders, col_result = st.columns([1, 1])

    with col_sliders:
        st.markdown("#### ❤️ Heart Rate")
        hr_mean  = st.slider("hr_mean",  50.0, 200.0, 80.0, 1.0)
        hr_std   = st.slider("hr_std",    0.0,  25.0,  5.0, 0.5)
        hr_delta = st.slider("hr_delta",  0.0,  30.0,  2.0, 0.5)

        st.markdown("#### 🤚 Hand Accelerometer")
        st.caption("Lying/Sitting ≈ 9.7 · Walking ≈ 10.9 · Running ≈ 16.8")
        h_mean   = st.slider("hand_acc_mean",       8.0, 20.0, 9.7, 0.1)
        h_std    = st.slider("hand_acc_std",         0.1, 6.0, 0.5, 0.1)
        h_energy = st.slider("hand_acc_energy",      40.0, 250.0, 60.0, 5.0)
        h_sma    = st.slider("hand_acc_sma",         12.0, 30.0, 14.5, 0.5)
        h_jerk   = st.slider("hand_acc_jerk_mean",   0.0, 4.0, 0.05, 0.05)

        st.markdown("#### 🦶 Ankle Accelerometer (most discriminative)")
        st.caption("Lying ≈ 10.0 · Walking ≈ 11.3 · Running ≈ 15.0 · Rope Jumping ≈ 12.3")
        a_mean   = st.slider("ankle_acc_mean",       8.0, 18.0, 10.0, 0.1)
        a_std    = st.slider("ankle_acc_std",         0.1, 5.0, 0.5, 0.1)
        a_energy = st.slider("ankle_acc_energy",      40.0, 200.0, 60.0, 5.0)

        st.markdown("#### 🔄 Gyroscopes (key for Sitting vs Standing)")
        st.caption("Lying ≈ 0.13 · Walking ≈ 2.0 · Running ≈ 2.7 · Rope Jumping ≈ 3.8")
        hg_mean  = st.slider("hand_gyro_mean",   0.0, 5.0, 0.13, 0.05)
        cg_mean  = st.slider("chest_gyro_mean",  0.0, 3.0, 0.05, 0.05)
        ag_mean  = st.slider("ankle_gyro_mean",  0.0, 4.0, 0.04, 0.05)

    # Build the complete feature row from slider values
    live_row = build_feature_row(
        hr_mean=hr_mean, hr_std=hr_std, hr_delta=hr_delta,
        hand_acc_mean=h_mean, hand_acc_std=h_std, hand_acc_energy=h_energy,
        hand_acc_sma=h_sma, hand_acc_jerk_mean=h_jerk, hand_acc_jerk_std=h_jerk * 0.5,
        chest_acc_mean=h_mean * 0.8, chest_acc_std=h_std * 0.8,
        chest_acc_energy=h_energy * 0.7,
        ankle_acc_mean=a_mean, ankle_acc_std=a_std, ankle_acc_energy=a_energy,
        hand_gyro_mean=hg_mean,  hand_gyro_std=hg_mean * 0.4,
        chest_gyro_mean=cg_mean, chest_gyro_std=cg_mean * 0.4,
        ankle_gyro_mean=ag_mean, ankle_gyro_std=ag_mean * 0.4,
        hand_mag_mean=30.0, chest_mag_mean=30.0, ankle_mag_mean=30.0,
    )

    activity, confidence, top5, pred_met = predict_activity(live_row)
    mi          = live_row["motion_intensity"]
    exertion, ex_score = get_exertion(hr_mean, mi, age=user_age, resting_hr=resting_hr)
    zone, z_color      = get_hr_zone(hr_mean, user_age, resting_hr)
    ex_color    = EXERTION_COLORS.get(exertion, "#00d4aa")
    calories_30 = calc_calories(activity, user_weight, 30)

    with col_result:
        emoji = ACTIVITY_EMOJI.get(activity, "🏃")
        st.markdown(
            f'<div class="prediction-box" style="margin-bottom:16px">'
            f'<div style="font-size:4rem;margin-bottom:8px">{emoji}</div>'
            f'<div style="font-size:2rem;font-weight:700;color:#00d4aa">{activity}</div>'
            f'<div style="color:#8b949e;margin-top:6px">Confidence: '
            f'<b style="color:#e6edf3;font-size:1.1rem">{confidence:.1%}</b></div>'
            f'<div style="margin-top:12px">'
            f'<span class="tag tag-green">{exertion.upper()} · {ex_score}/100</span>'
            f'<span style="color:{z_color};font-weight:700;margin-left:8px">{zone}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        for col, (val, color, label) in zip(
            st.columns(4),
            [(calories_30, "#ffd166", "Cal/30min"),
             (f"{hr_mean:.0f}", "#79c0ff", "BPM"),
             (f"{pred_met:.2f}", "#ff9f43", "MET"),
             (f"{confidence:.0%}", "#00d4aa", "Confidence")],
        ):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:{color}">{val}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("#### Prediction Confidence")
        names = list(top5.keys())
        probs = list(top5.values())
        fig = go.Figure(go.Bar(
            x=probs, y=names, orientation="h",
            marker=dict(
                color=["#00d4aa" if n == activity else "#21262d" for n in names],
                line=dict(color="#30363d", width=1),
            ),
            text=[f"{p:.1%}" for p in probs], textposition="outside",
            textfont=dict(color="#e6edf3", size=11),
        ))
        fig.update_layout(height=260, xaxis_range=[0, 1.05],
                          margin=dict(l=10, r=50, t=10, b=10), **DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Drift / reliability check
        st.markdown("#### 🔬 Input Reliability Check")
        if df_feat is not None and feature_cols:
            try:
                present = [c for c in feature_cols if c in df_feat.columns]
                means = df_feat[present].mean().to_dict()
                stds  = df_feat[present].std().to_dict()
                z_scores = [
                    abs(live_row.get(c, 0) - means.get(c, 0)) / max(stds.get(c, 1), 1e-9)
                    for c in present
                ]
                avg_z = sum(z_scores) / max(len(z_scores), 1)
                if avg_z < 1.5:
                    st.success(f"✅ Input looks similar to training data (z={avg_z:.2f}) — reliable.")
                elif avg_z < 2.5:
                    st.info(f"ℹ️ Somewhat different from training subjects (z={avg_z:.2f}).")
                else:
                    st.warning(f"⚠️ Very different from training data (z={avg_z:.2f}) — may be less accurate.")
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 · METABOLIC RING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 💍 Metabolic Ring")
    st.markdown('<p style="color:#8b949e;">Live metabolic indicators based on your current sensor input</p>',
                unsafe_allow_html=True)

    # Reuse slider values from Tab 3
    max_hr_ring  = max(160, 220 - user_age)
    hr_zone_pct  = min(100, int((hr_mean - resting_hr) / max(max_hr_ring - resting_hr, 1) * 100))
    motion_pct   = min(100, int(mi / 3.0 * 100))

    r1, r2, r3 = st.columns(3)
    with r1:
        st.plotly_chart(make_ring_gauge(hr_zone_pct, "Heart Rate Zone %", "#ff6b6b", 100),
                        use_container_width=True)
    with r2:
        st.plotly_chart(make_ring_gauge(ex_score, "Exertion Score", "#ffd166", 100),
                        use_container_width=True)
    with r3:
        st.plotly_chart(make_ring_gauge(motion_pct, "Motion Intensity", "#00d4aa", 100),
                        use_container_width=True)

    st.markdown(
        f'<div style="text-align:center;margin:8px 0 20px">'
        f'<span style="color:{z_color};font-size:1.1rem;font-weight:700;'
        f'background:#161b22;padding:8px 24px;border-radius:24px;border:1px solid {z_color}44">'
        f'🎯 HR Zone: {zone} · Max HR for age {user_age}: {max_hr_ring} bpm'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    st.divider()
    col_act, col_radar = st.columns(2)
    with col_act:
        emoji    = ACTIVITY_EMOJI.get(activity, "🏃")
        st.markdown(
            f'<div class="prediction-box">'
            f'<div style="font-size:3.5rem">{emoji}</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:#00d4aa;margin:10px 0">{activity}</div>'
            f'<div style="display:flex;justify-content:center;gap:20px;margin-top:12px">'
            f'<div><span style="color:#8b949e;font-size:0.8rem">MET</span><br>'
            f'<span style="font-size:1.4rem;font-weight:700;color:#ffd166">{pred_met:.2f}</span></div>'
            f'<div><span style="color:#8b949e;font-size:0.8rem">Cal/hr</span><br>'
            f'<span style="font-size:1.4rem;font-weight:700;color:#ff9f43">'
            f'{calc_calories(activity, user_weight, 60)}</span></div>'
            f'<div><span style="color:#8b949e;font-size:0.8rem">Zone</span><br>'
            f'<span style="font-size:1.1rem;font-weight:700;color:{z_color}">{zone}</span></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with col_radar:
        categories = ["Cardiovascular","Muscular Load","Energy Burn","Recovery Need","Metabolic Stress"]
        met_scores = [
            min(100, hr_zone_pct),
            min(100, int(h_mean / 2.0 * 100)),
            min(100, int(pred_met / 12.0 * 100)),
            max(0, 100 - int(ex_score)),
            min(100, int((hr_mean - 60) / 120 * 100)),
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=met_scores + [met_scores[0]],
            theta=categories + [categories[0]],
            fill="toself", fillcolor="rgba(0,212,170,0.15)",
            line=dict(color="#00d4aa", width=2),
            marker=dict(color="#00d4aa", size=6),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(range=[0,100], tickfont=dict(color="#8b949e",size=9),
                                gridcolor="#21262d"),
                angularaxis=dict(tickfont=dict(color="#e6edf3",size=10),
                                 gridcolor="#21262d"),
            ),
            height=320, showlegend=False,
            paper_bgcolor="#0d1117", font_color="#e6edf3",
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 · ENERGY STORY
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🔋 Energy Story")
    st.markdown('<p style="color:#8b949e;">Build your day — see calories burned & freshness battery</p>',
                unsafe_allow_html=True)

    default_schedule = [
        ("Lying",7),("Sitting",2),("Walking",1),("Sitting",4),
        ("Walking",0.5),("Sitting",1),("Sitting",3),
        ("Running",0.5),("Cycling",0.5),("Sitting",1),("Lying",6),
    ]

    with st.columns(2)[0]:
        n_activities = st.number_input("Number of activity blocks", 3, 12, 6)

    schedule = []
    st.markdown("#### Activity Timeline")
    s_cols = st.columns(3)
    for i in range(int(n_activities)):
        default = default_schedule[i % len(default_schedule)]
        with s_cols[i % 3]:
            act = st.selectbox(
                f"Block {i+1}", list(MET_VALUES.keys()),
                index=list(MET_VALUES.keys()).index(default[0]) if default[0] in MET_VALUES else 0,
                key=f"act_{i}",
            )
            dur = st.number_input(f"Duration (hrs)", 0.1, 8.0,
                                  float(min(default[1], 4.0)), 0.1, key=f"dur_{i}")
            schedule.append((act, dur))

    hours, cals_cum, cal_per_hour = [], [], []
    load_values, freshness_values = [], []
    running_cal = running_load = t = 0
    total_hrs = sum(d for _, d in schedule)

    for act, dur in schedule:
        cal_h = calc_calories(act, user_weight, 60)
        _, ex_sc = get_exertion(hr_mean, MET_VALUES.get(act, 2) / 4,
                                age=user_age, resting_hr=resting_hr)
        steps = max(1, int(dur * 4))
        for _ in range(steps):
            t            += dur / steps
            running_cal  += cal_h * (dur / steps)
            running_load += (ex_sc / 100.0) * ((dur / steps) / 60.0) * 10.0
            hours.append(round(t, 2))
            cals_cum.append(round(running_cal, 1))
            cal_per_hour.append(cal_h)
            load_values.append(round(running_load, 2))
            freshness_values.append(round(max(0, 100 - running_load), 1))

    total_cal = round(running_cal)
    avg_met   = round(sum(MET_VALUES.get(a, 3) * d for a, d in schedule) / max(total_hrs, 1), 1)
    active_hrs = round(sum(d for a, d in schedule if MET_VALUES.get(a, 0) >= 3.0), 1)
    goal_pct   = min(100, int(total_cal / daily_goal * 100))

    st.divider()
    for col, (val, color, label) in zip(
        st.columns(4),
        [(total_cal, "#ffd166", "Total Calories"),
         (avg_met,   "#00d4aa", "Avg MET"),
         (active_hrs,"#ff9f43", "Active Hours"),
         (f"{goal_pct}%","#ff6b6b", f"Daily Goal ({daily_goal} kcal)")],
    ):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color}">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### ⚡ Cumulative Energy")
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=hours, y=cals_cum, fill="tozeroy",
        fillcolor="rgba(0,212,170,0.15)",
        line=dict(color="#00d4aa", width=2.5), mode="lines",
        name="Calories Burned",
    ))
    fig_cal.add_trace(go.Scatter(
        x=hours, y=cal_per_hour,
        line=dict(color="#ffd166", width=1.5, dash="dot"),
        mode="lines", name="Cal/hr Rate", yaxis="y2",
    ))
    fig_cal.update_layout(
        title="Cumulative Energy Expenditure",
        xaxis_title="Time (hours)",
        yaxis=dict(title="Calories Burned", gridcolor="#21262d", linecolor="#30363d"),
        yaxis2=dict(title="Cal/hr Rate", overlaying="y", side="right",
                    gridcolor="#21262d", tickfont=dict(color="#ffd166")),
        height=340,
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="DM Sans"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("### 🔋 Freshness Battery")
    fig_fresh = go.Figure()
    fig_fresh.add_trace(go.Scatter(
        x=hours, y=freshness_values, fill="tozeroy",
        fillcolor="rgba(0,212,170,0.1)",
        line=dict(color="#00d4aa", width=2.5), mode="lines",
        name="Freshness %",
    ))
    fig_fresh.add_trace(go.Scatter(
        x=hours, y=load_values,
        line=dict(color="#ff6b6b", width=2, dash="dot"),
        mode="lines", name="Cumulative Load", yaxis="y2",
    ))
    fig_fresh.add_hline(y=50, line_dash="dash", line_color="#ffd166",
                        annotation_text="50% — consider rest",
                        annotation_font_color="#ffd166")
    fig_fresh.update_layout(
        xaxis_title="Time (hours)",
        yaxis=dict(title="Freshness %", range=[0, 105], gridcolor="#21262d", linecolor="#30363d"),
        yaxis2=dict(title="Cumulative Load", overlaying="y", side="right",
                    gridcolor="#21262d", tickfont=dict(color="#ff6b6b")),
        height=320,
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="DM Sans"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    )
    st.plotly_chart(fig_fresh, use_container_width=True)

    final_freshness = freshness_values[-1] if freshness_values else 100
    ff_color = "#00d4aa" if final_freshness > 60 else "#ffd166" if final_freshness > 30 else "#ff6b6b"
    ff_msg   = ("Well rested 💪" if final_freshness > 60
                else "Moderately fatigued 😐" if final_freshness > 30
                else "High fatigue — recover 😴")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{ff_color}">{final_freshness:.1f}%</div>'
            f'<div class="metric-label">End-of-Day Freshness</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card" style="text-align:left;padding:20px">'
            f'<span style="font-size:1.1rem">{ff_msg}</span><br>'
            f'<span style="color:#8b949e;font-size:0.85rem">'
            f'Based on Karvonen exertion formula · Age {user_age} · Resting HR {resting_hr} bpm'
            f'</span></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 · SENSOR EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 📡 Sensor Explorer")
    st.markdown('<p style="color:#8b949e;">Explore sensor patterns from the PAMAP2 training data</p>',
                unsafe_allow_html=True)

    if df_feat is not None and "activityID" in df_feat.columns:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            all_ids   = sorted(df_feat["activityID"].unique())
            act_names = [ACTIVITY_LABELS.get(a, str(a)) for a in all_ids]
            sel_acts  = st.multiselect("Filter by Activity", act_names, default=act_names[:4])
        with col_f2:
            num_cols = [c for c in df_feat.select_dtypes(include=[np.number]).columns
                        if c not in {"activityID", "reference_met"}]
            sel_feature = st.selectbox("Feature to explore", num_cols[:30])

        sel_ids     = [k for k, v in ACTIVITY_LABELS.items() if v in sel_acts]
        df_filtered = df_feat[df_feat["activityID"].isin(sel_ids)]

        if len(df_filtered) > 0:
            fig_dist = go.Figure()
            for act_id in sel_ids:
                act_name = ACTIVITY_LABELS.get(act_id, str(act_id))
                subset   = df_filtered[df_filtered["activityID"] == act_id][sel_feature].dropna()
                if len(subset) > 0:
                    fig_dist.add_trace(go.Violin(
                        y=subset, name=act_name,
                        box_visible=True, meanline_visible=True, opacity=0.8,
                    ))
            fig_dist.update_layout(
                title=f"Distribution of {sel_feature} by Activity",
                height=420, **DARK_LAYOUT, violingap=0.05,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            if "motion_intensity" in df_filtered.columns and "hr_mean" in df_filtered.columns:
                st.markdown("#### Motion Intensity vs Heart Rate")
                sample = df_filtered.sample(min(1500, len(df_filtered)), random_state=42).copy()
                sample["activity_name"] = sample["activityID"].map(ACTIVITY_LABELS)
                fig_sc = px.scatter(
                    sample, x="motion_intensity", y="hr_mean", color="activity_name",
                    opacity=0.65,
                    labels={"motion_intensity":"Motion Intensity","hr_mean":"Heart Rate (bpm)"},
                    color_discrete_sequence=px.colors.qualitative.Dark24,
                )
                fig_sc.update_layout(height=400, paper_bgcolor="#0d1117",
                                     plot_bgcolor="#0d1117", font_color="#e6edf3",
                                     legend_title="Activity")
                st.plotly_chart(fig_sc, use_container_width=True)

            st.markdown("#### Feature Correlation")
            corr_cols = [c for c in num_cols if any(k in c for k in ["mean","energy","hr_"])][:20]
            if len(corr_cols) >= 2:
                corr     = df_feat[corr_cols].dropna().corr()
                fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r",
                                     zmin=-1, zmax=1, text_auto=".1f", aspect="auto")
                fig_corr.update_layout(height=400, paper_bgcolor="#0d1117", font_color="#e6edf3")
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No data for selected activities.")
    else:
        st.info("Run the full pipeline to load feature data.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 · MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## 📊 Model Results")
    st.markdown('<p style="color:#8b949e;">All metrics loaded live from model_meta.json</p>',
                unsafe_allow_html=True)

    if not meta:
        st.info("No model trained yet. Run: `python -m src.train`")
    else:
        for col, (val, color, label) in zip(
            st.columns(4),
            [(f"{meta.get('macro_f1','?')}",  "#00d4aa", "Macro F1 (CV)"),
             (f"{meta.get('accuracy','?')}",   "#ffd166", "Accuracy (CV)"),
             (f"{meta.get('loso_summary',{}).get('macro_f1','?')}", "#ff9f43", "LOSO Macro F1"),
             (f"{len(meta.get('feature_columns',[]))}", "#79c0ff", "Features Used")],
        ):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:{color}">'
                    f'{val}</div><div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        if "mae" in meta:
            for col, (val, color, label) in zip(
                st.columns(2),
                [(f"{meta['mae']}", "#ff6b6b", "MET MAE"),
                 (f"{meta['r2']}",  "#ff6b6b", "MET R²")],
            ):
                with col:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value" style="color:{color}">'
                        f'{val}</div><div class="metric-label">{label}</div></div>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        col_left, col_right = st.columns(2)
        with col_left:
            per_f1 = meta.get("per_activity_f1", {})
            if per_f1:
                sorted_f1 = dict(sorted(per_f1.items(), key=lambda x: x[1]))
                fig_f1 = go.Figure(go.Bar(
                    x=list(sorted_f1.values()), y=list(sorted_f1.keys()),
                    orientation="h",
                    marker_color=["#00d4aa" if v >= 0.6 else "#ffd166" if v >= 0.4 else "#ff6b6b"
                                  for v in sorted_f1.values()],
                    text=[f"{v:.2f}" for v in sorted_f1.values()],
                    textposition="outside", textfont=dict(color="#e6edf3"),
                ))
                fig_f1.add_vline(x=0.5, line_dash="dash", line_color="#8b949e",
                                 annotation_text="0.5 threshold",
                                 annotation_font_color="#8b949e")
                fig_f1.update_layout(title="Per-Activity F1 Score",
                                     xaxis_range=[0, 1.1], height=460, **DARK_LAYOUT)
                st.plotly_chart(fig_f1, use_container_width=True)

        with col_right:
            feat_imp_list = meta.get("feature_importances", [])
            if feat_imp_list:
                fi_df = pd.DataFrame(feat_imp_list[:20]).sort_values("importance")
                fig_imp = go.Figure(go.Bar(
                    x=fi_df["importance"], y=fi_df["feature"], orientation="h",
                    marker_color=["#00d4aa" if v >= fi_df["importance"].max() * 0.6
                                  else "#ffd166" if v >= fi_df["importance"].max() * 0.3
                                  else "#30363d" for v in fi_df["importance"]],
                    text=fi_df["importance"].astype(str), textposition="outside",
                    textfont=dict(color="#e6edf3", size=10),
                ))
                fig_imp.update_layout(title="Feature Importance (Top 20)",
                                      height=460, **DARK_LAYOUT)
                st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("### Confusion Matrix")
        if os.path.exists("outputs/confusion_matrix.png"):
            st.image("outputs/confusion_matrix.png", use_container_width=True)
        else:
            st.info("Run `python -m src.evaluate` to generate the confusion matrix image.")

        st.markdown("### LOSO Per-Subject Results")
        loso_rows_data = meta.get("loso_subject_rows", [])
        if loso_rows_data:
            loso_df = pd.DataFrame(loso_rows_data)
            st.dataframe(loso_df, use_container_width=True, hide_index=True)
            loso_sorted = loso_df.sort_values("macro_f1")
            fig_loso = go.Figure(go.Bar(
                x=loso_sorted["subject"], y=loso_sorted["macro_f1"],
                marker_color=["#00d4aa" if v >= 0.5 else "#ff6b6b"
                              for v in loso_sorted["macro_f1"]],
                text=[f"{v:.2f}" for v in loso_sorted["macro_f1"]],
                textposition="outside", textfont=dict(color="#e6edf3"),
            ))
            fig_loso.add_hline(y=0.5, line_dash="dash", line_color="#8b949e")
            fig_loso.update_layout(title="LOSO Macro F1 per Subject",
                                   height=320,
                                   yaxis=dict(range=[0, 1.0], gridcolor="#21262d", linecolor="#30363d"),
                                   paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                                   font=dict(color="#e6edf3", family="DM Sans"),
                                   xaxis=dict(gridcolor="#21262d", linecolor="#30363d"))
            st.plotly_chart(fig_loso, use_container_width=True)

        ref_mets  = meta.get("reference_mets", [])
        pred_mets = meta.get("predicted_mets", [])
        if ref_mets and pred_mets:
            st.markdown("### MET Regressor — Predicted vs Reference")
            fig_met = px.scatter(
                x=ref_mets, y=pred_mets, opacity=0.5,
                labels={"x":"Reference MET","y":"Predicted MET"},
                color_discrete_sequence=["#ffd166"],
            )
            lo = min(min(ref_mets), min(pred_mets)) - 0.2
            hi = max(max(ref_mets), max(pred_mets)) + 0.2
            fig_met.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                         name="Ideal", line=dict(color="#00d4aa", dash="dash")))
            fig_met.update_layout(height=360, paper_bgcolor="#0d1117",
                                  plot_bgcolor="#0d1117", font_color="#e6edf3")
            st.plotly_chart(fig_met, use_container_width=True)

        diff_acts = meta.get("difficult_activities", [])
        if diff_acts:
            st.markdown("### Most Confused Activity Pairs")
            st.dataframe(pd.DataFrame(diff_acts), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 8 · ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown("## ℹ️ About & Limitations")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### About This Project")
        st.markdown(
            '<div class="metric-card" style="text-align:left;padding:24px">'
            '<p><b style="color:#00d4aa">Wearable Metabolic Twin</b> — AI-powered activity '
            'recognition and metabolic monitoring on PAMAP2.</p><br>'
            '<p style="color:#8b949e">IMU sensors (accel, gyro, mag) on hand, chest, ankle '
            'plus heart rate from 9 subjects performing 18 activities.</p><br>'
            '<p><b>Classifier:</b> LightGBM multiclass<br>'
            '<b>Regressor:</b> LightGBM MET estimator<br>'
            '<b>Validation:</b> GroupKFold(4) + LOSO<br>'
            '<b>Features:</b> 137 extracted → top 60 selected<br>'
            '<b>Exertion:</b> Karvonen HR-reserve formula<br>'
            '<b>Window:</b> 5s, 50% overlap at 10 Hz</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Run Pipeline")
        st.code(
            "pip install -r requirements.txt\n"
            "python -m src.preprocess\n"
            "python -m src.features\n"
            "python -m src.train\n"
            "python -m src.evaluate\n"
            "streamlit run app.py",
            language="bash",
        )

    with col_b:
        st.markdown("### Known Limitations")
        for color, title, desc in [
            ("🔴","Only 9 subjects","May not generalise to all body types"),
            ("🟡","Sitting vs Standing","Still challenging despite gyroscope features"),
            ("🟡","No real-time streaming","Sliders simulate sensors — not live BLE"),
            ("🟡","MET is estimated","HR-adjusted compendium, not measured VO2"),
            ("🟢","Gyroscope included","Key for separating similar low-motion activities"),
            ("🟢","LOSO validated","Tested on subjects the model has never seen"),
            ("🟢","Feature selection","Top 60 from 137 reduces noise and overfitting"),
        ]:
            st.markdown(
                f'<div class="metric-card" style="text-align:left;padding:16px;margin-bottom:10px">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
                f'<span style="font-size:1.1rem">{color}</span>'
                f'<b style="color:#e6edf3">{title}</b></div>'
                f'<p style="color:#8b949e;font-size:0.85rem;margin:0">{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Future Improvements")
    improvements = [
        "1D-CNN or LSTM on raw signals (~85–90% F1)",
        "Real-time BLE sensor streaming",
        "Per-user fine-tuning after calibration",
        "VO2max estimation from HR + motion",
        "Sleep stage detection from overnight data",
        "Magnetometer cross-axis features",
    ]
    imp_cols = st.columns(3)
    for i, imp in enumerate(improvements):
        with imp_cols[i % 3]:
            st.markdown(
                f'<div class="metric-card" style="text-align:left;padding:16px">'
                f'<span style="color:#00d4aa">→</span>'
                f'<span style="color:#e6edf3;font-size:0.85rem;margin-left:8px">{imp}</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown(
        '<div style="text-align:center;color:#8b949e;font-size:0.8rem;padding:20px">'
        'Built with ❤️ using <b style="color:#00d4aa">Streamlit</b> · '
        '<b style="color:#ffd166">LightGBM</b> · '
        '<b style="color:#ff6b6b">Plotly</b> · '
        '<b style="color:#79c0ff">PAMAP2 Dataset</b><br><br>'
        '<span style="font-family:\'Space Mono\',monospace">Wearable Metabolic Twin v2.0</span>'
        '</div>',
        unsafe_allow_html=True,
    )