# 🫀 Wearable Metabolic Twin

> An AI-powered activity recognition and metabolic monitoring dashboard built on the PAMAP2 wearable sensor dataset.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)](https://lightgbm.readthedocs.io)
[![Dataset](https://img.shields.io/badge/Dataset-PAMAP2-orange)](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)

---

## 📌 What is this project?

A **Wearable Metabolic Twin** is a virtual representation of your body's physical state — built from wearable sensor data. This project uses accelerometer, gyroscope, magnetometer, and heart rate data from 9 subjects performing 12 activities to:

- **Predict** what physical activity a person is doing
- **Estimate** their exertion level and heart rate zone
- **Track** cumulative energy expenditure over a session
- **Visualise** all of this in a live interactive dashboard

> **Note:** This is a proof-of-concept. In a real deployment, sensor data would stream live from a wearable device over Bluetooth. Here, pre-recorded PAMAP2 data is used to demonstrate the full pipeline.

---

## 📂 Project Structure

```
wearable_metabolic_twin/
│
├── app.py                  ← Main Streamlit dashboard (8 tabs)
├── requirements.txt        ← All Python dependencies
├── README.md               ← This file
│
├── src/
│   ├── config.py           ← Paths, window settings, activity labels, MET values
│   ├── preprocess.py       ← Reads raw .dat files, cleans, resamples to 10 Hz
│   ├── features.py         ← Sliding window feature extraction (137 features)
│   ├── train.py            ← LightGBM training, GroupKFold CV, LOSO validation
│   └── evaluate.py         ← Confusion matrix, plots, evaluation outputs
│
├── data/
│   ├── raw/
│   │   └── PAMAP2_Dataset/
│   │       └── Protocol/   ← Place downloaded .dat files here
│   └── processed/          ← clean_data.parquet and features.parquet saved here
│
├── artifacts/              ← Trained model files saved here
│   ├── activity_model.pkl
│   ├── met_regressor.pkl
│   ├── scaler.pkl
│   ├── feature_columns.json
│   ├── label_map.json
│   ├── model_meta.json
│   └── exertion_rules.json
│
└── outputs/                ← Saved charts (confusion matrix, EDA plots)
```

---

## 📊 Dataset

**PAMAP2 Physical Activity Monitoring**
- 🔗 [Download from UCI ML Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- 9 subjects, ~3.85 million rows
- 3 IMU sensors: hand, chest, ankle (accelerometer + gyroscope + magnetometer)
- Heart rate monitor (~9 Hz)
- 54 raw columns, sampled at 100 Hz
- 12 activity classes used in this project

| Activity | ID | Activity | ID |
|---|---|---|---|
| Lying | 1 | Ascending Stairs | 12 |
| Sitting | 2 | Descending Stairs | 13 |
| Standing | 3 | Vacuum Cleaning | 16 |
| Walking | 4 | Ironing | 17 |
| Running | 5 | Nordic Walking | 7 |
| Cycling | 6 | Rope Jumping | 24 |

---

## 🧠 Model & Approach

| Component | Choice | Reason |
|---|---|---|
| Classifier | LightGBM multiclass | CPU-safe, fast, handles tabular features well |
| Validation | GroupKFold (4-fold) + LOSO | Subjects never leak across train/test |
| Feature extraction | Sliding window (5s, 50% overlap) | Captures temporal motion patterns |
| Feature count | 137 extracted → top 60 selected | Reduces noise, faster inference |
| Exertion proxy | Karvonen HR-reserve formula | Interpretable, no extra labels needed |
| MET estimation | LightGBM regressor | Predicts metabolic equivalent from sensor features |

### Results (GroupKFold CV)

| Metric | Score |
|---|---|
| Accuracy | 65% |
| Macro F1 | 0.676 |
| Best activity (Rope Jumping) | F1 = 0.857 |
| Best activity (Running) | F1 = 0.797 |
| Hardest (Descending Stairs) | F1 = 0.461 |

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/wearable_metabolic_twin.git
cd wearable_metabolic_twin
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download PAMAP2 from [UCI ML Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) and place the Protocol folder at:

```
data/raw/PAMAP2_Dataset/Protocol/
```

It should contain files like `subject101.dat`, `subject102.dat`, etc.

### 4. Run the full pipeline

```bash
python -m src.preprocess    # Clean raw data → data/processed/clean_data.parquet
python -m src.features      # Extract features → data/processed/features.parquet
python -m src.train         # Train model → artifacts/
python -m src.evaluate      # Evaluate → outputs/
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🖥️ App Features (8 Tabs)

| Tab | What it shows |
|---|---|
| 📋 Overview | Dataset summary, model metrics, sensor placement diagram |
| 📂 Upload / Demo | Try pre-built activity demos OR upload your own CSV |
| ⚡ Live Prediction | Real-time sliders for sensor values → instant prediction |
| 🫀 Metabolic Twin Ring | Circular exertion gauge + activity display |
| 🔋 Energy Storyboard | Cumulative load curve + freshness battery |
| 📡 Sensor Explorer | Signal plots for wrist, chest, ankle |
| 📈 Model Performance | Confusion matrix, F1 scores, feature importance, LOSO |
| ℹ️ About | Limitations, future improvements, run commands |

---

## 📦 Requirements

```
pandas
numpy
scipy
scikit-learn
lightgbm
plotly
streamlit
joblib
pyarrow
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this project to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **Create app**
4. Select your repository and branch
5. Set entrypoint to `app.py`
6. Click **Deploy**

> **Important:** The `data/` folder and `artifacts/` folder must be committed to GitHub, OR you must run the pipeline after deployment. For Streamlit Cloud, commit the pre-trained `artifacts/` folder and `data/processed/features.parquet` so the app works without retraining.

---

## ⚠️ Known Limitations

- Only 9 subjects — may not generalise to all body types
- Sitting vs Standing is still challenging (similar motion patterns)
- Descending Stairs is confused with Vacuum Cleaning (low F1 = 0.46)
- No real-time BLE sensor streaming — sliders simulate sensor input
- MET is estimated via HR-adjusted compendium, not measured VO₂

---

## 🔮 Future Improvements

- 1D-CNN or LSTM on raw signals (expected ~85–90% F1)
- Real-time Bluetooth sensor streaming
- Per-user calibration after a short recording session
- VO₂max estimation from HR + motion trends
- Sleep stage detection from overnight accelerometer data

---

## 👩‍💻 Built By

**Vaishnavi** — First Year Engineering Student  
Project: Wearable Metabolic Twin  
Dataset: PAMAP2 Physical Activity Monitoring (UCI ML Repository)  
Tools: Python · Streamlit · LightGBM · Plotly · Pandas · SciPy

---

## 📄 License

This project is for educational purposes. The PAMAP2 dataset is provided by the UCI Machine Learning Repository.