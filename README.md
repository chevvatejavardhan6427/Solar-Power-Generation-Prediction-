# ☀️ Solar Lens – Solar Power Generation Prediction

A coverage-aware, time-safe solar-output prediction and deviation-triage MVP built for hackathon demonstration.

## What It Does

Solar Lens predicts **expected mean AC power per reporting inverter** at 15-minute intervals for two solar plants, then surfaces intervals where observed output deviates significantly from the model's expectation — inviting human investigation.

> ⚠️ **Limitation:** Detected deviations are statistical outliers. They require human investigation and do **not** prove equipment faults or failures.

## Quick Start

```bash
# 1. Clone / navigate to the project
cd "Solar Power Generation Prediction"

# 2. Install dependencies & run the full pipeline + dashboard
chmod +x run_demo.sh
./run_demo.sh
```

Or step by step:

```bash
pip install -r requirements.txt
mkdir -p artifacts

# Run pipeline
python3 -c "
import sys; sys.path.insert(0, '.')
from src.ingest import load_all_data
from src.intervals import build_intervals
from src.features import add_features
from src.train import train_and_evaluate
from src.deviations import detect_deviations

data = load_all_data('.')
intervals = build_intervals(data)
intervals = add_features(intervals)
for pid in sorted(intervals['plant_id'].unique()):
    r = train_and_evaluate(intervals, pid)
    d, b = detect_deviations(r['test_predictions'], r['val_residuals'])
    print(f'Plant {pid}: {r[\"champion_name\"]} | Test MAE={r[\"metrics\"][\"test\"][\"mae\"]:.2f} | Deviations={len(d)}')
"

# Launch dashboard
streamlit run app/dashboard.py
```

Then open **http://localhost:8501** in your browser.

## Run Tests

```bash
python3 -m pytest tests/ -v
```

## Project Structure

```
├── src/
│   ├── ingest.py        # Load CSVs, parse mixed timestamp formats
│   ├── intervals.py     # Plant-level 15-min mean AC + n_reporting + weather join
│   ├── features.py      # Cyclical hour features, chronological 70/15/15 split
│   ├── train.py         # Train LR, RF, SVR; champion selection on validation
│   └── deviations.py    # Validation-derived tolerance band + daylight gating
├── app/
│   └── dashboard.py     # Streamlit dashboard
├── tests/
│   └── test_pipeline.py # 31 tests covering all required scenarios
├── artifacts/           # Saved models, scalers, predictions (auto-generated)
├── requirements.txt
├── run_demo.sh          # One-command launcher
└── README.md
```

## Data

Source CSVs (in project root):
- `Plant_1_Generation_Data.csv` – 22 inverters, timestamp format DD-MM-YYYY HH:MM
- `Plant_1_Weather_Sensor_Data.csv` – 1 sensor, YYYY-MM-DD HH:MM:SS
- `Plant_2_Generation_Data.csv` – 22 inverters, YYYY-MM-DD HH:MM:SS
- `Plant_2_Weather_Sensor_Data.csv` – 1 sensor, YYYY-MM-DD HH:MM:SS

Data range: May 15 – June 17, 2020 (34 days, 15-minute intervals).

## Methodology

### Target
**Mean AC_POWER per reporting inverter** (not summed plant power), because missing inverter reports in Plant 2 would otherwise mimic a power drop. `n_reporting` is tracked at every timestamp.

### Features (leakage-safe)
| Feature | Note |
|---------|------|
| `irradiation` | Unit not documented in source data |
| `ambient_temperature` | From weather sensor |
| `module_temperature` | From weather sensor |
| `hour_sin` | sin(2π · hour_frac / 24) |
| `hour_cos` | cos(2π · hour_frac / 24) |

**Excluded:** DC_POWER, DAILY_YIELD, TOTAL_YIELD, lagged actual output, day-of-year.

### Chronological Split (no shuffling)
| Split | Proportion | Purpose |
|-------|-----------|---------|
| Training | 70% (earliest) | Fit models and scaler |
| Validation | 15% (next) | Select champion, calibrate tolerance band |
| Test | 15% (final) | Score champion once, detect deviations |

### Models
- **Linear Regression** – transparent baseline
- **Random Forest Regressor** (100 trees, seed=42) – nonlinear
- **SVR** (RBF kernel) – bounded search

All use StandardScaler fit on training data only. Champion selected by **lowest validation MAE**.

### Deviation Detection
- **Tolerance band:** 2.5th–97.5th percentile of validation residuals (NOT training)
- **Daylight gate:** Only flag when irradiation > 0.01 (unit not documented)
- **Label:** "Deviation Candidate – Needs Investigation" (never "fault")

## Dashboard Features

- 🔧 Plant selector and date-range filter
- 📊 Model comparison table (LR, RF, SVR validation metrics)
- 🏆 Champion KPI cards (Test MAE, RMSE, R²)
- 📈 Actual vs Predicted chart with validation-derived tolerance band
- 📉 Residual chart with tolerance bounds
- 📡 n_reporting / coverage indicator
- 🔍 Deviation table with full context (timestamp, actual, predicted, residual, threshold, irradiation, temperatures, n_reporting)
- 🔬 Per-inverter peer evidence table (optional drilldown)
- ⚠️ Clear limitation statement throughout

## Test Coverage

31 tests covering all 7 required scenarios:

| Scenario | Tests |
|----------|-------|
| Mixed timestamp formats | 5 |
| Missing weather | 3 |
| Incomplete inverter reporting | 4 |
| No time leakage | 6 |
| Validation-based threshold calibration | 3 |
| Daylight gating | 4 |
| Synthetic large residual event | 2 |
| Pipeline integrity | 4 |

## Results Summary

| Plant | Champion | Test MAE | Test RMSE | Test R² | Deviations |
|-------|----------|----------|-----------|---------|------------|
| 4135001 | Random Forest | 11.94 | 26.12 | 0.9951 | 30 |
| 4136001 | Random Forest | 33.45 | 83.56 | 0.8935 | 37 |

## Technology Stack

| Layer | Choice |
|-------|--------|
| Data & ML | Python, pandas, scikit-learn, NumPy, joblib |
| UI | Streamlit, Plotly |
| Quality | pytest (31 tests) |
| Persistence | CSV artifacts, joblib models |
