"""
Solar Lens – Streamlit Dashboard
=================================
Plant-level solar power prediction, deviation detection, and model comparison.

⚠ LIMITATION: Detected deviations are statistical outliers that require
human investigation. They do NOT prove equipment faults or failures.
"""

import sys
import os

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ingest import load_all_data
from src.intervals import build_intervals
from src.features import add_features, chrono_split, FEATURE_COLS, TARGET_COL
from src.train import train_and_evaluate
from src.deviations import detect_deviations

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Lens – Solar Power Prediction",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar Lens – Solar Power Generation Prediction")
st.markdown(
    "**Target:** Mean AC power per reporting inverter at 15-minute intervals.  \n"
    "**Purpose:** Compare observed output with model-expected output and surface "
    "intervals that deserve investigation.  \n"
    "⚠️ *Deviations are investigation candidates, not proof of equipment faults.*"
)


# ── Data Loading (cached) ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and processing data…")
def load_and_process():
    data = load_all_data(PROJECT_ROOT)
    intervals = build_intervals(data)
    intervals = add_features(intervals)
    return intervals


@st.cache_data(show_spinner="Training models… (this takes a few seconds on first run)")
def run_pipeline(_intervals_hash, intervals_bytes, plant_id):
    """Train models and detect deviations for one plant."""
    intervals = pd.read_json(intervals_bytes, orient="records")
    intervals["timestamp"] = pd.to_datetime(intervals["timestamp"])
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    result = train_and_evaluate(intervals, plant_id, artifacts_dir=artifacts_dir)
    deviations_df, band_info = detect_deviations(
        result["test_predictions"],
        result["val_residuals"],
    )
    # Convert non-serializable objects for caching
    return {
        "champion_name": result["champion_name"],
        "metrics": result["metrics"],
        "val_residuals": result["val_residuals"].tolist(),
        "test_predictions": result["test_predictions"].to_json(orient="records", date_format="iso"),
        "feature_cols": result["feature_cols"],
    }, deviations_df.to_json(orient="records", date_format="iso"), band_info


intervals = load_and_process()
plant_ids = sorted(intervals["plant_id"].unique())

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Controls")
    selected_plant = st.selectbox(
        "Select Plant",
        plant_ids,
        format_func=lambda x: f"Plant {x}",
    )

    # Run pipeline for selected plant
    intervals_json = intervals.to_json(orient="records", date_format="iso")
    intervals_hash = hash(intervals_json + str(selected_plant))
    result_cached, dev_json, band_info = run_pipeline(
        intervals_hash, intervals_json, selected_plant
    )

    # Reconstruct DataFrames
    test_df = pd.read_json(result_cached["test_predictions"], orient="records")
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    deviations_df = pd.read_json(dev_json, orient="records")
    if len(deviations_df) > 0:
        deviations_df["timestamp"] = pd.to_datetime(deviations_df["timestamp"])

    # Date range filter (for display, not retraining)
    min_date = test_df["timestamp"].min().date()
    max_date = test_df["timestamp"].max().date()
    date_range = st.date_input(
        "Test Period Date Filter",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.divider()

    # Coverage summary
    st.subheader("📡 Coverage Summary")
    plant_intervals = intervals[intervals["plant_id"] == selected_plant]
    st.metric("Total Intervals", f"{len(plant_intervals):,}")
    st.metric("Avg Inverters Reporting", f"{plant_intervals['n_reporting'].mean():.1f}")
    st.metric("Min Inverters Reporting", f"{plant_intervals['n_reporting'].min()}")
    weather_missing = plant_intervals["irradiation"].isnull().sum()
    st.metric("Missing Weather Rows", f"{weather_missing}")

    st.divider()
    st.warning(
        "**⚠️ Limitation:** Detected deviations are statistical outliers that "
        "require human investigation. They **do not** prove equipment faults."
    )

# ── Date Range Filter ──────────────────────────────────────────────────────
if len(date_range) == 2:
    d_start = pd.Timestamp(date_range[0])
    d_end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    d_start = pd.Timestamp(min_date)
    d_end = pd.Timestamp(max_date) + pd.Timedelta(days=1)

test_display = test_df[
    (test_df["timestamp"] >= d_start) & (test_df["timestamp"] < d_end)
].copy()

dev_display = deviations_df.copy()
if len(dev_display) > 0:
    dev_display = dev_display[
        (dev_display["timestamp"] >= d_start) & (dev_display["timestamp"] < d_end)
    ].copy()

# ══════════════════════════════════════════════════════════════════════════
#  MODEL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════
st.header("📊 Model Comparison (Validation Set)")

val_metrics = result_cached["metrics"]["validation"]
comparison_rows = []
for name, m in val_metrics.items():
    row = {
        "Model": f"🏆 {name} (Champion)" if name == result_cached["champion_name"] else name,
        "Val MAE": f"{m['mae']:.2f}",
        "Val RMSE": f"{m['rmse']:.2f}",
        "Val R²": f"{m['r2']:.4f}",
    }
    comparison_rows.append(row)

st.table(pd.DataFrame(comparison_rows).set_index("Model"))

# ══════════════════════════════════════════════════════════════════════════
#  CHAMPION MODEL KPI CARDS
# ══════════════════════════════════════════════════════════════════════════
st.header(f"🏆 Champion: {result_cached['champion_name']} – Final Test Metrics")
st.caption(
    "The champion was selected by lowest validation MAE. "
    "Test metrics below are from a single, final scoring on the unseen test period."
)

test_m = result_cached["metrics"]["test"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Test MAE", f"{test_m['mae']:.2f}")
col2.metric("Test RMSE", f"{test_m['rmse']:.2f}")
col3.metric("Test R²", f"{test_m['r2']:.4f}")
col4.metric("Deviation Candidates", f"{len(dev_display)}")

# ══════════════════════════════════════════════════════════════════════════
#  ACTUAL vs PREDICTED CHART with TOLERANCE BAND
# ══════════════════════════════════════════════════════════════════════════
st.header("📈 Actual vs Predicted Mean AC Power (Test Period)")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.65, 0.35],
    subplot_titles=(
        "Actual vs Predicted with Validation-Derived Tolerance Band",
        "Residuals (actual − predicted)",
    ),
)

# Actual line
fig.add_trace(
    go.Scatter(
        x=test_display["timestamp"],
        y=test_display["actual"],
        mode="lines",
        name="Actual Mean AC",
        line=dict(color="#1f77b4", width=1.5),
    ),
    row=1, col=1,
)

# Predicted line
fig.add_trace(
    go.Scatter(
        x=test_display["timestamp"],
        y=test_display["predicted"],
        mode="lines",
        name="Predicted Mean AC",
        line=dict(color="#ff7f0e", width=1.5),
    ),
    row=1, col=1,
)

# Tolerance band (predicted ± validation-derived bounds)
upper_band = test_display["predicted"] + band_info["upper"]
lower_band = test_display["predicted"] + band_info["lower"]

fig.add_trace(
    go.Scatter(
        x=test_display["timestamp"], y=upper_band,
        mode="lines", line=dict(width=0), showlegend=False,
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(
        x=test_display["timestamp"], y=lower_band,
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255,127,14,0.15)",
        name="Tolerance Band (val-derived)",
    ),
    row=1, col=1,
)

# Deviation markers
if len(dev_display) > 0:
    fig.add_trace(
        go.Scatter(
            x=dev_display["timestamp"], y=dev_display["actual"],
            mode="markers",
            name="Deviation Candidate",
            marker=dict(color="red", size=7, symbol="x"),
        ),
        row=1, col=1,
    )

# ── Residual subplot ──
fig.add_trace(
    go.Scatter(
        x=test_display["timestamp"], y=test_display["residual"],
        mode="lines", name="Residual",
        line=dict(color="#2ca02c", width=1),
    ),
    row=2, col=1,
)
fig.add_hline(y=band_info["upper"], line_dash="dash", line_color="orange",
              annotation_text=f"Upper ({band_info['upper']:.1f})", row=2, col=1)
fig.add_hline(y=band_info["lower"], line_dash="dash", line_color="orange",
              annotation_text=f"Lower ({band_info['lower']:.1f})", row=2, col=1)
fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

fig.update_layout(height=700, legend=dict(orientation="h", y=-0.07))
fig.update_yaxes(title_text="Mean AC Power (per inv.)", row=1, col=1)
fig.update_yaxes(title_text="Residual", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  n_reporting / Coverage Indicator
# ══════════════════════════════════════════════════════════════════════════
st.header("📡 Inverter Reporting Coverage (Test Period)")

max_inverters = int(test_display["n_reporting"].max()) if len(test_display) > 0 else 22

fig_nr = go.Figure()
fig_nr.add_trace(
    go.Scatter(
        x=test_display["timestamp"],
        y=test_display["n_reporting"],
        mode="lines",
        name="n_reporting",
        line=dict(color="#9467bd"),
        fill="tozeroy",
        fillcolor="rgba(148,103,189,0.2)",
    )
)
fig_nr.update_layout(
    height=250,
    yaxis_title="Inverters Reporting",
    yaxis=dict(range=[0, max_inverters + 2]),
    margin=dict(t=20),
)
st.plotly_chart(fig_nr, use_container_width=True)

if len(test_display) > 0:
    low_coverage = test_display[test_display["n_reporting"] < test_display["n_reporting"].max()]
    if len(low_coverage) > 0:
        st.info(
            f"⚠️ {len(low_coverage)} intervals in the test period have fewer than "
            f"{test_display['n_reporting'].max()} inverters reporting. "
            "Low coverage may explain some apparent deviations."
        )

# ══════════════════════════════════════════════════════════════════════════
#  DEVIATION TABLE
# ══════════════════════════════════════════════════════════════════════════
st.header("🔍 Deviation Candidates (Test Period, Daylight Only)")

st.info(
    f"**Tolerance Band:** [{band_info['lower']:.2f}, {band_info['upper']:.2f}] "
    f"(2.5th–97.5th percentile of validation residuals)  ·  "
    f"**Daylight gate:** irradiation > {band_info['irr_threshold']} (unit not documented)  ·  "
    f"**Deviations found:** {len(dev_display)}"
)

if len(dev_display) > 0:
    display_cols = [
        "timestamp", "actual", "predicted", "residual",
        "lower_bound", "upper_bound",
        "irradiation", "ambient_temperature", "module_temperature",
        "n_reporting", "flag",
    ]
    display_cols = [c for c in display_cols if c in dev_display.columns]
    fmt = {
        "actual": "{:.2f}", "predicted": "{:.2f}", "residual": "{:.2f}",
        "lower_bound": "{:.2f}", "upper_bound": "{:.2f}",
        "irradiation": "{:.4f}",
        "ambient_temperature": "{:.2f}", "module_temperature": "{:.2f}",
    }
    st.dataframe(
        dev_display[display_cols].style.format(
            {k: v for k, v in fmt.items() if k in display_cols}
        ),
        use_container_width=True,
        height=400,
    )
else:
    st.success("No deviation candidates detected in the selected date range.")

# ══════════════════════════════════════════════════════════════════════════
#  OPTIONAL: Per-Inverter Peer Evidence
# ══════════════════════════════════════════════════════════════════════════
st.header("🔬 Per-Inverter Peer Evidence (Optional)")
st.caption(
    "Select a deviation timestamp to inspect inverter-level AC_POWER. "
    "Large differences between inverters may support investigation."
)

if len(dev_display) > 0:
    ts_options = dev_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    selected_ts = st.selectbox(
        "Select a deviation timestamp:",
        ts_options,
    )
    if selected_ts:
        ts = pd.Timestamp(selected_ts)
        # Load raw generation data for that timestamp
        raw_data = load_all_data(PROJECT_ROOT)
        plant_key = "plant_1" if selected_plant == 4135001 else "plant_2"
        gen_raw = raw_data[plant_key]["gen"]
        inv_at_ts = gen_raw[gen_raw["DATE_TIME"] == ts][
            ["SOURCE_KEY", "AC_POWER", "DC_POWER"]
        ].copy()
        if len(inv_at_ts) > 0:
            mean_ac = inv_at_ts["AC_POWER"].mean()
            inv_at_ts["deviation_from_peer_mean"] = inv_at_ts["AC_POWER"] - mean_ac
            inv_at_ts = inv_at_ts.sort_values("deviation_from_peer_mean")
            st.caption(
                f"Inverter-level data at {selected_ts} | "
                f"Plant mean AC: {mean_ac:.2f} | "
                f"Inverters reporting: {len(inv_at_ts)}"
            )
            st.dataframe(
                inv_at_ts.style.format({
                    "AC_POWER": "{:.2f}",
                    "DC_POWER": "{:.2f}",
                    "deviation_from_peer_mean": "{:.2f}",
                }),
                use_container_width=True,
            )
        else:
            st.warning("No inverter-level data found for this timestamp.")
else:
    st.caption("No deviation candidates available. Select a different plant or date range.")

# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**⚠️ Important Limitation:** All deviations shown are *statistical outliers* "
    "relative to the model's validation-derived tolerance band. They require human "
    "investigation and **do not** constitute proof of equipment faults, failures, or "
    "any specific root cause. The irradiation unit is **not documented** in the source "
    "data. Use this dashboard as a screening tool only."
)
st.caption("Solar Lens MVP · Built for hackathon demonstration")
