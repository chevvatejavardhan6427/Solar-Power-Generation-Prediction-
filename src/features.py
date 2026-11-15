"""
Solar Lens – Feature Engineering & Chronological Split
======================================================
Adds cyclical hour features and performs a leakage-safe chronological
70 / 15 / 15 split.

Feature list (leakage-safe):
  - irradiation           (unit not documented)
  - ambient_temperature
  - module_temperature
  - hour_sin              sin(2π · hour_frac / 24)
  - hour_cos              cos(2π · hour_frac / 24)

Excluded by design:
  DC_POWER, DAILY_YIELD, TOTAL_YIELD, lagged actual output, day-of-year.
"""

import numpy as np
import pandas as pd

# ── Feature contract ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "irradiation",          # unit not documented
    "ambient_temperature",
    "module_temperature",
    "hour_sin",
    "hour_cos",
]

TARGET_COL = "mean_ac_power"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical hour-of-day features.

    ``hour_frac`` includes fractional minutes, e.g. 14:15 → 14.25.
    """
    df = df.copy()
    hour_frac = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    return df


def chrono_split(
    df: pd.DataFrame,
    plant_id,
) -> tuple:
    """
    Chronological 70 / 15 / 15 split for a single plant.

    No shuffling – the earliest 70 % is training, the next 15 % is
    validation, and the final 15 % is the untouched test set.

    Parameters
    ----------
    df : pd.DataFrame
        Full interval DataFrame (may contain multiple plants).
    plant_id : int or str
        Plant ID to filter on.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    pdf = df[df["plant_id"] == plant_id].sort_values("timestamp").reset_index(drop=True)
    n = len(pdf)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = pdf.iloc[:train_end].copy()
    val_df = pdf.iloc[train_end:val_end].copy()
    test_df = pdf.iloc[val_end:].copy()

    return train_df, val_df, test_df
