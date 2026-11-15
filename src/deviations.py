"""
Solar Lens – Deviation Detection
=================================
Calibrate a robust tolerance band from **validation** residuals and flag
significant deviations on the test set during daylight hours only.

Tolerance band:
    2.5th and 97.5th percentile of validation residuals (actual − predicted).
    This is a distribution-free, robust approach.

Daylight gate:
    irradiation > IRR_THRESHOLD (default 0.01).
    The unit of irradiation is **not documented** in the source data.

Labels:
    "Deviation Candidate – Needs Investigation"
    We NEVER label an event as an equipment fault.
"""

import numpy as np
import pandas as pd

# Default irradiation threshold for daylight gating.
# Only timestamps with irradiation above this value are evaluated for
# deviations.  The unit of irradiation is NOT documented in the source data.
IRR_THRESHOLD = 0.01


def calibrate_band(val_residuals: np.ndarray) -> dict:
    """
    Compute the tolerance band from validation residuals.

    Uses the 2.5th and 97.5th percentile — a robust, distribution-free
    approach that is not sensitive to outliers.

    Parameters
    ----------
    val_residuals : array-like
        (actual − predicted) residuals on the **validation** set.

    Returns
    -------
    dict  ``{"lower": float, "upper": float}``
    """
    lower = float(np.percentile(val_residuals, 2.5))
    upper = float(np.percentile(val_residuals, 97.5))
    return {"lower": lower, "upper": upper}


def detect_deviations(
    test_preds_df: pd.DataFrame,
    val_residuals: np.ndarray,
    irr_threshold: float = IRR_THRESHOLD,
) -> tuple:
    """
    Detect deviations on the test set.

    A deviation is flagged when **both** conditions hold:
      1. ``irradiation > irr_threshold``  (daylight gate)
      2. ``residual < lower_bound``  OR  ``residual > upper_bound``

    Parameters
    ----------
    test_preds_df : pd.DataFrame
        Must include: timestamp, plant_id, actual, predicted, residual,
        irradiation, ambient_temperature, module_temperature, n_reporting.
    val_residuals : np.ndarray
        Validation residuals from the champion model.
    irr_threshold : float
        Minimum irradiation to consider (daylight gating).

    Returns
    -------
    (deviations_df, band_info)
    """
    band = calibrate_band(val_residuals)
    band_info = {**band, "irr_threshold": irr_threshold}

    df = test_preds_df.copy()

    # Daylight gate
    daylight_mask = df["irradiation"] > irr_threshold

    # Deviation gate
    deviation_mask = (
        (df["residual"] < band["lower"]) | (df["residual"] > band["upper"])
    )

    # Combined: must be daylight AND outside tolerance band
    flagged = df[daylight_mask & deviation_mask].copy()
    flagged["lower_bound"] = band["lower"]
    flagged["upper_bound"] = band["upper"]
    flagged["flag"] = "Deviation Candidate \u2013 Needs Investigation"

    # Output columns
    out_cols = [
        "timestamp", "plant_id", "actual", "predicted", "residual",
        "lower_bound", "upper_bound",
        "irradiation", "ambient_temperature", "module_temperature",
        "n_reporting", "flag",
    ]
    out_cols = [c for c in out_cols if c in flagged.columns]
    flagged = flagged[out_cols].reset_index(drop=True)

    return flagged, band_info
