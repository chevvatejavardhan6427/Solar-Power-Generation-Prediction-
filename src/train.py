"""
Solar Lens – Model Training & Evaluation
=========================================
Train Linear Regression, Random Forest Regressor, and SVR per plant.
Champion selection on validation MAE; final test scored exactly once.

All models use the same feature contract and StandardScaler (fit on
train only).
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import FEATURE_COLS, TARGET_COL, chrono_split


def _metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, and R² for a set of predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    plant_id,
    artifacts_dir: str = "artifacts",
) -> dict:
    """
    Train three models for one plant, select champion by validation MAE,
    score champion on test set (once), and persist artifacts.

    Parameters
    ----------
    df : pd.DataFrame
        Full interval DataFrame with features already added.
    plant_id : int
        Which plant to train on.
    artifacts_dir : str
        Where to save model / scaler joblib files.

    Returns
    -------
    dict with keys:
        models, champion_name, champion_model, scaler, metrics,
        feature_cols, val_residuals, test_predictions
    """
    train_df, val_df, test_df = chrono_split(df, plant_id)

    # Drop rows where any feature or the target is missing
    for split in (train_df, val_df, test_df):
        split.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    # ── Scale features (fit on train only) ──────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ── Define models ───────────────────────────────────────────────────
    model_defs = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "SVR": SVR(kernel="rbf"),
    }

    trained_models = {}
    val_metrics = {}

    for name, model in model_defs.items():
        model.fit(X_train_s, y_train)
        preds_val = model.predict(X_val_s)
        trained_models[name] = model
        val_metrics[name] = _metrics(y_val, preds_val)

    # ── Champion = lowest validation MAE ────────────────────────────────
    champion_name = min(val_metrics, key=lambda k: val_metrics[k]["mae"])
    champion_model = trained_models[champion_name]

    # Validation residuals for tolerance-band calibration
    val_preds = champion_model.predict(X_val_s)
    val_residuals = y_val - val_preds  # actual − predicted

    # ── Test-set scoring (exactly once) ─────────────────────────────────
    test_preds = champion_model.predict(X_test_s)
    test_metrics = _metrics(y_test, test_preds)

    # Build test predictions DataFrame
    test_predictions = test_df.copy()
    test_predictions["predicted"] = test_preds
    test_predictions["actual"] = y_test
    test_predictions["residual"] = y_test - test_preds

    # ── Persist artifacts ───────────────────────────────────────────────
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(
        champion_model,
        os.path.join(artifacts_dir, f"plant_{plant_id}_champion.joblib"),
    )
    joblib.dump(
        scaler,
        os.path.join(artifacts_dir, f"plant_{plant_id}_scaler.joblib"),
    )
    # Save test predictions to CSV for auditability
    test_predictions.to_csv(
        os.path.join(artifacts_dir, f"plant_{plant_id}_test_predictions.csv"),
        index=False,
    )

    return {
        "models": trained_models,
        "champion_name": champion_name,
        "champion_model": champion_model,
        "scaler": scaler,
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "feature_cols": FEATURE_COLS,
        "val_residuals": val_residuals,
        "test_predictions": test_predictions,
    }
