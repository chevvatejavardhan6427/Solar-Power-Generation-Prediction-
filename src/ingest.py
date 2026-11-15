"""
Solar Lens – Data Ingestion
===========================
Load generation and weather CSVs for both plants.

Handles mixed timestamp formats safely:
  Plant 1 generation: DD-MM-YYYY HH:MM
  All others:          YYYY-MM-DD HH:MM:SS

The function ``load_all_data`` accepts either a directory containing
the four CSVs (auto-detected) or explicit file paths.
"""

import os
import pandas as pd


def _parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse DATE_TIME column handling both DD-MM-YYYY HH:MM
    and YYYY-MM-DD HH:MM:SS safely, without relying on ``dayfirst``
    heuristics that can produce warnings.
    """
    # Strip whitespace / carriage returns
    cleaned = series.astype(str).str.strip().str.replace("\r", "", regex=False)

    # Try ISO-like format first (YYYY-MM-DD HH:MM:SS) – most files
    try:
        return pd.to_datetime(cleaned, format="%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        pass

    # Try DD-MM-YYYY HH:MM (Plant 1 generation)
    try:
        return pd.to_datetime(cleaned, format="%d-%m-%Y %H:%M")
    except (ValueError, TypeError):
        pass

    # Fallback: let pandas infer with dayfirst=True
    return pd.to_datetime(cleaned, dayfirst=True)


def _load_csv(path: str) -> pd.DataFrame:
    """Load a single CSV, strip whitespace, parse timestamps."""
    df = pd.read_csv(path, encoding="utf-8")
    # Handle \r in column names (Windows line endings)
    df.columns = [c.strip().replace("\r", "") for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.replace("\r", "", regex=False)
    df["DATE_TIME"] = _parse_datetime(df["DATE_TIME"])
    return df


def _find_csv(directory: str, pattern: str) -> str:
    """Find a CSV file matching a pattern in the directory."""
    for f in os.listdir(directory):
        if pattern.lower() in f.lower() and f.endswith(".csv"):
            return os.path.join(directory, f)
    raise FileNotFoundError(f"No CSV matching '{pattern}' in {directory}")


def load_all_data(data_dir: str) -> dict:
    """
    Load all four CSVs.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the CSV files.
        Files are matched by name pattern (Plant_1_Generation, etc.).

    Returns
    -------
    dict with structure::

        {
            "plant_1": {"gen": DataFrame, "weather": DataFrame},
            "plant_2": {"gen": DataFrame, "weather": DataFrame},
        }
    """
    return {
        "plant_1": {
            "gen": _load_csv(_find_csv(data_dir, "Plant_1_Generation")),
            "weather": _load_csv(_find_csv(data_dir, "Plant_1_Weather")),
        },
        "plant_2": {
            "gen": _load_csv(_find_csv(data_dir, "Plant_2_Generation")),
            "weather": _load_csv(_find_csv(data_dir, "Plant_2_Weather")),
        },
    }
