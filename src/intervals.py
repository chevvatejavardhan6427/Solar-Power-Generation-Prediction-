"""
Solar Lens – Interval Builder
==============================
Aggregate inverter-level generation data to plant-level 15-minute intervals.

For each (plant_id, timestamp):
  - ``mean_ac_power``: mean AC_POWER across all reporting inverters
  - ``n_reporting``: count of inverters with non-null AC_POWER

Then join with weather sensor data on (plant_id, timestamp).
"""

import pandas as pd


def _aggregate_generation(gen_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each plant_id + timestamp: compute mean AC_POWER across reporting
    inverters and count n_reporting (number of inverters with non-null
    AC_POWER at that timestamp).
    """
    agg = (
        gen_df
        .groupby(["PLANT_ID", "DATE_TIME"])
        .agg(
            mean_ac_power=("AC_POWER", "mean"),
            n_reporting=("AC_POWER", "count"),  # count ignores NaN
        )
        .reset_index()
    )
    return agg


def _prepare_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename weather columns for the merge."""
    wdf = weather_df[["PLANT_ID", "DATE_TIME",
                       "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
                       "IRRADIATION"]].copy()
    wdf = wdf.rename(columns={
        "AMBIENT_TEMPERATURE": "ambient_temperature",
        "MODULE_TEMPERATURE": "module_temperature",
        "IRRADIATION": "irradiation",
    })
    return wdf


def build_intervals(data: dict) -> pd.DataFrame:
    """
    Build a single DataFrame of plant-level 15-minute intervals with weather.

    Parameters
    ----------
    data : dict
        Output of ``load_all_data()``.

    Returns
    -------
    pd.DataFrame with columns:
        plant_id, timestamp, mean_ac_power, n_reporting,
        ambient_temperature, module_temperature, irradiation
    """
    frames = []
    for plant_key in ("plant_1", "plant_2"):
        gen_df = data[plant_key]["gen"]
        weather_df = data[plant_key]["weather"]

        gen_agg = _aggregate_generation(gen_df)
        weather = _prepare_weather(weather_df)

        merged = gen_agg.merge(
            weather,
            on=["PLANT_ID", "DATE_TIME"],
            how="left",  # keep all generation rows; weather may be missing
        )

        merged = merged.rename(columns={
            "PLANT_ID": "plant_id",
            "DATE_TIME": "timestamp",
        })
        frames.append(merged)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["plant_id", "timestamp"]).reset_index(drop=True)
    return result
