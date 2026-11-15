"""
data_loader.py - Robust loader for NASA Giovanni CSV time series.

Handles:
- Header metadata parsing from Giovanni CSV format
- Missing/fill value filtering (without inventing or simulating data)
- Strict unit conversions:
    * Heat: Kelvin (K) -> Celsius (°C) by subtracting 273.15
    * Rain: mm/hr -> mm/month by multiplying by 24 and the exact days in each month
    * Soil: kg m-2 (retained as is)
- Monthly climatology and anomaly computation
"""

import os
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np


# Metadata definitions for each variable
VARIABLE_INFO = {
    "heat": {
        "file": "chennai_heat.csv",
        "name": "Surface Skin Temperature",
        "short_name": "Heat",
        "question": "Heat (Surface Skin Temperature)",
        "unit": "°C",
        "raw_unit": "K",
        "source": "NASA GLDAS (Global Land Data Assimilation System) Noah Model 0.25°",
        "giovanni_id": "GLDAS_NOAH025_M_2_1_AvgSurfT_inst",
        "definition": (
            "Surface skin temperature is the radiometric temperature of the land surface "
            "(the top millimeters of vegetation, bare soil, or asphalt). It reflects direct radiative "
            "heating and cooling, which differs from the standard 2-meter ambient air temperature."
        ),
        "color": "#E63946",
    },
    "rain": {
        "file": "chennai_rain.csv",
        "name": "Precipitation",
        "short_name": "Rain",
        "question": "Rain (Monthly Total Rainfall)",
        "unit": "mm/month",
        "raw_unit": "mm/hr",
        "source": "NASA GPM (Global Precipitation Measurement) IMERG Final Run 0.1°/0.25°",
        "giovanni_id": "GPM_3IMERGM_07_precipitation",
        "definition": (
            "Monthly accumulated precipitation estimated from a constellation of satellites "
            "and ground rain gauges. Converted from the Giovanni average rate (mm/hr) to total "
            "monthly accumulation (mm) based on hours in each specific month."
        ),
        "color": "#1D3557",
    },
    "soil": {
        "file": "chennai_soil.csv",
        "name": "Root Zone Soil Moisture",
        "short_name": "Soil dryness",
        "question": "Soil Dryness / Moisture (Root Zone)",
        "unit": "kg m⁻²",
        "raw_unit": "kg m⁻²",
        "source": "NASA GLDAS Noah Land Surface Model 0.25°",
        "giovanni_id": "GLDAS_NOAH025_M_2_1_RootMoist_inst",
        "definition": (
            "Root zone soil moisture represents the total volume of water stored in the upper "
            "~100 cm of soil where plant roots extract moisture. Lower values indicate soil dryness "
            "and potential agricultural drought stress."
        ),
        "color": "#2A9D8F",
    },
}


def load_giovanni_series(
    filepath: str,
    var_key: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robustly reads a NASA Giovanni CSV file, skipping arbitrary metadata header lines,
    cleaning values, and applying unit conversions strictly as specified.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    var_key : str
        One of 'heat', 'rain', or 'soil'.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:
        ['time', 'raw_value', 'value', 'month', 'climatology', 'anomaly']
    metadata : dict
        Header metadata extracted from the file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required Giovanni data file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header_idx = None
    metadata = {}
    fill_value = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Parse metadata lines preceding table data
        if ":" in line and header_idx is None:
            parts = line.split(":", 1)
            k = parts[0].strip().strip('"')
            v = parts[1].strip().strip('"').strip()
            metadata[k] = v
            if "Fill Value" in k:
                try:
                    fill_value = float(v.split(",")[0].strip())
                except ValueError:
                    fill_value = None

        # Check for column header row starting with "time"
        tokens = [t.strip().lower() for t in stripped.split(",")]
        if tokens and tokens[0] == "time":
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not locate the 'time' column header in file: {filepath}")

    # Read CSV data starting from the identified table header
    df = pd.read_csv(filepath, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]

    time_col = [c for c in df.columns if c.lower() == "time"][0]
    df["time"] = pd.to_datetime(df[time_col])

    # Find the data column (the non-time column)
    val_cols = [c for c in df.columns if c.lower() != "time"]
    if not val_cols:
        raise ValueError(f"No numeric data column found in {filepath}")
    raw_col_name = val_cols[0]

    # Convert to numeric, strictly dropping invalid / fill values
    df["raw_value"] = pd.to_numeric(df[raw_col_name], errors="coerce")

    # Filter out missing fill values without inventing, simulating, or filling in data
    if fill_value is not None:
        df = df[df["raw_value"] != fill_value]
    df = df[df["raw_value"] > -9000].dropna(subset=["raw_value"]).copy()

    # Sort chronologically and reset index
    df = df.sort_values("time").reset_index(drop=True)

    # Unit conversions strictly per specification:
    if var_key == "heat":
        # Convert heat from K to °C (subtract 273.15)
        df["value"] = df["raw_value"] - 273.15
        df["unit"] = "°C"
    elif var_key == "rain":
        # Convert rain from mm/hr to mm per month
        # (multiply by 24 and by the number of days in that month)
        days_in_month = df["time"].dt.days_in_month
        df["value"] = df["raw_value"] * 24.0 * days_in_month
        df["unit"] = "mm/month"
    elif var_key == "soil":
        # Keep soil moisture in kg m-2
        df["value"] = df["raw_value"]
        df["unit"] = "kg m⁻²"
    else:
        raise ValueError(f"Unsupported variable key '{var_key}'")

    # Seasonal cycle removal:
    # Monthly anomaly = observed month value minus that calendar month's long-term average
    df["month"] = df["time"].dt.month
    monthly_climatology = df.groupby("month")["value"].transform("mean")
    df["climatology"] = monthly_climatology
    df["anomaly"] = df["value"] - df["climatology"]

    return df, metadata


def load_all_datasets(data_dir: str = "data") -> Dict[str, Dict[str, Any]]:
    """
    Loads all three Chennai Giovanni time series into a clean dictionary.
    """
    results = {}
    for key, info in VARIABLE_INFO.items():
        path = os.path.join(data_dir, info["file"])
        df, meta = load_giovanni_series(path, key)
        results[key] = {
            "df": df,
            "metadata": meta,
            "info": info,
        }
    return results
