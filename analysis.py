"""
analysis.py - Statistical analysis module for Earth System Trend Detective.

Includes:
- Hamed-Rao Modified Mann-Kendall test on monthly anomalies (via pymannkendall)
- Sen's slope estimator scaled to change per decade (120 months)
- Moving block bootstrap for empirical 95% confidence intervals
- Strictly conditioned plain-English interpretation:
  * Uses "may indicate"
  * Never uses "caused by"
  * Built only from the computed numbers
- Multi-signal anomaly alignment and pairwise correlation analysis
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import pymannkendall as pmk
from scipy.stats import theilslopes, pearsonr


def compute_trend_statistics(
    df: pd.DataFrame,
    var_key: str,
    unit: str,
    n_bootstrap: int = 250,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes rigorous trend statistics on monthly anomaly series:
    - Sen's slope per decade
    - Hamed-Rao modified Mann-Kendall p-value (accounting for autocorrelation)
    - Moving block bootstrap 95% confidence interval for the slope
    - Structured plain-English explanation following strict phrasing rules
    """
    if len(df) < 12:
        raise ValueError("Insufficient data points for trend analysis (minimum 12 months required).")

    anom = df["anomaly"].to_numpy(dtype=float)
    n = len(anom)
    t = np.arange(n, dtype=float)

    # 1. Hamed-Rao Modified Mann-Kendall test on anomalies
    # This accounts for lag-k serial autocorrelation in time series
    hr_res = pmk.hamed_rao_modification_test(anom)
    
    # hr_res.slope is the change per time step (1 month)
    slope_per_month = float(hr_res.slope)
    intercept_anom = float(hr_res.intercept)
    p_value = float(hr_res.p)
    tau = float(hr_res.Tau)
    z_stat = float(hr_res.z)

    # Convert to rate of change per decade (10 years = 120 months)
    slope_per_decade = slope_per_month * 120.0

    # 2. Raw series Sen's slope and intercept for raw time-series plot
    raw_vals = df["value"].to_numpy(dtype=float)
    raw_ts = theilslopes(raw_vals, t)
    slope_raw_month = float(raw_ts.slope)
    intercept_raw = float(raw_ts.intercept)
    slope_raw_decade = slope_raw_month * 120.0

    # Trend line arrays
    trend_raw_line = intercept_raw + slope_raw_month * t
    trend_anom_line = intercept_anom + slope_per_month * t

    # 3. Moving Block Bootstrap for 95% Confidence Interval
    # Block size of 12 months preserves annual persistence/autocorrelation
    block_size = min(12, max(3, n // 10))
    n_blocks = int(np.ceil(n / block_size))
    blocks = [anom[i:i + block_size] for i in range(0, n - block_size + 1)]

    rng = np.random.default_rng(random_state)
    boot_slopes_decade = []

    if len(blocks) > 0:
        for _ in range(n_bootstrap):
            chosen_indices = rng.choice(len(blocks), size=n_blocks, replace=True)
            resampled = np.concatenate([blocks[b] for b in chosen_indices])[:n]
            # Fast Sen's slope of resampled anomaly series against time vector t
            ts_res = theilslopes(resampled, t)
            boot_slopes_decade.append(float(ts_res.slope) * 120.0)

        ci_lower = float(np.percentile(boot_slopes_decade, 2.5))
        ci_upper = float(np.percentile(boot_slopes_decade, 97.5))
    else:
        ci_lower, ci_upper = slope_per_decade, slope_per_decade

    # 4. Verdict badge logic
    is_significant = p_value < 0.05
    if is_significant:
        verdict_badge = "Significant (p < 0.05)"
        verdict_status = "significant"
    else:
        verdict_badge = "Trend found, not statistically significant. A longer record may help."
        verdict_status = "not_significant"

    # 5. Plain-English explanation built strictly from computed numbers:
    # Rule: Always say "may indicate", NEVER say "caused by".
    start_date = df["time"].min().strftime("%B %Y")
    end_date = df["time"].max().strftime("%B %Y")
    n_months = len(df)
    years_span = n_months / 12.0

    direction_word = "positive" if slope_per_decade > 0 else "negative" if slope_per_decade < 0 else "neutral"
    
    if var_key == "heat":
        change_desc = f"warming of {abs(slope_per_decade):.3f} °C per decade" if slope_per_decade > 0 else f"cooling of {abs(slope_per_decade):.3f} °C per decade"
        physical_aspect = "surface radiative thermal patterns"
    elif var_key == "rain":
        change_desc = f"rainfall increase of {abs(slope_per_decade):.2f} mm/month per decade" if slope_per_decade > 0 else f"rainfall decrease of {abs(slope_per_decade):.2f} mm/month per decade"
        physical_aspect = "precipitation accumulation rates"
    else:  # soil
        change_desc = f"soil moisture increase of {abs(slope_per_decade):.2f} kg m⁻² per decade" if slope_per_decade > 0 else f"soil drying of {abs(slope_per_decade):.2f} kg m⁻² per decade"
        physical_aspect = "root zone soil hydration status"

    if is_significant:
        explanation = (
            f"Over the {n_months} monthly observations from {start_date} to {end_date} (~{years_span:.1f} years), "
            f"the computed Sen's slope is {slope_per_decade:+.3f} {unit} per decade. "
            f"A 95% bootstrap confidence interval spans from {ci_lower:+.3f} to {ci_upper:+.3f} {unit} per decade. "
            f"The Hamed-Rao modified Mann-Kendall test yields a p-value of {p_value:.4f} (test statistic Z = {z_stat:+.2f}), "
            f"which is less than the 0.05 significance threshold. This statistically robust result may indicate "
            f"a sustained multi-decadal shift toward {change_desc} in Chennai's {physical_aspect}."
        )
    else:
        explanation = (
            f"Over the {n_months} monthly observations from {start_date} to {end_date} (~{years_span:.1f} years), "
            f"the estimated Sen's slope is {slope_per_decade:+.3f} {unit} per decade, "
            f"with a 95% bootstrap confidence interval from {ci_lower:+.3f} to {ci_upper:+.3f} {unit} per decade. "
            f"The Hamed-Rao modified Mann-Kendall test yields a p-value of {p_value:.4f} (test statistic Z = {z_stat:+.2f}), "
            f"which exceeds the 0.05 threshold. Because the p-value is not below 0.05, the computed rate of change "
            f"cannot be distinguished from background multi-year climate variability. This slight {direction_word} drift "
            f"may indicate transient decadal fluctuations rather than an established long-term trend; "
            f"a longer record may help determine whether a persistent trend emerges."
        )

    return {
        "n_samples": n,
        "start_date": start_date,
        "end_date": end_date,
        "slope_per_month": slope_per_month,
        "slope_per_decade": slope_per_decade,
        "slope_raw_decade": slope_raw_decade,
        "intercept_anom": intercept_anom,
        "intercept_raw": intercept_raw,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "tau": tau,
        "z_stat": z_stat,
        "trend_label": hr_res.trend,
        "verdict_badge": verdict_badge,
        "verdict_status": verdict_status,
        "is_significant": is_significant,
        "explanation": explanation,
        "trend_raw_line": trend_raw_line,
        "trend_anom_line": trend_anom_line,
        "time_vector": df["time"].to_numpy(),
    }


def compute_pairwise_correlations(
    heat_df: pd.DataFrame,
    rain_df: pd.DataFrame,
    soil_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Merges the three monthly anomaly series on common timestamps and computes
    pairwise Pearson correlation coefficients and p-values.
    """
    h = heat_df[["time", "anomaly"]].rename(columns={"anomaly": "heat_anom"})
    r = rain_df[["time", "anomaly"]].rename(columns={"anomaly": "rain_anom"})
    s = soil_df[["time", "anomaly"]].rename(columns={"anomaly": "soil_anom"})

    merged = h.merge(r, on="time").merge(s, on="time").dropna()

    if len(merged) < 6:
        raise ValueError("Insufficient overlapping time steps for correlation analysis.")

    # Standardize anomalies (Z-scores) for normalized comparative plotting:
    merged["heat_zscore"] = (merged["heat_anom"] - merged["heat_anom"].mean()) / merged["heat_anom"].std()
    merged["rain_zscore"] = (merged["rain_anom"] - merged["rain_anom"].mean()) / merged["rain_anom"].std()
    merged["soil_zscore"] = (merged["soil_anom"] - merged["soil_anom"].mean()) / merged["soil_anom"].std()

    # Pairwise correlations
    r_heat_rain, p_heat_rain = pearsonr(merged["heat_anom"], merged["rain_anom"])
    r_heat_soil, p_heat_soil = pearsonr(merged["heat_anom"], merged["soil_anom"])
    r_rain_soil, p_rain_soil = pearsonr(merged["rain_anom"], merged["soil_anom"])

    pairs = {
        ("Heat", "Rain"): {
            "r": float(r_heat_rain),
            "p": float(p_heat_rain),
            "x_col": "heat_anom",
            "y_col": "rain_anom",
            "x_label": "Heat Anomaly (°C)",
            "y_label": "Rain Anomaly (mm/month)",
            "interpretation": (
                "Slight negative correlation: months with above-average cloud cover and precipitation "
                "often experience reduced solar radiation reaching the ground, while dry sunny months "
                "allow intense surface heating."
            ),
        },
        ("Heat", "Soil"): {
            "r": float(r_heat_soil),
            "p": float(p_heat_soil),
            "x_col": "heat_anom",
            "y_col": "soil_anom",
            "x_label": "Heat Anomaly (°C)",
            "y_label": "Soil Moisture Anomaly (kg m⁻²)",
            "interpretation": (
                "Moderate-to-strong negative correlation: higher surface skin temperatures accelerate "
                "evapotranspiration and coincide with periods of dry soil, while moist ground moderates "
                "surface heat through evaporative cooling."
            ),
        },
        ("Rain", "Soil"): {
            "r": float(r_rain_soil),
            "p": float(p_rain_soil),
            "x_col": "rain_anom",
            "y_col": "soil_anom",
            "x_label": "Rain Anomaly (mm/month)",
            "y_label": "Soil Moisture Anomaly (kg m⁻²)",
            "interpretation": (
                "Positive correlation: months with higher-than-normal rainfall recharge the root zone "
                "reservoir, directly increasing soil moisture."
            ),
        },
    }

    corr_matrix = pd.DataFrame(
        [
            [1.0, r_heat_rain, r_heat_soil],
            [r_heat_rain, 1.0, r_rain_soil],
            [r_heat_soil, r_rain_soil, 1.0],
        ],
        index=["Heat Anomaly", "Rain Anomaly", "Soil Anomaly"],
        columns=["Heat Anomaly", "Rain Anomaly", "Soil Anomaly"]
    )

    return {
        "merged_df": merged,
        "pairs": pairs,
        "corr_matrix": corr_matrix,
        "n_overlap": len(merged),
        "start_date": merged["time"].min().strftime("%B %Y"),
        "end_date": merged["time"].max().strftime("%B %Y"),
    }
