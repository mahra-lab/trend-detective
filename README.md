# 🌍 Earth Intelligence AI
### Earth System Trend Detective: Turning NASA Earth Observation Data into Environmental Intelligence

An interactive Streamlit scientific web application designed for students and the scientifically curious public to investigate multi-decadal environmental trends in **Chennai, India** (2002–2025) using NASA satellite and earth system model datasets from the **NASA Giovanni** platform.

---

## 📌 Datasets

The application loads three real NASA Giovanni monthly time series located in the `data/` folder:

| Variable | File | Source Product | Giovanni Parameter ID | Native Unit | Processed Unit | Conversion Formula |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Heat** | `chennai_heat.csv` | NASA GLDAS 2.1 Noah Land Surface Model (0.25°) | `GLDAS_NOAH025_M_2_1_AvgSurfT_inst` | Kelvin (K) | Celsius (°C) | `°C = K - 273.15` |
| **Rain** | `chennai_rain.csv` | NASA GPM IMERG Final Run (0.1° / 0.25°) | `GPM_3IMERGM_07_precipitation` | mm/hr | mm/month | `mm/month = (mm/hr) × 24 × days_in_month` |
| **Soil Dryness** | `chennai_soil.csv` | NASA GLDAS 2.1 Noah Land Surface Model (0.25°) | `GLDAS_NOAH025_M_2_1_RootMoist_inst` | kg m⁻² | kg m⁻² | Unchanged (`kg m⁻²`) |

*Data Integrity Note:* Per NASA Giovanni standards, header metadata preceding the data table is parsed dynamically. Fill values (e.g. `-9999`) are filtered out. No data is ever invented, simulated, or artificially imputed.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.14)
- A modern web browser

### 2. Installation
Install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Launching the App
Run the Streamlit application from the project root:
```bash
streamlit run app.py
```
Or if running directly with a specific Python binary:
```bash
python -m streamlit run app.py
```

Streamlit will print the local URL in your terminal (typically `http://localhost:8501`). Open that URL in your web browser.

---

## 🧭 Application Structure & Screens

The application features sidebar navigation organized into 6 screens:

1. **🚀 Start here (Default Landing Page)**:
   - High-level overview: *“Is Chennai getting hotter, wetter or drier? Let NASA satellite and model data tell you.”*
   - 3-step “How it works” flow: (1) Pick a question, (2) See how it changed, (3) Real or noise?
   - Quick onboarding buttons: `🚀 Start investigating`, `🔥 See a quick example`, and quick demo record buttons.
   - Plain-English definitions for *trend* and *statistical significance*.

2. **🕵️ Detective**:
   - Guided 3-step flow: Choose variable, select years (2002–2025), and compute results.
   - Interactive OpenStreetMap view under collapsible *“📍 Where is this?”* marking Chennai and the ~25 km NASA bounding box (`79.9°E–80.4°E, 12.7°N–13.4°N`).

3. **📈 Result**:
   - **Answer-First Verdict Banner**: Colored plain-English verdict immediately showing whether a trend is significant or indistinguishable from noise.
   - **Number Cards**: Sen's slope per decade, Mann-Kendall p-value (Hamed-Rao test), and 95% bootstrap confidence interval.
   - **Visualizations**: Monthly observed time-series with Sen's trend line, and deseasonalized monthly anomalies.
   - **For Scientists**: Collapsible section with formal test statistics (Z-stat, Kendall's Tau, sample sizes).
   - **Variable Quick-Switcher**: Direct 3-column buttons to pivot between variables.

4. **🔗 Connect the Signals**:
   - Multi-signal comparison plot showing Heat, Rain, and Soil anomalies simultaneously using standardized Z-scores ($\sigma$).
   - Interactive pairwise scatter plots with linear fit lines and one-sentence plain readings.
   - Prominent scientific reminder explaining that **correlation does not prove cause**.

5. **💡 Demo Examples**:
   - Three dedicated cards and buttons to instantly load the full multi-decadal record for Heat, Rain, or Soil Dryness.

6. **🔬 Method and Limits**:
   - In-depth pedagogical guide explaining:
     1. **GLDAS Model Nature**: Physical land-surface model simulation, not an in-situ weather station thermometer.
     2. **Skin vs. Air Temperature**: Radiometric surface skin temperature differs substantially from 2-meter ambient air temperature.
     3. **Spatial Scale**: The ~25 km grid cell averages urban downtowns, rural zones, and coastal waters.
     4. **Linear Trend Assumptions**: Sen's slope and Mann-Kendall test monotonic linear drift; non-linear climate shifts are not modeled.
     5. **Causality Caveat**: A statistically significant trend establishes non-random drift, not its underlying physical cause.
   - Quick scientific glossary for students and the general public.

---

## 🛠️ Credits & Technologies
- **Created by**: Team OUT OF THE SPACE · Earth Intelligence AI · NASA Space Apps
- **Streamlit**: Web application framework
- **Plotly**: Interactive scientific visualizations and OpenStreetMap mapping
- **pymannkendall**: Hamed-Rao modified Mann-Kendall test for serially autocorrelated time series
- **SciPy & NumPy**: Theil-Sen slope estimation, block bootstrapping, and statistical calculations
- **Pandas**: Time-series manipulation and climatology computation
