"""
Earth Intelligence AI
----------------------
A friendly, accessible Streamlit web app designed for students and the public
to investigate multi-decadal environmental trends in Chennai, India using NASA Giovanni
satellite and land surface model data (2002–2025).

Sub-branding:
"Earth System Trend Detective: turning NASA Earth observation data into environmental intelligence."

UI Architecture:
- Light theme with strong contrast on all headings, cards, and sidebars (.streamlit/config.toml)
- on_click callbacks for reliable state-based navigation
- 6 screens: Start here, Detective, Result, Connect the signals, Demo examples, Method and limits
- Clean error-free rendering with modern Streamlit width controls and robust date clamping
- Team OUT OF THE SPACE · Earth Intelligence AI · NASA Space Apps footer
"""

import os
import sys
from pathlib import Path
from datetime import datetime, date

# Ensure current script directory is in sys.path for robust imports regardless of CWD
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

DATA_DIR = str(APP_DIR / "data")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load data and analysis modules (NO changes to calculations)
from data_loader import load_all_datasets, VARIABLE_INFO
from analysis import compute_trend_statistics, compute_pairwise_correlations

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Earth Intelligence AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for theme-adaptive typography, cards, and badges
st.markdown("""
<style>
    /* Google Fonts & Base Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Headings follow theme textColor with strong contrast */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: var(--text-color, #0F172A);
        line-height: 1.2;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        font-weight: 500;
        color: var(--text-color, #334155);
        opacity: 0.9;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }
    .brand-subhead {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0E7490;
        margin-bottom: 0.4rem;
        letter-spacing: -0.01em;
    }

    /* Step / Workflow Cards */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 14px;
        padding: 20px 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        height: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .step-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    }
    .step-badge {
        display: inline-block;
        background-color: #E2E8F0;
        color: #0F172A;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 12px;
    }
    .step-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .step-desc {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Concept & Definition Boxes */
    .concept-box {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #0E7490;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.95rem;
        color: #1E293B;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .concept-box-title {
        font-weight: 700;
        color: #0E7490;
        margin-bottom: 4px;
        font-size: 1rem;
    }

    /* Variable Selection Cards */
    .var-choice-card {
        border: 2px solid #CBD5E1;
        border-radius: 16px;
        padding: 22px 18px;
        text-align: center;
        background: #FFFFFF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        margin-bottom: 12px;
    }
    .var-choice-card.active-heat {
        border-color: #EA580C;
        background-color: #FFF7ED;
    }
    .var-choice-card.active-rain {
        border-color: #2563EB;
        background-color: #EFF6FF;
    }
    .var-choice-card.active-soil {
        border-color: #16A34A;
        background-color: #F0FDF4;
    }
    .var-icon {
        font-size: 2.3rem;
        margin-bottom: 8px;
    }
    .var-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .var-def {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.4;
        min-height: 40px;
    }

    /* Result Banner (Answer First) */
    .verdict-banner-amber {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 2px solid #F59E0B;
        border-radius: 16px;
        padding: 24px;
        margin: 15px 0 24px 0;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.12);
    }
    .verdict-banner-green {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #22C55E;
        border-radius: 16px;
        padding: 24px;
        margin: 15px 0 24px 0;
        box-shadow: 0 4px 14px rgba(34, 197, 94, 0.12);
    }
    .verdict-tag {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 999px;
        margin-bottom: 8px;
    }
    .verdict-tag-amber {
        background-color: #FDE68A;
        color: #92400E;
    }
    .verdict-tag-green {
        background-color: #BBF7D0;
        color: #166534;
    }
    .verdict-headline {
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.35;
        margin-bottom: 8px;
    }
    .verdict-text-amber { color: #78350F; }
    .verdict-text-green { color: #14532D; }

    /* Number Cards */
    .metric-card-simple {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 14px;
        padding: 18px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label-simple {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .metric-value-simple {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .metric-expl-simple {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.35;
    }

    /* Caution & Highlight Box */
    .caution-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 20px 0;
        color: #991B1B;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Glossary Card */
    .glossary-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    .glossary-term {
        font-weight: 700;
        color: #0F172A;
        font-size: 1rem;
        margin-bottom: 4px;
    }
    .glossary-def {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 28px 0 14px 0;
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748B;
        border-top: 1px solid #E2E8F0;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# CACHED DATA LOADERS & COMPUTATIONS
# -------------------------------------------------------------
@st.cache_data
def get_cached_datasets():
    return load_all_datasets(DATA_DIR)

@st.cache_data
def get_cached_stats(df_subset, var_key, unit):
    return compute_trend_statistics(df_subset, var_key, unit, n_bootstrap=250)

@st.cache_data
def get_cached_correlations(heat_df, rain_df, soil_df):
    return compute_pairwise_correlations(heat_df, rain_df, soil_df)


# Load datasets
try:
    datasets = get_cached_datasets()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()


# -------------------------------------------------------------
# HELPER FUNCTIONS FOR COMPATIBILITY & CLEAN RENDERING
# -------------------------------------------------------------
def to_pydatetime(dt_val) -> datetime:
    """Safely converts string, date, or Timestamp to pure python datetime.datetime."""
    if hasattr(dt_val, "to_pydatetime"):
        return dt_val.to_pydatetime()
    elif isinstance(dt_val, datetime):
        return datetime(dt_val.year, dt_val.month, dt_val.day, dt_val.hour, dt_val.minute, dt_val.second)
    elif isinstance(dt_val, date):
        return datetime(dt_val.year, dt_val.month, dt_val.day)
    parsed = pd.to_datetime(dt_val)
    if hasattr(parsed, "to_pydatetime"):
        return parsed.to_pydatetime()
    return datetime(parsed.year, parsed.month, parsed.day)

def show_chart(fig):
    """Renders a Plotly figure with full container width across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

def action_button(label, key=None, on_click=None, args=None, type="secondary"):
    """Renders a full-width button across Streamlit versions without warnings."""
    try:
        return st.button(label, key=key, width="stretch", type=type, on_click=on_click, args=args)
    except TypeError:
        return st.button(label, key=key, use_container_width=True, type=type, on_click=on_click, args=args)

def build_chennai_map():
    """Constructs the Chennai map figure with version-adaptive scattermap trace."""
    fig_map = go.Figure()
    box_lons = [79.9, 80.4, 80.4, 79.9, 79.9]
    box_lats = [12.7, 12.7, 13.4, 13.4, 12.7]

    if hasattr(go, "Scattermap"):
        fig_map.add_trace(go.Scattermap(
            lon=box_lons,
            lat=box_lats,
            mode="lines",
            fill="toself",
            fillcolor="rgba(14, 116, 144, 0.18)",
            line=dict(color="#0E7490", width=2.5),
            name="Giovanni 25 km Box"
        ))
        fig_map.add_trace(go.Scattermap(
            lat=[13.0827],
            lon=[80.2707],
            mode="markers+text",
            marker=dict(size=14, color="#DC2626"),
            text=["Chennai (City Center)"],
            textposition="top right",
            name="Chennai"
        ))
        fig_map.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lat=13.05, lon=80.15),
                zoom=8.5
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=340
        )
    else:
        fig_map.add_trace(go.Scattermapbox(
            lon=box_lons,
            lat=box_lats,
            mode="lines",
            fill="toself",
            fillcolor="rgba(14, 116, 144, 0.18)",
            line=dict(color="#0E7490", width=2.5),
            name="Giovanni 25 km Box"
        ))
        fig_map.add_trace(go.Scattermapbox(
            lat=[13.0827],
            lon=[80.2707],
            mode="markers+text",
            marker=dict(size=14, color="#DC2626"),
            text=["Chennai (City Center)"],
            textposition="top right",
            name="Chennai"
        ))
        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=13.05, lon=80.15),
                zoom=8.5
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=340
        )
    return fig_map


# -------------------------------------------------------------
# SESSION STATE & CALLBACK FUNCTIONS
# -------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Start here"
if "selected_var" not in st.session_state:
    st.session_state["selected_var"] = "heat"
if "date_range" not in st.session_state:
    st.session_state["date_range"] = (datetime(2002, 1, 1), datetime(2025, 12, 1))

# Callback functions that update session state BEFORE rerunning
def set_page(page_name: str):
    """Callback to switch page cleanly before rerun."""
    st.session_state["page"] = page_name

def set_var(var_key: str, reset_range: bool = False):
    """Callback to change active variable cleanly and clamp date range."""
    st.session_state["selected_var"] = var_key
    if var_key in datasets:
        v_df = datasets[var_key]["df"]
        v_min = to_pydatetime(v_df["time"].min())
        v_max = to_pydatetime(v_df["time"].max())
        if reset_range:
            st.session_state["date_range"] = (v_min, v_max)
            st.session_state["guided_date_slider"] = (v_min, v_max)
        else:
            curr_start, curr_end = st.session_state.get("date_range", (v_min, v_max))
            curr_start = to_pydatetime(curr_start)
            curr_end = to_pydatetime(curr_end)
            clamped_start = max(v_min, min(curr_start, v_max))
            clamped_end = min(v_max, max(curr_end, v_min))
            if clamped_start >= clamped_end:
                clamped_start, clamped_end = v_min, v_max
            st.session_state["date_range"] = (clamped_start, clamped_end)
            st.session_state["guided_date_slider"] = (clamped_start, clamped_end)

def set_var_and_page(var_key: str, page_name: str, reset_range: bool = False):
    """Callback to change variable and switch page cleanly before rerun."""
    set_var(var_key, reset_range=reset_range)
    st.session_state["page"] = page_name


# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.markdown("### 🌍 Earth Intelligence AI")
st.sidebar.caption("NASA Giovanni satellite & model data for Chennai (2002–2025)")
st.sidebar.markdown("---")

nav_screens = [
    "Start here",
    "Detective",
    "Result",
    "Connect the signals",
    "Demo examples",
    "Method and limits"
]

screen_labels = {
    "Start here": "🚀 Start here",
    "Detective": "🕵️ Detective",
    "Result": "📈 Result",
    "Connect the signals": "🔗 Connect the signals",
    "Demo examples": "💡 Demo examples",
    "Method and limits": "🔬 Method and limits"
}

# Ensure page in session state is valid
if st.session_state["page"] not in nav_screens:
    st.session_state["page"] = "Start here"

# Radio widget directly bound to st.session_state["page"] via key="page"
st.sidebar.radio(
    "Navigation",
    nav_screens,
    key="page",
    format_func=lambda s: screen_labels.get(s, s)
)

st.sidebar.markdown("---")
# Quick Variable Indicator in sidebar
st.sidebar.markdown("**Current Variable:**")
var_icons = {"heat": "🔥 Heat", "rain": "🌧️ Rain", "soil": "🌱 Soil dryness"}
st.sidebar.info(f"{var_icons.get(st.session_state['selected_var'], '🔥 Heat')}")

st.sidebar.markdown("---")
st.sidebar.caption("Team OUT OF THE SPACE · Earth Intelligence AI · NASA Space Apps")


# Read current active page
current_page = st.session_state["page"]


# -------------------------------------------------------------
# SCREEN 1: START HERE (DEFAULT LANDING PAGE)
# -------------------------------------------------------------
if current_page == "Start here":
    st.markdown('<div class="hero-title">Earth Intelligence AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subhead">'
        'Earth System Trend Detective: turning NASA Earth observation data into environmental intelligence.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">Is Chennai getting hotter, wetter or drier? '
        'Let NASA satellite and model data tell you.</div>',
        unsafe_allow_html=True
    )

    # 3-step "How it works" row of cards
    st.markdown("#### How it works")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="step-card">
            <span class="step-badge">1. Pick a question</span>
            <div class="step-title">Choose what to investigate</div>
            <div class="step-desc">Pick between <b>Heat</b> (surface temperature), <b>Rain</b> (monthly rainfall), or <b>Soil dryness</b>.</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="step-card">
            <span class="step-badge">2. See how it changed</span>
            <div class="step-title">Explore 2002 to 2025</div>
            <div class="step-desc">Inspect 24 years of real monthly NASA measurements with repeating seasonal cycles removed.</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="step-card">
            <span class="step-badge">3. Real or noise?</span>
            <div class="step-title">Test the significance</div>
            <div class="step-desc">Find out if the change is real or just random noise using standard scientific statistics.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Action Buttons: on_click callbacks ensure clean navigation before rerun
    btn_col1, btn_col2, _ = st.columns([1.2, 1.2, 1.6])
    with btn_col1:
        action_button(
            "🚀 Start investigating",
            key="btn_start_investigating",
            type="primary",
            on_click=set_page,
            args=("Detective",)
        )
    with btn_col2:
        action_button(
            "🔥 See a quick example",
            key="btn_quick_example",
            on_click=set_var_and_page,
            args=("heat", "Result", True)
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Demo Quick Launch Row
    st.markdown("##### 💡 Or jump directly into a demonstration record:")
    d1, d2, d3 = st.columns(3)
    with d1:
        action_button("🔥 Heat Analysis (GLDAS)", key="home_demo_heat", on_click=set_var_and_page, args=("heat", "Result", True))
    with d2:
        action_button("🌧️ Rain Analysis (GPM)", key="home_demo_rain", on_click=set_var_and_page, args=("rain", "Result", True))
    with d3:
        action_button("🌱 Soil Dryness Analysis (GLDAS)", key="home_demo_soil", on_click=set_var_and_page, args=("soil", "Result", True))

    st.markdown("---")

    # Plain Language Explanations
    c_box1, c_box2 = st.columns(2)
    with c_box1:
        st.markdown("""
        <div class="concept-box">
            <div class="concept-box-title">📈 What is a trend?</div>
            A trend is a steady change in one direction over many years, like temperature slowly rising.
        </div>
        """, unsafe_allow_html=True)

    with c_box2:
        st.markdown("""
        <div class="concept-box">
            <div class="concept-box-title">🎯 What does significant mean?</div>
            Scientists call a change significant when it is very unlikely to be just chance. We test this with a standard statistics method called the Mann-Kendall test.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.caption("Data: NASA GLDAS and GPM IMERG via NASA Giovanni. Region: Chennai (about 25 km box).")


# -------------------------------------------------------------
# SCREEN 2: DETECTIVE (GUIDED 3-STEP FLOW)
# -------------------------------------------------------------
elif current_page == "Detective":
    st.markdown('<div class="hero-title">🕵️ Environmental Detective</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Follow the three steps below to set up your investigation for Chennai.</div>',
        unsafe_allow_html=True
    )

    # Step 1: What do you want to investigate?
    st.markdown("### Step 1: What do you want to investigate?")
    st.caption("Click a card to choose the environmental variable you want to study:")

    v_cols = st.columns(3)
    curr_var = st.session_state["selected_var"]

    with v_cols[0]:
        is_heat = (curr_var == "heat")
        border_class = "active-heat" if is_heat else ""
        btn_label = "✅ Heat Selected" if is_heat else "Select Heat"
        st.markdown(f"""
        <div class="var-choice-card {border_class}">
            <div class="var-icon">🔥</div>
            <div class="var-name">Heat</div>
            <div class="var-def">Surface skin temperature measured by NASA land models and thermal satellites.</div>
        </div>
        """, unsafe_allow_html=True)
        action_button(
            btn_label,
            key="sel_btn_heat",
            type="primary" if is_heat else "secondary",
            on_click=set_var,
            args=("heat",)
        )

    with v_cols[1]:
        is_rain = (curr_var == "rain")
        border_class = "active-rain" if is_rain else ""
        btn_label = "✅ Rain Selected" if is_rain else "Select Rain"
        st.markdown(f"""
        <div class="var-choice-card {border_class}">
            <div class="var-icon">🌧️</div>
            <div class="var-name">Rain</div>
            <div class="var-def">Monthly precipitation estimated from NASA rain satellites and ground gauges.</div>
        </div>
        """, unsafe_allow_html=True)
        action_button(
            btn_label,
            key="sel_btn_rain",
            type="primary" if is_rain else "secondary",
            on_click=set_var,
            args=("rain",)
        )

    with v_cols[2]:
        is_soil = (curr_var == "soil")
        border_class = "active-soil" if is_soil else ""
        btn_label = "✅ Soil Dryness Selected" if is_soil else "Select Soil Dryness"
        st.markdown(f"""
        <div class="var-choice-card {border_class}">
            <div class="var-icon">🌱</div>
            <div class="var-name">Soil dryness</div>
            <div class="var-def">Moisture stored in the upper 100 cm root zone where plants and trees drink.</div>
        </div>
        """, unsafe_allow_html=True)
        action_button(
            btn_label,
            key="sel_btn_soil",
            type="primary" if is_soil else "secondary",
            on_click=set_var,
            args=("soil",)
        )

    st.markdown("---")

    # Step 2: Choose the years
    st.markdown("### Step 2: Choose the years")
    st.caption("Drag the slider handles to investigate a specific time period or keep the full 2002–2025 timeline:")

    full_df = datasets[st.session_state["selected_var"]]["df"]
    min_date = to_pydatetime(full_df["time"].min())
    max_date = to_pydatetime(full_df["time"].max())

    # Pre-clamp stored slider state to avoid Streamlit bounds check exception
    if "guided_date_slider" in st.session_state:
        s_val = st.session_state["guided_date_slider"]
        if isinstance(s_val, (list, tuple)) and len(s_val) == 2:
            s0 = to_pydatetime(s_val[0])
            s1 = to_pydatetime(s_val[1])
            s0_c = max(min_date, min(s0, max_date))
            s1_c = min(max_date, max(s1, min_date))
            if s0_c >= s1_c:
                s0_c, s1_c = min_date, max_date
            if (s0_c, s1_c) != (s0, s1):
                st.session_state["guided_date_slider"] = (s0_c, s1_c)

    # Ensure slider bounds are respected
    current_date_range = st.session_state.get("date_range", (min_date, max_date))
    clamped_start = max(min_date, min(to_pydatetime(current_date_range[0]), max_date))
    clamped_end = min(max_date, max(to_pydatetime(current_date_range[1]), min_date))
    if clamped_start >= clamped_end:
        clamped_start, clamped_end = min_date, max_date

    slider_range = st.slider(
        "Time window to analyze:",
        min_value=min_date,
        max_value=max_date,
        value=(clamped_start, clamped_end),
        format="MMM YYYY",
        key="guided_date_slider"
    )
    st.session_state["date_range"] = (to_pydatetime(slider_range[0]), to_pydatetime(slider_range[1]))

    st.markdown("---")

    # Step 3: Show me the result (on_click callback ensures navigation to Result)
    st.markdown("### Step 3: Show me the result")
    st.caption(f"Ready! Click below to compute the trend for **{VARIABLE_INFO[st.session_state['selected_var']]['name']}** ({slider_range[0].strftime('%b %Y')} to {slider_range[1].strftime('%b %Y')}).")

    action_button(
        "📊 Show me the result 👉",
        key="btn_detective_show_result",
        type="primary",
        on_click=set_page,
        args=("Result",)
    )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Collapsed Map Section: "Where is this?"
    with st.expander("📍 Where is this?"):
        st.markdown(
            "NASA Giovanni calculates area averages across a **~25 km box** centered over Chennai "
            "(bounding box `79.9°E–80.4°E, 12.7°N–13.4°N`)."
        )
        show_chart(build_chennai_map())


# -------------------------------------------------------------
# SCREEN 3: RESULT (ANSWER FIRST)
# -------------------------------------------------------------
elif current_page == "Result":
    var_key = st.session_state["selected_var"]
    info = VARIABLE_INFO[var_key]
    full_df = datasets[var_key]["df"]

    start_dt, end_dt = st.session_state["date_range"]
    start_dt = to_pydatetime(start_dt)
    end_dt = to_pydatetime(end_dt)

    mask = (full_df["time"] >= pd.to_datetime(start_dt)) & (full_df["time"] <= pd.to_datetime(end_dt))
    df_subset = full_df[mask].copy().reset_index(drop=True)

    if len(df_subset) < 12:
        st.warning("⚠️ Please choose a date range with at least 12 months of observations to calculate reliable trend statistics.")
        wb1, wb2 = st.columns(2)
        with wb1:
            action_button("🕵️ Adjust dates in Detective", key="warn_go_detective", type="primary", on_click=set_page, args=("Detective",))
        with wb2:
            action_button("🔄 Reset to full 2002–2025 period", key="warn_reset_dates", on_click=set_var, args=(var_key, True))
        st.stop()

    # Compute statistics (unchanged functions and calculation)
    stats = get_cached_stats(df_subset, var_key, info["unit"])

    st.markdown(f'<div class="hero-title">📈 Result: {info["name"]}</div>', unsafe_allow_html=True)
    st.caption(f"Investigating {stats['start_date']} to {stats['end_date']} ({stats['n_samples']} months) in Chennai")

    # 1. THE ANSWER FIRST: Large Colored Verdict Banner in Plain English
    if stats["is_significant"]:
        st.markdown(f"""
        <div class="verdict-banner-green">
            <span class="verdict-tag verdict-tag-green">Verdict</span>
            <div class="verdict-headline verdict-text-green">
                Significant trend found. {info['short_name'].title()} in Chennai shows a clear change that is very unlikely to be random chance.
            </div>
            <div style="font-size:0.95rem; color:#166534;">
                The p-value is <b>{stats['p_value']:.4f}</b> (below 0.05), which scientists consider statistically significant.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-banner-amber">
            <span class="verdict-tag verdict-tag-amber">Verdict</span>
            <div class="verdict-headline verdict-text-amber">
                No clear trend. {info['short_name'].title()} in Chennai has not changed in a way we can separate from normal year-to-year variation.
            </div>
            <div style="font-size:0.95rem; color:#92400E;">
                The p-value is <b>{stats['p_value']:.4f}</b> (above 0.05). A longer record may help determine if a subtle trend exists.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. THREE SIMPLE NUMBER CARDS WITH ONE-LINE PLAIN MEANING UNDER EACH
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card-simple">
            <div>
                <div class="metric-label-simple">Rate of change per decade</div>
                <div class="metric-value-simple">{stats['slope_per_decade']:+.3f} <span style="font-size:0.95rem; font-weight:600; color:#64748B;">{info['unit']}/decade</span></div>
            </div>
            <div class="metric-expl-simple">Calculated with Sen's slope — a method that ignores freak storms or heatwaves.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        p_val = stats["p_value"]
        p_text = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
        st.markdown(f"""
        <div class="metric-card-simple">
            <div>
                <div class="metric-label-simple">Chance this is just noise (p-value)</div>
                <div class="metric-value-simple">{p_text}</div>
            </div>
            <div class="metric-expl-simple">Below 0.05 means significant. Above 0.05 means it could just be natural noise.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        ci_str = f"[{stats['ci_lower']:+.3f}, {stats['ci_upper']:+.3f}]"
        st.markdown(f"""
        <div class="metric-card-simple">
            <div>
                <div class="metric-label-simple">Likely range of the change (95% confidence)</div>
                <div class="metric-value-simple" style="font-size:1.35rem;">{ci_str}</div>
            </div>
            <div class="metric-expl-simple">We are 95% confident the true multi-year rate of change falls within this window.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 3. CHARTS (EACH WITH A ONE-LINE CAPTION)
    st.subheader("1. Monthly Observations Over Time")
    st.caption("The line shows monthly values; the dashed line is the overall direction.")

    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=df_subset["time"],
        y=df_subset["value"],
        mode="lines+markers",
        name=f"Observed ({info['unit']})",
        line=dict(color=info["color"], width=2),
        marker=dict(size=3.5),
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Value:</b> %{y:.2f} " + info["unit"] + "<extra></extra>"
    ))
    fig_raw.add_trace(go.Scatter(
        x=df_subset["time"],
        y=stats["trend_raw_line"],
        mode="lines",
        name=f"Trend line ({stats['slope_raw_decade']:+.2f} {info['unit']}/decade)",
        line=dict(color="#0F172A", width=2.5, dash="dash"),
        hovertemplate="<b>Trend Fit:</b> %{y:.2f} " + info["unit"] + "<extra></extra>"
    ))
    fig_raw.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(title="Year", gridcolor="#F1F5F9"),
        yaxis=dict(title=f"{info['name']} ({info['unit']})", gridcolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    show_chart(fig_raw)

    st.subheader("2. Seasonal Cycle Removed (Monthly Anomalies)")
    st.caption("This chart removes the normal seasons so the real change is easier to see.")

    fig_anom = go.Figure()
    anom_colors = np.where(df_subset["anomaly"] >= 0, "#EF4444" if var_key == "heat" else "#2563EB", "#3B82F6" if var_key == "heat" else "#F97316")

    fig_anom.add_trace(go.Bar(
        x=df_subset["time"],
        y=df_subset["anomaly"],
        name="Monthly Departure from Normal",
        marker_color=anom_colors,
        opacity=0.75,
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Anomaly:</b> %{y:+.2f} " + info["unit"] + "<extra></extra>"
    ))
    fig_anom.add_trace(go.Scatter(
        x=df_subset["time"],
        y=stats["trend_anom_line"],
        mode="lines",
        name=f"Anomaly Trend ({stats['slope_per_decade']:+.3f} {info['unit']}/decade)",
        line=dict(color="#0F172A", width=3),
        hovertemplate="<b>Anomaly Trend:</b> %{y:+.2f} " + info["unit"] + "<extra></extra>"
    ))
    fig_anom.add_hline(y=0, line_width=1.5, line_color="#94A3B8")

    fig_anom.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(title="Year", gridcolor="#F1F5F9"),
        yaxis=dict(title=f"Anomaly ({info['unit']})", gridcolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    show_chart(fig_anom)

    # 4. COLLAPSED "FOR SCIENTISTS" SECTION
    with st.expander("🔬 For scientists"):
        st.markdown("**Formal Statistical Summary:**")
        st.info(stats["explanation"])
        st.markdown(f"""
        - **Hamed-Rao Modified Test Statistic (Z):** `{stats['z_stat']:+.3f}`
        - **Kendall's Tau (Rank Correlation):** `{stats['tau']:+.3f}`
        - **Test Trend Category:** `{stats['trend_label']}`
        - **Dataset ID:** `{info['giovanni_id']}` ({info['source']})
        - **Number of Monthly Observations Analyzed:** `{stats['n_samples']}`
        - **Method:** Hamed & Rao (1998) variance correction for lag autocorrelation + 250-iteration moving block bootstrap for 95% CI.
        """)

    # Quick Switcher with on_click callbacks (arranged in 3 columns)
    st.markdown("---")
    st.markdown("##### 🔄 Switch Variable:")
    c_sw1, c_sw2, c_sw3 = st.columns(3)
    with c_sw1:
        action_button("🔥 Switch to Heat", key="res_btn_heat", on_click=set_var, args=("heat",))
    with c_sw2:
        action_button("🌧️ Switch to Rain", key="res_btn_rain", on_click=set_var, args=("rain",))
    with c_sw3:
        action_button("🌱 Switch to Soil Dryness", key="res_btn_soil", on_click=set_var, args=("soil",))


# -------------------------------------------------------------
# SCREEN 4: CONNECT THE SIGNALS
# -------------------------------------------------------------
elif current_page == "Connect the signals":
    st.markdown('<div class="hero-title">🔗 Connect the Signals</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Do heat, rain and soil moisture move together? '
        'Let’s see how Chennai’s climate signals connect over time.</div>',
        unsafe_allow_html=True
    )

    corr_data = get_cached_correlations(
        datasets["heat"]["df"],
        datasets["rain"]["df"],
        datasets["soil"]["df"]
    )
    merged_df = corr_data["merged_df"]

    # Multi-variable Timeline (Z-Scores)
    st.subheader("1. All Three Signals on One Timeline")
    st.caption("Each signal is standardized (Z-score) so we can compare temperature, rain, and soil moisture on the same graph.")

    fig_multi = go.Figure()
    fig_multi.add_trace(go.Scatter(
        x=merged_df["time"], y=merged_df["heat_zscore"],
        mode="lines", name="Heat Anomaly", line=dict(color="#EA580C", width=1.8),
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Heat:</b> %{y:+.2f}σ<extra></extra>"
    ))
    fig_multi.add_trace(go.Scatter(
        x=merged_df["time"], y=merged_df["rain_zscore"],
        mode="lines", name="Rain Anomaly", line=dict(color="#2563EB", width=1.8),
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Rain:</b> %{y:+.2f}σ<extra></extra>"
    ))
    fig_multi.add_trace(go.Scatter(
        x=merged_df["time"], y=merged_df["soil_zscore"],
        mode="lines", name="Soil Moisture Anomaly", line=dict(color="#16A34A", width=2.2),
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Soil:</b> %{y:+.2f}σ<extra></extra>"
    ))
    fig_multi.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
    fig_multi.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(title="Year", gridcolor="#F1F5F9"),
        yaxis=dict(title="Departure from Normal (Standard Deviations)", gridcolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    show_chart(fig_multi)

    st.markdown("---")

    # 2. Pairwise Correlations with One-Sentence Reading Under Each
    st.subheader("2. How Closely Do Pairs Move Together?")
    st.caption(f"Calculated across {corr_data['n_overlap']} overlapping monthly measurements (2002 to 2025).")

    p_cols = st.columns(3)
    pairs = corr_data["pairs"]

    plain_readings = {
        ("Heat", "Rain"): "When months are rainier, cloud cover keeps the surface cooler.",
        ("Heat", "Soil"): "When it is hotter, the soil tends to be drier.",
        ("Rain", "Soil"): "When it rains more, the soil absorbs water and becomes moister."
    }

    for col, (pair_name, pair_info) in zip(p_cols, pairs.items()):
        with col:
            r = pair_info["r"]
            color = "#EA580C" if r < 0 else "#2563EB"
            reading = plain_readings.get(pair_name, pair_info["interpretation"])
            st.markdown(f"""
            <div class="metric-card-simple" style="margin-bottom:12px;">
                <div>
                    <div class="metric-label-simple">{pair_name[0]} vs. {pair_name[1]}</div>
                    <div class="metric-value-simple" style="color: {color};">r = {r:+.2f}</div>
                </div>
                <div class="metric-expl-simple"><b>{reading}</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Scatter Plot with NumPy linear fit (no extra dependencies)
            x_vals = merged_df[pair_info["x_col"]].to_numpy(dtype=float)
            y_vals = merged_df[pair_info["y_col"]].to_numpy(dtype=float)
            poly_fit = np.polyfit(x_vals, y_vals, 1)
            x_range = np.linspace(x_vals.min(), x_vals.max(), 50)
            y_fit = poly_fit[0] * x_range + poly_fit[1]

            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(color="#1E3A8A", opacity=0.55, size=6),
                name="Monthly Anomaly",
                hovertemplate=f"<b>{pair_info['x_label']}:</b> %{{x:+.2f}}<br><b>{pair_info['y_label']}:</b> %{{y:+.2f}}<extra></extra>"
            ))
            fig_scat.add_trace(go.Scatter(
                x=x_range,
                y=y_fit,
                mode="lines",
                line=dict(color="#DC2626", width=2),
                name="Linear Fit",
                hoverinfo="skip"
            ))
            fig_scat.update_layout(
                template="plotly_white",
                height=260,
                margin=dict(l=10, r=10, t=20, b=20),
                xaxis=dict(title=pair_info["x_label"], gridcolor="#F1F5F9"),
                yaxis=dict(title=pair_info["y_label"], gridcolor="#F1F5F9"),
                showlegend=False
            )
            show_chart(fig_scat)

    # Highlighted Caveat Box: Correlation Does Not Prove Cause
    st.markdown("""
    <div class="caution-box">
        <b>⚠️ Remember: Correlation does not prove cause!</b><br>
        Just because two things rise or fall together does not mean one caused the other. 
        For example, rain brings cloud cover that blocks sunlight, making rain and heat seem linked, 
        even though neither single-handedly controls the other. Both respond to regional monsoon winds and seasons.
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# SCREEN 5: DEMO EXAMPLES
# -------------------------------------------------------------
elif current_page == "Demo examples":
    st.markdown('<div class="hero-title">💡 Demo Examples</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Explore pre-loaded multi-decadal satellite records for Chennai with a single click.</div>',
        unsafe_allow_html=True
    )
    st.markdown("Each demo loads the full observational time series, computes the seasonal anomalies, and evaluates statistical significance:")

    de1, de2, de3 = st.columns(3)
    with de1:
        st.markdown("""
        <div class="step-card">
            <span class="step-badge" style="background-color:#FFEDD5; color:#9A3412;">NASA GLDAS 2.1</span>
            <div class="step-title">🔥 Heat (Surface Skin Temp)</div>
            <div class="step-desc">24 years (2002–2025) of monthly land surface temperature measurements over Chennai.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        action_button("📊 Load Heat Results", key="demo_btn_heat", type="primary", on_click=set_var_and_page, args=("heat", "Result", True))

    with de2:
        st.markdown("""
        <div class="step-card">
            <span class="step-badge" style="background-color:#DBEAFE; color:#1E40AF;">NASA GPM IMERG</span>
            <div class="step-title">🌧️ Rain (Monthly Rainfall)</div>
            <div class="step-desc">Monthly precipitation accumulation through September 2025 from NASA's rain constellation.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        action_button("📊 Load Rain Results", key="demo_btn_rain", type="primary", on_click=set_var_and_page, args=("rain", "Result", True))

    with de3:
        st.markdown("""
        <div class="step-card">
            <span class="step-badge" style="background-color:#DCFCE7; color:#166534;">NASA GLDAS 2.1</span>
            <div class="step-title">🌱 Soil Dryness (Root Zone)</div>
            <div class="step-desc">Root zone soil moisture stored in the top 100 cm of soil across Chennai (2002–2025).</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        action_button("📊 Load Soil Dryness Results", key="demo_btn_soil", type="primary", on_click=set_var_and_page, args=("soil", "Result", True))


# -------------------------------------------------------------
# SCREEN 6: METHOD AND LIMITS (SIMPLIFIED & GLOSSARY)
# -------------------------------------------------------------
elif current_page == "Method and limits":
    st.markdown('<div class="hero-title">🔬 Method and Limits</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">How we did the math, what the data can tell you, '
        'and the limits you should always keep in mind.</div>',
        unsafe_allow_html=True
    )

    # 5 Key Caveats in Simple Short Sentences
    st.subheader("5 Things to Know About This Data")

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("""
        <div class="glossary-card">
            <div class="glossary-term">1. GLDAS is a model, not a thermometer</div>
            <div class="glossary-def">
                GLDAS is a computer model driven by satellites, radar, and weather calculations. 
                It is not a direct thermometer stuck in the ground. While very useful, models can have minor biases.
            </div>
        </div>
        <div class="glossary-card">
            <div class="glossary-term">2. Skin temperature is not air temperature</div>
            <div class="glossary-def">
                Skin temperature measures the warmth of the ground surface (roofs, roads, and soil). 
                On a sunny afternoon in Chennai, dark asphalt can hit 45°C even when the air you feel is 34°C.
            </div>
        </div>
        <div class="glossary-card">
            <div class="glossary-term">3. The area is a 25 km box</div>
            <div class="glossary-def">
                The satellite pixel averages a 25 km by 25 km area. It blends downtown Chennai, suburbs, 
                farmland, and coastal waters together. Small neighborhood microclimates are smoothed out.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_l2:
        st.markdown("""
        <div class="glossary-card">
            <div class="glossary-term">4. Only straight-line trends are tested</div>
            <div class="glossary-def">
                Our tests look for steady, one-directional changes over time. 
                They do not test for sudden jumps, multi-year waves, or cyclic patterns like ocean oscillations.
            </div>
        </div>
        <div class="glossary-card">
            <div class="glossary-term">5. A trend does not prove a cause</div>
            <div class="glossary-def">
                Even when a trend is statistically significant, statistics alone cannot prove why it happened. 
                A trend could be influenced by city growth, global warming, or natural multi-decade cycles.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Short Glossary of Terms
    st.subheader("📖 Quick Glossary of Terms")
    st.caption("Simple definitions for the scientific terms used in Earth Intelligence AI:")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
        <div class="glossary-card">
            <div class="glossary-term">Trend</div>
            <div class="glossary-def">A steady change in one direction over many years, like temperature slowly rising or falling.</div>
        </div>
        <div class="glossary-card">
            <div class="glossary-term">Anomaly</div>
            <div class="glossary-def">How much a specific month was above or below its normal long-term average for that time of year.</div>
        </div>
        <div class="glossary-card">
            <div class="glossary-term">p-value</div>
            <div class="glossary-def">A score from 0 to 1 measuring the chance that a pattern is pure coincidence. Below 0.05 means significant (unlikely to be random noise).</div>
        </div>
        """, unsafe_allow_html=True)

    with g2:
        st.markdown("""
        <div class="glossary-card">
            <div class="glossary-term">Confidence Interval (95%)</div>
            <div class="glossary-def">A range of values that we are 95% confident contains the true rate of change over time.</div>
        </div>
        <div class="glossary-card">
            <div class="glossary-term">Sen's Slope</div>
            <div class="glossary-def">A smart way to calculate rate of change that finds the median slope between all pairs of months, so freak storms or extreme days don’t distort the trend.</div>
        </div>
        <div class="glossary-card">
            <div class="glossary-term">Mann-Kendall Test</div>
            <div class="glossary-def">A standard test recommended by scientists that checks if numbers consistently move up or down over time, corrected for seasonal repeating patterns.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("💡 **Data Acknowledgement:** Earth Intelligence AI utilizes NASA Goddard Earth Sciences (GES DISC) Giovanni data for the GLDAS Noah Land Surface Model and GPM IMERG Precipitation.")


# -------------------------------------------------------------
# GLOBAL FOOTER
# -------------------------------------------------------------
st.markdown('<div class="app-footer">Team OUT OF THE SPACE · Earth Intelligence AI · NASA Space Apps</div>', unsafe_allow_html=True)
