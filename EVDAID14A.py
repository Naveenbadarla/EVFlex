import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime
import base64


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="E.ON EV Charging Optimisation",
    layout="wide",
    page_icon="⚡",
)


# =============================================================================
# GLOBAL PREMIUM E.ON STYLING
# =============================================================================
st.markdown(
    """
    <style>

    /* Overall background and typography */
    body {
        background-color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    .main {
        padding-top: 10px;
    }

    /* Center app width a bit more */
    .block-container {
        max-width: 1300px;
        padding-top: 0rem;
        padding-bottom: 4rem;
    }

    /* Clean card-like containers */
    .eon-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(15, 23, 42, 0.06);
    }

    .eon-card h3 {
        margin-top: 0;
    }

    /* Metrics tweaks */
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid rgba(15, 23, 42, 0.06);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #fafafa;
        border-right: 1px solid rgba(148, 163, 184, 0.3);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Sidebar headers */
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        font-weight: 600;
    }

    /* Sliders & number inputs minimal style */
    input[type=number], .stNumberInput input, .stTextInput input {
        border-radius: 10px !important;
    }

    /* File uploader */
    .stFileUploader > label {
        font-weight: 500;
    }

    /* Expander minimal */
    details {
        border-radius: 12px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# LOGO LOADING
# =============================================================================
def load_logo(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Adjust filename if needed
logo_base64 = load_logo("eon_logo.png")


# =============================================================================
# PREMIUM MINIMAL E.ON HEADER
# =============================================================================
st.markdown(
    """
    <style>

    @keyframes subtleShift {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    .eon-header {
        width: 100%;
        padding: 26px 40px;
        border-radius: 18px;
        margin-bottom: 32px;
        display: flex;
        align-items: center;
        justify-content: flex-start;

        background: linear-gradient(90deg,
            #E2000F 0%,
            #D9001A 40%,
            #C90024 100%
        );
        background-size: 200% 200%;
        animation: subtleShift 12s ease-in-out infinite;
    }

    .eon-header-logo {
        width: 170px;
        margin-right: 40px;
        filter: drop-shadow(0 3px 8px rgba(0,0,0,0.18));
    }

    .eon-header-title {
        font-size: 36px;
        font-weight: 600;
        color: white;
        margin: 0;
        letter-spacing: -0.4px;
    }

    .eon-header-sub {
        font-size: 18px;
        font-weight: 300;
        color: rgba(255,255,255,0.9);
        margin-top: 6px;
    }

    .eon-nav {
        margin-top: 12px;
        display: flex;
        gap: 26px;
    }

    .eon-nav a {
        font-size: 15px;
        color: rgba(255,255,255,0.92);
        text-decoration: none;
        transition: 0.25s;
        padding-bottom: 4px;
        border-bottom: 2px solid transparent;
    }

    .eon-nav a:hover {
        border-bottom: 2px solid rgba(255,255,255,0.85);
    }

    @media (max-width: 768px) {
        .eon-header {
            flex-direction: column;
            padding: 24px 20px;
            text-align: center;
        }
        .eon-header-logo {
            margin-right: 0;
            margin-bottom: 12px;
            width: 150px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="eon-header">
        <img src="data:image/png;base64,{logo_base64}" class="eon-header-logo">

        <div>
            <div class="eon-header-title">EV Charging Optimisation Dashboard</div>
            <div class="eon-header-sub">
                Full-Year Day-Ahead & Intraday Market-Based Smart Charging
            </div>

            <div class="eon-nav">
                <a href="#overview">Overview</a>
                <a href="#prices">Market Prices</a>
                <a href="#results">Results</a>
                <a href="#details">Details</a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SYNTHETIC PRICE GENERATION (used if no price files uploaded)
# =============================================================================
def get_synthetic_daily_price_profiles():
    hours = np.arange(24)
    base = 70
    morning_peak = 15 * np.sin((hours - 7) / 24 * 2 * np.pi) ** 2
    evening_peak = 35 * np.sin((hours - 18) / 24 * 2 * np.pi) ** 4
    da_hourly = base + morning_peak + evening_peak

    rng = np.random.default_rng(42)
    da_q = np.repeat(da_hourly, 4)
    id_q = da_q + rng.normal(0, 6, size=96)

    return da_hourly, id_q


# =============================================================================
# CHARGING FREQUENCY RULES
# =============================================================================
DAY_NAME_TO_IDX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def compute_charging_days(pattern, num_days, custom=None, weekly=None):
    if custom is None:
        custom = []

    if pattern == "Every day":
        return list(range(num_days))

    if pattern == "Every other day":
        return list(range(0, num_days, 2))

    if pattern == "Weekdays only (Mon–Fri)":
        return [d for d in range(num_days) if (d % 7) < 5]

    if pattern == "Weekends only (Sat–Sun)":
        return [d for d in range(num_days) if (d % 7) >= 5]

    if pattern == "Custom weekdays":
        allowed = {DAY_NAME_TO_IDX[d] for d in custom}
        return [d for d in range(num_days) if (d % 7) in allowed]

    if pattern == "Custom: X sessions per week":
        weekly = int(weekly)
        out = []
        weeks = num_days // 7
        for w in range(weeks):
            for s in range(weekly):
                day = w * 7 + int(np.floor(s * 7 / weekly))
                out.append(day)
        return out

    return list(range(num_days))


# =============================================================================
# TIME WINDOW PARSER (15-min slots)
# =============================================================================
def build_available_quarters(arrival, departure):
    aq = int(arrival * 4) % 96
    dq = int(departure * 4) % 96
    if aq == dq:
        return list(range(96)), aq, dq
    if aq < dq:
        return list(range(aq, dq)), aq, dq
    return list(range(aq, 96)) + list(range(0, dq)), aq, dq


# =============================================================================
# COSTING LOGIC
# =============================================================================
def apply_tariffs(price_kwh, grid, taxes, vat):
    return (price_kwh + grid + taxes) * (1 + vat / 100)


def compute_da_indexed_baseline_daily(E, Pmax, quarters, da_day, grid, taxes, vat):
    da_q = np.repeat(da_day / 1000, 4)
    remain = E
    cost = 0.0
    emax = Pmax * 0.25
    for q in quarters:
        if remain <= 1e-6:
            break
        e = min(emax, remain)
        cost += e * apply_tariffs(da_q[q], grid, taxes, vat)
        remain -= e
    if remain > 1e-6:
        raise ValueError("Charging window too short to deliver required energy.")
    return cost


def compute_optimised_daily_cost(E, Pmax, quarters, price_q, grid, taxes, vat):
    pq = price_q / 1000
    emax = Pmax * 0.25
    if len(quarters) * emax < E - 1e-6:
        raise ValueError("Charging window too short for requested energy.")
    sorted_q = sorted(quarters, key=lambda x: pq[x])
    remain = E
    cost = 0.0
    for q in sorted_q:
        if remain <= 1e-6:
            break
        e = min(emax, remain)
        cost += e * apply_tariffs(pq[q], grid, taxes, vat)
        remain -= e
    return cost


# =============================================================================
# FILE LOADER
# =============================================================================
def load_price_series_from_csv(upload, multiple):
    df = pd.read_csv(upload)
    if "price" in df.columns:
        series = df["price"].values
    else:
        series = df.select_dtypes(include="number").iloc[:, 0].values
    if len(series) % multiple != 0:
        return None, 0
    return series.astype(float), len(series) // multiple


# =============================================================================
# SIDEBAR (minimal, but functional)
# =============================================================================
st.sidebar.title("Simulation Settings")

st.sidebar.subheader("EV & Charging")
energy = st.sidebar.number_input("Energy per session (kWh)", 1.0, 200.0, 40.0, 1.0)
power = st.sidebar.number_input("Max charging power (kW)", 1.0, 50.0, 11.0, 0.5)

arrival = st.sidebar.slider("Arrival time [h]", 0.0, 24.0, 18.0, 0.25)
departure = st.sidebar.slider("Departure time [h]", 0.0, 24.0, 7.0, 0.25)

st.sidebar.subheader("Charging Frequency")
freq = st.sidebar.selectbox(
    "Pattern",
    [
        "Every day",
        "Every other day",
        "Weekdays only (Mon–Fri)",
        "Weekends only (Sat–Sun)",
        "Custom weekdays",
        "Custom: X sessions per week",
    ],
)

custom_days = None
sessions_week = None
if freq == "Custom weekdays":
    custom_days = st.sidebar.multiselect(
        "Select weekdays",
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        default=["Mon", "Wed", "Fri"],
    )
if freq == "Custom: X sessions per week":
    sessions_week = st.sidebar.number_input("Sessions per week", 1, 14, 3)

st.sidebar.subheader("Tariffs")
flat_price = st.sidebar.number_input("Flat all-in price (€/kWh)", 0.0, 1.0, 0.35, 0.01)
grid = st.sidebar.number_input("Grid network charges (€/kWh)", 0.0, 1.0, 0.11, 0.01)
taxes = st.sidebar.number_input("Taxes & levies (€/kWh)", 0.0, 1.0, 0.05, 0.01)
vat = st.sidebar.number_input("VAT (%)", 0.0, 30.0, 19.0, 1.0)

st.sidebar.subheader("Market Data")
da_file = st.sidebar.file_uploader("DA hourly prices CSV (€/MWh)")
id_file = st.sidebar.file_uploader("ID 15-min prices CSV (€/MWh)")


# =============================================================================
# MARKET DATA LOADING
# =============================================================================
da_day_syn, id_day_syn = get_synthetic_daily_price_profiles()

if da_file:
    da_series, da_days = load_price_series_from_csv(da_file, 24)
else:
    da_series, da_days = np.tile(da_day_syn, 365), 365

if id_file:
    id_series, id_days = load_price_series_from_csv(id_file, 96)
else:
    id_series, id_days = np.tile(id_day_syn, 365), 365

num_days = min(da_days, id_days)
da_year = da_series[: num_days * 24]
id_year = id_series[: num_days * 96]

st.sidebar.info(f"Using **{num_days} days** of DA & ID price data.")


# =============================================================================
# OVERVIEW
# =============================================================================
st.markdown("<h2 id='overview'></h2>", unsafe_allow_html=True)

overview_col1, overview_col2 = st.columns([2, 1])

with overview_col1:
    st.markdown(
        """
        <div class="eon-card">
            <h3>Overview</h3>
            <p style="margin-bottom:0;">
            This dashboard compares four EV charging cost scenarios over a full year 
            using hourly day-ahead and 15-minute intraday price signals:
            </p>
            <ul>
              <li><b>Flat retail:</b> constant €/kWh, no market exposure</li>
              <li><b>DA-indexed:</b> hourly wholesale pass-through, no optimisation</li>
              <li><b>DA-optimised:</b> smart charging using DA prices</li>
              <li><b>DA+ID-optimised:</b> smart charging using min(DA, ID)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_col2:
    st.markdown(
        f"""
        <div class="eon-card">
            <h3>Scenario Snapshot</h3>
            <p><b>Energy/session:</b> {energy:.1f} kWh</p>
            <p><b>Max power:</b> {power:.1f} kW</p>
            <p><b>Arrival:</b> {arrival:.2f} h</p>
            <p><b>Departure:</b> {departure:.2f} h</p>
            <p><b>Tariff flat:</b> {flat_price:.2f} €/kWh</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# FULL-YEAR PLOT SECTION
# =============================================================================
st.markdown("<h2 id='prices'></h2>", unsafe_allow_html=True)
st.markdown('<div class="eon-card">', unsafe_allow_html=True)
st.subheader("Market Price Curves (Full Year)")

timestamps = pd.date_range("2023-01-01", periods=num_days * 96, freq="15min")
da_quarter = np.repeat(da_year, 4)[: num_days * 96]
id_quarter = id_year[: num_days * 96]
effective = np.minimum(da_quarter, id_quarter)

df_plot = pd.DataFrame(
    {
        "timestamp": timestamps,
        "DA (€/MWh)": da_quarter,
        "ID (€/MWh)": id_quarter,
        "Effective min(DA, ID) (€/MWh)": effective,
    }
)

fig = px.line(
    df_plot,
    x="timestamp",
    y=["DA (€/MWh)", "ID (€/MWh)", "Effective min(DA, ID) (€/MWh)"],
    title="Day-Ahead & Intraday Prices (zoom & pan)",
)
fig.update_layout(
    xaxis_rangeslider_visible=True,
    height=450,
    legend=dict(orientation="h", y=-0.2),
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# COST CALCULATION
# =============================================================================
quarters, _, _ = build_available_quarters(arrival, departure)
charging_days = compute_charging_days(freq, num_days, custom_days, sessions_week)
sessions = len(charging_days)

flat_annual = flat_price * energy * sessions
da_index_annual = 0.0
da_opt_annual = 0.0
da_id_annual = 0.0

try:
    for d in charging_days:
        da_day = da_year[d * 24 : (d + 1) * 24]
        id_day = id_year[d * 96 : (d + 1) * 96]
        da_q_day = np.repeat(da_day, 4)
        effective_day = np.minimum(da_q_day, id_day)

        da_index_annual += compute_da_indexed_baseline_daily(
            energy, power, quarters, da_day, grid, taxes, vat
        )

        da_opt_annual += compute_optimised_daily_cost(
            energy, power, quarters, da_q_day, grid, taxes, vat
        )

        da_id_annual += compute_optimised_daily_cost(
            energy, power, quarters, effective_day, grid, taxes, vat
        )
except ValueError as e:
    st.error(str(e))


# =============================================================================
# RESULTS SECTION
# =============================================================================
st.markdown("<h2 id='results'></h2>", unsafe_allow_html=True)
st.markdown('<div class="eon-card">', unsafe_allow_html=True)
st.subheader("Annual Cost by Customer Type")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Flat retail", f"{flat_annual:,.0f} €")
r2.metric("DA-indexed", f"{da_index_annual:,.0f} €")
r3.metric("DA-optimised", f"{da_opt_annual:,.0f} €")
r4.metric("DA+ID-optimised", f"{da_id_annual:,.0f} €")

st.markdown("---")
st.subheader("Savings vs Flat Retail")

def savings(new, base):
    return base - new, 100 * (1 - new / base) if base > 0 else 0.0

s2, p2 = savings(da_index_annual, flat_annual)
s3, p3 = savings(da_opt_annual, flat_annual)
s4, p4 = savings(da_id_annual, flat_annual)

c1, c2, c3 = st.columns(3)
c1.metric("DA-indexed", f"{s2:,.0f} € / yr", f"{p2:.1f} %")
c2.metric("DA-optimised", f"{s3:,.0f} € / yr", f"{p3:.1f} %")
c3.metric("DA+ID-optimised", f"{s4:,.0f} € / yr", f"{p4:.1f} %")

st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# DETAILS SECTION
# =============================================================================
st.markdown("<h2 id='details'></h2>", unsafe_allow_html=True)
st.markdown('<div class="eon-card">', unsafe_allow_html=True)
st.subheader("Model Assumptions & Notes")

st.markdown(
    f"""
- Time resolution: **15 minutes (96 slots/day)**  
- DA: **hourly prices**, ID: **15-min prices**, both in €/MWh  
- Effective price for DA+ID = **min(DA, ID)** per 15-min slot  
- Dynamic tariffs applied as:  
  **(wholesale price + grid charges + taxes & levies) × (1 + VAT)**  
- Flat retail scenario assumes an all-in €/kWh price (no extra VAT in the model).  
- Charging pattern: **{freq}**, resulting in **{sessions} sessions** over {num_days} days of price data.  
- Arrival / departure window is the same every charging day, with smart charging reshaping load within that window.  
"""
)

st.markdown("</div>", unsafe_allow_html=True)
