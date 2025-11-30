import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime
import base64


# =====================================================================================
# Load logo for animated header
# =====================================================================================
def load_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


logo_base64 = load_logo("eon_logo.png")


# =====================================================================================
# Inject ADVANCED ANIMATED E.ON HEADER (CSS + HTML)
# =====================================================================================
st.markdown("""
    <style>
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes fadeInSlide {
            0% { opacity: 0; transform: translateY(-20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        @keyframes logoGlow {
            0%, 100% { filter: drop-shadow(0 0 5px rgba(255,255,255,0.3)); }
            50% { filter: drop-shadow(0 0 15px rgba(255,255,255,0.8)); }
        }

        /* Main animated header container */
        .eon-header {
            width: 100%;
            background: linear-gradient(90deg, #E2000F, #FF3A3A, #FF6A6A, #E2000F);
            background-size: 300% 300%;
            animation: gradientShift 12s ease infinite;
            padding: 25px 40px;
            border-radius: 14px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 999;
            animation: fadeInSlide 1.2s ease-out;
        }

        /* Logo animation */
        .eon-header-logo {
            width: 160px;
            height: auto;
            margin-right: 25px;
            animation: logoGlow 4s ease-in-out infinite;
        }

        /* Text styling */
        .eon-header-text h1 {
            color: white;
            margin: 0;
            font-size: 38px;
            font-weight: 700;
            line-height: 1.1;
        }

        .eon-header-text h3 {
            color: rgba(255,255,255,0.9);
            margin-top: 6px;
            font-size: 20px;
            font-weight: 300;
        }

        /* Navigation bar */
        .eon-nav {
            margin-top: 14px;
            display: flex;
            gap: 22px;
        }

        .eon-nav a {
            color: white;
            font-size: 16px;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            transition: 0.25s;
            font-weight: 400;
        }

        .eon-nav a:hover {
            background: rgba(255,255,255,0.25);
        }

        /* Responsive layout */
        @media (max-width: 768px) {
            .eon-header {
                flex-direction: column;
                text-align: center;
                padding: 25px 20px;
            }
            .eon-header-logo {
                margin-right: 0;
                margin-bottom: 15px;
            }
        }

    </style>
""", unsafe_allow_html=True)


st.markdown(
    f"""
    <div class="eon-header">
        <img src="data:image/png;base64,{logo_base64}" class="eon-header-logo">
        <div class="eon-header-text">
            <h1>EV Charging Optimisation Dashboard</h1>
            <h3>Full-Year Day-Ahead & Intraday Market Simulation</h3>
            <div class="eon-nav">
                <a href="#overview">Overview</a>
                <a href="#prices">Market Prices</a>
                <a href="#results">Results</a>
                <a href="#details">Details</a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# =====================================================================================
# SYNTHETIC PRICE GENERATION (used if no price files uploaded)
# =====================================================================================
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


# =====================================================================================
# CHARGING FREQUENCY RULES
# =====================================================================================
DAY_NAME_TO_IDX = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}

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


# =====================================================================================
# TIME WINDOW PARSER
# =====================================================================================
def build_available_quarters(arrival, departure):
    aq = int(arrival * 4) % 96
    dq = int(departure * 4) % 96
    if aq == dq:
        return list(range(96)), aq, dq
    if aq < dq:
        return list(range(aq, dq)), aq, dq
    return list(range(aq, 96)) + list(range(0, dq)), aq, dq


# =====================================================================================
# COSTING LOGIC
# =====================================================================================
def apply_tariffs(price_kwh, grid, taxes, vat):
    return (price_kwh + grid + taxes) * (1 + vat/100)


def compute_da_indexed_baseline_daily(E, Pmax, quarters, da_day, grid, taxes, vat):
    da_q = np.repeat(da_day/1000, 4)
    remain = E
    cost = 0
    emax = Pmax * 0.25
    for q in quarters:
        if remain <= 0: break
        e = min(emax, remain)
        cost += e * apply_tariffs(da_q[q], grid, taxes, vat)
        remain -= e
    if remain > 0:
        raise ValueError("Time window too short.")
    return cost


def compute_optimised_daily_cost(E, Pmax, quarters, price_q, grid, taxes, vat):
    pq = price_q / 1000
    emax = Pmax * 0.25
    if len(quarters) * emax < E:
        raise ValueError("Insufficient charging window.")
    sorted_q = sorted(quarters, key=lambda x: pq[x])
    remain = E
    cost = 0
    for q in sorted_q:
        if remain <= 0: break
        e = min(emax, remain)
        cost += e * apply_tariffs(pq[q], grid, taxes, vat)
        remain -= e
    return cost


# =====================================================================================
# FILE LOADER
# =====================================================================================
def load_price_series_from_csv(upload, multiple):
    df = pd.read_csv(upload)
    if "price" in df.columns:
        series = df["price"].values
    else:
        series = df.select_dtypes(include="number").iloc[:,0].values
    if len(series) % multiple != 0:
        return None, 0
    return series, len(series)//multiple


# =====================================================================================
# SIDEBAR SETTINGS
# =====================================================================================
st.markdown("<h2 id='overview'></h2>", unsafe_allow_html=True)
st.header("Overview")

st.sidebar.header("EV Settings")
energy = st.sidebar.number_input("Energy per session (kWh)", 1.0, 200.0, 40.0)
power = st.sidebar.number_input("Max Charging Power (kW)", 1.0, 50.0, 11.0)

arrival = st.sidebar.slider("Arrival time", 0.0, 24.0, 18.0, 0.25)
departure = st.sidebar.slider("Departure time", 0.0, 24.0, 7.0, 0.25)

st.sidebar.header("Charging Frequency")
freq = st.sidebar.selectbox("Pattern", 
    ["Every day","Every other day","Weekdays only (Mon–Fri)",
     "Weekends only (Sat–Sun)", "Custom weekdays","Custom: X sessions per week"])

custom_days = None
sessions_week = None
if freq == "Custom weekdays":
    custom_days = st.sidebar.multiselect("Days", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
if freq == "Custom: X sessions per week":
    sessions_week = st.sidebar.number_input("Sessions per week",1,14,3)

st.sidebar.header("Tariffs")
flat_price = st.sidebar.number_input("Flat price €/kWh", 0.0, 1.0, 0.35)
grid = st.sidebar.number_input("Grid charges €/kWh", 0.0,1.0,0.11)
taxes = st.sidebar.number_input("Taxes & levies €/kWh",0.0,1.0,0.05)
vat = st.sidebar.number_input("VAT %",0.0,30.0,19.0)

st.sidebar.header("Market Data")
da_file = st.sidebar.file_uploader("DA hourly CSV")
id_file = st.sidebar.file_uploader("ID 15-min CSV")


# =====================================================================================
# MARKET DATA LOADING
# =====================================================================================
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
da_year = da_series[:num_days * 24]
id_year = id_series[:num_days * 96]

st.success(f"{num_days} days loaded.")


# =====================================================================================
# FULL-YEAR PLOT SECTION
# =====================================================================================
st.markdown("<h2 id='prices'></h2>", unsafe_allow_html=True)
st.header("Market Prices (Zoomable)")

timestamps = pd.date_range("2023-01-01", periods=num_days*96, freq="15min")
da_quarter = np.repeat(da_year,4)[:num_days*96]
id_quarter = id_year[:num_days*96]
effective = np.minimum(da_quarter, id_quarter)

df_plot = pd.DataFrame({
    "timestamp": timestamps,
    "DA (€/MWh)": da_quarter,
    "ID (€/MWh)": id_quarter,
    "Effective price": effective
})

fig = px.line(df_plot, x="timestamp",
              y=["DA (€/MWh)","ID (€/MWh)","Effective price"],
              title="Full-Year DA & ID Prices")
fig.update_layout(xaxis_rangeslider_visible=True, height=500)
st.plotly_chart(fig, use_container_width=True)


# =====================================================================================
# COST CALCULATION
# =====================================================================================
quarters,_,_ = build_available_quarters(arrival, departure)
charging_days = compute_charging_days(freq, num_days, custom_days, sessions_week)
sessions = len(charging_days)

flat_annual = flat_price * energy * sessions
da_index_annual = 0
da_opt_annual = 0
da_id_annual = 0

try:
    for d in charging_days:
        da_day = da_year[d*24:(d+1)*24]
        id_day = id_year[d*96:(d+1)*96]
        da_q_day = np.repeat(da_day,4)
        effective_day = np.minimum(da_q_day, id_day)

        da_index_annual += compute_da_indexed_baseline_daily(
            energy, power, quarters, da_day, grid, taxes, vat)

        da_opt_annual += compute_optimised_daily_cost(
            energy, power, quarters, da_q_day, grid, taxes, vat)

        da_id_annual += compute_optimised_daily_cost(
            energy, power, quarters, effective_day, grid, taxes, vat)

except ValueError as e:
    st.error(str(e))


# =====================================================================================
# RESULTS SECTION
# =====================================================================================
st.markdown("<h2 id='results'></h2>", unsafe_allow_html=True)
st.header("Annual Cost Comparison")

col1,col2 = st.columns(2)
col3,col4 = st.columns(2)

col1.metric("Flat price", f"{flat_annual:,.0f} €")
col2.metric("DA-indexed", f"{da_index_annual:,.0f} €")
col3.metric("DA-optimised", f"{da_opt_annual:,.0f} €")
col4.metric("DA+ID optimised", f"{da_id_annual:,.0f} €")

st.subheader("Savings vs Flat")
def savings(new, base): 
    return base-new, 100*(1-new/base)

s2, p2 = savings(da_index_annual, flat_annual)
s3, p3 = savings(da_opt_annual, flat_annual)
s4, p4 = savings(da_id_annual, flat_annual)

c1,c2,c3 = st.columns(3)
c1.metric("DA-indexed", f"{s2:,.0f} €", f"{p2:.1f}%")
c2.metric("DA-optimised", f"{s3:,.0f} €", f"{p3:.1f}%")
c3.metric("DA+ID optimised", f"{s4:,.0f} €", f"{p4:.1f}%")

st.success("All calculations complete.")


# =====================================================================================
# DETAILS SECTION
# =====================================================================================
st.markdown("<h2 id='details'></h2>", unsafe_allow_html=True)
st.header("Details & Model Assumptions")

st.write("""
- Full year based on uploaded DA/ID price data  
- 15-min charging resolution  
- Tariffs applied as: (price + grid + taxes) × (1 + VAT)  
- Charging windows defined by arrival → departure  
- Smart optimisation chooses cheapest available 15-min slots  
""")


