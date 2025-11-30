import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime


# ==========================================================================
# ---------------------- FULL E.ON THEME STYLING ---------------------------
# ==========================================================================

eon_logo = "https://upload.wikimedia.org/wikipedia/commons/1/17/E.ON_Logo_2021.png"

st.markdown(
    f"""
    <style>
        /* Global font */
        html, body, [class*="css"]  {{
            font-family: "Source Sans Pro", sans-serif;
        }}

        /* ---------------- HEADER ---------------- */
        .eon-header {{
            background-color: #E2000F;
            padding: 25px 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            color: white;
        }}

        .eon-header img {{
            height: 60px;
            margin-right: 22px;
        }}

        .eon-title {{
            font-size: 34px;
            font-weight: 700;
            margin: 0;
        }}

        .eon-subtitle {{
            font-size: 18px;
            font-weight: 300;
            color: #FFEAEA;
            margin-top: 4px;
        }}

        @media (max-width: 600px) {{
            .eon-header {{
                flex-direction: column;
                text-align: center;
            }}
            .eon-header img {{
                margin-bottom: 12px;
            }}
        }}

        /* ---------------- SIDEBAR ---------------- */
        section[data-testid="stSidebar"] {{
            background-color: #FDECEC !important;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: #E2000F !important;
        }}
        .css-1cpxqw2, .css-1p4jv9z, label {{
            color: #B00000 !important;
            font-weight: 600 !important;
        }}

        /* ---------------- BUTTONS ---------------- */
        .stButton button {{
            background-color: #E2000F;
            color: white;
            border-radius: 6px;
            padding: 8px 18px;
            border: none;
            font-weight: 600;
            height: 40px;
        }}
        .stButton button:hover {{
            background-color: #C40000;
        }}

        /* ---------------- SLIDERS ---------------- */
        .stSlider > div {{
            background-color: #E2000F20;
            border-radius: 8px;
            padding: 8px;
        }}
        div[data-baseweb="slider"] > div > div {{
            background-color: #E2000F !important;
        }}

        /* ---------------- INPUT FIELDS ---------------- */
        input[type="number"], input[type="text"] {{
            border: 1.5px solid #E2000F50 !important;
            border-radius: 5px !important;
        }}

        /* ---------------- SELECTBOX ---------------- */
        .stSelectbox div[data-baseweb="select"] > div {{
            border: 1.5px solid #E2000F50 !important;
            border-radius: 5px !important;
        }}

        /* ---------------- CARDS ---------------- */
        .eon-card {{
            padding: 20px;
            margin-top: 15px;
            background-color: white;
            border-radius: 10px;
            border-left: 5px solid #E2000F;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }}

        /* ---------------- FOOTER ---------------- */
        .eon-footer {{
            margin-top: 50px;
            text-align: center;
            color: #888;
            font-size: 14px;
            padding: 15px;
            border-top: 1px solid #EEE;
        }}

    </style>

    <div class="eon-header">
        <img src="{eon_logo}">
        <div>
            <div class="eon-title">EV Charging Optimisation Dashboard</div>
            <div class="eon-subtitle">Full-Year Market Simulation · DA · ID · Smart Charging</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Price generation (synthetic fallback)
# ==========================================================================
def get_synthetic_daily_price_profiles():
    hours = np.arange(24)
    base = 70
    morning_peak = 15 * np.sin((hours - 7) / 24 * 2 * np.pi) ** 2
    evening_peak = 35 * np.sin((hours - 18) / 24 * 2 * np.pi) ** 4
    da_hourly = base + morning_peak + evening_peak

    rng = np.random.default_rng(42)
    da_quarter = np.repeat(da_hourly, 4)
    id_quarter = da_quarter + rng.normal(0, 6, size=96)
    return da_hourly, id_quarter


# ==========================================================================
# Charging frequency model
# ==========================================================================
DAY_NAME_TO_IDX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

def compute_charging_days(pattern, num_days, custom_weekdays=None, sessions_per_week=None):
    if custom_weekdays is None:
        custom_weekdays = []

    if pattern == "Every day":
        return list(range(num_days))

    if pattern == "Every other day":
        return list(range(0, num_days, 2))

    if pattern == "Weekdays only (Mon–Fri)":
        return [d for d in range(num_days) if (d % 7) in {0,1,2,3,4}]

    if pattern == "Weekends only (Sat–Sun)":
        return [d for d in range(num_days) if (d % 7) in {5,6}]

    if pattern == "Custom weekdays":
        target = {DAY_NAME_TO_IDX[x] for x in custom_weekdays}
        return [d for d in range(num_days) if (d % 7) in target]

    if pattern == "Custom: X sessions per week":
        if not sessions_per_week:
            return []
        days = []
        sessions_per_week = int(sessions_per_week)
        num_weeks = num_days // 7
        for w in range(num_weeks):
            for s in range(sessions_per_week):
                day_offset = int(np.floor(s * 7 / sessions_per_week))
                d = w*7 + day_offset
                if d < num_days:
                    days.append(d)
        return days

    return list(range(num_days))


# ==========================================================================
# Charging window (15-minute resolution)
# ==========================================================================
def build_available_quarters(arrival_h, departure_h):
    a = int(round(arrival_h * 4)) % 96
    d = int(round(departure_h * 4)) % 96

    if a == d:
        return list(range(96)), a, d
    if a < d:
        return list(range(a, d)), a, d
    else:
        return list(range(a, 96)) + list(range(0, d)), a, d


# ==========================================================================
# Tariff
# ==========================================================================
def apply_tariffs(price_kwh, grid, taxes, vat):
    pre = price_kwh + grid + taxes
    return pre * (1 + vat / 100)


# ==========================================================================
# Cost models
# ==========================================================================
def compute_da_indexed_baseline_daily(E_kwh, P_max, quarters, da_hourly, grid, taxes, vat):
    da_kwh = da_hourly / 1000
    da_q = np.repeat(da_kwh, 4)

    energy = E_kwh
    cost = 0
    Emax_q = P_max * 0.25

    for q in quarters:
        if energy <= 0:
            break
        e = min(Emax_q, energy)
        cost += e * apply_tariffs(da_q[q], grid, taxes, vat)
        energy -= e

    if energy > 0:
        raise ValueError("Charging window too short for requested energy.")

    return cost


def compute_optimised_daily_cost(E_kwh, P_max, quarters, price_q, grid, taxes, vat):
    price_kwh = price_q / 1000
    Emax_q = P_max * 0.25

    max_energy = len(quarters) * Emax_q
    if max_energy < E_kwh:
        raise ValueError("Charging window too short.")

    sorted_q = sorted(quarters, key=lambda q: price_kwh[q])

    energy = E_kwh
    cost = 0
    for q in sorted_q:
        if energy <= 0:
            break
        e = min(Emax_q, energy)
        cost += e * apply_tariffs(price_kwh[q], grid, taxes, vat)
        energy -= e

    return cost


# ==========================================================================
# CSV loader
# ==========================================================================
def load_price_series_from_csv(uploaded, multiple, name):
    df = pd.read_csv(uploaded)
    if "price" in df.columns:
        series = df["price"].values
    else:
        series = df.select_dtypes(include="number").iloc[:,0].values

    if len(series) % multiple != 0:
        st.error(f"{name}: length {len(series)} not divisible by {multiple}.")
        return None, 0

    return series.astype(float), len(series) // multiple


# ==========================================================================
# MAIN APP
# ==========================================================================
def main():

    # Sidebar inputs --------------------------------------------------------
    st.sidebar.header("EV Inputs")
    E_kwh = st.sidebar.number_input("Energy per session (kWh)", 1.0, 200.0, 40.0)
    P_max = st.sidebar.number_input("Max charging power (kW)", 1.0, 50.0, 11.0)

    st.sidebar.header("Arrival / Departure")
    arrival = st.sidebar.slider("Arrival time [h]", 0.0, 24.0, 18.0, 0.25)
    departure = st.sidebar.slider("Departure time [h]", 0.0, 24.0, 7.0, 0.25)

    st.sidebar.header("Charging Pattern")
    pattern = st.sidebar.selectbox("Choose pattern", [
        "Every day", "Every other day",
        "Weekdays only (Mon–Fri)", "Weekends only (Sat–Sun)",
        "Custom weekdays", "Custom: X sessions per week"
    ])

    custom_days = []
    sessions_week = None
    if pattern == "Custom weekdays":
        custom_days = st.sidebar.multiselect("Choose days", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    if pattern == "Custom: X sessions per week":
        sessions_week = st.sidebar.number_input("Sessions per week", 1, 14, 3)

    st.sidebar.header("Tariffs")
    flat_price = st.sidebar.number_input("Flat retail (€/kWh)", 0.0, 1.0, 0.35)
    grid = st.sidebar.number_input("Grid charges (€/kWh)", 0.0, 1.0, 0.11)
    taxes = st.sidebar.number_input("Taxes & levies (€/kWh)", 0.0, 1.0, 0.05)
    vat = st.sidebar.number_input("VAT (%)", 0.0, 30.0, 19.0)

    st.sidebar.header("Upload Market Data")
    da_file = st.sidebar.file_uploader("DA hourly (CSV)", type="csv")
    id_file = st.sidebar.file_uploader("ID 15-min (CSV)", type="csv")

    # Load prices -----------------------------------------------------------
    da_daily, id_daily = get_synthetic_daily_price_profiles()

    if da_file:
        da_series, da_days = load_price_series_from_csv(da_file, 24, "DA")
    else:
        da_series = np.tile(da_daily, 365)
        da_days = 365

    if id_file:
        id_series, id_days = load_price_series_from_csv(id_file, 96, "ID")
    else:
        id_series = np.tile(id_daily, 365)
        id_days = 365

    num_days = min(da_days, id_days)
    da_year = da_series[:num_days*24]
    id_year = id_series[:num_days*96]

    st.sidebar.success(f"Loaded {num_days} valid days.")

    # Plot full-year prices -------------------------------------------------
    st.subheader("📈 Full-Year DA + ID Prices (Zoomable)")

    timestamps = pd.date_range("2023-01-01", periods=num_days*96, freq="15min")

    da_q = np.repeat(da_year, 4)[:num_days*96]
    id_q = id_year[:num_days*96]
    eff_q = np.minimum(da_q, id_q)

    df_plot = pd.DataFrame({
        "timestamp": timestamps,
        "DA (€/MWh)": da_q,
        "ID (€/MWh)": id_q,
        "Effective min(DA,ID)": eff_q
    })

    fig = px.line(
        df_plot,
        x="timestamp",
        y=["DA (€/MWh)", "ID (€/MWh)", "Effective min(DA,ID)"],
        title="Full-Year DA / ID Price Curve",
    )
    fig.update_layout(xaxis_rangeslider_visible=True, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Charging windows ------------------------------------------------------
    quarters, _, _ = build_available_quarters(arrival, departure)
    charging_days = compute_charging_days(pattern, num_days, custom_days, sessions_week)
    n_sessions = len(charging_days)

    # Cost calculations -----------------------------------------------------
    flat_annual = E_kwh * flat_price * n_sessions
    da_indexed_annual = 0
    da_opt_annual = 0
    da_id_annual = 0

    try:
        for d in charging_days:
            da_day = da_year[d*24:(d+1)*24]
            id_day = id_year[d*96:(d+1)*96]
            da_q_day = np.repeat(da_day, 4)
            eff_day = np.minimum(da_q_day, id_day)

            da_indexed_annual += compute_da_indexed_baseline_daily(E_kwh, P_max, quarters, da_day, grid, taxes, vat)
            da_opt_annual += compute_optimised_daily_cost(E_kwh, P_max, quarters, da_q_day, grid, taxes, vat)
            da_id_annual += compute_optimised_daily_cost(E_kwh, P_max, quarters, eff_day, grid, taxes, vat)

    except ValueError as e:
        st.error(str(e))
        return

    # Results ---------------------------------------------------------------
    st.markdown('<div class="eon-card">', unsafe_allow_html=True)
    st.subheader("Annual Cost Comparison")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col1.metric("Flat retail", f"{flat_annual:,.0f} €")
    col2.metric("DA-indexed", f"{da_indexed_annual:,.0f} €")
    col3.metric("DA-optimised", f"{da_opt_annual:,.0f} €")
    col4.metric("DA+ID-optimised", f"{da_id_annual:,.0f} €")
    st.markdown('</div>', unsafe_allow_html=True)

    # Savings ---------------------------------------------------------------
    st.markdown('<div class="eon-card">', unsafe_allow_html=True)
    st.subheader("Savings vs Flat Retail")

    save_da = flat_annual - da_indexed_annual
    save_daopt = flat_annual - da_opt_annual
    save_eff = flat_annual - da_id_annual

    s1, s2, s3 = st.columns(3)
    s1.metric("DA-indexed", f"{save_da:,.0f} €", f"{100*(1-da_indexed_annual/flat_annual):.1f}%")
    s2.metric("DA-optimised", f"{save_daopt:,.0f} €", f"{100*(1-da_opt_annual/flat_annual):.1f}%")
    s3.metric("DA+ID-optimised", f"{save_eff:,.0f} €", f"{100*(1-da_id_annual/flat_annual):.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer ---------------------------------------------------------------
    st.markdown(
        """
        <div class="eon-footer">
            © 2025 E.ON SE · Internal Model Prototype
        </div>
        """,
        unsafe_allow_html=True
    )


# Run app
if __name__ == "__main__":
    main()
