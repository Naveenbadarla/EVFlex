import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime
# ---------------------- FULL E.ON THEME STYLING ---------------------------

eon_logo = "https://upload.wikimedia.org/wikipedia/commons/1/17/E.ON_Logo_2021.png"

st.markdown(
    f"""
    <style>
        /* Global font + reset */
        html, body, [class*="css"]  {{
            font-family: "Source Sans Pro", sans-serif;
        }}

        /* -------------------------- HEADER BAR -------------------------- */
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

        /* Mobile responsiveness */
        @media (max-width: 600px) {{
            .eon-header {{
                flex-direction: column;
                text-align: center;
            }}

            .eon-header img {{
                margin-bottom: 12px;
            }}
        }}

        /* -------------------------- SIDEBAR -------------------------- */
        section[data-testid="stSidebar"] {{
            background-color: #FDECEC !important;
        }}

        section[data-testid="stSidebar"] .css-1d391kg {{
            background-color: #FDECEC;
        }}

        /* Sidebar title text */
        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {{
            color: #E2000F !important;
        }}

        /* Sidebar labels */
        .css-1cpxqw2, .css-1p4jv9z, label {{
            color: #B00000 !important;
            font-weight: 600 !important;
        }}

        /* -------------------------- BUTTONS -------------------------- */
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
            color: white;
        }}

        /* -------------------------- SLIDERS -------------------------- */
        .stSlider > div {{
            background-color: #E2000F20;
            border-radius: 8px;
            padding: 8px;
        }}

        .stSlider .css-1aumxhk, .stSlider .css-1ptx7rd {{
            color: #E2000F !important;
        }}

        div[data-baseweb="slider"] > div > div {{
            background-color: #E2000F !important; /* slider color */
        }}

        /* -------------------------- NUMERIC INPUTS -------------------------- */
        input[type="number"], input[type="text"] {{
            border: 1.5px solid #E2000F50 !important;
            border-radius: 5px !important;
        }}

        /* -------------------------- SELECTBOX -------------------------- */
        .stSelectbox div[data-baseweb="select"] > div {{
            border: 1.5px solid #E2000F50 !important;
            border-radius: 5px !important;
        }}

        /* -------------------------- CARDS -------------------------- */
        .eon-card {{
            padding: 20px;
            margin-top: 15px;
            background-color: white;
            border-radius: 10px;
            border-left: 5px solid #E2000F;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }}

        /* -------------------------- FOOTER -------------------------- */
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


# =====================================================================================
# Synthetic DAILY DA + ID generator (used only when user does NOT upload files)
# =====================================================================================
def get_synthetic_daily_price_profiles():
    hours = np.arange(24)
    base = 70
    morning_peak = 15 * np.sin((hours - 7) / 24 * 2 * np.pi) ** 2
    evening_peak = 35 * np.sin((hours - 18) / 24 * 2 * np.pi) ** 4
    da_hourly = base + morning_peak + evening_peak  # €/MWh

    rng = np.random.default_rng(42)
    da_quarter = np.repeat(da_hourly, 4)
    noise = rng.normal(0, 6, size=96)
    id_quarter = da_quarter + noise

    return da_hourly, id_quarter


# =====================================================================================
# Charging days selector (every day, weekdays, custom...)
# =====================================================================================
DAY_NAME_TO_IDX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def compute_charging_days(pattern, num_days, custom_weekdays=None, sessions_per_week=None):
    if custom_weekdays is None:
        custom_weekdays = []

    if pattern == "Every day":
        return list(range(num_days))

    if pattern == "Every other day":
        return list(range(0, num_days, 2))

    if pattern == "Weekdays only (Mon–Fri)":
        return [d for d in range(num_days) if (d % 7) in {0, 1, 2, 3, 4}]

    if pattern == "Weekends only (Sat–Sun)":
        return [d for d in range(num_days) if (d % 7) in {5, 6}]

    if pattern == "Custom weekdays":
        dows = {DAY_NAME_TO_IDX[x] for x in custom_weekdays}
        return [d for d in range(num_days) if (d % 7) in dows]

    if pattern == "Custom: X sessions per week":
        if not sessions_per_week:
            return []
        sessions_per_week = int(sessions_per_week)
        days = []
        num_weeks = num_days // 7
        for w in range(num_weeks):
            for s in range(sessions_per_week):
                day_in_week = int(np.floor(s * 7 / sessions_per_week))
                d = w * 7 + day_in_week
                if d < num_days:
                    days.append(d)
        return days

    return list(range(num_days))


# =====================================================================================
# Build daily 15-minute charging windows
# =====================================================================================
def build_available_quarters(arrival_time_h, departure_time_h):
    arrival_q = int(round(arrival_time_h * 4)) % 96
    departure_q = int(round(departure_time_h * 4)) % 96

    if arrival_q == departure_q:
        return list(range(96)), arrival_q, departure_q

    if arrival_q < departure_q:
        return list(range(arrival_q, departure_q)), arrival_q, departure_q
    else:
        return list(range(arrival_q, 96)) + list(range(0, departure_q)), arrival_q, departure_q


# =====================================================================================
# Tariff application
# =====================================================================================
def apply_tariffs(price_kwh, grid, taxes, vat):
    pre_vat = price_kwh + grid + taxes
    return pre_vat * (1 + vat / 100)


# =====================================================================================
# Charging cost models (baseline & optimised)
# =====================================================================================
def compute_da_indexed_baseline_daily(E_kwh, P_max, quarters, da_day, grid, taxes, vat):
    da_kwh = da_day / 1000.0
    da_q = np.repeat(da_kwh, 4)

    E = E_kwh
    cost = 0.0
    Emax_q = P_max * 0.25

    for q in quarters:
        if E <= 0:
            break
        e = min(Emax_q, E)
        unit = apply_tariffs(da_q[q], grid, taxes, vat)
        cost += e * unit
        E -= e

    if E > 0:
        raise ValueError("Charging window too short to deliver required energy.")

    return cost


def compute_optimised_daily_cost(E_kwh, P_max, quarters, price_q_day, grid, taxes, vat):
    price_kwh = price_q_day / 1000.0
    Emax_q = P_max * 0.25

    max_possible = len(quarters) * Emax_q
    if max_possible < E_kwh:
        raise ValueError("Charging window too short for that much energy.")

    sorted_q = sorted(quarters, key=lambda q: price_kwh[q])

    E = E_kwh
    cost = 0

    for q in sorted_q:
        if E <= 0:
            break
        e = min(Emax_q, E)
        unit = apply_tariffs(price_kwh[q], grid, taxes, vat)
        cost += e * unit
        E -= e

    return cost


# =====================================================================================
# File loading helper
# =====================================================================================
def load_price_series_from_csv(uploaded, multiple, label):
    df = pd.read_csv(uploaded)
    if "price" in df.columns:
        series = df["price"].values
    else:
        series = df.select_dtypes(include="number").iloc[:, 0].values

    if len(series) % multiple != 0:
        st.error(f"{label}: Length {len(series)} is not divisible by {multiple}.")
        return None, 0

    return series.astype(float), len(series) // multiple


# =====================================================================================
# Streamlit App
# =====================================================================================
def main():
    st.title("EV Charging Cost Model with Full-Year DA + ID + Plotly Zoom Viewer")

    # ==========================================================================
    # Sidebar: Inputs
    # ==========================================================================
    st.sidebar.header("EV Inputs")
    energy_kwh = st.sidebar.number_input("Energy per session (kWh)", 1.0, 200.0, 40.0)
    max_power_kw = st.sidebar.number_input("Max charging power (kW)", 1.0, 50.0, 11.0)

    st.sidebar.header("Arrival/Departure")
    arrival_h = st.sidebar.slider("Arrival hour", 0.0, 24.0, 18.0, 0.25)
    departure_h = st.sidebar.slider("Departure hour", 0.0, 24.0, 7.0, 0.25)

    st.sidebar.header("Charging Frequency")
    freq = st.sidebar.selectbox(
        "Pattern",
        ["Every day", "Every other day", "Weekdays only (Mon–Fri)",
         "Weekends only (Sat–Sun)", "Custom weekdays", "Custom: X sessions per week"]
    )

    custom_days = []
    sessions_week = None
    if freq == "Custom weekdays":
        custom_days = st.sidebar.multiselect("Choose days", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    if freq == "Custom: X sessions per week":
        sessions_week = st.sidebar.number_input("Sessions per week", 1, 14, 3)

    # Tariffs
    st.sidebar.header("Tariffs")
    flat_price = st.sidebar.number_input("Flat retail price (€/kWh)", 0.0, 1.0, 0.35)
    grid = st.sidebar.number_input("Grid network charges (€/kWh)", 0.0, 1.0, 0.11)
    taxes = st.sidebar.number_input("Taxes & levies (€/kWh)", 0.0, 1.0, 0.05)
    vat = st.sidebar.number_input("VAT (%)", 0.0, 30.0, 19.0)

    # Upload DA/ID
    st.sidebar.header("Upload Market Data")
    da_file = st.sidebar.file_uploader("DA hourly prices CSV", type="csv")
    id_file = st.sidebar.file_uploader("ID 15-min prices CSV", type="csv")

    # ==========================================================================
    # Load or generate yearly price arrays
    # ==========================================================================
    da_daily_syn, id_daily_syn = get_synthetic_daily_price_profiles()

    if da_file is not None:
        da_series, num_da_days = load_price_series_from_csv(da_file, 24, "DA")
    else:
        da_series = np.tile(da_daily_syn, 365)
        num_da_days = 365

    if id_file is not None:
        id_series, num_id_days = load_price_series_from_csv(id_file, 96, "ID")
    else:
        id_series = np.tile(id_daily_syn, 365)
        num_id_days = 365

    num_days = min(num_da_days, num_id_days)
    da_year = da_series[: num_days * 24]
    id_year = id_series[: num_days * 96]

    st.sidebar.success(f"{num_days} valid days of price data loaded.")

    # ==========================================================================
    # Plotly full-year zoomable chart
    # ==========================================================================
    st.subheader("📈 Full-Year DA + ID Prices (Zoom & Pan)")

    timestamps = pd.date_range(start="2023-01-01", periods=num_days * 96, freq="15min")

    da_quarter_year = np.repeat(da_year, 4)[: num_days * 96]
    id_quarter_year = id_year[: num_days * 96]
    effective = np.minimum(da_quarter_year, id_quarter_year)

    df_plot = pd.DataFrame({
        "timestamp": timestamps,
        "DA (€/MWh)": da_quarter_year,
        "ID (€/MWh)": id_quarter_year,
        "Effective min(DA,ID)": effective,
    })

    fig = px.line(
        df_plot,
        x="timestamp",
        y=["DA (€/MWh)", "ID (€/MWh)", "Effective min(DA,ID)"],
        title="Full-Year DA/ID Price Curve",
    )
    fig.update_layout(
        xaxis_rangeslider_visible=True,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # Determine charging windows
    # ==========================================================================
    quarters, _, _ = build_available_quarters(arrival_h, departure_h)
    charging_days = compute_charging_days(freq, num_days, custom_days, sessions_week)
    days_per_year = len(charging_days)

    # ==========================================================================
    # Compute costs over full year
    # ==========================================================================
    flat_annual = flat_price * energy_kwh * days_per_year
    da_index_annual = 0
    da_opt_annual = 0
    da_id_annual = 0

    try:
        for d in charging_days:
            da_day = da_year[d * 24:(d + 1) * 24]
            id_day = id_year[d * 96:(d + 1) * 96]
            da_q_day = np.repeat(da_day, 4)
            effective_day = np.minimum(da_q_day, id_day)

            # Scenario 2
            da_index_annual += compute_da_indexed_baseline_daily(
                energy_kwh, max_power_kw, quarters, da_day, grid, taxes, vat
            )

            # Scenario 3
            da_opt_annual += compute_optimised_daily_cost(
                energy_kwh, max_power_kw, quarters, da_q_day, grid, taxes, vat
            )

            # Scenario 4
            da_id_annual += compute_optimised_daily_cost(
                energy_kwh, max_power_kw, quarters, effective_day, grid, taxes, vat
            )

    except ValueError as e:
        st.error(str(e))
        return

    # ==========================================================================
    # Display results
    # ==========================================================================
    st.subheader("Annual Cost Comparison")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.metric("1) Flat retail", f"{flat_annual:,.0f} €")
    col2.metric("2) DA-indexed", f"{da_index_annual:,.0f} €")
    col3.metric("3) DA-optimised", f"{da_opt_annual:,.0f} €")
    col4.metric("4) DA+ID-optimised", f"{da_id_annual:,.0f} €")

    st.subheader("Savings vs Flat Retail")
    def savings(new, base):
        return base - new, 100 * (1 - new / base)

    s2, p2 = savings(da_index_annual, flat_annual)
    s3, p3 = savings(da_opt_annual, flat_annual)
    s4, p4 = savings(da_id_annual, flat_annual)

    c1, c2, c3 = st.columns(3)
    c1.metric("DA-indexed", f"{s2:,.0f} €", f"{p2:.1f}%")
    c2.metric("DA-optimised", f"{s3:,.0f} €", f"{p3:.1f}%")
    c3.metric("DA+ID-optimised", f"{s4:,.0f} €", f"{p4:.1f}%")

    st.success("Model complete. Zoomable full-year price chart is active.")

if __name__ == "__main__":
    main()
