import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="EV Hourly Optimizer", layout="wide")

# ============================================================
# DSO PRESETS (HT/ST/NT grid-fee hours + NNE prices)
# ============================================================
MODULE3_PRESETS = {
    "Westnetz": {
        "nne_ht": 0.1565,
        "nne_st": 0.0953,
        "nne_nt": 0.0095,
        "ht_hours": [15,16,17,18,19,20],
        "nt_hours": [0,1,2,3,4,5,6,13,14,21,22,23],
    },
    "Avacon": {
        "nne_ht": 0.0841,
        "nne_st": 0.0604,
        "nne_nt": 0.0060,
        "ht_hours": [16,17,18,19],
        "nt_hours": [0,1,2,3,4,5,6,22,23],
    },
    "Netze BW": {
        "nne_ht": 0.1320,
        "nne_st": 0.0757,
        "nne_nt": 0.0303,
        "ht_hours": [17,18,19,20],
        "nt_hours": [0,1,2,3,4,5,6,22,23],
    },
}

DATA_DIR = Path("data")

# ============================================================
# UTILITIES FOR PRICES
# ============================================================

def load_clean_price_csv(path: Path) -> pd.Series:
    """
    Load a 'datetime, price_eur_mwh' CSV and return a pandas Series
    indexed by datetime, values in €/kWh.
    """
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise ValueError(f"'datetime' column not found in {path}")
    # Try to detect price column
    price_col = None
    for c in df.columns:
        if c.lower().startswith("price"):
            price_col = c
            break
    if price_col is None:
        raise ValueError(f"No price column found in {path}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    # Convert €/MWh → €/kWh
    df["price_eur_kwh"] = df[price_col] / 1000.0
    return df.set_index("datetime")["price_eur_kwh"]

def hourly_profile_from_series(series: pd.Series) -> np.ndarray:
    """
    Take a datetime-indexed series (hourly or more frequent),
    return a 24-element numpy array with mean price per hour-of-day.
    """
    s = series.copy()
    # If not hourly, resample to hourly mean
    s = s.resample("1H").mean()
    df = s.to_frame("price")
    df["hour"] = df.index.hour
    hourly = df.groupby("hour")["price"].mean()
    # Ensure we have all 24 hours
    hourly = hourly.reindex(range(24), fill_value=hourly.mean())
    return hourly.values

def synthetic_da_id_profiles() -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback synthetic 24h DA and ID price profiles in €/kWh.
    DA: with realistic shape.
    ID: slightly lower than DA in some hours.
    """
    da = np.array([
        0.28,0.25,0.22,0.19,0.17,0.15,0.14,0.15,0.18,0.20,0.25,0.30,
        0.32,0.34,0.33,0.31,0.28,0.26,0.25,0.24,0.23,0.22,0.22,0.24
    ])
    # ID a bit lower where DA is high
    id_prices = da - np.clip(da - da.min(), 0, 0.03)
    id_prices = np.maximum(id_prices, 0.0)
    return da, id_prices

def get_da_id_profiles(price_source, year, da_file, id_file):
    """
    Returns (da_24h, id_24h) numpy arrays in €/kWh.
    """
    if price_source == "Synthetic example":
        return synthetic_da_id_profiles()

    if price_source == "Built-in historical (data/…csv)":
        da_path = DATA_DIR / f"da_{year}.csv"
        id_path = DATA_DIR / f"id_{year}.csv"
        if not da_path.exists() or not id_path.exists():
            st.warning(f"DA/ID files for year {year} not found in data/. Falling back to synthetic.")
            return synthetic_da_id_profiles()
        da_series = load_clean_price_csv(da_path)
        id_series = load_clean_price_csv(id_path)
        da_24 = hourly_profile_from_series(da_series)
        id_24 = hourly_profile_from_series(id_series)
        return da_24, id_24

    if price_source == "Upload DA+ID CSVs":
        if da_file is None or id_file is None:
            st.warning("Please upload both DA and ID CSV files. Using synthetic prices until then.")
            return synthetic_da_id_profiles()
        da_series = load_clean_price_csv(da_file)
        id_series = load_clean_price_csv(id_file)
        da_24 = hourly_profile_from_series(da_series)
        id_24 = hourly_profile_from_series(id_series)
        return da_24, id_24

    # fallback
    return synthetic_da_id_profiles()

# ============================================================
# OPTIMISATION
# ============================================================

def build_nne_curve(dso):
    nne = []
    for h in range(24):
        if h in dso["ht_hours"]:
            nne.append(dso["nne_ht"])
        elif h in dso["nt_hours"]:
            nne.append(dso["nne_nt"])
        else:
            nne.append(dso["nne_st"])
    return np.array(nne)

def allowed_hours_array(arrival_hour, departure_hour):
    allowed = np.zeros(24, dtype=bool)
    for h in range(24):
        if arrival_hour <= departure_hour:
            allowed[h] = (arrival_hour <= h < departure_hour)
        else:
            allowed[h] = (h >= arrival_hour or h < departure_hour)
    return allowed

def optimise_charging(cost_curve: np.ndarray,
                      allowed_hours: np.ndarray,
                      session_kwh: float,
                      charger_power: float) -> np.ndarray:
    if session_kwh <= 0:
        return np.zeros(24)
    remaining = session_kwh
    charge = np.zeros(24)
    cheap_hours = sorted(
        [h for h in range(24) if allowed_hours[h]],
        key=lambda h: cost_curve[h]
    )
    for h in cheap_hours:
        if remaining <= 0:
            break
        max_hourly = charger_power
        charge[h] = min(max_hourly, remaining)
        remaining -= charge[h]
    return charge

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.title("🚗 EV Hourly Optimizer")

# EV energy + frequency
ev_annual_kwh = st.sidebar.number_input(
    "EV annual charging need (kWh/year)",
    min_value=0.0, value=2000.0, step=100.0
)

charger_power = st.sidebar.number_input(
    "Charger power (kW)",
    min_value=1.0, value=11.0, step=1.0
)

freq_choice = st.sidebar.selectbox(
    "Charging frequency",
    [
        "Every day",
        "Every 2 days",
        "Weekdays only",
        "Weekends only",
        "Custom days per week",
    ]
)

custom_days = 7
if freq_choice == "Custom days per week":
    custom_days = st.sidebar.slider("Days per week", 1, 7, 3)

if freq_choice == "Every day":
    sessions_per_year = 365
elif freq_choice == "Every 2 days":
    sessions_per_year = 365 / 2.0
elif freq_choice == "Weekdays only":
    sessions_per_year = 5 * 52
elif freq_choice == "Weekends only":
    sessions_per_year = 2 * 52
else:
    sessions_per_year = custom_days * 52

default_session_kwh = ev_annual_kwh / sessions_per_year if sessions_per_year > 0 else 0.0

session_kwh = st.sidebar.number_input(
    "kWh to charge per session",
    min_value=0.0,
    value=float(round(default_session_kwh, 2)),
    step=0.1,
    help="Energy the EV needs each time it is plugged in."
)

arrival_hour = st.sidebar.slider("Arrival hour", 0, 23, 18)
departure_hour = st.sidebar.slider("Departure hour", 0, 23, 7)

energy_flat = st.sidebar.number_input(
    "Retail energy price (€/kWh, flat tariff)",
    min_value=0.01, value=0.30, step=0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown("### DA & ID price source")

price_source = st.sidebar.radio(
    "Select price source",
    [
        "Synthetic example",
        "Built-in historical (data/…csv)",
        "Upload DA+ID CSVs",
    ]
)

year = None
da_file = None
id_file = None

if price_source == "Built-in historical (data/…csv)":
    year = st.sidebar.selectbox("Historical year", [2022, 2023, 2024])
elif price_source == "Upload DA+ID CSVs":
    da_file = st.sidebar.file_uploader("Upload DA price CSV", type=["csv"])
    id_file = st.sidebar.file_uploader("Upload ID price CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Module 3 – Time-variable grid fee")

dso_name = st.sidebar.selectbox("Select DSO", list(MODULE3_PRESETS.keys()))
dso = MODULE3_PRESETS[dso_name]

nne_curve = build_nne_curve(dso)

st.sidebar.markdown("### ID flexibility")
id_flex_share = st.sidebar.slider(
    "Fraction of charging benefiting from ID (0–1)",
    0.0, 1.0, 0.5
)

# ============================================================
# LOAD PRICES (24h DA & ID)
# ============================================================

da_24, id_24 = get_da_id_profiles(price_source, year, da_file, id_file)

# ID improvement: where ID < DA
id_discount = np.clip(da_24 - id_24, 0, None)
# Only some fraction of kWh get full ID discount
effective_da_id = da_24 - id_discount * id_flex_share

# ============================================================
# ALLOWED HOURS & FEASIBILITY
# ============================================================

allowed = allowed_hours_array(arrival_hour, departure_hour)
available_hours = allowed.sum()
max_possible_kwh = available_hours * charger_power

if session_kwh > max_possible_kwh + 1e-6:
    st.sidebar.warning(
        f"Requested {session_kwh:.1f} kWh per session but max possible with "
        f"{charger_power} kW & {available_hours}h window is {max_possible_kwh:.1f} kWh. "
        "Charging will be capped at the physical maximum."
    )
    session_kwh = max_possible_kwh

# ============================================================
# COST CURVES PER SCENARIO (€/kWh)
# ============================================================

# 1) Baseline: flat energy + standard NNE
baseline_curve = np.full(24, energy_flat + dso["nne_st"])

# 2) DA only: DA energy + standard NNE
da_only_curve = da_24 + dso["nne_st"]

# 3) Module 3 only: flat energy + NNE curve
m3_curve = energy_flat + nne_curve

# 4) DA + Module 3
da_m3_curve = da_24 + nne_curve

# 5) DA + ID + Module 3
full_curve = effective_da_id + nne_curve

# ============================================================
# OPTIMISE DAILY CHARGING PROFILES
# ============================================================

charge_baseline = optimise_charging(baseline_curve, allowed, session_kwh, charger_power)
charge_da_only = optimise_charging(da_only_curve, allowed, session_kwh, charger_power)
charge_m3 = optimise_charging(m3_curve, allowed, session_kwh, charger_power)
charge_da_m3 = optimise_charging(da_m3_curve, allowed, session_kwh, charger_power)
charge_full = optimise_charging(full_curve, allowed, session_kwh, charger_power)

# ============================================================
# COST PER SESSION & PER YEAR
# ============================================================

def session_cost(charge, curve):
    return float(np.sum(charge * curve))

session_cost_baseline = session_cost(charge_baseline, baseline_curve)
session_cost_da_only = session_cost(charge_da_only, da_only_curve)
session_cost_m3 = session_cost(charge_m3, m3_curve)
session_cost_da_m3 = session_cost(charge_da_m3, da_m3_curve)
session_cost_full = session_cost(charge_full, full_curve)

annual_baseline = session_cost_baseline * sessions_per_year
annual_da = session_cost_da_only * sessions_per_year
annual_m3 = session_cost_m3 * sessions_per_year
annual_da_m3 = session_cost_da_m3 * sessions_per_year
annual_full = session_cost_full * sessions_per_year

# ============================================================
# MAIN PAGE OUTPUT
# ============================================================

st.title("🚗 EV Hourly Charging Optimizer (DA + ID + Module 3)")

st.write(
    f"**Sessions per year:** {sessions_per_year:.1f}  \n"
    f"**kWh per session:** {session_kwh:.2f}  \n"
    f"**Charger power:** {charger_power:.1f} kW  \n"
    f"**Arrival:** {arrival_hour:02d}:00  –  **Departure:** {departure_hour:02d}:00  \n"
    f"**Price source:** {price_source}  \n"
    f"**DSO:** {dso_name}"
)

df = pd.DataFrame([
    ["Baseline (flat energy + flat NNE)", annual_baseline],
    ["DA only", annual_da],
    ["Module 3 only", annual_m3],
    ["DA + Module 3", annual_da_m3],
    ["DA + ID + Module 3 (Full)", annual_full],
], columns=["Scenario", "Annual Cost (€)"])

best = df.loc[df["Annual Cost (€)"].idxmin()]

st.subheader("Annual Cost Comparison")
st.dataframe(df.style.format({"Annual Cost (€)": "{:.0f}"}), use_container_width=True)

st.success(
    f"**Best option:** {best['Scenario']} – ~{best['Annual Cost (€)']:.0f} €/year"
)

# ============================================================
# CHARGING SCHEDULE VISUALISATION
# ============================================================

chart_df = pd.DataFrame({
    "Hour": list(range(24)),
    "Baseline": charge_baseline,
    "DA only": charge_da_only,
    "Module 3": charge_m3,
    "DA + M3": charge_da_m3,
    "Full (DA+ID+M3)": charge_full,
})

st.subheader("Charging Schedule per Day (kWh per hour)")

melted = chart_df.melt("Hour", var_name="Scenario", value_name="kWh")
chart = (
    alt.Chart(melted)
    .mark_bar()
    .encode(
        x=alt.X("Hour:O"),
        y=alt.Y("kWh:Q"),
        color="Scenario:N",
        tooltip=["Hour", "Scenario", "kWh"]
    )
    .properties(height=300)
)

st.altair_chart(chart, use_container_width=True)

st.markdown(
    """
    **Notes**  
    * DA/ID prices come from historical data (if provided) or a synthetic profile.  
    * Module 3 changes only the NNE (grid-fee) component, not the retail energy price.  
    * This app ignores PV and home battery on purpose – it focuses only on EV charging cost optimisation.
    """
)
