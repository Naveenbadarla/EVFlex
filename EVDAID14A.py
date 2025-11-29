import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.title("🚗 EV Hourly Optimizer")

# ---- EV Energy ----
ev_annual_kwh = st.sidebar.number_input(
    "EV annual charging need (kWh/year)",
    min_value=0.0, value=2000.0, step=100.0
)

charger_power = st.sidebar.number_input(
    "Charger power (kW)",
    min_value=1.0, value=11.0, step=1.0
)

# ---- Charging frequency ----
freq_choice = st.sidebar.selectbox(
    "Charging frequency",
    [
        "Every day",
        "Every 2 days",
        "Weekdays only",
        "Weekends only",
        "Custom days per week"
    ]
)

custom_days = 7
if freq_choice == "Custom days per week":
    custom_days = st.sidebar.slider("Days per week", 1, 7, 3)

# Compute sessions per year
if freq_choice == "Every day":
    sessions_per_year = 365
elif freq_choice == "Every 2 days":
    sessions_per_year = 365 / 2
elif freq_choice == "Weekdays only":
    sessions_per_year = 5 * 52
elif freq_choice == "Weekends only":
    sessions_per_year = 2 * 52
else:
    sessions_per_year = custom_days * 52

# ---- Charging need per session ----
default_session_kwh = ev_annual_kwh / sessions_per_year

session_kwh = st.sidebar.number_input(
    "kWh to charge per session",
    min_value=0.1,
    value=round(default_session_kwh, 2),
    step=0.1,
    help="Energy that the EV needs each time it is plugged in."
)

# Arrival/departure
arrival_hour = st.sidebar.slider("Arrival hour", 0, 23, 18)
departure_hour = st.sidebar.slider("Departure hour", 0, 23, 7)

# ---- Energy prices ----
energy_flat = st.sidebar.number_input(
    "Retail energy price (€/kWh)",
    min_value=0.01, value=0.30, step=0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Day-Ahead prices (€/kWh)")

# Simple synthetic DA curve for 24 hours
default_da = [0.28,0.25,0.22,0.19,0.17,0.15,0.14,0.15,0.18,0.20,0.25,0.30,
              0.32,0.34,0.33,0.31,0.28,0.26,0.25,0.24,0.23,0.22,0.22,0.24]

# Editable slider curve
da_prices = []
for h in range(24):
    da_prices.append(
        st.sidebar.number_input(f"DA price {h:02d}:00", 0.01, 1.0, default_da[h], 0.01)
    )

# ---- Intraday optimisation ----
st.sidebar.markdown("### Intraday optimisation")
id_discount = st.sidebar.number_input(
    "ID price improvement (€/kWh)",
    min_value=0.0, value=0.02, step=0.01,
    help="How much ID can reduce DA price for flexible charging hours."
)

id_flex_share = st.sidebar.slider(
    "Fraction of charging benefiting from ID (0–1)", 0.0, 1.0, 0.5
)

# ---- Module 3 (grid fees) ----
st.sidebar.markdown("---")
st.sidebar.markdown("### Module 3 – Time-variable grid fee")

dso_name = st.sidebar.selectbox("Select DSO", list(MODULE3_PRESETS.keys()))
dso = MODULE3_PRESETS[dso_name]

nne_ht = dso["nne_ht"]
nne_st = dso["nne_st"]
nne_nt = dso["nne_nt"]
ht_hours = dso["ht_hours"]
nt_hours = dso["nt_hours"]

# ============================================================
# HOURLY COST ARRAYS
# ============================================================
nne = []
for h in range(24):
    if h in ht_hours:
        nne.append(nne_ht)
    elif h in nt_hours:
        nne.append(nne_nt)
    else:
        nne.append(nne_st)

da_energy = np.array(da_prices)
id_energy = np.maximum(0, da_energy - id_discount)

# ============================================================
# ALLOWED HOURS
# ============================================================
allowed = np.zeros(24, dtype=bool)
for h in range(24):
    if arrival_hour <= departure_hour:
        allowed[h] = (arrival_hour <= h < departure_hour)
    else:
        allowed[h] = (h >= arrival_hour or h < departure_hour)

# ============================================================
# OPTIMISATION FUNCTION
# ============================================================
def optimise_charging(cost_curve, allowed_hours, session_kwh, charger_power):
    remaining = session_kwh
    charge = np.zeros(24)

    # Sort cheapest allowed hours
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
# COST SCENARIO DEFINITIONS
# ============================================================

# 1) Baseline: retail energy + standard NNE
cost_baseline_curve = energy_flat + nne_st
baseline_energy_cost = session_kwh * energy_flat

charge_baseline = optimise_charging(
    [cost_baseline_curve]*24, allowed, session_kwh, charger_power
)
cost_baseline = np.sum(charge_baseline * (energy_flat + nne_st))

# 2) DA only
cost_da_curve = da_energy + nne_st
charge_da = optimise_charging(cost_da_curve, allowed, session_kwh, charger_power)
cost_da = np.sum(charge_da * (da_energy + nne_st))

# 3) Module 3 only
cost_m3_curve = energy_flat + np.array(nne)
charge_m3 = optimise_charging(cost_m3_curve, allowed, session_kwh, charger_power)
cost_m3 = np.sum(charge_m3 * cost_m3_curve)

# 4) DA + Module 3
cost_da_m3_curve = da_energy + np.array(nne)
charge_da_m3 = optimise_charging(cost_da_m3_curve, allowed, session_kwh, charger_power)
cost_da_m3 = np.sum(charge_da_m3 * cost_da_m3_curve)

# 5) DA + ID + Module 3 (full optimisation)
cost_full_curve = id_energy + np.array(nne)
charge_full = optimise_charging(cost_full_curve, allowed, session_kwh, charger_power)
cost_full = np.sum(charge_full * cost_full_curve)

# Multiply by number of sessions per year
annual_baseline = cost_baseline * sessions_per_year
annual_da = cost_da * sessions_per_year
annual_m3 = cost_m3 * sessions_per_year
annual_da_m3 = cost_da_m3 * sessions_per_year
annual_full = cost_full * sessions_per_year

# ============================================================
# OUTPUT TABLE
# ============================================================

df = pd.DataFrame([
    ["Baseline", annual_baseline],
    ["DA only", annual_da],
    ["Module 3 only", annual_m3],
    ["DA + Module 3", annual_da_m3],
    ["DA + ID + Module 3 (Full)", annual_full],
], columns=["Scenario", "Annual Cost (€)"])

best = df.loc[df["Annual Cost (€)"].idxmin()]

# ============================================================
# MAIN PAGE
# ============================================================

st.title("🚗 EV Hourly Charging Optimizer (DA + ID + Module 3)")

st.write(f"""
**Charging sessions per year:** {sessions_per_year:.0f}  
**kWh per session:** {session_kwh:.2f}  
**DSO:** {dso_name}  
""")

st.subheader("Annual Cost Comparison (€)")
st.dataframe(df.style.format({"Annual Cost (€)": "{:.0f}"}), use_container_width=True)

st.success(
    f"**Best Option:** {best['Scenario']} – ~{best['Annual Cost (€)']:.0f} €/year"
)

# ============================================================
# CHARGING GRAPH
# ============================================================
chart_df = pd.DataFrame({
    "Hour": list(range(24)),
    "Baseline": charge_baseline,
    "DA only": charge_da,
    "Module 3": charge_m3,
    "DA + M3": charge_da_m3,
    "Full": charge_full,
})

st.subheader("Charging Schedule per Day (kWh per hour)")
chart_df = chart_df.melt("Hour", var_name="Scenario", value_name="kWh")
chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(x="Hour:O", y="kWh:Q", color="Scenario:N")
    .properties(height=300)
)
st.altair_chart(chart, use_container_width=True)
