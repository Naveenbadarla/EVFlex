import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

# ============================================================
# BASIC CONFIG
# ============================================================

st.set_page_config(
    page_title="EV Flex – DA/ID Optimiser",
    layout="wide",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# HELPER FUNCTIONS – PRICES
# ============================================================

def generate_synthetic_full_year_prices(year: int = 2023):
    """
    Generate synthetic EPEX-like DA & ID prices for a full year (15-min resolution).
    Saves da_{year}_full.csv and id_{year}_full.csv in data/.
    """
    start = pd.Timestamp(f"{year}-01-01 00:00")
    end = pd.Timestamp(f"{year}-12-31 23:45")
    times = pd.date_range(start, end, freq="15min")

    rng = np.random.default_rng(2023)

    da_prices = []
    id_prices = []

    for ts in times:
        # seasonal factor: winter high, summer low
        doy = ts.timetuple().tm_yday
        seasonal = 0.5 * (1 + np.cos((doy - 190) / 365 * 2 * np.pi))  # 0..1
        base_level = 60 + 120 * seasonal  # 60–180 €/MWh

        h = ts.hour + ts.minute / 60

        # Intraday shape for DA
        if 0 <= h < 5:
            da = base_level + rng.normal(10, 10)
        elif 5 <= h < 9:
            da = base_level + 60 + rng.normal(0, 15)  # morning ramp
        elif 9 <= h < 14:
            da = base_level - 30 + rng.normal(0, 20)  # PV dip
        elif 14 <= h < 17:
            da = base_level + rng.normal(0, 15)
        elif 17 <= h < 21:
            da = base_level + 80 + rng.normal(0, 25)  # evening peak
        else:
            da = base_level + 20 + rng.normal(0, 15)

        # winter spikes
        if seasonal > 0.8 and rng.random() < 0.01:
            da += rng.uniform(150, 300)

        da = float(np.clip(da, -50, 500))

        # Intraday: DA + spread
        if 0 <= h < 5:
            spread = rng.normal(-15, 7)
        elif 5 <= h < 9:
            spread = rng.normal(-5, 10)
        elif 9 <= h < 14:
            spread = rng.normal(-30, 15)
        elif 14 <= h < 17:
            spread = rng.normal(-10, 10)
        elif 17 <= h < 21:
            spread = rng.normal(5, 20)   # can be more expensive
        else:
            spread = rng.normal(-5, 10)

        id_p = da + spread

        # occasional deep negative or spikes
        if rng.random() < 0.02:
            if rng.random() < 0.6:
                id_p -= rng.uniform(30, 120)
            else:
                id_p += rng.uniform(150, 350)

        id_p = float(np.clip(id_p, -150, 800))

        da_prices.append(da)
        id_prices.append(id_p)

    da_df = pd.DataFrame({"datetime": times, "price_eur_mwh": da_prices})
    id_df = pd.DataFrame({"datetime": times, "price_eur_mwh": id_prices})

    da_path = DATA_DIR / "da_2023_full.csv"
    id_path = DATA_DIR / "id_2023_full.csv"
    da_df.to_csv(da_path, index=False)
    id_df.to_csv(id_path, index=False)


@st.cache_data
def load_full_year_prices(year: int = 2023):
    da_path = DATA_DIR / f"da_{year}_full.csv"
    id_path = DATA_DIR / f"id_{year}_full.csv"

    if not da_path.exists() or not id_path.exists():
        generate_synthetic_full_year_prices(year)

    da_full = pd.read_csv(da_path, parse_dates=["datetime"])
    id_full = pd.read_csv(id_path, parse_dates=["datetime"])

    da_full["price_eur_kwh"] = da_full["price_eur_mwh"] / 1000.0
    id_full["price_eur_kwh"] = id_full["price_eur_mwh"] / 1000.0

    return da_full, id_full


# ============================================================
# HELPER FUNCTIONS – EV / OPTIMISATION
# ============================================================

def allowed_slots_96(arrival_hour: int, departure_hour: int) -> np.ndarray:
    """
    Boolean mask of allowed charging slots (96) for arrival -> next-day departure.
    """
    arr_idx = arrival_hour * 4
    dep_idx = departure_hour * 4
    allowed = np.zeros(96, dtype=bool)

    if dep_idx > arr_idx:
        allowed[arr_idx:dep_idx] = True
    else:
        allowed[arr_idx:] = True
        allowed[:dep_idx] = True
    return allowed


def optimise_schedule(prices: np.ndarray,
                      allowed: np.ndarray,
                      need_kwh: float,
                      charger_power: float) -> np.ndarray:
    """
    Greedy: fill cheapest allowed slots first.
    """
    sched = np.zeros(96)
    remaining = need_kwh
    order = np.argsort(prices)

    for i in order:
        if not allowed[i]:
            continue
        if remaining <= 0:
            break
        take = min(charger_power, remaining)
        sched[i] = take
        remaining -= take

    return sched


def baseline_schedule(arrival_hour: int,
                      need_kwh: float,
                      charger_power: float) -> np.ndarray:
    """
    Baseline behaviour: start charging immediately at arrival time,
    fill forward, then wrap after midnight if needed.
    """
    sched = np.zeros(96)
    remaining = need_kwh
    start = arrival_hour * 4

    # first from arrival to midnight
    idx = start
    while remaining > 0 and idx < 96:
        take = min(charger_power, remaining)
        sched[idx] = take
        remaining -= take
        idx += 1

    # then from midnight until before arrival
    idx = 0
    while remaining > 0 and idx < start:
        take = min(charger_power, remaining)
        sched[idx] = take
        remaining -= take
        idx += 1

    return sched


def soc_to_energy(battery_kwh: float, soc_from: float, soc_to: float) -> float:
    return max(0.0, battery_kwh * (soc_to - soc_from))


# ============================================================
# DAILY EV OPTIMISER PAGE
# ============================================================

def page_daily_optimizer():
    st.title("🚗 Daily EV Smart Charging – 15-min Resolution")

    da_full, id_full = load_full_year_prices(2023)

    # --- Sidebar inputs ---
    st.sidebar.header("EV & Charging Setup")
    battery_kwh = st.sidebar.number_input("Battery size (kWh)", 20.0, 120.0, 60.0, step=1.0)
    charger_power = st.sidebar.number_input("Charger power (kW)", 2.0, 22.0, 7.0, step=0.5)
    arrival_hour = st.sidebar.slider("Arrival time (hour of day)", 0, 23, 18)
    departure_hour = st.sidebar.slider("Departure time next day (hour)", 0, 23, 7)
    arrival_soc = st.sidebar.slider("Arrival SOC (%)", 0, 100, 30) / 100
    target_soc = st.sidebar.slider("Target SOC (%)", 20, 100, 90) / 100

    st.sidebar.header("Tariff / Price Settings")
    flat_retail = st.sidebar.number_input("Flat retail price (€/kWh)", 0.05, 1.0, 0.30, step=0.01)
    da_markup = st.sidebar.number_input("DA retail markup (€/kWh)", 0.0, 0.5, 0.12, step=0.01)

    st.sidebar.header("Select day from full-year prices")
    selected_date = st.sidebar.date_input(
        "Day to analyse",
        value=pd.to_datetime("2023-03-15").date(),
        min_value=da_full["datetime"].min().date(),
        max_value=da_full["datetime"].max().date()
    )

    # --- Prepare price series for this day ---
    mask_day = da_full["datetime"].dt.date == selected_date
    da_day = da_full.loc[mask_day].copy()
    id_day = id_full.loc[mask_day].copy()

    if len(da_day) != 96:
        st.error("Selected day does not have 96 quarter-hours in DA data.")
        return

    # convert to arrays (€/kWh)
    da_96 = da_day["price_eur_kwh"].values
    id_96 = id_day["price_eur_kwh"].values

    # Effective DA+ID curve (simple model: use ID if cheaper)
    id_discount = np.clip(da_96 - id_96, 0, None)
    effective_da_id_96 = da_96 - id_discount  # pure wholesale effect

    # DA retail curve
    da_retail_96 = da_96 + da_markup

    # --- EV energy need & allowed slots ---
    session_kwh = soc_to_energy(battery_kwh, arrival_soc, target_soc)
    allowed_96 = allowed_slots_96(arrival_hour, departure_hour)

    st.subheader("Session Inputs")
    st.write(f"Selected day: **{selected_date}**")
    st.write(f"Session energy need: **{session_kwh:.1f} kWh** "
             f"(from {arrival_soc*100:.0f}% to {target_soc*100:.0f}%)")
    st.write(f"Arrival: **{arrival_hour}:00**, Departure next day: **{departure_hour}:00**")

    # --- Schedules ---
    baseline_96 = baseline_schedule(arrival_hour, session_kwh, charger_power)
    cost_baseline = baseline_96.sum() * flat_retail

    cost_da_indexed = (baseline_96 * da_retail_96).sum()

    smart_da_96 = optimise_schedule(da_retail_96, allowed_96, session_kwh, charger_power)
    cost_smart_da = (smart_da_96 * da_retail_96).sum()

    retail_da_id_96 = effective_da_id_96 + da_markup
    smart_full_96 = optimise_schedule(retail_da_id_96, allowed_96, session_kwh, charger_power)
    cost_smart_full = (smart_full_96 * retail_da_id_96).sum()

    total_kwh = session_kwh

    # --- Result table ---
    st.subheader("💰 Cost Comparison for this Charging Session")
    df_res = pd.DataFrame({
        "Scenario": [
            "Baseline (flat retail)",
            "Retail DA-indexed",
            "Smart DA optimisation",
            "Smart DA+ID optimisation",
        ],
        "Cost (€)": [
            cost_baseline,
            cost_da_indexed,
            cost_smart_da,
            cost_smart_full,
        ],
        "Effective price (€/kWh)": [
            cost_baseline / total_kwh if total_kwh > 0 else np.nan,
            cost_da_indexed / total_kwh if total_kwh > 0 else np.nan,
            cost_smart_da / total_kwh if total_kwh > 0 else np.nan,
            cost_smart_full / total_kwh if total_kwh > 0 else np.nan,
        ]
    })

    st.dataframe(df_res.style.format({"Cost (€)": "{:.2f}", "Effective price (€/kWh)": "{:.3f}"}))

    best_row = df_res.iloc[df_res["Cost (€)"].idxmin()]
    st.success(
        f"Best scenario: **{best_row['Scenario']}** → "
        f"{best_row['Cost (€)']:.2f} € ({best_row['Effective price (€/kWh)']:.3f} €/kWh)"
    )

    # --- Charging pattern chart ---
    # build timeline starting at arrival and going 96 slots
    start_ts = pd.Timestamp(f"{selected_date} {arrival_hour:02d}:00")
    times_96 = [start_ts + pd.Timedelta(minutes=15 * i) for i in range(96)]

    charging_df = pd.DataFrame({
        "Datetime": times_96,
        "Baseline (kWh)": baseline_96,
        "DA-indexed (kWh)": baseline_96,  # same schedule, different price
        "Smart DA (kWh)": smart_da_96,
        "Smart DA+ID (kWh)": smart_full_96,
    })

    charging_long = charging_df.melt("Datetime", var_name="Scenario", value_name="kWh")

    st.subheader("🔌 Charging Pattern (Arrival → Departure Window)")
    charging_chart = (
        alt.Chart(charging_long)
        .mark_bar()
        .encode(
            x=alt.X("Datetime:T", axis=alt.Axis(format="%m-%d %H:%M", labelAngle=-45, tickCount=24)),
            y=alt.Y("kWh:Q", title="Charging (kWh)"),
            color="Scenario:N",
            tooltip=["Datetime", "Scenario", alt.Tooltip("kWh", format=".3f")],
        )
        .properties(height=300)
    )
    st.altair_chart(charging_chart, use_container_width=True)

    # ============================================================
    # EXPANDER 1 — DA vs ID PRICE CURVES
    # ============================================================
    with st.expander("📈 Prices: DA vs ID (15-minute resolution)"):

        price_df = pd.DataFrame({
            "Datetime": times_96,
            "DA (€/kWh)": da_96,
            "ID (€/kWh)": id_96,
        })
        price_long = price_df.melt("Datetime", var_name="Curve", value_name="€/kWh")

        price_chart = (
            alt.Chart(price_long)
            .mark_line()
            .encode(
                x=alt.X("Datetime:T",
                        axis=alt.Axis(format="%m-%d %H:%M", labelAngle=-45, tickCount=24)),
                y=alt.Y("€/kWh:Q", title="Price (€/kWh)"),
                color="Curve:N",
                tooltip=["Datetime", "Curve", alt.Tooltip("€/kWh", format=".4f")],
            )
            .properties(height=280)
        )
        st.altair_chart(price_chart, use_container_width=True)

    # ============================================================
    # EXPANDER 2 — CLEAR CHARGING COMPARISON (DA vs DA+ID)
    # ============================================================
    with st.expander("🔌 Clear Charging Comparison: Smart DA vs Smart DA+ID"):

        comp_rows = []
        for i in range(96):
            da_val = float(smart_da_96[i])
            full_val = float(smart_full_96[i])
            if da_val > 0 or full_val > 0:
                ts = times_96[i]
                label = ts.strftime("%m-%d %H:%M")
                comp_rows.append({
                    "Time": label,
                    "Smart DA (kWh)": da_val,
                    "Smart DA+ID (kWh)": full_val
                })

        if not comp_rows:
            st.info("No charging occurs during this session.")
        else:
            cmp_df = pd.DataFrame(comp_rows)
            cmp_long = cmp_df.melt("Time", var_name="Scenario", value_name="kWh")

            cmp_chart = (
                alt.Chart(cmp_long)
                .mark_bar()
                .encode(
                    x=alt.X("Time:O", title="Time of Day"),
                    y=alt.Y("kWh:Q", title="Charging (kWh)"),
                    color=alt.Color("Scenario:N",
                                    scale=alt.Scale(range=["#1f77b4", "#d62728"])),
                    tooltip=["Time", "Scenario", alt.Tooltip("kWh", format=".3f")],
                )
                .properties(height=320)
            )
            st.altair_chart(cmp_chart, use_container_width=True)

    # ============================================================
    # EXPANDER 3 — PRICES + CHARGING OVERLAY
    # ============================================================
    with st.expander("📉 Prices + Charging Overlay (All Scenarios)"):

        def overlay_chart(prices, charging, title):
            df = pd.DataFrame({
                "Datetime": times_96,
                "Price (€/kWh)": prices,
                "Charging (kWh)": charging,
            })

            price_line = alt.Chart(df).mark_line(color="orange").encode(
                x=alt.X("Datetime:T",
                        axis=alt.Axis(format="%m-%d %H:%M", labelAngle=-45, tickCount=24)),
                y=alt.Y("Price (€/kWh):Q"),
            )

            bars = alt.Chart(df).mark_bar(opacity=0.5, color="steelblue").encode(
                x="Datetime:T",
                y=alt.Y("Charging (kWh):Q"),
            )

            return (price_line + bars).properties(height=260, title=title)

        st.altair_chart(
            overlay_chart(da_retail_96, baseline_96, "Retail DA-indexed: Prices + Charging"),
            use_container_width=True,
        )
        st.altair_chart(
            overlay_chart(da_retail_96, smart_da_96, "Smart DA: Prices + Charging"),
            use_container_width=True,
        )
        st.altair_chart(
            overlay_chart(retail_da_id_96, smart_full_96, "Smart DA+ID: Prices + Charging"),
            use_container_width=True,
        )


# ============================================================
# FULL-YEAR SIMULATION PAGE
# ============================================================

def page_full_year_simulation():
    st.title("📅 Full-Year EV Smart Charging Simulation (365 days)")

    da_full, id_full = load_full_year_prices(2023)

    st.sidebar.header("Full-Year Simulation Settings")
    battery_kwh = st.sidebar.number_input("Battery size (kWh)", 20.0, 120.0, 60.0, step=1.0)
    charger_power = st.sidebar.number_input("Charger power (kW)", 2.0, 22.0, 7.0, step=0.5)
    arrival_mean = st.sidebar.slider("Average arrival hour", 0, 23, 18)
    departure_mean = st.sidebar.slider("Average departure hour", 0, 23, 7)
    target_soc = st.sidebar.slider("Target SOC (%)", 50, 100, 85) / 100
    sessions_per_week = st.sidebar.slider("Charging sessions per week", 1, 7, 4)
    flat_retail = st.sidebar.number_input("Flat retail price (€/kWh)", 0.05, 1.0, 0.30, step=0.01)
    da_markup = st.sidebar.number_input("DA markup (€/kWh)", 0.0, 0.5, 0.12, step=0.01)

    rng = np.random.default_rng(42)

    results = []

    for day_idx in range(365):
        da_day = da_full.iloc[day_idx * 96:(day_idx + 1) * 96].copy()
        id_day = id_full.iloc[day_idx * 96:(day_idx + 1) * 96].copy()

        # decide if this is a charging day
        if rng.random() > sessions_per_week / 7:
            continue

        # arrival & departure (some noise)
        arr = int(np.clip(rng.normal(arrival_mean, 1.5), 0, 23))
        dep = int(np.clip(rng.normal(departure_mean, 1.5), 0, 23))

        allowed = allowed_slots_96(arr, dep)

        # SOC & session energy
        arrival_soc = rng.uniform(0.20, 0.60)
        session_kwh = soc_to_energy(battery_kwh, arrival_soc, target_soc)
        if session_kwh <= 0:
            continue

        da_96 = da_day["price_eur_kwh"].values
        id_96 = id_day["price_eur_kwh"].values
        da_retail_96 = da_96 + da_markup

        # Baseline schedule
        baseline_96 = baseline_schedule(arr, session_kwh, charger_power)
        cost_baseline = baseline_96.sum() * flat_retail

        # DA-indexed
        cost_da_indexed = (baseline_96 * da_retail_96).sum()

        # Smart DA
        smart_da_96 = optimise_schedule(da_retail_96, allowed, session_kwh, charger_power)
        cost_smart_da = (smart_da_96 * da_retail_96).sum()

        # Smart DA+ID
        id_discount = np.clip(da_96 - id_96, 0, None)
        eff_da_id_96 = da_96 - id_discount + da_markup
        smart_full_96 = optimise_schedule(eff_da_id_96, allowed, session_kwh, charger_power)
        cost_smart_full = (smart_full_96 * eff_da_id_96).sum()

        results.append({
            "date": da_day["datetime"].iloc[0].date(),
            "baseline": cost_baseline,
            "da_indexed": cost_da_indexed,
            "smart_da": cost_smart_da,
            "smart_full": cost_smart_full,
            "kwh": session_kwh,
        })

    if not results:
        st.warning("No charging sessions simulated (check sessions per week and SOC settings).")
        return

    df_year = pd.DataFrame(results)
    total_kwh = df_year["kwh"].sum()

    annual = pd.DataFrame({
        "Scenario": ["Baseline (flat)", "Retail DA-indexed", "Smart DA", "Smart DA+ID"],
        "Annual cost (€)": [
            df_year["baseline"].sum(),
            df_year["da_indexed"].sum(),
            df_year["smart_da"].sum(),
            df_year["smart_full"].sum(),
        ],
    })
    annual["Effective price (€/kWh)"] = annual["Annual cost (€)"] / total_kwh

    st.subheader("📊 Annual Cost Comparison")
    st.dataframe(
        annual.style.format({"Annual cost (€)": "{:.0f}", "Effective price (€/kWh)": "{:.3f}"})
    )

    best_row = annual.iloc[annual["Annual cost (€)"].idxmin()]
    st.success(
        f"Best scenario: **{best_row['Scenario']}** → "
        f"{best_row['Annual cost (€)']:.0f} € / year "
        f"({best_row['Effective price (€/kWh)']:.3f} €/kWh)"
    )

    # simple bar chart
    chart = (
        alt.Chart(annual)
        .mark_bar()
        .encode(
            x="Scenario:N",
            y=alt.Y("Annual cost (€):Q"),
            tooltip=["Scenario", alt.Tooltip("Annual cost (€)", format=".0f"),
                     alt.Tooltip("Effective price (€/kWh)", format=".3f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


# ============================================================
# PRICE MANAGER PAGE (simple)
# ============================================================

def page_price_manager():
    st.title("💾 Price Manager – Import & Inspect DA/ID Data")

    st.write(
        "This is a simplified price manager. "
        "You can upload custom DA / ID CSV files (full-year, 15-minute) to inspect them. "
        "The optimiser still uses the synthetic prices generated in data/ for now."
    )

    uploaded = st.file_uploader("Upload DA or ID CSV (with datetime, price_eur_mwh columns)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df.head())
        if "datetime" in df.columns:
            try:
                df["datetime"] = pd.to_datetime(df["datetime"])
                st.line_chart(df.set_index("datetime")[df.columns[1]])
            except Exception as e:
                st.error(f"Could not parse datetime: {e}")


# ============================================================
# MAIN NAVIGATION
# ============================================================

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Daily EV Optimizer", "Full-Year Simulation", "Price Manager"],
    )

    if page == "Daily EV Optimizer":
        page_daily_optimizer()
    elif page == "Full-Year Simulation":
        page_full_year_simulation()
    elif page == "Price Manager":
        page_price_manager()


if __name__ == "__main__":
    main()
