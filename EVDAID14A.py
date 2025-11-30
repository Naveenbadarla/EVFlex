import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="EV Optimizer (15-min Precision)",
    layout="wide"
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# PRICE UTILITIES (15-MIN RESOLUTION)
# ============================================================

def load_price_15min(filepath):
    """
    Robust loader for cleaned 15-min files created by Price Manager 2.0.
    Accepts any column naming pattern as long as it contains:
        - datetime column
        - price column
    """

    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect datetime column
    dt_candidates = [
        c for c in df.columns
        if any(k in c for k in ["datetime", "date", "time"])
    ]

    if not dt_candidates:
        raise ValueError("No datetime column found in uploaded file.")

    dt_col = dt_candidates[0]

    # Parse datetimes
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])

    # Detect price column
    price_candidates = [c for c in df.columns if "price" in c]

    if not price_candidates:
        raise ValueError("No price column found in cleaned file.")

    price_col = price_candidates[0]

    # Build cleaned series
    s = (
        df.set_index(dt_col)[price_col]
        .astype(float)
        .sort_index()
    )

    return s



def series_to_96_slots(series_15min: pd.Series) -> np.ndarray:
    """Extract first 96 quarter-hours (1 day)."""
    if len(series_15min) < 96:
        raise ValueError("Price series must contain at least 96 quarter-hours.")
    return series_15min.iloc[:96].values


def synthetic_da_id_profiles_96():
    """
    Synthetic fallback: 24-hour DA & ID → repeat each hour ×4 to produce 96 slots.
    """
    da = np.array([
        0.28, 0.25, 0.22, 0.19, 0.17, 0.15, 0.14, 0.15,
        0.18, 0.20, 0.25, 0.30, 0.32, 0.34, 0.33, 0.31,
        0.28, 0.26, 0.25, 0.24, 0.23, 0.22, 0.22, 0.24
    ])
    id_prices = da - np.clip(da - da.min(), 0, 0.03)
    id_prices = np.maximum(id_prices, 0.0)

    return np.repeat(da, 4), np.repeat(id_prices, 4)


def get_da_id_profiles_96(price_source, year, da_file, id_file):
    """Return (da_96, id_96) based on user selection."""

    # -------------------------------------------------------
    # 1. SYNTHETIC EXAMPLE
    # -------------------------------------------------------
    if price_source == "Synthetic example":
        return synthetic_da_id_profiles_96()

    # -------------------------------------------------------
    # 2. BUILT-IN HISTORICAL (data/*.csv)
    # -------------------------------------------------------
    if price_source == "Built-in historical (data/…csv)":
        da_path = DATA_DIR / f"da_{year}.csv"
        id_path = DATA_DIR / f"id_{year}.csv"

        if not da_path.exists() or not id_path.exists():
            st.warning("Missing DA/ID files → using synthetic fallback.")
            return synthetic_da_id_profiles_96()

        da_series = load_price_15min(da_path)
        id_series = load_price_15min(id_path)

        # Convert €/MWh → €/kWh
        da_96 = series_to_96_slots(da_series) / 1000.0
        id_96 = series_to_96_slots(id_series) / 1000.0
        return da_96, id_96

    # -------------------------------------------------------
    # 3. UPLOAD RAW DA+ID CSVs
    # -------------------------------------------------------
    if price_source == "Upload DA+ID CSVs":
        if da_file is None or id_file is None:
            st.warning("Please upload both DA and ID price files.")
            return synthetic_da_id_profiles_96()

        da_series = load_price_15min(da_file)
        id_series = load_price_15min(id_file)

        # Convert €/MWh → €/kWh
        da_96 = series_to_96_slots(da_series) / 1000.0
        id_96 = series_to_96_slots(id_series) / 1000.0
        return da_96, id_96

    # -------------------------------------------------------
    # FALLBACK (should not be reached)
    # -------------------------------------------------------
    return synthetic_da_id_profiles_96()


# ============================================================
# EV OPTIMISATION UTILITIES (15-MIN / 96 SLOTS)
# ============================================================

def allowed_slots_96(arrival_hour: int, departure_hour: int) -> np.ndarray:
    """
    Returns a 96-length boolean vector where True means the EV is plugged in.
    Slot numbering:
        0   = 00:00–00:15
        1   = 00:15–00:30
        ...
        95  = 23:45–24:00
    """
    allowed = np.zeros(96, dtype=bool)
    start = arrival_hour * 4
    end = departure_hour * 4

    for s in range(96):
        if start <= end:
            allowed[s] = (start <= s < end)
        else:
            # Wraps across midnight
            allowed[s] = (s >= start or s < end)

    return allowed


def optimise_charging_96(cost_curve_96: np.ndarray,
                         allowed_96: np.ndarray,
                         session_kwh: float,
                         charger_power: float) -> np.ndarray:
    """
    Greedy cheapest-slot optimisation at 15-min resolution.
    cost_curve_96: €/kWh (length 96)
    allowed_96: boolean (length 96)
    """
    if session_kwh <= 0:
        return np.zeros(96)

    charge = np.zeros(96)
    remaining = session_kwh
    slot_energy = charger_power * 0.25  # 15 min = 0.25 hour

    # Sort allowed slots by cheapest price
    cheap_slots = sorted(
        [s for s in range(96) if allowed_96[s]],
        key=lambda s: cost_curve_96[s]
    )

    for s in cheap_slots:
        if remaining <= 0:
            break
        charge[s] = min(slot_energy, remaining)
        remaining -= charge[s]

    return charge


def session_cost_96(charge_96: np.ndarray, retail_curve_96: np.ndarray) -> float:
    """Cost per session = Σ(slot_kWh × retail_price)."""
    return float(np.sum(charge_96 * retail_curve_96))


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select page",
    ["EV Optimizer (15-min)", "Price Manager (15-min)"]
)


# ============================================================
# EV OPTIMIZER PAGE (15-min ENGINE)
# ============================================================

if page == "EV Optimizer (15-min)":

    st.sidebar.title("🚗 EV Optimizer (15-min resolution)")

    # -------------------------------------------
    # Charger power
    # -------------------------------------------
    charger_power = st.sidebar.number_input(
        "Charger power (kW)",
        min_value=1.0, max_value=22.0,
        value=11.0, step=1.0
    )

    # -------------------------------------------
    # Charging frequency
    # -------------------------------------------
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
        custom_days = st.sidebar.slider("Days per week", 1, 7, 4)

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

    # -------------------------------------------
    # Arrival / Departure Window
    # -------------------------------------------
    arrival_hour = st.sidebar.slider("Arrival hour", 0, 23, 18)
    departure_hour = st.sidebar.slider("Departure hour", 0, 23, 7)

    # -------------------------------------------
    # Baseline flat retail price
    # -------------------------------------------
    energy_flat = st.sidebar.number_input(
        "Flat retail price (€/kWh)",
        min_value=0.01, max_value=1.00,
        value=0.30, step=0.01
    )

    # -------------------------------------------
    # Charging energy method
    # -------------------------------------------
    st.sidebar.markdown("### Charging energy method")

    energy_method = st.sidebar.radio(
        "How to define energy per session?",
        ["SOC-based charging", "Manual kWh per session"]
    )

    # Needed later for SOC simulation
    battery_capacity = None
    arrival_soc = None
    target_soc = None

    # Placeholder to fill later
    session_kwh = 0.0

    # -------------------------------------------
    # When SOC-based mode is used
    # -------------------------------------------
    if energy_method == "SOC-based charging":
        battery_capacity = st.sidebar.number_input(
            "EV battery capacity (kWh)",
            min_value=10.0, max_value=200.0,
            value=60.0, step=1.0
        )

        arrival_soc = st.sidebar.slider(
            "Arrival SOC (%)", 0, 100, 30
        )
        target_soc = st.sidebar.slider(
            "Target SOC (%)", 0, 100, 80
        )

        if target_soc <= arrival_soc:
            st.sidebar.warning("Target SOC ≤ arrival SOC → no charging needed.")
            required_kwh = 0.0
        else:
            required_kwh = battery_capacity * (target_soc - arrival_soc) / 100.0

        session_kwh = required_kwh

        # ---- SOC FEEDBACK ----
        st.sidebar.markdown("#### SOC Feedback")
        st.sidebar.write(f"Arrival SOC: **{arrival_soc}%**")
        st.sidebar.progress(arrival_soc / 100)

        st.sidebar.write(f"Target SOC: **{target_soc}%**")
        st.sidebar.progress(target_soc / 100)

        st.sidebar.write(f"Energy required: **{required_kwh:.1f} kWh**")

    else:
        # -------------------------------------------
        # Manual kWh per session
        # -------------------------------------------
        session_kwh = st.sidebar.number_input(
            "kWh per charging session",
            min_value=0.0, max_value=200.0,
            value=20.0, step=0.5
        )
            # -------------------------------------------
    # WHOLESALE PRICE SOURCE (DA & ID, 15-MIN)
    # -------------------------------------------
    st.sidebar.markdown("### Wholesale DA & ID price source")

    price_source = st.sidebar.radio(
        "Source",
        [
            "Synthetic example",
            "Built-in historical (data/…csv)",
            "Upload DA+ID CSVs"
        ]
    )

    year = None
    da_file = None
    id_file = None

    if price_source == "Built-in historical (data/…csv)":
        year = st.sidebar.selectbox("Historical year", [2022, 2023, 2024])

    elif price_source == "Upload DA+ID CSVs":
        da_file = st.sidebar.file_uploader("Upload DA CSV", type=["csv"])
        id_file = st.sidebar.file_uploader("Upload ID CSV", type=["csv"])

    # -------------------------------------------
    # ID FLEXIBILITY (Smart DA+ID scenario)
    # -------------------------------------------
    id_flex_share = st.sidebar.slider(
        "ID flexibility share",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.05,
        help="Fraction of charging that can benefit from ID discounts."
    )

    # -------------------------------------------
    # RETAIL MARKUP (grid fees + taxes)
    # -------------------------------------------
    st.sidebar.markdown("### Retail markup (grid fees + taxes)")

    da_retail_markup = st.sidebar.number_input(
        "Markup (€/kWh)",
        min_value=0.00, max_value=1.00,
        value=0.12, step=0.01,
        help="All non-energy components (grid fees, taxes, levies, margin)."
    )

    # ============================================================
    # LOAD 96-SLOT WHOLESALE DA & ID PROFILES
    # ============================================================
    da_96, id_96 = get_da_id_profiles_96(
        price_source, year, da_file, id_file
    )

    # ID discount effect (Smart DA+ID scenario)
    id_discount_96 = np.clip(da_96 - id_96, 0, None)
    effective_da_id_96 = da_96 - id_discount_96 * id_flex_share

    # ============================================================
    # ALLOWED SLOTS (15-MIN) & FEASIBILITY CHECK
    # ============================================================
    allowed_96 = allowed_slots_96(arrival_hour, departure_hour)
    available_slots = allowed_96.sum()  # number of 15-min slots EV is connected

    slot_energy = charger_power * 0.25  # kWh per slot
    max_possible_kwh = available_slots * slot_energy

    if session_kwh > max_possible_kwh:
        st.sidebar.warning(
            f"Requested {session_kwh:.1f} kWh per session exceeds "
            f"maximum possible {max_possible_kwh:.1f} kWh "
            f"in the arrival→departure window. "
            "Capping to the physical maximum."
        )
        session_kwh = max_possible_kwh

    # ============================================================
    # RETAIL PRICE CURVES FOR BILLING (96 SLOTS)
    # ============================================================
    # Baseline: flat retail
    baseline_curve_96 = np.full(96, energy_flat)

    # DA-indexed retail tariff (DA + markup)
    da_indexed_curve_96 = da_96 + da_retail_markup

    # Smart DA & Smart DA+ID retail curves:
    retail_da_96 = da_96 + da_retail_markup
    retail_da_id_96 = effective_da_id_96 + da_retail_markup
        # ============================================================
    # CHARGING PROFILES (15-MIN SMART SCHEDULING)
    # ============================================================

    # Baseline (flat retail price)
    # --- Baseline charging: fill from arrival index forward ---
    charge_baseline_96 = np.zeros(96)
    remaining = session_kwh
    
    # iterate forward starting at arrival index
    start = arrival_hour * 4
    idx = start
    
    while remaining > 0 and idx < 96:
        take = min(charger_power, remaining)
        charge_baseline_96[idx] = take
        remaining -= take
        idx += 1
    
    # if still energy left, continue after midnight (wrap-around)
    idx = 0
    while remaining > 0 and idx < start:
        take = min(charger_power, remaining)
        charge_baseline_96[idx] = take
        remaining -= take
        idx += 1


    # Retail DA-indexed (no smart shifting → same pattern as baseline)
    charge_da_indexed_96 = charge_baseline_96.copy()

    # Smart DA optimisation (uses wholesale DA prices)
    charge_smart_da_96 = optimise_charging_96(
        da_96, allowed_96, session_kwh, charger_power
    )

    # Smart DA+ID optimisation (uses DA+ID effective prices)
    charge_smart_full_96 = optimise_charging_96(
        effective_da_id_96, allowed_96, session_kwh, charger_power
    )

    # ============================================================
    # SESSION COSTS (with RETAIL billing)
    # ============================================================

    session_cost_baseline = session_cost_96(charge_baseline_96, baseline_curve_96)
    session_cost_da_indexed = session_cost_96(charge_da_indexed_96, da_indexed_curve_96)
    session_cost_smart_da = session_cost_96(charge_smart_da_96, retail_da_96)
    session_cost_smart_full = session_cost_96(charge_smart_full_96, retail_da_id_96)

    # ============================================================
    # ANNUAL COSTS
    # ============================================================

    annual_baseline = session_cost_baseline * sessions_per_year
    annual_da_indexed = session_cost_da_indexed * sessions_per_year
    annual_smart_da = session_cost_smart_da * sessions_per_year
    annual_smart_full = session_cost_smart_full * sessions_per_year

    total_kwh_year = sessions_per_year * session_kwh

    def eff_price(annual_cost):
        if total_kwh_year == 0:
            return 0.0
        return annual_cost / total_kwh_year

    # ============================================================
    # MAIN OUTPUT SUMMARY
    # ============================================================

    st.title("🚗 EV Optimizer — 15-minute Resolution")

    st.markdown(
        f"**Sessions/year:** {sessions_per_year:.1f}  \n"
        f"**Energy per session:** {session_kwh:.2f} kWh  \n"
        f"**Annual charging energy:** {total_kwh_year:.0f} kWh  \n"
        f"**Arrival → Departure:** {arrival_hour:02d}:00 → {departure_hour:02d}:00  \n"
        f"**Retail markup:** {da_retail_markup:.2f} €/kWh"
    )

    # ============================================================
    # ============================================================
# VIEW MODE: SINGLE DAY vs FULL YEAR
# ============================================================

mode = st.radio(
    "Simulation mode",
    ["Single Day", "Full Year"],
    help="Single Day = optimise a typical day. Full Year = simulate all 365 days."
)

if mode == "Single Day":
    # COST COMPARISON TABLE
        # ============================================================

        results_df = pd.DataFrame([
            ["Baseline (flat retail)", annual_baseline, eff_price(annual_baseline)],
            ["Retail DA-indexed", annual_da_indexed, eff_price(annual_da_indexed)],
            ["Smart DA optimisation", annual_smart_da, eff_price(annual_smart_da)],
            ["Smart DA+ID optimisation", annual_smart_full, eff_price(annual_smart_full)],
        ], columns=["Scenario", "Annual Cost (€)", "Effective Price (€/kWh)"])

        best = results_df.loc[results_df["Annual Cost (€)"].idxmin()]

        st.subheader("Annual Cost Comparison")
        st.dataframe(
            results_df.style.format({
                "Annual Cost (€)": "{:.0f}",
                "Effective Price (€/kWh)": "{:.3f}"
            }),
            use_container_width=True
        )

        st.success(
            f"**Best scenario:** {best['Scenario']} → "
            f"**{best['Annual Cost (€)']:.0f} € / year** "
            f"({best['Effective Price (€/kWh)']:.3f} €/kWh)"
        )

        # ============================================================
        # COST BAR CHART
        # ============================================================

        st.subheader("Cost Comparison (Bar Chart)")

        cost_chart = (
            alt.Chart(results_df)
            .mark_bar()
            .encode(
                x=alt.X("Scenario:N", sort=None),
                y=alt.Y("Annual Cost (€):Q"),
                tooltip=[
                    "Scenario",
                    alt.Tooltip("Annual Cost (€):Q", format=".0f"),
                    alt.Tooltip("Effective Price (€/kWh):Q", format=".3f")
                ]
            )
            .properties(height=330)
        )

        st.altair_chart(cost_chart, use_container_width=True)
            # ============================================================
        # SIMPLE CHARGING PATTERN VISUALISATION (15-min → hourly)
        # ============================================================

        st.subheader("🔌 Charging Pattern (Arrival → Departure Window)")

        # ------------------------------
        # Convert 96-slot (15-min) charge arrays into 24 hourly totals
        # ------------------------------
        # Each hour = sum of its 4 quarter-hours
        baseline_hourly = charge_baseline_96.reshape(24, 4).sum(axis=1)
        da_indexed_hourly = charge_da_indexed_96.reshape(24, 4).sum(axis=1)
        smart_da_hourly = charge_smart_da_96.reshape(24, 4).sum(axis=1)
        smart_full_hourly = charge_smart_full_96.reshape(24, 4).sum(axis=1)

        # ------------------------------
        # Build the arrival → departure hour window (correct order)
        # ------------------------------
        if arrival_hour <= departure_hour:
            hours_window = list(range(arrival_hour, departure_hour))
        else:
            hours_window = list(range(arrival_hour, 24)) + list(range(0, departure_hour))

        # ------------------------------
        # Build DataFrame for the chart
        # ------------------------------
        charging_pattern_df = pd.DataFrame({
            "Hour": hours_window,
            "Baseline (kWh)": [baseline_hourly[h] for h in hours_window],
            "DA-indexed (kWh)": [da_indexed_hourly[h] for h in hours_window],
            "Smart DA (kWh)": [smart_da_hourly[h] for h in hours_window],
            "Smart DA+ID (kWh)": [smart_full_hourly[h] for h in hours_window],
        })

        charging_long = charging_pattern_df.melt(
            id_vars="Hour",
            var_name="Scenario",
            value_name="kWh"
        )

        # ------------------------------
        # Cleanest view: one panel per scenario
        # ------------------------------
        charging_chart = (
            alt.Chart(charging_long)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Hour:O",
                    title="Hour of Day",
                    sort=hours_window  # preserve arrival→departure order
                ),
                y=alt.Y("kWh:Q", title="Charging (kWh)"),
                color="Scenario:N",
                column=alt.Column(
                    "Scenario:N",
                    header=alt.Header(labelAngle=0, labelFontSize=12)
                ),
                tooltip=["Scenario", "Hour", alt.Tooltip("kWh:Q", format=".2f")]
            )
            .properties(height=300)
        )

        st.altair_chart(charging_chart, use_container_width=True)

        st.markdown(
            """
            **How to read this chart:**

            - Only hours where the EV is connected are shown (between arrival and departure).
            - Each panel is one scenario:
                - Baseline (flat retail)
                - Retail DA-indexed
                - Smart DA
                - Smart DA+ID
            - Bars show how many kWh are charged in each **hour**, internally based on 15-minute optimisation.
            """
        )
        # ============================================================
    # EXPANDER 1 — DA vs ID PRICE CURVES
    # ============================================================

    with st.expander("📈 Prices: DA vs ID (15-minute resolution)"):

        price_df = pd.DataFrame({
            "Datetime": pd.date_range("2023-01-01", periods=96, freq="15min"),
            "DA (€/kWh)": da_96,
            "ID (€/kWh)": id_96
        })

        price_long = price_df.melt("Datetime", var_name="Curve", value_name="€/kWh")

        price_chart = (
            alt.Chart(price_long)
            .mark_line(point=False)
            .encode(
                x="Datetime:T",
                y="€/kWh:Q",
                color="Curve:N",
                tooltip=["Datetime", "Curve", alt.Tooltip("€/kWh", format=".4f")]
            )
            .properties(height=300)
        )

        st.altair_chart(price_chart, use_container_width=True)

        st.markdown("""
        **Interpretation:**
        - DA curve shows base day-ahead prices.
        - ID curve is more volatile and shows real intraday opportunities.
        - Smart DA+ID will shift charging into low-ID dips.
        """)

    # ============================================================
    # EXPANDER 2 — MUCH CLEARER Charging Comparison (DA vs DA+ID)
    # ============================================================

    with st.expander("🔌 Clear Charging Comparison: Smart DA vs Smart DA+ID"):

        # Build a cleaned dataframe only with hours where at least one scenario charges
        comp_rows = []
        for h in range(96):
            da_val = float(charge_smart_da_96[h])
            full_val = float(charge_smart_full_96[h])
            if da_val > 0 or full_val > 0:
                comp_rows.append({
                    "Hour": h,
                    "Smart DA (kWh)": da_val,
                    "Smart DA+ID (kWh)": full_val
                })

        if len(comp_rows) == 0:
            st.info("No charging occurs during the selected window.")
        else:
            compare_df = pd.DataFrame(comp_rows)

            # Convert quarter-hour index to hour of day text
            compare_df["HourLabel"] = compare_df["Hour"].apply(lambda x: f"{(x//4):02d}:{(x%4)*15:02d}")

            # Melt for grouped bar format
            compare_long = compare_df.melt("HourLabel", var_name="Scenario", value_name="kWh")

            clear_chart = (
                alt.Chart(compare_long)
                .mark_bar()
                .encode(
                    x=alt.X("HourLabel:O", title="Time of Day"),
                    y=alt.Y("kWh:Q", title="Charging (kWh)"),
                    color=alt.Color("Scenario:N", scale=alt.Scale(range=["#1f77b4", "#d62728"])),
                    tooltip=["HourLabel", "Scenario", alt.Tooltip("kWh", format=".3f")]
                )
                .properties(height=320)
            )

            st.altair_chart(clear_chart, use_container_width=True)

            st.markdown("""
            ### ✔ Interpretation
            - Each time-of-day slot shows **two bars**: Smart DA (blue) and Smart DA+ID (red).
            - Higher bar = more kWh charged in that 15-min slot.
            - If red ≠ blue, ID optimisation is doing something different.
            - Days with large ID dips will show big movement in Smart DA+ID.
            """)


        # ============================================================
    # EXPANDER 3 — PRICES + CHARGING OVERLAY
    # ============================================================

    with st.expander("📉 Prices + Charging Overlay (All Scenarios)"):

        def overlay_chart(prices, charging, title):
            df = pd.DataFrame({
                "Datetime": pd.date_range("2023-01-01", periods=96, freq="15min"),
                "Price (€/kWh)": prices,
                "Charging (kWh)": charging
            })

            price_line = alt.Chart(df).mark_line(color="orange").encode(
                x="Datetime:T",
                y="Price (€/kWh):Q"
            )

            charge_bars = alt.Chart(df).mark_bar(opacity=0.5, color="steelblue").encode(
                x="Datetime:T",
                y="Charging (kWh):Q"
            )

            return (price_line + charge_bars).properties(height=280, title=title)

        st.altair_chart(
            overlay_chart(da_96, charge_da_indexed_96, "Retail DA-indexed: Prices + Charging"),
            use_container_width=True
        )

        st.altair_chart(
            overlay_chart(da_96, charge_smart_da_96, "Smart DA: Prices + Charging"),
            use_container_width=True
        )

        st.altair_chart(
            overlay_chart(effective_da_id_96, charge_smart_full_96, "Smart DA+ID: Prices + Charging"),
            use_container_width=True
        )

        st.markdown("""
        **Interpretation:**
        - Blue bars = charging amount  
        - Orange line = price  
        - Smart DA puts bars where orange line is lowest  
        - Smart DA+ID uses an even more volatile price curve → should shift bars further into ID dips  
        - If plots look identical, ID dips may be small or time window may not include them  
        """)


    # ============================================================
# FULL-YEAR SIMULATION MODE (15-MIN DA/ID FOR 365 DAYS)
# ============================================================

if mode == "Full Year":

    st.header("📅 Full-Year EV Smart Charging Simulation (365 days)")

    # ---------- USER PARAMETERS ----------
    battery_kwh = st.number_input("Battery size (kWh)", 20, 120, 60)
    charger_power = st.number_input("Charger power (kW)", 2, 22, 7)
    arrival_mean = st.slider("Average arrival hour", 0, 23, 18)
    departure_mean = st.slider("Average departure hour", 0, 23, 7)
    target_soc = st.slider("Target SOC (%)", 50, 100, 85) / 100
    sessions_per_week = st.slider("Charging sessions per week", 1, 7, 4)
    markup = st.number_input("DA markup (€/kWh)", 0.00, 0.50, 0.12)
    retail_flat = st.number_input("Flat retail price (€/kWh)", 0.10, 1.00, 0.30)

    # ---------- LOAD FULL-YEAR DATA (synthetic or uploaded) ----------
    da_full = pd.read_csv("data/da_2023_full.csv", parse_dates=["datetime"])
    id_full = pd.read_csv("data/id_2023_full.csv", parse_dates=["datetime"])

    da_full["price"] = da_full["price_eur_mwh"] / 1000.0
    id_full["price"] = id_full["price_eur_mwh"] / 1000.0

    # Make sure both datasets cover the year
    if len(da_full) != 35040 or len(id_full) != 35040:
        st.error("❌ Full-year datasets must contain 35,040 rows (96×365).")
        st.stop()

    # ---------- SIMULATION LOOP ----------

    results = []
    rng = np.random.default_rng(42)

    for day in range(365):

        # Extract 96 quarter-hours for the day
        da_day = da_full.iloc[day*96:(day+1)*96].copy()
        id_day = id_full.iloc[day*96:(day+1)*96].copy()

        # Decide if today is a charging day
        if rng.random() > sessions_per_week / 7:
            continue  # Skip today

        # Arrival & departure time (randomized around mean)
        arr = int(np.clip(rng.normal(arrival_mean, 1.5), 0, 23))
        dep = int(np.clip(rng.normal(departure_mean, 1.5), 0, 23))

        # Overnight wrap
        arr_idx = arr * 4
        dep_idx = dep * 4

        allowed = np.zeros(96, dtype=bool)
        if dep_idx > arr_idx:
            allowed[arr_idx:dep_idx] = True
        else:
            allowed[arr_idx:] = True
            allowed[:dep_idx] = True

        # Session energy need
        arrival_soc = rng.uniform(0.20, 0.60)
        session_kwh = max(0, battery_kwh * (target_soc - arrival_soc))

        # ---------- BASELINE (flat retail) ----------
        remaining = session_kwh
        base_charge = np.zeros(96)
        idx = arr_idx
        while remaining > 0 and idx < 96:
            take = min(charger_power, remaining)
            base_charge[idx] = take
            remaining -= take
            idx += 1
        idx = 0
        while remaining > 0 and idx < arr_idx:
            take = min(charger_power, remaining)
            base_charge[idx] = take
            remaining -= take
            idx += 1

        base_cost = base_charge.sum() * retail_flat

        # ---------- RETAIL DA-INDEXED ----------
        da_retail = da_day["price"].values + markup
        da_indexed_cost = (base_charge * da_retail).sum()

        # ---------- SMART DA ----------
        def optimise(prices, allowed, need_kwh):
            sched = np.zeros(96)
            remaining = need_kwh
            order = np.argsort(prices)
            for i in order:
                if not allowed[i]:
                    continue
                take = min(charger_power, remaining)
                sched[i] = take
                remaining -= take
                if remaining <= 0:
                    break
            return sched

        smart_da_charge = optimise(da_retail, allowed, session_kwh)
        smart_da_cost = (smart_da_charge * da_retail).sum()

        # ---------- SMART DA+ID ----------
        # ID discount
        id_discount = np.clip(da_day["price"].values - id_day["price"].values, 0, None)
        effective_da_id = da_day["price"].values - id_discount + markup
        smart_full_charge = optimise(effective_da_id, allowed, session_kwh)
        smart_full_cost = (smart_full_charge * effective_da_id).sum()

        # ---------- STORE DAILY RESULT ----------
        results.append({
            "date": da_day["datetime"].iloc[0].date(),
            "baseline": base_cost,
            "da_indexed": da_indexed_cost,
            "smart_da": smart_da_cost,
            "smart_full": smart_full_cost,
            "kwh": session_kwh,
        })

    # ---------- AGGREGATE ----------
    df_year = pd.DataFrame(results)
    total_kwh = df_year["kwh"].sum()

    annual = {
        "Scenario": ["Baseline (flat)", "Retail DA-indexed", "Smart DA", "Smart DA+ID"],
        "Annual Cost (€)": [
            df_year["baseline"].sum(),
            df_year["da_indexed"].sum(),
            df_year["smart_da"].sum(),
            df_year["smart_full"].sum(),
        ],
        "Effective Price (€/kWh)": [
            df_year["baseline"].sum() / total_kwh,
            df_year["da_indexed"].sum() / total_kwh,
            df_year["smart_da"].sum() / total_kwh,
            df_year["smart_full"].sum() / total_kwh,
        ]
    }

    df_annual = pd.DataFrame(annual)
    st.subheader("📊 Annual Cost Comparison (365-day simulation)")
    st.dataframe(df_annual)

    best = df_annual.iloc[df_annual["Annual Cost (€)"].idxmin()]
    st.success(f"Best scenario: **{best['Scenario']}** → {best['Annual Cost (€)']:.0f} € / year "
               f"({best['Effective Price (€/kWh)']:.3f} €/kWh)")

    # ============================================================
    # OPTIONAL: RANDOM ARRIVAL / SOC SIMULATION (MONTE CARLO)
    # ============================================================

    st.subheader("🎲 Random Arrival Simulation (Optional)")

    simulate = st.checkbox("Enable random arrival & SOC simulation")

    if simulate:

        # Number of Monte Carlo samples
        n_sim = st.slider(
            "Number of simulated charging sessions",
            min_value=100, max_value=2000,
            value=500, step=100
        )

        # Arrival time variation (± hours)
        arrival_jitter = st.slider(
            "Arrival time jitter (± hours)",
            min_value=0.0, max_value=4.0,
            value=2.0, step=0.5,
            help="0 = always same arrival hour; larger = more random arrival times."
        )

        # SOC variation only applies if SOC-based mode is used
        if energy_method == "SOC-based charging" and battery_capacity is not None:
            soc_jitter = st.slider(
                "Arrival SOC jitter (± %)",
                min_value=0, max_value=50,
                value=10, step=5,
                help="Random variation around the chosen arrival SOC."
            )
        else:
            soc_jitter = 0

        rng = np.random.default_rng()

        # Cost arrays
        costs_baseline = []
        costs_da_idx = []
        costs_smart_da = []
        costs_smart_full = []

        for _ in range(n_sim):

            # ------------------------
            # Random arrival hour
            # ------------------------
            if arrival_jitter > 0:
                a = arrival_hour + rng.normal(0, arrival_jitter / 2.0)
                a = int(np.round(a)) % 24
            else:
                a = arrival_hour

            # Allowed slots for simulated arrival time
            allowed_i = allowed_slots_96(a, departure_hour)
            avail_slots_i = allowed_i.sum()
            max_kwh_i = avail_slots_i * slot_energy

            # ------------------------
            # Random SOC (if enabled)
            # ------------------------
            if energy_method == "SOC-based charging" and battery_capacity:
                if soc_jitter > 0:
                    arr_soc_i = arrival_soc + rng.normal(0, soc_jitter / 2.0)
                    arr_soc_i = float(np.clip(arr_soc_i, 0, target_soc))
                else:
                    arr_soc_i = float(arrival_soc)

                req_kwh_i = battery_capacity * (target_soc - arr_soc_i) / 100.0
                req_kwh_i = max(req_kwh_i, 0.0)
            else:
                req_kwh_i = session_kwh

            # Cap by physical max
            req_kwh_i = min(req_kwh_i, max_kwh_i)

            # ------------------------
            # Run all 4 optimisations
            # ------------------------
            c_base = optimise_charging_96(baseline_curve_96, allowed_i, req_kwh_i, charger_power)
            c_daidx = c_base.copy()
            c_da = optimise_charging_96(da_96, allowed_i, req_kwh_i, charger_power)
            c_full = optimise_charging_96(effective_da_id_96, allowed_i, req_kwh_i, charger_power)

            # ------------------------
            # Retail billed session cost
            # ------------------------
            costs_baseline.append(session_cost_96(c_base, baseline_curve_96))
            costs_da_idx.append(session_cost_96(c_daidx, da_indexed_curve_96))
            costs_smart_da.append(session_cost_96(c_da, retail_da_96))
            costs_smart_full.append(session_cost_96(c_full, retail_da_id_96))

        # ------------------------
        # Summaries
        # ------------------------
        def summarize(costs):
            arr = np.array(costs)
            return float(arr.mean()), float(arr.min()), float(arr.max())

        mean_b, min_b, max_b = summarize(costs_baseline)
        mean_da, min_da, max_da = summarize(costs_da_idx)
        mean_sda, min_sda, max_sda = summarize(costs_smart_da)
        mean_sfull, min_sfull, max_sfull = summarize(costs_smart_full)

        # Annualised mean cost (for reference)
        annual_mean_b = mean_b * sessions_per_year
        annual_mean_da = mean_da * sessions_per_year
        annual_mean_sda = mean_sda * sessions_per_year
        annual_mean_sfull = mean_sfull * sessions_per_year

        # ------------------------
        # Table Output
        # ------------------------
        sim_df = pd.DataFrame([
            ["Baseline (flat retail)",       mean_b,  min_b,  max_b,  annual_mean_b],
            ["Retail DA-indexed",            mean_da, min_da, max_da, annual_mean_da],
            ["Smart DA optimisation",        mean_sda, min_sda, max_sda, annual_mean_sda],
            ["Smart DA+ID optimisation",     mean_sfull, min_sfull, max_sfull, annual_mean_sfull],
        ], columns=[
            "Scenario",
            "Mean cost/session (€)",
            "Min cost/session (€)",
            "Max cost/session (€)",
            "Mean annual cost (€)"
        ])

        st.markdown("### Simulation Results")
        st.dataframe(
            sim_df.style.format({
                "Mean cost/session (€)": "{:.2f}",
                "Min cost/session (€)": "{:.2f}",
                "Max cost/session (€)": "{:.2f}",
                "Mean annual cost (€)": "{:.0f}",
            }),
            use_container_width=True
        )

        st.markdown(
            "_These simulated values account for variability in arrival times "
            "and arrival SOC levels, giving a more realistic picture of annual cost._"
        )
                # ============================================================
        # VISUAL 1: Histogram of simulated arrival hours
        # ============================================================

        st.subheader("📊 Distribution of Simulated Arrival Hours")

        arrival_hours_list = []

        # regenerate all arrival hours without recalculating costs
        for _ in range(n_sim):
            if arrival_jitter > 0:
                a = arrival_hour + rng.normal(0, arrival_jitter / 2.0)
                a = int(np.round(a)) % 24
            else:
                a = arrival_hour
            arrival_hours_list.append(a)

        arrival_df = pd.DataFrame({"ArrivalHour": arrival_hours_list})

        hist_arrival = (
            alt.Chart(arrival_df)
            .mark_bar()
            .encode(
                x=alt.X("ArrivalHour:O", bin=False, title="Hour of Arrival"),
                y=alt.Y("count()", title="Frequency")
            )
            .properties(height=200)
        )

        st.altair_chart(hist_arrival, use_container_width=True)

        # ============================================================
        # VISUAL 2: Scatter plot (Arrival Hour → Cost Per Session)
        # ============================================================

        st.subheader("📈 Arrival Hour vs Cost Per Session")

        # Recalculate arrival hours and costs (minimal overhead)
        arrival_list_scatter = []
        cost_list_scatter = []

        for _ in range(n_sim):

            # Random arrival
            if arrival_jitter > 0:
                a = arrival_hour + rng.normal(0, arrival_jitter / 2.0)
                a = int(np.round(a)) % 24
            else:
                a = arrival_hour

            arrival_list_scatter.append(a)

            # Allowed slots for this session
            allowed_i = allowed_slots_96(a, departure_hour)
            avail_slots_i = allowed_i.sum()
            max_kwh_i = avail_slots_i * slot_energy

            # Required energy (SOC or manual)
            if energy_method == "SOC-based charging" and battery_capacity:
                if soc_jitter > 0:
                    arr_soc_i = arrival_soc + rng.normal(0, soc_jitter / 2.0)
                    arr_soc_i = float(np.clip(arr_soc_i, 0, target_soc))
                else:
                    arr_soc_i = float(arrival_soc)

                req_kwh_i = battery_capacity * (target_soc - arr_soc_i) / 100.0
                req_kwh_i = max(req_kwh_i, 0.0)
            else:
                req_kwh_i = session_kwh

            req_kwh_i = min(req_kwh_i, max_kwh_i)

            # Cost for *best* scenario (smart DA+ID)
            c_full = optimise_charging_96(effective_da_id_96, allowed_i, req_kwh_i, charger_power)
            cost = session_cost_96(c_full, retail_da_id_96)
            cost_list_scatter.append(cost)

        scatter_df = pd.DataFrame({
            "ArrivalHour": arrival_list_scatter,
            "Cost": cost_list_scatter
        })

        scatter_plot = (
            alt.Chart(scatter_df)
            .mark_circle(size=60, opacity=0.6)
            .encode(
                x=alt.X("ArrivalHour:O", title="Arrival Hour"),
                y=alt.Y("Cost:Q", title="Cost per Session (€)"),
                tooltip=["ArrivalHour", alt.Tooltip("Cost", format=".2f")]
            )
            .properties(height=250)
        )

        st.altair_chart(scatter_plot, use_container_width=True)

        st.markdown(
            "_This scatter shows how arrival-time variability affects cost. "
            "Later arrivals typically reduce flexibility and increase cost._"
        )

        # ============================================================
        # VISUAL 3: Histogram of Simulated Arrival SOC (if SOC mode)
        # ============================================================

        if energy_method == "SOC-based charging" and battery_capacity is not None:

            st.subheader("🔋 Distribution of Simulated Arrival SOC")

            soc_values_list = []

            # Regenerate random SOC values without recomputing costs
            for _ in range(n_sim):
                if soc_jitter > 0:
                    arr_soc_i = arrival_soc + rng.normal(0, soc_jitter / 2.0)
                    arr_soc_i = float(np.clip(arr_soc_i, 0, target_soc))
                else:
                    arr_soc_i = float(arrival_soc)

                soc_values_list.append(arr_soc_i)

            soc_df = pd.DataFrame({"ArrivalSOC": soc_values_list})

            soc_hist = (
                alt.Chart(soc_df)
                .mark_bar()
                .encode(
                    x=alt.X("ArrivalSOC:Q", bin=alt.Bin(maxbins=20), title="Arrival SOC (%)"),
                    y=alt.Y("count()", title="Frequency")
                )
                .properties(height=200)
            )

            st.altair_chart(soc_hist, use_container_width=True)

            st.markdown(
                "_This histogram shows how much arrival SOC varies when jitter is applied. "
                "Lower arrival SOC → more required charging → potentially higher costs._"
            )

# ============================================================
# PRICE MANAGER 2.0 — Robust DA/ID Import + Cleaning System
# ============================================================

if page == "Price Manager (15-min)":

    st.title("📈 Price Manager 2.0 — Robust 15-min DA/ID Loader")

    st.markdown("""
    Price Manager 2.0 allows you to upload *any form* of Day-Ahead (DA) or Intraday (ID)
    price files and automatically converts them into **96×15-minute** price profiles.

    Supported inputs:
    - ENTSO-E CSV / Excel
    - EPEX / EEX hourly DA data
    - GDM datasets
    - Files with MTU ranges (e.g., “2023-01-01 00:00 – 01:00”)
    - Files in €/MWh or €/kWh
    - Files with irregular timestamps
    """)

    # -------------------------------------------
    # Select DA or ID
    # -------------------------------------------
    price_type = st.selectbox(
        "Select price type",
        ["Day-Ahead (DA)", "Intraday (ID)"]
    )

    raw_file = st.file_uploader(
        "Upload raw DA/ID file (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload raw file — all formats supported."
    )

    if raw_file is not None:

        # AUTO LOAD CSV OR EXCEL
        try:
            if raw_file.name.endswith(".csv"):
                df_raw = pd.read_csv(raw_file)
            else:
                df_raw = pd.read_excel(raw_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

        st.subheader("🔍 Raw File Preview")
        st.dataframe(df_raw.head(), use_container_width=True)

        # ============================================================
        # STEP 1 — Detect datetime column
        # ============================================================

        dt_candidates = [
            c for c in df_raw.columns
            if any(k in c.lower() for k in ["date", "time", "mtu"])
        ]

        if not dt_candidates:
            st.error("❌ Could not detect a datetime column.")
            st.stop()

        dt_col = dt_candidates[0]

        # Try to split MTU ranges
        df_raw[dt_col] = df_raw[dt_col].astype(str).str.split("–").str[0].str.strip()

        # Parse datetime
        df_raw["datetime"] = pd.to_datetime(df_raw[dt_col], errors="coerce")

        # Drop invalid rows
        df_raw = df_raw.dropna(subset=["datetime"])

        # ============================================================
        # STEP 2 — Detect price column
        # ============================================================

        price_candidates = [
            c for c in df_raw.columns
            if any(k in c.lower() for k in ["price", "eur", "mwh", "kwh", "value"])
        ]

        if not price_candidates:
            st.error("❌ No price column detected.")
            st.stop()

        price_col = price_candidates[-1]  # usually last one

        # Convert to numeric
        df_raw["price"] = pd.to_numeric(df_raw[price_col], errors="coerce")
        df_raw = df_raw.dropna(subset=["price"])

        st.success(f"Detected datetime column: **{dt_col}**")
        st.success(f"Detected price column: **{price_col}**")

        # ============================================================
        # STEP 3 — Convert ALL inputs → €/MWh uniformly
        # ============================================================

        # If values look like €/kWh (e.g. <10), multiply
        if df_raw["price"].mean() < 10:
            df_raw["price_eur_mwh"] = df_raw["price"] * 1000
        else:
            df_raw["price_eur_mwh"] = df_raw["price"]

        st.info(f"Price range: {df_raw['price_eur_mwh'].min():.2f} – {df_raw['price_eur_mwh'].max():.2f} €/MWh")

        # ============================================================
        # STEP 4 — Normalize to EXACT 15-minute resolution
        # ============================================================

        s = df_raw.set_index("datetime")["price_eur_mwh"].sort_index()

        # Detect if hourly
        freq_guess = pd.infer_freq(s.index)
        st.write("Detected frequency:", freq_guess)

        if freq_guess in ["H", "60T"]:
            st.warning("Hourly data detected → converting to 15-min resolution (×4).")
            s_15 = s.resample("15min").ffill()
        else:
            s_15 = s.resample("15min").mean().ffill()

        # Final validity check
        if len(s_15) < 96:
            st.error("❌ Could not extract one full 96 quarter-hour day from this file.")
            st.stop()

        # Extract FIRST 96 values (typical day)
        day_values = s_15.iloc[:96].reset_index()
        day_values.columns = ["datetime", "price_eur_mwh"]

        st.subheader("🧹 Cleaned 96×15-min Profile")
        st.dataframe(day_values.head(), use_container_width=True)

        # ============================================================
        # STEP 5 — Save cleaned file
        # ============================================================

        year_save = st.number_input(
            "Assign cleaned dataset to year:",
            min_value=2000, max_value=2100, value=2023
        )

        if st.button("💾 Save cleaned 15-min file"):
            fname = "da" if price_type.startswith("Day") else "id"
            out_path = DATA_DIR / f"{fname}_{year_save}.csv"
            day_values.to_csv(out_path, index=False)
            st.success(f"Saved to `{out_path}`")

    # ============================================================
    # FILE MANAGEMENT (DELETE / DOWNLOAD)
    # ============================================================

    st.subheader("📂 Data folder contents")

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        st.info("No CSV files in /data yet.")
    else:
        for f in files:
            col1, col2, col3 = st.columns([4,1,1])

            with col1:
                st.write(f"**{f.name}**")

            with col2:
                st.download_button("⬇️", data=open(f,"rb").read(),
                                   file_name=f.name, key=f"dl_{f.name}")

            with col3:
                if st.button("🗑️", key=f"del_{f.name}"):
                    f.unlink()
                    st.success(f"Deleted {f.name}. Refresh the page.")

# ============================================================
# FOOTER (APPEARS ON EVERY PAGE)
# ============================================================

def app_footer():
    st.markdown(
        """
        <hr>

        <div style="text-align:center; color:#666; font-size:13px;">
        EV Optimizer – 15-Minute DA/ID Engine<br>
        Smart Day-Ahead & Intraday Scheduling<br>
        Includes SOC Feedback & Random Arrival Simulation<br><br>
        Version 2.0 · Built with Streamlit · Altair · NumPy · Pandas
        </div>
        """,
        unsafe_allow_html=True
    )

# Display footer on all pages
app_footer()










