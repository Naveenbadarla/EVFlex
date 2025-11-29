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

def load_price_15min(source) -> pd.Series:
    """
    Load a raw CSV: detect datetime + price columns; convert €/MWh → €/kWh;
    return a cleaned 15-minute price series indexed by datetime.
    """
    df = pd.read_csv(source)

    # Detect datetime column
    dt_col = None
    for c in df.columns:
        if ("MTU" in c) or ("date" in c.lower()) or ("time" in c.lower()):
            dt_col = c
            break
    if dt_col is None:
        raise ValueError("No datetime column found in uploaded file.")

    # Detect price column (€/MWh)
    price_col = None
    for c in df.columns:
        if ("price" in c.lower()) or ("MWh" in c):
            price_col = c
            break
    if price_col is None:
        raise ValueError("No price column found.")

    # Clean datetime
    try:
        dt_clean = df[dt_col].astype(str).str.split("-").str[0].str.strip()
        df["datetime"] = pd.to_datetime(dt_clean)
    except:
        df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")

    # Convert €/MWh → €/kWh
    df["price_eur_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["datetime", "price_eur_mwh"])
    df["price_eur_kwh"] = df["price_eur_mwh"] / 1000.0

    # Resample to EXACT 15-minute resolution
    s15 = (
        df.set_index("datetime")["price_eur_kwh"]
        .resample("15min")
        .mean()
    ).dropna()

    return s15


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
    if price_source == "Synthetic example":
        return synthetic_da_id_profiles_96()

    if price_source == "Built-in historical (data/…csv)":
        da_path = DATA_DIR / f"da_{year}.csv"
        id_path = DATA_DIR / f"id_{year}.csv"
        if not da_path.exists() or not id_path.exists():
            st.warning("Missing DA/ID files → using synthetic prices.")
            return synthetic_da_id_profiles_96()
        da_series = load_price_15min(da_path)
        id_series = load_price_15min(id_path)
        return series_to_96_slots(da_series), series_to_96_slots(id_series)

    if price_source == "Upload DA+ID CSVs":
        if da_file is None or id_file is None:
            st.warning("Please upload both DA & ID files or use another source.")
            return synthetic_da_id_profiles_96()
        da_series = load_price_15min(da_file)
        id_series = load_price_15min(id_file)
        return series_to_96_slots(da_series), series_to_96_slots(id_series)

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
    charge_baseline_96 = optimise_charging_96(
        baseline_curve_96, allowed_96, session_kwh, charger_power
    )

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
# PRICE MANAGER PAGE (15-MIN CLEANER VERSION)
# ============================================================

if page == "Price Manager (15-min)":

    st.title("📈 Price Manager – Import & Clean 15-min DA/ID")

    st.markdown(
        """
        Upload raw ENTSO-E (or similar) CSV files that contain **15-minute**
        Day-Ahead (DA) or Intraday (ID) price data.

        This tool will:
        - Detect datetime and price columns  
        - Convert €/MWh → €/kWh  
        - Resample to an exact 15-minute frequency  
        - Save a clean file to `/data` for historical use in the optimizer  

        ⚠️ This is a simplified utility:
        - No charts  
        - No advanced analysis  
        - Only preview + cleaning + saving  
        """
    )

    # -------------------------------------------
    # Select whether DA or ID file is being uploaded
    # -------------------------------------------
    upload_type = st.selectbox(
        "Select price type to import",
        ["Day-Ahead (DA)", "Intraday (ID)"]
    )

    # File upload
    raw_file = st.file_uploader(
        "Upload raw price CSV",
        type=["csv"],
        help="Upload DA/ID CSV file containing 15-min prices."
    )

    if raw_file is not None:
        df_raw = pd.read_csv(raw_file)

        st.subheader("🔍 Raw File Preview")
        st.dataframe(df_raw.head(), use_container_width=True)

        # -------------------------------------------
        # Auto-detect datetime & price columns
        # -------------------------------------------
        dt_col = None
        price_col = None

        # Detect datetime column
        for c in df_raw.columns:
            if ("MTU" in c) or ("date" in c.lower()) or ("time" in c.lower()):
                dt_col = c
                break

        # Detect price column
        for c in df_raw.columns:
            if ("MWh" in c) or ("price" in c.lower()):
                price_col = c
                break

        if dt_col is None or price_col is None:
            st.error("❌ Could not find datetime or price column automatically.")
        else:
            st.success(f"Detected datetime column: **{dt_col}**")
            st.success(f"Detected price column: **{price_col}**")

            # -------------------------------------------
            # Clean datetime column
            # ENTSO-E format like: "2023-07-01 00:00 – 00:15"
            # -------------------------------------------
            try:
                dt_clean = df_raw[dt_col].astype(str).str.split("-").str[0].str.strip()
                df_raw["datetime"] = pd.to_datetime(dt_clean)
            except:
                df_raw["datetime"] = pd.to_datetime(df_raw[dt_col], errors="coerce")

            # Convert €/MWh → €/kWh
            df_raw["price_eur_mwh"] = pd.to_numeric(df_raw[price_col], errors="coerce")
            df = df_raw.dropna(subset=["datetime", "price_eur_mwh"]).copy()
            df["price_eur_kwh"] = df["price_eur_mwh"] / 1000.0

            st.subheader("🧹 Cleaned Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # -------------------------------------------
            # EXACT 15-min resample
            # -------------------------------------------
            df_15 = (
                df.set_index("datetime")["price_eur_kwh"]
                .resample("15min")
                .mean()
            ).dropna()

            if len(df_15) < 96:
                st.warning(
                    "⚠️ This file does not contain a full 96 quarter-hours. "
                    "Make sure your raw file has complete 15-min data."
                )

            # -------------------------------------------
            # Choose which year this dataset belongs to
            # -------------------------------------------
            year_save = st.number_input(
                "Assign this dataset to year:",
                min_value=2000, max_value=2100,
                value=2023, step=1
            )

            # -------------------------------------------
            # Save cleaned file
            # -------------------------------------------
            if st.button("💾 Save cleaned 15-min price file"):
                name = "da" if upload_type.startswith("Day") else "id"
                out_path = DATA_DIR / f"{name}_{int(year_save)}.csv"

                out_df = pd.DataFrame({
                    "datetime": df_15.index,
                    "price_eur_mwh": df_15.values * 1000,  # store as €/MWh like ENTSO-E
                })

                out_df.to_csv(out_path, index=False)
                st.success(f"Saved cleaned file to: `{out_path}`")
    # -------------------------------------------
    # SHOW EXISTING CLEANED FILES
    # -------------------------------------------

    st.subheader("📂 Existing cleaned 15-min price files")

    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        st.info("No cleaned DA/ID files found in `/data` yet.")
    else:
        for f in files:
            st.write(f"• `{f.name}`")
    # ============================================================
    # DELETE FILES UI
    # ============================================================

    st.subheader("🗑️ Delete files from /data")

    delete_mode = st.checkbox("Enable delete mode")

    if delete_mode:

        st.warning("Delete mode is enabled. Be careful — deleted files cannot be recovered.")

        data_files = sorted(DATA_DIR.glob("*.csv"))

        if not data_files:
            st.info("No files available to delete.")
        else:
            for f in data_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{f.name}**")
                with col2:
                    if st.button("Delete", key=f"del_{f.name}"):
                        try:
                            f.unlink()
                            st.success(f"Deleted: {f.name}")
                        except Exception as e:
                            st.error(f"Failed to delete {f.name}: {e}")

            st.info("Refresh the page to update the file list after deleting.")

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










