import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="EV Hourly Optimizer (Simplified)",
    layout="wide"
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# PRICE UTILITIES
# ============================================================

def load_clean_price_csv(source) -> pd.Series:
    """
    Load a 'datetime, price_eur_mwh' CSV and return a Series
    indexed by datetime with values in €/kWh.
    """
    df = pd.read_csv(source)
    if "datetime" not in df.columns:
        raise ValueError("'datetime' column not found in CSV")

    price_col = None
    for c in df.columns:
        if c.lower().startswith("price"):
            price_col = c
            break
    if price_col is None:
        raise ValueError("No price column found in CSV (need something like 'price_eur_mwh').")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # €/MWh → €/kWh
    df["price_eur_kwh"] = df[price_col] / 1000.0
    return df.set_index("datetime")["price_eur_kwh"]


def hourly_profile_from_series(series: pd.Series) -> np.ndarray:
    """
    Convert any timestamped price series to a 24-element (hour 0–23)
    average price profile in €/kWh.
    """
    s = series.resample("1H").mean()
    df = s.to_frame("price")
    df["hour"] = df.index.hour
    hourly = df.groupby("hour")["price"].mean()
    hourly = hourly.reindex(range(24), fill_value=hourly.mean())
    return hourly.values


def synthetic_da_id_profiles() -> tuple[np.ndarray, np.ndarray]:
    """
    Simple synthetic 24h DA and ID price profiles in €/kWh
    as a fallback.
    """
    da = np.array([
        0.28, 0.25, 0.22, 0.19, 0.17, 0.15, 0.14, 0.15,
        0.18, 0.20, 0.25, 0.30, 0.32, 0.34, 0.33, 0.31,
        0.28, 0.26, 0.25, 0.24, 0.23, 0.22, 0.22, 0.24
    ])
    # ID a bit cheaper when DA is high
    id_prices = da - np.clip(da - da.min(), 0, 0.03)
    id_prices = np.maximum(id_prices, 0.0)
    return da, id_prices


def get_da_id_profiles(price_source, year, da_file, id_file):
    """
    Returns (da_24, id_24) in €/kWh.
    """
    if price_source == "Synthetic example":
        return synthetic_da_id_profiles()

    if price_source == "Built-in historical (data/…csv)":
        da_path = DATA_DIR / f"da_{year}.csv"
        id_path = DATA_DIR / f"id_{year}.csv"
        if not da_path.exists() or not id_path.exists():
            st.warning(f"DA/ID files for year {year} missing in /data, using synthetic.")
            return synthetic_da_id_profiles()
        da_series = load_clean_price_csv(da_path)
        id_series = load_clean_price_csv(id_path)
        return hourly_profile_from_series(da_series), hourly_profile_from_series(id_series)

    if price_source == "Upload DA+ID CSVs":
        if da_file is None or id_file is None:
            st.warning("Upload both DA and ID CSV files or switch source.")
            return synthetic_da_id_profiles()
        da_series = load_clean_price_csv(da_file)
        id_series = load_clean_price_csv(id_file)
        return hourly_profile_from_series(da_series), hourly_profile_from_series(id_series)

    return synthetic_da_id_profiles()


# ============================================================
# EV OPTIMISATION UTILITIES
# ============================================================

def allowed_hours_array(arrival_hour: int, departure_hour: int) -> np.ndarray:
    """
    Boolean array (length 24) indicating when the EV is connected.
    """
    allowed = np.zeros(24, dtype=bool)
    for h in range(24):
        if arrival_hour <= departure_hour:
            allowed[h] = (arrival_hour <= h < departure_hour)
        else:
            # Window wraps around midnight
            allowed[h] = (h >= arrival_hour or h < departure_hour)
    return allowed


def optimise_charging(cost_curve: np.ndarray,
                      allowed: np.ndarray,
                      session_kwh: float,
                      charger_power: float) -> np.ndarray:
    """
    Greedy cheapest-hour optimisation for a single day.
    cost_curve: €/kWh for each hour 0..23.
    """
    if session_kwh <= 0:
        return np.zeros(24)

    charge = np.zeros(24)
    remaining = session_kwh

    # Only allowed hours, sorted by cost
    cheap_hours = sorted(
        [h for h in range(24) if allowed[h]],
        key=lambda h: cost_curve[h]
    )

    for h in cheap_hours:
        if remaining <= 0:
            break
        max_hourly = charger_power
        charge[h] = min(max_hourly, remaining)
        remaining -= charge[h]

    return charge


def session_cost(charge: np.ndarray, retail_curve: np.ndarray) -> float:
    """
    Cost = Σ charge[h] * retail_price[h].
    """
    return float(np.sum(charge * retail_curve))
    # ============================================================
# NAVIGATION
# ============================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page", ["EV Optimizer", "Price Manager"])


# ============================================================
# EV OPTIMIZER PAGE
# ============================================================

if page == "EV Optimizer":

    st.sidebar.title("🚗 EV Hourly Optimizer (Simplified)")

    # -----------------------------
    # BASIC EV PARAMETERS
    # -----------------------------
    charger_power = st.sidebar.number_input(
        "Charger power (kW)",
        min_value=1.0, max_value=22.0,
        value=11.0, step=1.0
    )

    # Charging frequency (sessions/year)
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

    # Arrival / departure window
    arrival_hour = st.sidebar.slider("Arrival hour", 0, 23, 18)
    departure_hour = st.sidebar.slider("Departure hour", 0, 23, 7)

    # Baseline flat retail price
    energy_flat = st.sidebar.number_input(
        "Flat retail price (€/kWh)",
        min_value=0.01, max_value=1.00,
        value=0.30, step=0.01,
        help="Price you pay in a standard flat tariff."
    )

    # -----------------------------
    # CHARGING ENERGY METHOD
    # -----------------------------
    st.sidebar.markdown("### Charging energy method")

    energy_method = st.sidebar.radio(
        "How to define charged energy per session?",
        ["SOC-based charging", "Manual kWh per session"]
    )

    # Placeholder; will be set below
    session_kwh = 0.0

    if energy_method == "SOC-based charging":
        battery_capacity = st.sidebar.number_input(
            "EV battery capacity (kWh)",
            min_value=10.0, max_value=200.0,
            value=60.0, step=1.0
        )

        arrival_soc = st.sidebar.slider(
            "Arrival SOC (%)", min_value=0, max_value=100,
            value=30, step=1
        )
        target_soc = st.sidebar.slider(
            "Target SOC at departure (%)", min_value=0, max_value=100,
            value=80, step=1
        )

        if target_soc < arrival_soc:
            st.sidebar.warning("Target SOC is below arrival SOC → no charging needed.")
            required_kwh = 0.0
        else:
            required_kwh = battery_capacity * (target_soc - arrival_soc) / 100.0

        session_kwh = required_kwh

    else:
        # Manual kWh per session
        session_kwh = st.sidebar.number_input(
            "kWh per charging session (manual)",
            min_value=0.0, max_value=200.0,
            value=20.0, step=0.5,
            help="Energy to charge each time the car is plugged in."
        )

    # -----------------------------
    # PRICE SOURCE SELECTION
    # -----------------------------
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

    # -----------------------------
    # ID FLEXIBILITY (for smart DA+ID scenario)
    # -----------------------------
    id_flex_share = st.sidebar.slider(
        "ID flexibility share",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.05,
        help="Fraction of charging that can exploit intraday (ID) discounts."
    )

    # -----------------------------
    # RETAIL MARKUP (grid fees + taxes)
    # -----------------------------
    st.sidebar.markdown("### Retail markup (grid fees + taxes)")

    da_retail_markup = st.sidebar.number_input(
        "Markup (€/kWh)",
        min_value=0.00, max_value=1.00,
        value=0.12, step=0.01,
        help="All non-energy components (grid fees, taxes, levies, margin)."
    )
        # ============================================================
    # LOAD WHOLESALE PRICES (DA & ID)
    # ============================================================
    da_24, id_24 = get_da_id_profiles(
        price_source, year, da_file, id_file
    )

    # Compute ID discount effect (only in smart DA+ID scenario)
    id_discount = np.clip(da_24 - id_24, 0, None)
    effective_da_id = da_24 - id_discount * id_flex_share

    # ============================================================
    # ALLOWED HOURS WINDOW
    # ============================================================
    allowed = allowed_hours_array(arrival_hour, departure_hour)
    available_hours = allowed.sum()  # number of hours EV is home
    max_possible_kwh = available_hours * charger_power

    # Safety: limit session_kwh to max possible
    if session_kwh > max_possible_kwh:
        st.sidebar.warning(
            f"Requested charge {session_kwh:.1f} kWh exceeds "
            f"maximum possible {max_possible_kwh:.1f} kWh "
            f"in the arrival→departure window. "
            "Capping it to the maximum possible."
        )
        session_kwh = max_possible_kwh

    # ============================================================
    # PRICE CURVES FOR BILLING
    # ============================================================
    # Baseline uses a flat retail price
    baseline_curve = np.full(24, energy_flat)

    # Retail DA-indexed tariff (DA + markup)
    da_indexed_curve = da_24 + da_retail_markup

    # Smart scheduling uses wholesale curves, but billing uses retail
    retail_da_curve = da_24 + da_retail_markup
    retail_da_id_curve = effective_da_id + da_retail_markup

    # ============================================================
    # CHARGING PROFILES (PHYSICAL SCHEDULING)
    # ============================================================

    # Baseline (flat retail; no optimisation)
    charge_baseline = optimise_charging(
        baseline_curve, allowed, session_kwh, charger_power
    )

    # Retail DA-indexed tariff → behaves like baseline (same charging pattern)
    charge_da_indexed = charge_baseline.copy()

    # Smart DA optimisation → schedule using DA wholesale
    charge_smart_da = optimise_charging(
        da_24, allowed, session_kwh, charger_power
    )

    # Smart DA+ID optimisation → schedule using DA+ID wholesale
    charge_smart_full = optimise_charging(
        effective_da_id, allowed, session_kwh, charger_power
    )

    # ============================================================
    # COST PER SESSION & PER YEAR (WITH RETAIL BILLING)
    # ============================================================

    session_cost_baseline = session_cost(charge_baseline, baseline_curve)
    session_cost_da_indexed = session_cost(charge_da_indexed, da_indexed_curve)
    session_cost_smart_da = session_cost(charge_smart_da, retail_da_curve)
    session_cost_smart_full = session_cost(charge_smart_full, retail_da_id_curve)

    annual_baseline = session_cost_baseline * sessions_per_year
    annual_da_indexed = session_cost_da_indexed * sessions_per_year
    annual_smart_da = session_cost_smart_da * sessions_per_year
    annual_smart_full = session_cost_smart_full * sessions_per_year

    total_kwh_year = sessions_per_year * session_kwh

    def eff_price(annual_cost):
        return annual_cost / total_kwh_year if total_kwh_year > 0 else 0.0

    # ============================================================
    # MAIN OUTPUT SUMMARY
    # ============================================================

    st.title("🚗 EV Hourly Optimizer (Simplified)")

    st.markdown(
        f"**Sessions/year:** {sessions_per_year:.1f}  \n"
        f"**Charged energy per session:** {session_kwh:.2f} kWh  \n"
        f"**Total annual charging energy:** {total_kwh_year:.0f} kWh  \n"
        f"**Arrival → Departure:** {arrival_hour:02d}:00 → {departure_hour:02d}:00  \n"
        f"**Retail markup:** {da_retail_markup:.2f} €/kWh"
    )

    # ============================================================
    # RESULTS TABLE
    # ============================================================

    results_df = pd.DataFrame([
        ["Baseline (flat retail)", annual_baseline, eff_price(annual_baseline)],
        ["Retail DA-indexed tariff", annual_da_indexed, eff_price(annual_da_indexed)],
        ["Smart DA optimisation", annual_smart_da, eff_price(annual_smart_da)],
        ["Smart DA+ID optimisation", annual_smart_full, eff_price(annual_smart_full)],
    ], columns=["Scenario", "Annual Cost (€)", "Effective price (€/kWh)"])

    best = results_df.loc[results_df["Annual Cost (€)"].idxmin()]

    st.subheader("Annual Cost Comparison")
    st.dataframe(
        results_df.style.format({
            "Annual Cost (€)": "{:.0f}",
            "Effective price (€/kWh)": "{:.3f}"
        }),
        use_container_width=True
    )

    st.success(
        f"**Best scenario:** {best['Scenario']} → "
        f"**{best['Annual Cost (€)']:.0f} € per year** "
        f"({best['Effective price (€/kWh)']:.3f} €/kWh)"
    )

    # ============================================================
    # COST COMPARISON BAR CHART
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
                alt.Tooltip("Effective price (€/kWh):Q", format=".3f")
            ]
        )
        .properties(height=330)
    )

    st.altair_chart(cost_chart, use_container_width=True)
        # ============================================================
    # SIMPLE & CORRECT CHARGING PATTERN VISUALISATION
    # ============================================================

    st.subheader("🔌 Charging Pattern (Arrival → Departure Window)")

    # Build correct chronological window (handles midnight wrap)
    if arrival_hour <= departure_hour:
        hours_window = list(range(arrival_hour, departure_hour))
    else:
        hours_window = list(range(arrival_hour, 24)) + list(range(0, departure_hour))

    # Preserve exact correct order
    hours_mod = hours_window[:]  # copy for chart sorting

    # Build visualisation DataFrame
    charging_pattern_df = pd.DataFrame({
        "Hour": hours_mod,
        "Baseline (kWh)": [charge_baseline[h] for h in hours_mod],
        "DA-indexed (kWh)": [charge_da_indexed[h] for h in hours_mod],
        "Smart DA (kWh)": [charge_smart_da[h] for h in hours_mod],
        "Smart DA+ID (kWh)": [charge_smart_full[h] for h in hours_mod],
    })

    # Reshape long format
    charging_long = charging_pattern_df.melt(
        id_vars="Hour",
        var_name="Scenario",
        value_name="kWh"
    )

    # 🌟 GROUPED BAR CHART — one panel per scenario (super clear)
    charging_chart = (
        alt.Chart(charging_long)
        .mark_bar()
        .encode(
            x=alt.X(
                "Hour:O",
                title="Hour of Day",
                sort=hours_mod   # 🟦 Force correct hour order
            ),
            y=alt.Y("kWh:Q", title="kWh charged"),
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
        ### How to read this chart

        - Each panel shows **one scenario**.
        - Bars represent **how much energy the EV charges** in each allowed hour.
        - Hours are displayed correctly from **arrival → next-day departure**.
        - Smart charging scenarios shift into the **cheapest hours**.
        - DA-indexed tariff uses **flat pattern** (like baseline) — because retail DA-indexed does not influence the physical pattern.
        """
    )

# ============================================================
# PRICE MANAGER PAGE
# ============================================================

if page == "Price Manager":

    st.title("📈 Price Manager – DA/ID Data Loader (Simplified)")

    st.markdown(
        """
        Upload raw ENTSO-E or compatible CSV files containing wholesale prices.
        This page will:

        - Parse and clean the datetime column  
        - Convert €/MWh → €/kWh  
        - Resample to hourly  
        - Save as `data/da_YEAR.csv` or `data/id_YEAR.csv`  

        No charts are shown here (simplified view).
        """
    )

    # ----------------------------------
    # INPUT: SELECT DA OR ID
    # ----------------------------------
    upload_type = st.selectbox(
        "Select price type",
        ["Day-Ahead (DA)", "Intraday (ID)"]
    )

    raw_file = st.file_uploader(
        "Upload raw price CSV",
        type=["csv"],
        help="Upload ENTSO-E or similar DA/ID CSV."
    )

    if raw_file is not None:

        df_raw = pd.read_csv(raw_file)
        st.subheader("🔍 Raw file preview")
        st.dataframe(df_raw.head(), use_container_width=True)

        # ----------------------------------
        # AUTO-DETECT datetime and price columns
        # ----------------------------------
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
            st.error(
                "❌ Could not detect datetime or price columns automatically.\n"
                "Please ensure columns contain 'MTU', 'date', 'time', 'price', or 'MWh'."
            )

        else:
            st.success(f"Detected datetime column: **{dt_col}**")
            st.success(f"Detected price column: **{price_col}**")

            # ----------------------------------
            # CLEAN & PARSE DATETIME
            # ----------------------------------
            df = df_raw[[dt_col, price_col]].copy()

            try:
                # ENTSO-E style: "2023-01-01 00:00 - 01:00"
                dt_clean = df[dt_col].astype(str).str.split("-").str[0].str.strip()
                df["datetime"] = pd.to_datetime(dt_clean)
            except Exception:
                df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")

            df["price_eur_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
            df = df.dropna(subset=["datetime", "price_eur_mwh"]).sort_values("datetime")

            st.subheader("🧹 Cleaned data preview")
            st.dataframe(df.head(), use_container_width=True)

            # ----------------------------------
            # RESAMPLE TO HOURLY €/kWh
            # ----------------------------------
            df_hourly = (
                df.set_index("datetime")["price_eur_mwh"]
                .resample("1H")
                .mean() / 1000.0
            )

            df_hourly = df_hourly.dropna()

            # ----------------------------------
            # SAVE CLEANED DATA TO /data
            # ----------------------------------
            year_save = st.number_input(
                "Assign this dataset to year:",
                min_value=2000, max_value=2100,
                value=2023, step=1
            )

            if st.button("💾 Save cleaned data to /data folder"):
                name = "da" if upload_type.startswith("Day") else "id"
                out_path = DATA_DIR / f"{name}_{int(year_save)}.csv"

                # Save original-style cleaned data (€/MWh)
                df[["datetime", "price_eur_mwh"]].to_csv(out_path, index=False)

                st.success(f"Saved cleaned file to: `{out_path}`")

    # ----------------------------------
    # SHOW EXISTING CLEANED FILES
    # ----------------------------------
    st.subheader("📂 Existing historical files in /data")

    files = list(DATA_DIR.glob("*.csv"))
    if not files:
        st.info("No CSV files found in `/data` yet.")
    else:
        for f in files:
            st.write("•", f.name)





