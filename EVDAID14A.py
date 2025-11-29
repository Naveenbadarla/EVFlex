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
    Load a 'datetime, price_eur_mwh' CSV and return a cleaned 15-min series
    indexed by datetime with values in €/kWh.
    """
    df = pd.read_csv(source)

    # Detect datetime column
    dt_col = None
    for c in df.columns:
        if ("MTU" in c) or ("date" in c.lower()) or ("time" in c.lower()):
            dt_col = c
            break
    if dt_col is None:
        raise ValueError("No datetime column found.")

    # Detect price column
    price_col = None
    for c in df.columns:
        if ("MWh" in c) or ("price" in c.lower()):
            price_col = c
            break
    if price_col is None:
        raise ValueError("No price column found.")

    # Clean datetime (handle ENTSO-E MTU)
    try:
        dt_clean = df[dt_col].astype(str).split("-").str[0].str.strip()
        df["datetime"] = pd.to_datetime(dt_clean)
    except Exception:
        df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")

    # Convert €/MWh → €/kWh
    df["price_eur_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["datetime", "price_eur_mwh"])
    df["price_eur_kwh"] = df["price_eur_mwh"] / 1000.0

    # Resample to exactly 15 minutes
    s = df.set_index("datetime")["price_eur_kwh"].resample("15min").mean()
    return s.dropna()


def series_to_96_slots(series_15min: pd.Series) -> np.ndarray:
    """
    Extract exactly 96 quarter-hours from the first full day of data.
    """
    if len(series_15min) < 96:
        raise ValueError("Series must contain at least 96 quarter-hours.")
    return series_15min.iloc[:96].values  # (96,)


def synthetic_da_id_profiles_96():
    """
    Synthetic fallback: repeat the hourly DA/ID x4 to produce 96 slots.
    """
    da = np.array([
        0.28, 0.25, 0.22, 0.19, 0.17, 0.15, 0.14, 0.15,
        0.18, 0.20, 0.25, 0.30, 0.32, 0.34, 0.33, 0.31,
        0.28, 0.26, 0.25, 0.24, 0.23, 0.22, 0.22, 0.24
    ])
    id_prices = da - np.clip(da - da.min(), 0, 0.03)
    id_prices = np.maximum(id_prices, 0.0)

    da_96 = np.repeat(da, 4)
    id_96 = np.repeat(id_prices, 4)
    return da_96, id_96


def get_da_id_profiles_96(price_source, year, da_file, id_file):
    """
    Returns (da_96, id_96) in €/kWh (length 96 each).
    """
    if price_source == "Synthetic example":
        return synthetic_da_id_profiles_96()

    if price_source == "Built-in historical (data/…csv)":
        da_path = DATA_DIR / f"da_{year}.csv"
        id_path = DATA_DIR / f"id_{year}.csv"
        if not da_path.exists() or not id_path.exists():
            st.warning("Missing DA/ID files → using synthetic 96-slot prices.")
            return synthetic_da_id_profiles_96()
        da_series = load_price_15min(da_path)
        id_series = load_price_15min(id_path)
        return series_to_96_slots(da_series), series_to_96_slots(id_series)

    if price_source == "Upload DA+ID CSVs":
        if da_file is None or id_file is None:
            st.warning("Upload both DA and ID CSV or change source.")
            return synthetic_da_id_profiles_96()
        da_series = load_price_15min(da_file)
        id_series = load_price_15min(id_file)
        return series_to_96_slots(da_series), series_to_96_slots(id_series)

    return synthetic_da_id_profiles_96()
    # ============================================================
# EV OPTIMISATION UTILITIES (15-MIN SLOTS)
# ============================================================

def allowed_slots_96(arrival_hour: int, departure_hour: int) -> np.ndarray:
    """
    Boolean array (length 96) indicating when the EV is connected.
    Slots:
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
            # Window wraps past midnight
            allowed[s] = (s >= start or s < end)

    return allowed


def optimise_charging_96(cost_curve_96: np.ndarray,
                         allowed_96: np.ndarray,
                         session_kwh: float,
                         charger_power: float) -> np.ndarray:
    """
    Greedy cheapest-slot optimisation for a single day at 15-min resolution.
    cost_curve_96: €/kWh for each slot (0..95).
    """
    if session_kwh <= 0:
        return np.zeros(96)

    charge = np.zeros(96)
    remaining = session_kwh
    slot_energy = charger_power * 0.25  # kWh per 15-min slot

    # Cheapest allowed slots first
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
    """
    Cost = Σ (charge[s] * retail_price[s]) over 96 slots.
    """
    return float(np.sum(charge_96 * retail_curve_96))


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page", ["EV Optimizer (15-min)", "Price Manager"])


# ============================================================
# EV OPTIMIZER PAGE (15-MIN ENGINE)
# ============================================================

if page == "EV Optimizer (15-min)":

    st.sidebar.title("🚗 EV Optimizer (15-min precision)")

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

    # Arrival / departure window (whole hours)
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
    # CHARGING ENERGY METHOD (SOC vs kWh/session)
    # -----------------------------
    st.sidebar.markdown("### Charging energy method")

    energy_method = st.sidebar.radio(
        "How to define charged energy per session?",
        ["SOC-based charging", "Manual kWh per session"]
    )

    session_kwh = 0.0  # will be set below

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

        if target_soc <= arrival_soc:
            st.sidebar.warning("Target SOC is <= arrival SOC → no charging needed.")
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
    # PRICE SOURCE SELECTION (DA & ID, 15-min)
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
    # LOAD WHOLESALE PRICES (DA & ID, 96 SLOTS)
    # ============================================================
    da_96, id_96 = get_da_id_profiles_96(
        price_source, year, da_file, id_file
    )

    # Compute ID discount effect (only in smart DA+ID scenario)
    id_discount_96 = np.clip(da_96 - id_96, 0, None)
    effective_da_id_96 = da_96 - id_discount_96 * id_flex_share

    # ============================================================
    # ALLOWED SLOTS WINDOW (96 QUARTER-HOURS)
    # ============================================================
    allowed_96 = allowed_slots_96(arrival_hour, departure_hour)
    available_slots = allowed_96.sum()  # number of 15-min intervals EV is home

    slot_energy = charger_power * 0.25  # kWh per 15-min

    max_possible_kwh = available_slots * slot_energy

    # Safety: limit session_kwh to max possible (your chosen Option A)
    if session_kwh > max_possible_kwh:
        st.sidebar.warning(
            f"Requested charge {session_kwh:.1f} kWh exceeds "
            f"maximum possible {max_possible_kwh:.1f} kWh "
            f"in the arrival→departure window. "
            "Capping it to the maximum possible."
        )
        session_kwh = max_possible_kwh

    # ============================================================
    # PRICE CURVES FOR BILLING (96 SLOTS)
    # ============================================================

    # Baseline uses a flat retail price (16 slots per hour)
    baseline_curve_96 = np.full(96, energy_flat)

    # Retail DA-indexed tariff (DA + markup) – used for DA-indexed scenario
    da_indexed_curve_96 = da_96 + da_retail_markup

    # Retail curves for smart scenarios:
    retail_da_96 = da_96 + da_retail_markup
    retail_da_id_96 = effective_da_id_96 + da_retail_markup

    # ============================================================
    # CHARGING PROFILES (15-MIN SMART SCHEDULING)
    # ============================================================

    # Baseline (flat retail; no optimisation)
    charge_baseline_96 = optimise_charging_96(
        baseline_curve_96, allowed_96, session_kwh, charger_power
    )

    # Retail DA-indexed (no smart shifting; behaves like baseline)
    charge_da_indexed_96 = charge_baseline_96.copy()

    # Smart DA optimisation (schedule using wholesale DA)
    charge_smart_da_96 = optimise_charging_96(
        da_96, allowed_96, session_kwh, charger_power
    )

    # Smart DA+ID optimisation (schedule using DA+ID wholesale)
    charge_smart_full_96 = optimise_charging_96(
        effective_da_id_96, allowed_96, session_kwh, charger_power
    )

    # ============================================================
    # SESSION COST (WITH RETAIL BILLING)
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
        return annual_cost / total_kwh_year if total_kwh_year > 0 else 0.0

    # ============================================================
    # SUMMARY DISPLAY
    # ============================================================

    st.title("🚗 EV Optimizer (15-min Resolution)")

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
    # SIMPLE CHARGING PATTERN VISUALISATION (15-min → hourly)
    # ============================================================

    st.subheader("🔌 Charging Pattern (Arrival → Departure Window)")

    # ------------------------------
    # Convert 96-slot (15-min) charge arrays into 24 hourly totals
    # ------------------------------
    # reshape(24,4) sums each group of 4 quarter-hours → 1 hour
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
    # Build DataFrame for chart
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
    # Cleanest visualization: one panel per scenario
    # ------------------------------
    charging_chart = (
        alt.Chart(charging_long)
        .mark_bar()
        .encode(
            x=alt.X(
                "Hour:O",
                title="Hour of Day",
                sort=hours_window      # ensures correct chronological order
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
        ### How to read this chart
        - Only the hours **between arrival and departure** are shown.
        - Each scenario appears in its **own panel** (easier to compare).
        - Smart DA and Smart DA+ID shift charging into the **cheapest 15-minute blocks**, aggregated hourly for clarity.
        - Baseline and DA-indexed scenarios usually look similar because retail DA-indexing does not change physical charging behavior.
        """
    )
    # ============================================================
# PRICE MANAGER PAGE (15-MIN SIMPLIFIED)
# ============================================================

if page == "Price Manager":

    st.title("📈 Price Manager – 15-min DA/ID Loader")

    st.markdown(
        """
        Upload raw ENTSO-E or compatible CSVs containing **15-min DA/ID prices**.

        This tool will:
        - Parse datetime  
        - Convert €/MWh → €/kWh  
        - Align everything to exact **15-minute intervals**  
        - Save as `data/da_YEAR.csv` or `data/id_YEAR.csv`  

        ⚠️ In this simplified version:
        - No charts
        - No visualisation
        - Only preview + cleaning + saving
        """
    )

    # -----------------------------
    # SELECT PRICE TYPE
    # -----------------------------
    upload_type = st.selectbox(
        "Select price type",
        ["Day-Ahead (DA)", "Intraday (ID)"]
    )

    raw_file = st.file_uploader(
        "Upload raw price CSV",
        type=["csv"],
        help="Upload ENTSO-E or any CSV containing 15-min DA/ID prices."
    )

    if raw_file is not None:

        df_raw = pd.read_csv(raw_file)

        st.subheader("🔍 Raw file preview")
        st.dataframe(df_raw.head(), use_container_width=True)

        # -----------------------------
        # AUTO-DETECT datetime & price columns
        # -----------------------------
        dt_col = None
        price_col = None

        # Detect datetime
        for c in df_raw.columns:
            if ("MTU" in c) or ("date" in c.lower()) or ("time" in c.lower()):
                dt_col = c
                break

        # Detect price (€/MWh)
        for c in df_raw.columns:
            if ("MWh" in c) or ("price" in c.lower()):
                price_col = c
                break

        if dt_col is None or price_col is None:
            st.error(
                "❌ Could not detect datetime or price columns. "
                "Please clean the file manually."
            )
        else:
            st.success(f"Detected datetime column: **{dt_col}**")
            st.success(f"Detected price column: **{price_col}**")

            # -----------------------------
            # CLEAN DATETIME
            # -----------------------------
            try:
                # ENTSO-E MTU format: "2023-01-01 00:00 - 01:00"
                dt_clean = df_raw[dt_col].astype(str).str.split("-").str[0].str.strip()
                df_raw["datetime"] = pd.to_datetime(dt_clean)
            except Exception:
                df_raw["datetime"] = pd.to_datetime(df_raw[dt_col], errors="coerce")

            # -----------------------------
            # CONVERT €/MWh → €/kWh
            # -----------------------------
            df_raw["price_eur_mwh"] = pd.to_numeric(df_raw[price_col], errors="coerce")
            df = df_raw.dropna(subset=["datetime", "price_eur_mwh"]).copy()
            df["price_eur_kwh"] = df["price_eur_mwh"] / 1000.0

            st.subheader("🧹 Cleaned data preview")
            st.dataframe(df.head(), use_container_width=True)

            # -----------------------------
            # RESAMPLE TO EXACT 15-MIN FREQUENCY
            # -----------------------------
            df_15 = (
                df.set_index("datetime")["price_eur_kwh"]
                .resample("15min")
                .mean()
            ).dropna()

            if len(df_15) < 96:
                st.warning(
                    "The data does not contain at least 96 quarter-hours. "
                    "Make sure you uploaded a full day."
                )

            # -----------------------------
            # SAVE CLEANED DATA TO /data
            # -----------------------------
            year_save = st.number_input(
                "Assign this dataset to year:",
                min_value=2000, max_value=2100,
                value=2023, step=1
            )

            if st.button("💾 Save cleaned 15-min price data"):
                name = "da" if upload_type.startswith("Day") else "id"
                out_path = DATA_DIR / f"{name}_{int(year_save)}.csv"

                out_df = pd.DataFrame({
                    "datetime": df_15.index,
                    "price_eur_kwh": df_15.values * 1000   # store back as €/MWh like ENTSO-E
                })

                out_df.to_csv(out_path, index=False)
                st.success(f"Saved cleaned file to: `{out_path}`")

    # -----------------------------
    # SHOW EXISTING CLEANED FILES
    # -----------------------------
    st.subheader("📂 Existing cleaned files in /data")

    files = list(DATA_DIR.glob("*.csv"))
    if not files:
        st.info("No cleaned 15-min price files yet.")
    else:
        for f in files:
            st.write("•", f.name)

# ============================================================
# FOOTER (OPTIONAL)
# ============================================================

def app_footer():
    st.markdown(
        """
        <hr>

        <div style="font-size:13px; color:#666; text-align:center;">
        EV Optimizer – 15-Minute Resolution<br>
        Smart DA & Smart DA+ID Scheduling Engine<br>
        Built with Streamlit · Altair · NumPy · Pandas<br><br>
        Version: 2.0 (15-min upgrade)
        </div>
        """,
        unsafe_allow_html=True
    )

app_footer()






