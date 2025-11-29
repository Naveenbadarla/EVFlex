import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import io

st.set_page_config(
    page_title="EV Hourly Optimizer (DA + ID + Retail Markup)",
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


def synthetic_da_id_profiles():
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

def allowed_hours_array(arrival_hour, departure_hour):
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


def optimise_charging(cost_curve, allowed, session_kwh, charger_power):
    """
    Greedy cheapest-hour optimisation for a single day.
    cost_curve: €/kWh for each hour 0..23.
    """
    if session_kwh <= 0:
        return np.zeros(24)

    charge = np.zeros(24)
    remaining = session_kwh

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


def session_cost(charge, retail_curve):
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

    st.sidebar.title("🚗 EV Hourly Optimizer")

    # ---- EV inputs ----
    ev_annual_kwh = st.sidebar.number_input(
        "EV annual need (kWh/year)", 0.0, value=2000.0, step=100.0
    )
    charger_power = st.sidebar.number_input(
        "Charger power (kW)", 1.0, 22.0, value=11.0, step=1.0
    )

    freq_choice = st.sidebar.selectbox(
        "Charging frequency",
        ["Every day", "Every 2 days", "Weekdays only",
         "Weekends only", "Custom days per week"]
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

    default_session_kwh = ev_annual_kwh / sessions_per_year if sessions_per_year else 0.0
    session_kwh = st.sidebar.number_input(
        "kWh per charging session",
        0.0, 200.0,
        value=float(round(default_session_kwh, 2)),
        step=0.1,
        help="Energy the EV needs each time it is plugged in."
    )

    arrival_hour = st.sidebar.slider("Arrival hour", 0, 23, 18)
    departure_hour = st.sidebar.slider("Departure hour", 0, 23, 7)

    energy_flat = st.sidebar.number_input(
        "Flat retail price (€/kWh)",
        0.01, 1.0,
        value=0.30,
        step=0.01,
        help="Price you pay in baseline flat tariff."
    )

    # ---- Price source ----
    st.sidebar.markdown("### Price data source")
    price_source = st.sidebar.radio(
        "Source",
        ["Synthetic example", "Built-in historical (data/…csv)", "Upload DA+ID CSVs"]
    )

    year = None
    da_file = None
    id_file = None
    if price_source == "Built-in historical (data/…csv)":
        year = st.sidebar.selectbox("Year", [2022, 2023, 2024])
    elif price_source == "Upload DA+ID CSVs":
        da_file = st.sidebar.file_uploader("Upload DA CSV", type=["csv"])
        id_file = st.sidebar.file_uploader("Upload ID CSV", type=["csv"])

    # ---- ID flexibility ----
    id_flex_share = st.sidebar.slider(
        "ID flexibility share",
        0.0, 1.0,
        value=0.5,
        help="Fraction of charging that can exploit ID discount."
    )

    # ---- Retail markup ----
    st.sidebar.markdown("### Retail markup (grid fees + taxes)")
    da_retail_markup = st.sidebar.number_input(
        "Markup (€/kWh)",
        0.00, 1.00,
        value=0.12,
        step=0.01,
        help="Typical all-in markup in DE ≈ 0.10–0.16 €/kWh."
    )

    # === LOAD WHOLESALE PRICES ===
    da_24, id_24 = get_da_id_profiles(price_source, year, da_file, id_file)

    # ID discount (wholesale)
    id_discount = np.clip(da_24 - id_24, 0, None)
    effective_da_id = da_24 - id_discount * id_flex_share

    # Allowed hours & feasibility
    allowed = allowed_hours_array(arrival_hour, departure_hour)
    available_hours = allowed.sum()
    max_kwh = available_hours * charger_power
    if session_kwh > max_kwh:
        st.sidebar.warning(
            f"Session kWh {session_kwh:.2f} > max possible "
            f"{max_kwh:.2f} kWh → capping."
        )
        session_kwh = max_kwh

    # ============================================================
    # PRICE CURVES (WHOLESALE + RETAIL)
    # ============================================================

    # Retail dynamic tariffs (no smart scheduling)
    da_indexed_curve = da_24 + da_retail_markup
    da_id_indexed_curve = effective_da_id + da_retail_markup

    # Wholesale curves for scheduling
    wholesale_da_curve = da_24
    wholesale_da_id_curve = effective_da_id

    # Retail curves for smart billing
    retail_da_curve = wholesale_da_curve + da_retail_markup
    retail_da_id_curve = wholesale_da_id_curve + da_retail_markup

    # ============================================================
    # CHARGING PROFILES (PER DAY)
    # ============================================================

    # Baseline: no smart shifting, just flat retail price
    baseline_curve = np.full(24, energy_flat)
    charge_baseline = optimise_charging(
        baseline_curve, allowed, session_kwh, charger_power
    )

    # Retail DA & DA+ID tariffs: same physical behaviour as baseline
    charge_da_indexed = charge_baseline.copy()
    charge_da_id_indexed = charge_baseline.copy()

    # Smart DA/DA+ID: schedule using wholesale curves
    charge_da_only = optimise_charging(
        wholesale_da_curve, allowed, session_kwh, charger_power
    )
    charge_full = optimise_charging(
        wholesale_da_id_curve, allowed, session_kwh, charger_power
    )

    # ============================================================
    # COST PER SESSION & PER YEAR
    # ============================================================

    session_cost_baseline = session_cost(charge_baseline, baseline_curve)
    session_cost_da_indexed = session_cost(charge_da_indexed, da_indexed_curve)
    session_cost_da_id_indexed = session_cost(charge_da_id_indexed, da_id_indexed_curve)

    # Smart optimisation: billing with retail curves
    session_cost_smart_da = session_cost(charge_da_only, retail_da_curve)
    session_cost_smart_full = session_cost(charge_full, retail_da_id_curve)

    annual_baseline = session_cost_baseline * sessions_per_year
    annual_da_indexed = session_cost_da_indexed * sessions_per_year
    annual_da_id_indexed = session_cost_da_id_indexed * sessions_per_year
    annual_smart_da = session_cost_smart_da * sessions_per_year
    annual_smart_full = session_cost_smart_full * sessions_per_year

    total_kwh_year = sessions_per_year * session_kwh

    def eff_price(annual):
        return annual / total_kwh_year if total_kwh_year > 0 else 0.0

    # ============================================================
    # MAIN OUTPUT: SUMMARY + TABLE
    # ============================================================

    st.title("🚗 EV Hourly Optimizer (Wholesale Scheduling + Retail Billing)")

    st.markdown(
        f"**Sessions/year:** {sessions_per_year:.1f}  \n"
        f"**kWh/session:** {session_kwh:.2f}  \n"
        f"**Total kWh/year (model):** {total_kwh_year:.0f}  \n"
        f"**Arrival–Departure:** {arrival_hour:02d}:00 → {departure_hour:02d}:00  \n"
        f"**Markup:** {da_retail_markup:.3f} €/kWh"
    )

    results_df = pd.DataFrame([
        ["Baseline (flat retail)", annual_baseline, eff_price(annual_baseline)],
        ["Retail DA-indexed (no shifting)", annual_da_indexed, eff_price(annual_da_indexed)],
        ["Retail DA+ID-indexed (no shifting)", annual_da_id_indexed, eff_price(annual_da_id_indexed)],
        ["Smart DA (wholesale scheduling + retail billing)", annual_smart_da, eff_price(annual_smart_da)],
        ["Smart DA+ID (wholesale scheduling + retail billing)", annual_smart_full, eff_price(annual_smart_full)],
    ], columns=["Scenario", "Annual Cost (€)", "Eff. Price (€/kWh)"])

    best = results_df.loc[results_df["Annual Cost (€)"].idxmin()]

    st.subheader("Annual Cost Comparison")
    st.dataframe(
        results_df.style.format({
            "Annual Cost (€)": "{:.0f}",
            "Eff. Price (€/kWh)": "{:.3f}"
        }),
        use_container_width=True
    )

    st.success(
        f"**Best scenario:** {best['Scenario']}  "
        f"→ **{best['Annual Cost (€)']:.0f} € / year**, "
        f"**{best['Eff. Price (€/kWh)']:.3f} €/kWh**."
    )

    # Cost bar visual
    st.subheader("Cost Comparison (Bar Chart)")
    cost_chart = (
        alt.Chart(results_df)
        .mark_bar()
        .encode(
            x=alt.X("Scenario:N", sort=None),
            y=alt.Y("Annual Cost (€):Q"),
            tooltip=["Scenario", alt.Tooltip("Annual Cost (€):Q", format=".0f")]
        )
        .properties(height=300)
    )
    st.altair_chart(cost_chart, use_container_width=True)

    # ============================================================
    # NEW VISUALS: CHARGING & PRICE PATTERNS
    # ============================================================

    # Window from arrival to departure (may cross midnight)
    st.subheader("🔌 Charging Pattern (Arrival → Departure Window)")

    start = arrival_hour
    end = departure_hour if departure_hour > arrival_hour else departure_hour + 24
    hours_window = list(range(start, end))
    hours_mod = [h % 24 for h in hours_window]

    charge_window_df = pd.DataFrame({
        "Hour": hours_mod,
        "Baseline": [charge_baseline[h % 24] for h in hours_window],
        "Smart DA": [charge_da_only[h % 24] for h in hours_window],
        "Smart DA+ID": [charge_full[h % 24] for h in hours_window],
    })

    charge_long = charge_window_df.melt("Hour", var_name="Scenario", value_name="kWh")

    charge_heat = (
        alt.Chart(charge_long)
        .mark_rect()
        .encode(
            x=alt.X("Hour:O", title="Hour of Day"),
            y=alt.Y("Scenario:N", title="Scenario"),
            color=alt.Color("kWh:Q", scale=alt.Scale(scheme="blues"), title="kWh"),
            tooltip=["Scenario", "Hour", alt.Tooltip("kWh:Q", format=".2f")]
        )
        .properties(height=200)
    )
    st.altair_chart(charge_heat, use_container_width=True)

    st.subheader("💶 Wholesale Price Pattern (DA & DA+ID) in Same Window")

    price_window_df = pd.DataFrame({
        "Hour": hours_mod,
        "DA Price (€/kWh)": [da_24[h % 24] for h in hours_window],
        "DA+ID Price (€/kWh)": [effective_da_id[h % 24] for h in hours_window],
    })

    price_long = price_window_df.melt("Hour", var_name="Price Type", value_name="Price €/kWh")

    price_heat = (
        alt.Chart(price_long)
        .mark_rect()
        .encode(
            x=alt.X("Hour:O", title="Hour of Day"),
            y=alt.Y("Price Type:N", title="Price Signal"),
            color=alt.Color("Price €/kWh:Q", scale=alt.Scale(scheme="reds")),
            tooltip=["Price Type", "Hour", alt.Tooltip("Price €/kWh:Q", format=".3f")]
        )
        .properties(height=150)
    )
    st.altair_chart(price_heat, use_container_width=True)

    # Line chart for charging across full 24h
    st.subheader("📊 Charging per Hour (Full 24h)")

    charge_24_df = pd.DataFrame({
        "Hour": list(range(24)),
        "Baseline": charge_baseline,
        "Smart DA": charge_da_only,
        "Smart DA+ID": charge_full,
    })
    charge_24_long = charge_24_df.melt("Hour", var_name="Scenario", value_name="kWh")

    charge_line = (
        alt.Chart(charge_24_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("Hour:O"),
            y=alt.Y("kWh:Q"),
            color="Scenario:N",
            tooltip=["Scenario", "Hour", alt.Tooltip("kWh:Q", format=".2f")]
        )
        .properties(height=250)
    )
    st.altair_chart(charge_line, use_container_width=True)

    # Line chart for DA & DA+ID prices full 24h
    st.subheader("📈 Wholesale Price Curves (Full 24h)")

    price_24_df = pd.DataFrame({
        "Hour": list(range(24)),
        "DA Price (€/kWh)": da_24,
        "DA+ID Price (€/kWh)": effective_da_id,
    })
    price_24_long = price_24_df.melt("Hour", var_name="Price Type", value_name="Price €/kWh")

    price_line = (
        alt.Chart(price_24_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("Hour:O"),
            y=alt.Y("Price €/kWh:Q"),
            color="Price Type:N",
            tooltip=["Price Type", "Hour", alt.Tooltip("Price €/kWh:Q", format=".3f")]
        )
        .properties(height=250)
    )
    st.altair_chart(price_line, use_container_width=True)

    # ============================================================
    # CSV DOWNLOADS
    # ============================================================

    st.subheader("📥 Download Data")

    # Charging schedule CSV (24h)
    charge_csv = charge_24_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download charging schedule (24h CSV)",
        data=charge_csv,
        file_name="ev_charging_schedule_24h.csv",
        mime="text/csv",
    )

    # Price curves CSV (24h)
    price_csv = price_24_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download wholesale price curves (24h CSV)",
        data=price_csv,
        file_name="wholesale_prices_24h.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        **Interpretation hints**  
        - Heatmap top: where kWh are charged in each scenario within the connection window.  
        - Heatmap bottom: DA vs DA+ID price signals that drive smart charging.  
        - Smart DA uses only DA prices for scheduling, Smart DA+ID uses DA+ID.  
        - All retail costs (including markup) are included in the annual cost table above.
        """
    )

# ============================================================
# PRICE MANAGER PAGE
# ============================================================

if page == "Price Manager":
    st.title("📈 Price Manager – DA/ID Data Cleaner & Loader")

    st.markdown(
        "Upload raw ENTSO-E (or similar) CSVs, clean them, "
        "visualise and save as `data/da_YEAR.csv` or `data/id_YEAR.csv`."
    )

    upload_type = st.selectbox("Price type", ["Day-Ahead (DA)", "Intraday (ID)"])
    raw_file = st.file_uploader("Upload raw CSV", type=["csv"])

    if raw_file is not None:
        df_raw = pd.read_csv(raw_file)
        st.subheader("Raw file preview")
        st.dataframe(df_raw.head(), use_container_width=True)

        # Auto-detect datetime & price columns
        dt_col = None
        price_col = None

        for c in df_raw.columns:
            if "MTU" in c or "date" in c.lower() or "time" in c.lower():
                dt_col = c
                break

        for c in df_raw.columns:
            if "MWh" in c or "price" in c.lower():
                price_col = c
                break

        if dt_col is None or price_col is None:
            st.error(
                "Could not auto-detect datetime/price columns. "
                "Rename columns or adjust detection logic."
            )
        else:
            st.success(f"Detected datetime='{dt_col}', price='{price_col}'")

            df = df_raw[[dt_col, price_col]].copy()

            # ENTSO-E often: "2023-01-01 00:00 - 01:00"
            try:
                dt_str = df[dt_col].astype(str).str.split("-").str[0].str.strip()
                df["datetime"] = pd.to_datetime(dt_str)
            except Exception:
                df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")

            df["price_eur_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
            df = df.dropna(subset=["datetime", "price_eur_mwh"]).sort_values("datetime")

            st.subheader("Cleaned data preview")
            st.dataframe(df.head(), use_container_width=True)

            # Hourly in €/kWh
            df_hourly = df.set_index("datetime")["price_eur_mwh"].resample("1H").mean() / 1000.0
            df_hourly = df_hourly.dropna()

            st.subheader("Hourly prices (€/kWh)")
            st.line_chart(df_hourly)

            avg24 = df_hourly.groupby(df_hourly.index.hour).mean()
            st.subheader("24h average price profile (€/kWh)")
            st.bar_chart(avg24)

            year_save = st.number_input("Assign year", 2000, 2100, value=2023, step=1)

            if st.button("💾 Save cleaned data to /data"):
                name = "da" if upload_type.startswith("Day") else "id"
                out_path = DATA_DIR / f"{name}_{int(year_save)}.csv"
                df[["datetime", "price_eur_mwh"]].to_csv(out_path, index=False)
                st.success(f"Saved cleaned data to: {out_path}")

    st.subheader("Existing historical files in /data")
    files = list(DATA_DIR.glob("*.csv"))
    if not files:
        st.info("No CSV files in /data yet.")
    else:
        for f in files:
            st.write("•", f.name)

    st.markdown(
        """
        **Tips**  
        - Download data from ENTSO-E Transparency for area DE-LU.  
        - Units: €/MWh, any resolution (15min or 60min).  
        - This tool converts to hourly €/kWh and stores clean files for the Optimizer.
        """
    )
