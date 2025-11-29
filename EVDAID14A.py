import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="EV Hourly Optimizer (DA + ID + Retail Markup)", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# PRICE UTILITIES
# ============================================================

def load_clean_price_csv(source) -> pd.Series:
    """
    Load a 'datetime, price_eur_mwh' CSV and return a pandas Series indexed by
    datetime, values in €/kWh.
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
        raise ValueError("No price column found in CSV.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    df["price_eur_kwh"] = df[price_col] / 1000.0
    return df.set_index("datetime")["price_eur_kwh"]


def hourly_profile_from_series(series: pd.Series) -> np.ndarray:
    """
    Converts any timestamped price series to a 24-element (hour 0–23)
    average price profile.
    """
    s = series.resample("1H").mean()
    df = s.to_frame("price")
    df["hour"] = df.index.hour
    hourly = df.groupby("hour")["price"].mean()
    hourly = hourly.reindex(range(24), fill_value=hourly.mean())
    return hourly.values


def synthetic_da_id_profiles():
    """
    A realistic fallback 24h price shape.
    """
    da = np.array([
        0.28,0.25,0.22,0.19,0.17,0.15,0.14,0.15,
        0.18,0.20,0.25,0.30,0.32,0.34,0.33,0.31,
        0.28,0.26,0.25,0.24,0.23,0.22,0.22,0.24
    ])
    id_prices = da - np.clip(da - da.min(), 0, 0.03)
    id_prices = np.maximum(id_prices, 0)
    return da, id_prices


def get_da_id_profiles(price_source, year, da_file, id_file):
    """Return (da_24, id_24) in €/kWh."""
    if price_source == "Synthetic example":
        return synthetic_da_id_profiles()

    if price_source == "Built-in historical (data/…csv)":
        da_path = DATA_DIR / f"da_{year}.csv"
        id_path = DATA_DIR / f"id_{year}.csv"
        if not da_path.exists() or not id_path.exists():
            st.warning(f"Historical data for year {year} missing, using synthetic.")
            return synthetic_da_id_profiles()
        da_series = load_clean_price_csv(da_path)
        id_series = load_clean_price_csv(id_path)
        return hourly_profile_from_series(da_series), hourly_profile_from_series(id_series)

    if price_source == "Upload DA+ID CSVs":
        if da_file is None or id_file is None:
            st.warning("Upload both DA and ID CSV files.")
            return synthetic_da_id_profiles()
        da_series = load_clean_price_csv(da_file)
        id_series = load_clean_price_csv(id_file)
        return hourly_profile_from_series(da_series), hourly_profile_from_series(id_series)

    return synthetic_da_id_profiles()


# ============================================================
# EV OPTIMISATION UTILITIES
# ============================================================

def allowed_hours_array(arrival_hour, departure_hour):
    """Return boolean array [24] for when EV is connected."""
    allowed = np.zeros(24, dtype=bool)
    for h in range(24):
        if arrival_hour <= departure_hour:
            allowed[h] = (arrival_hour <= h < departure_hour)
        else:
            allowed[h] = (h >= arrival_hour or h < departure_hour)
    return allowed


def optimise_charging(cost_curve, allowed, session_kwh, charger_power):
    """Greedy cheapest-hour optimiser."""
    if session_kwh <= 0:
        return np.zeros(24)

    charge = np.zeros(24)
    remaining = session_kwh

    cheap_hours = sorted([h for h in range(24) if allowed[h]], key=lambda h: cost_curve[h])

    for h in cheap_hours:
        if remaining <= 0:
            break
        max_hourly = charger_power
        charge[h] = min(max_hourly, remaining)
        remaining -= charge[h]

    return charge


def session_cost(charge, retail_curve):
    """Cost = Σ charge[h] * retail_price[h]."""
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

    st.sidebar.title("🚗 EV Hourly Optimizer (DA + ID + Markup)")

    # ---- EV inputs ----
    ev_annual_kwh = st.sidebar.number_input("EV annual need (kWh/year)", 0.0, value=2000.0)
    charger_power = st.sidebar.number_input("Charger power (kW)", 1.0, 22.0, value=11.0)

    freq_choice = st.sidebar.selectbox(
        "Charging frequency",
        ["Every day","Every 2 days","Weekdays only","Weekends only","Custom days per week"]
    )

    custom_days = 7
    if freq_choice == "Custom days per week":
        custom_days = st.sidebar.slider("Days per week", 1, 7, 4)

    if freq_choice == "Every day": sessions_per_year = 365
    elif freq_choice == "Every 2 days": sessions_per_year = 365/2
    elif freq_choice == "Weekdays only": sessions_per_year = 5*52
    elif freq_choice == "Weekends only": sessions_per_year = 2*52
    else: sessions_per_year = custom_days*52

    default_session_kwh = ev_annual_kwh / sessions_per_year if sessions_per_year else 0

    session_kwh = st.sidebar.number_input(
        "kWh per charging session", 0.0, 200.0, value=round(default_session_kwh, 2)
    )

    arrival_hour = st.sidebar.slider("Arrival hour", 0, 23, 18)
    departure_hour = st.sidebar.slider("Departure hour", 0, 23, 7)

    # Retail flat price
    energy_flat = st.sidebar.number_input(
        "Flat retail price (€/kWh)", 0.01, 1.0, value=0.30
    )

    # ---- Price source ----
    st.sidebar.markdown("### Price data source")
    price_source = st.sidebar.radio("Source", [
        "Synthetic example",
        "Built-in historical (data/…csv)",
        "Upload DA+ID CSVs"
    ])

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
        "ID flexibility share", 0.0, 1.0, value=0.5,
        help="Fraction of charging that can benefit from ID discount."
    )

    # ---- Retail markup ----
    st.sidebar.markdown("### Retail markup (grid fees + taxes)")
    da_retail_markup = st.sidebar.number_input(
        "Markup (€/kWh)", 0.00, 1.00, value=0.12,
        help="Typical total markup ≈ 0.10–0.16 €/kWh."
    )

    # === LOAD PRICES ===
    da_24, id_24 = get_da_id_profiles(price_source, year, da_file, id_file)

    # ID discount (wholesale)
    id_discount = np.clip(da_24 - id_24, 0, None)
    effective_da_id = da_24 - id_discount * id_flex_share

    # Allowed hours check
    allowed = allowed_hours_array(arrival_hour, departure_hour)
    available_hours = allowed.sum()
    max_kwh = available_hours * charger_power
    if session_kwh > max_kwh:
        st.warning(f"Session kWh capped to {max_kwh:.2f}")
        session_kwh = max_kwh

    # ================================================================
    # COST CURVES (WHOLESALE + MARKUP)
    # ================================================================

    # Retail curves (no optimisation behaviour)
    da_indexed_curve = da_24 + da_retail_markup
    da_id_indexed_curve = effective_da_id + da_retail_markup

    # Smart wholesale curves = for scheduling only
    wholesale_da_curve = da_24
    wholesale_da_id_curve = effective_da_id

    # Retails for smart billing:
    retail_da_curve = wholesale_da_curve + da_retail_markup
    retail_da_id_curve = wholesale_da_id_curve + da_retail_markup

    # ================================================================
    # CHARGING PROFILES
    # ================================================================

    # Baseline = no smart shifting
    baseline_curve = np.full(24, energy_flat)
    charge_baseline = optimise_charging(baseline_curve, allowed, session_kwh, charger_power)

    # Retail DA/DA+ID tariffs (no smart shifting)
    charge_da_indexed = charge_baseline.copy()
    charge_da_id_indexed = charge_baseline.copy()

    # Smart wholesale optimization
    charge_da_only = optimise_charging(wholesale_da_curve, allowed, session_kwh, charger_power)
    charge_full = optimise_charging(wholesale_da_id_curve, allowed, session_kwh, charger_power)

    # ================================================================
    # COST PER SESSION
    # ================================================================

    session_cost_baseline = session_cost(charge_baseline, baseline_curve)
    session_cost_da_indexed = session_cost(charge_da_indexed, da_indexed_curve)
    session_cost_da_id_indexed = session_cost(charge_da_id_indexed, da_id_indexed_curve)

    # SMART (billing includes markup)
    session_cost_smart_da = session_cost(charge_da_only, retail_da_curve)
    session_cost_smart_full = session_cost(charge_full, retail_da_id_curve)

    # Annual
    annual_baseline = session_cost_baseline * sessions_per_year
    annual_da_indexed = session_cost_da_indexed * sessions_per_year
    annual_da_id_indexed = session_cost_da_id_indexed * sessions_per_year
    annual_smart_da = session_cost_smart_da * sessions_per_year
    annual_smart_full = session_cost_smart_full * sessions_per_year

    total_kwh_year = sessions_per_year * session_kwh

    def eff(annual):
        return annual/total_kwh_year if total_kwh_year > 0 else 0

    # ================================================================
    # OUTPUT
    # ================================================================

    st.title("🚗 EV Hourly Optimizer (Wholesale Scheduling + Retail Billing)")

    df = pd.DataFrame([
        ["Baseline (flat retail)", annual_baseline, eff(annual_baseline)],
        ["Retail DA-indexed (no shifting)", annual_da_indexed, eff(annual_da_indexed)],
        ["Retail DA+ID-indexed (no shifting)", annual_da_id_indexed, eff(annual_da_id_indexed)],
        ["Smart DA (wholesale scheduling + retail billing)", annual_smart_da, eff(annual_smart_da)],
        ["Smart DA+ID (wholesale scheduling + retail billing)", annual_smart_full, eff(annual_smart_full)],
    ], columns=["Scenario", "Annual Cost (€)", "Eff. Price (€/kWh)"])

    best = df.loc[df["Annual Cost (€)"].idxmin()]

    st.subheader("Annual Cost Comparison")
    st.dataframe(df.style.format({
        "Annual Cost (€)": "{:.0f}",
        "Eff. Price (€/kWh)": "{:.3f}"
    }), use_container_width=True)

    st.success(
        f"BEST: **{best['Scenario']}**, cost **{best['Annual Cost (€)']:.0f} €**, "
        f"effective price **{best['Eff. Price (€/kWh)']:.3f} €/kWh**."
    )

    st.subheader("Charging Schedules")

    chart_df = pd.DataFrame({
        "Hour": list(range(24)),
        "Baseline": charge_baseline,
        "Smart DA": charge_da_only,
        "Smart DA+ID": charge_full,
    })

    chart = (
        alt.Chart(chart_df.melt("Hour"))
        .mark_bar()
        .encode(
            x="Hour:O",
            y="value:Q",
            color="variable:N"
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)


# ============================================================
# PRICE MANAGER PAGE
# ============================================================

if page == "Price Manager":
    st.title("📈 Price Manager – DA/ID Cleaner & Loader")

    upload_type = st.selectbox("Data type", ["Day-Ahead (DA)", "Intraday (ID)"])
    raw_file = st.file_uploader("Upload raw CSV", type=["csv"])

    if raw_file:
        df_raw = pd.read_csv(raw_file)
        st.write("Raw preview:", df_raw.head())

        # ---- Auto-detect datetime and price columns ----
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

        if not dt_col or not price_col:
            st.error("Could not auto-detect datetime or price column.")
        else:
            df = df_raw[[dt_col, price_col]].copy()

            try:
                df["datetime"] = df[dt_col].astype(str).str.split("-").str[0].str.strip()
                df["datetime"] = pd.to_datetime(df["datetime"])
            except:
                df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")

            df["price_eur_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
            df = df.dropna(subset=["datetime", "price_eur_mwh"]).sort_values("datetime")

            st.write("Cleaned preview:", df.head())

            df_hourly = (
                df.set_index("datetime")["price_eur_mwh"]
                .resample("1H").mean() / 1000.0
            )
            st.write("Hourly prices (€/kWh):")
            st.line_chart(df_hourly)

            avg24 = df_hourly.groupby(df_hourly.index.hour).mean()
            st.write("24h profile:")
            st.bar_chart(avg24)

            year = st.number_input("Assign year", 2000, 2100, value=2023)

            if st.button("Save cleaned data"):
                name = "da" if upload_type.startswith("Day") else "id"
                out = DATA_DIR / f"{name}_{year}.csv"
                df[["datetime","price_eur_mwh"]].to_csv(out, index=False)
                st.success(f"Saved to {out}")

    st.subheader("Available historical data:")
    for f in DATA_DIR.glob("*.csv"):
        st.write("-", f.name)
