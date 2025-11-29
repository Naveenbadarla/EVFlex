import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="EV Hourly Optimizer (DA + ID)", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# PRICE UTILITIES (shared)
# ============================================================

def load_clean_price_csv(source) -> pd.Series:
    """
    Load a 'datetime, price_eur_mwh' CSV from a Path or file-like
    and return a pandas Series indexed by datetime, values in €/kWh.
    """
    df = pd.read_csv(source)
    if "datetime" not in df.columns:
        raise ValueError("'datetime' column not found in CSV")

    # Try to detect price column
    price_col = None
    for c in df.columns:
        if c.lower().startswith("price"):
            price_col = c
            break
    if price_col is None:
        raise ValueError("No price column found (expected something like 'price_eur_mwh').")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Convert €/MWh -> €/kWh
    df["price_eur_kwh"] = df[price_col] / 1000.0
    return df.set_index("datetime")["price_eur_kwh"]


def hourly_profile_from_series(series: pd.Series) -> np.ndarray:
    """
    Take a datetime-indexed series (hourly or more frequent),
    return a 24-element numpy array with mean price per hour-of-day.
    """
    s = series.copy()
    # Resample to hourly if needed
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
    DA: simple realistic shape.
    ID: slightly lower where DA is high.
    """
    da = np.array([
        0.28, 0.25, 0.22, 0.19, 0.17, 0.15, 0.14, 0.15,
        0.18, 0.20, 0.25, 0.30, 0.32, 0.34, 0.33, 0.31,
        0.28, 0.26, 0.25, 0.24, 0.23, 0.22, 0.22, 0.24
    ])
    # ID slightly cheaper in high-price hours
    id_prices = da - np.clip(da - da.min(), 0, 0.03)
    id_prices = np.maximum(id_prices, 0.0)
    return da, id_prices


def get_da_id_profiles(price_source, year, da_file, id_file):
    """
    Returns (da_24h, id_24h) numpy arrays in €/kWh.
    """
    # Synthetic fallback
    if price_source == "Synthetic example":
        return synthetic_da_id_profiles()

    # Built-in historical
    if price_source == "Built-in historical (data/…csv)":
        if year is None:
            return synthetic_da_id_profiles()
        da_path = DATA_DIR / f"da_{year}.csv"
        id_path = DATA_DIR / f"id_{year}.csv"
        if not da_path.exists() or not id_path.exists():
            st.warning(f"DA/ID files for year {year} not found in data/. Using synthetic prices.")
            return synthetic_da_id_profiles()
        da_series = load_clean_price_csv(da_path)
        id_series = load_clean_price_csv(id_path)
        da_24 = hourly_profile_from_series(da_series)
        id_24 = hourly_profile_from_series(id_series)
        return da_24, id_24

    # User-uploaded CSVs
    if price_source == "Upload DA+ID CSVs":
        if da_file is None or id_file is None:
            st.warning("Please upload both DA and ID CSV files. Using synthetic prices until then.")
            return synthetic_da_id_profiles()
        da_series = load_clean_price_csv(da_file)
        id_series = load_clean_price_csv(id_file)
        da_24 = hourly_profile_from_series(da_series)
        id_24 = hourly_profile_from_series(id_series)
        return da_24, id_24

    # Safety fallback
    return synthetic_da_id_profiles()


# ============================================================
# EV + OPTIMISATION UTILITIES
# ============================================================

def allowed_hours_array(arrival_hour: int, departure_hour: int) -> np.ndarray:
    """
    Boolean array of length 24 indicating when the car is at home.
    """
    allowed = np.zeros(24, dtype=bool)
    for h in range(24):
        if arrival_hour <= departure_hour:
            allowed[h] = (arrival_hour <= h < departure_hour)
        else:
            # Wrap around midnight
            allowed[h] = (h >= arrival_hour or h < departure_hour)
    return allowed


def optimise_charging(cost_curve: np.ndarray,
                      allowed_hours: np.ndarray,
                      session_kwh: float,
                      charger_power: float) -> np.ndarray:
    """
    Greedy optimisation: fill the cheapest allowed hours first.
    cost_curve: €/kWh for each hour 0..23.
    """
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


def session_cost(charge: np.ndarray, curve: np.ndarray) -> float:
    """
    Cost of one charging session in €.
    charge[h] in kWh, curve[h] in €/kWh.
    """
    return float(np.sum(charge * curve))


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page", ["EV Optimizer", "Price Manager"])

# ============================================================
# EV OPTIMIZER PAGE
# ============================================================

if page == "EV Optimizer":
    st.sidebar.markdown("---")
    st.sidebar.title("🚗 EV Hourly Optimizer (DA + ID)")

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
        "Flat retail energy price (€/kWh)",
        min_value=0.01, value=0.30, step=0.01,
        help="Average retail price you pay without DA/ID optimisation."
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

    st.sidebar.markdown("### ID flexibility")

    id_flex_share = st.sidebar.slider(
        "Fraction of charging benefiting from ID (0–1)",
        0.0, 1.0, 0.5,
        help="0 = ignore ID, 1 = all DA-chosen charging can be price-improved by ID."
    )

    # LOAD PRICES (24h DA & ID)
    da_24, id_24 = get_da_id_profiles(price_source, year, da_file, id_file)

    # Discount where ID < DA
    id_discount = np.clip(da_24 - id_24, 0, None)
    # Only some fraction of kWh gets that benefit
    effective_da_id = da_24 - id_discount * id_flex_share

    # ALLOWED HOURS & FEASIBILITY
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

    # COST CURVES PER SCENARIO (€/kWh)
    # 1) Baseline: flat retail price
    baseline_curve = np.full(24, energy_flat)

    # 2) DA only: DA curve (assumed effective energy price)
    da_only_curve = da_24

    # 3) DA + ID: effective DA+ID curve
    full_curve = effective_da_id

    # OPTIMISE DAILY CHARGING PROFILES
    charge_baseline = optimise_charging(baseline_curve, allowed, session_kwh, charger_power)
    charge_da_only = optimise_charging(da_only_curve, allowed, session_kwh, charger_power)
    charge_full = optimise_charging(full_curve, allowed, session_kwh, charger_power)

    # COST PER SESSION & PER YEAR
    session_cost_baseline = session_cost(charge_baseline, baseline_curve)
    session_cost_da_only = session_cost(charge_da_only, da_only_curve)
    session_cost_full = session_cost(charge_full, full_curve)

    annual_baseline = session_cost_baseline * sessions_per_year
    annual_da = session_cost_da_only * sessions_per_year
    annual_full = session_cost_full * sessions_per_year

    total_kwh_year = sessions_per_year * session_kwh if sessions_per_year > 0 else 0.0

    def eff_price(annual_cost):
        return annual_cost / total_kwh_year if total_kwh_year > 0 else 0.0

    # MAIN PAGE OUTPUT
    st.title("🚗 EV Hourly Charging Optimizer (DA + ID)")

    st.write(
        f"**Sessions per year:** {sessions_per_year:.1f}  \n"
        f"**kWh per session:** {session_kwh:.2f}  \n"
        f"**Charger power:** {charger_power:.1f} kW  \n"
        f"**Arrival:** {arrival_hour:02d}:00  –  **Departure:** {departure_hour:02d}:00  \n"
        f"**Price source:** {price_source}"
    )

    df = pd.DataFrame([
        ["Baseline (flat retail price)", annual_baseline, eff_price(annual_baseline)],
        ["DA only", annual_da, eff_price(annual_da)],
        ["DA + ID (full optimisation)", annual_full, eff_price(annual_full)],
    ], columns=["Scenario", "Annual Cost (€)", "Effective price (€/kWh)"])

    best = df.loc[df["Annual Cost (€)"].idxmin()]

    st.subheader("Annual Cost Comparison")
    st.dataframe(
        df.style.format({
            "Annual Cost (€)": "{:.0f}",
            "Effective price (€/kWh)": "{:.3f}"
        }),
        use_container_width=True
    )

    st.success(
        f"**Best option:** {best['Scenario']} – ~{best['Annual Cost (€)']:.0f} €/year "
        f"({best['Effective price (€/kWh)']:.3f} €/kWh)."
    )

    # CHARGING SCHEDULE VISUALISATION
    chart_df = pd.DataFrame({
        "Hour": list(range(24)),
        "Baseline": charge_baseline,
        "DA only": charge_da_only,
        "DA + ID": charge_full,
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
        * Baseline assumes your flat 'retail' price is what you pay without optimisation.  
        * DA and DA+ID scenarios use DA/ID price profiles (historical or uploaded) as effective energy prices.  
        * No grid-fee / §14a logic is included in this version – it's pure energy price optimisation for the EV.
        """
    )

# ============================================================
# PRICE MANAGER PAGE
# ============================================================

if page == "Price Manager":
    st.title("📈 Price Manager – DA/ID Data Tools")

    st.markdown("""
    Use this page to upload, clean, visualize, and save DA/ID price data.
    
    Supported:
    - Raw ENTSO-E CSVs
    - Any table with datetime + €/MWh columns
    - Auto-detect time & price columns
    - Auto-resample to hourly
    - Auto-convert to €/kWh
    - Auto-save as **data/da_YEAR.csv** or **data/id_YEAR.csv**
    """)

    st.subheader("1. Upload Raw Price Data")

    upload_type = st.selectbox(
        "Type of price",
        ["Day-Ahead (DA)", "Intraday (ID)"]
    )
    raw_file = st.file_uploader("Upload raw CSV file", type=["csv"])

    if raw_file:
        df_raw = pd.read_csv(raw_file)
        st.write("### Raw file preview:")
        st.dataframe(df_raw.head(), use_container_width=True)
        
        st.subheader("2. Auto-clean raw file")

        # Auto-detect datetime and price columns
        datetime_col = None
        price_col = None

        for c in df_raw.columns:
            if "MTU" in c or "date" in c.lower() or "time" in c.lower():
                datetime_col = c
                break

        for c in df_raw.columns:
            if "MWh" in c or "price" in c.lower():
                price_col = c
                break
        
        if not datetime_col or not price_col:
            st.error("Could not detect datetime/price columns. Please rename your columns or adjust the detection.")
        else:
            st.success(f"Detected datetime='{datetime_col}', price='{price_col}'")

            df = df_raw[[datetime_col, price_col]].copy()

            # handle ENTSO-E MTU format "2023-01-01 00:00 - 01:00"
            try:
                df["datetime"] = df[datetime_col].astype(str).str.split("-").str[0].str.strip()
                df["datetime"] = pd.to_datetime(df["datetime"])
            except Exception:
                df["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")

            df["price_eur_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
            df = df.dropna(subset=["datetime", "price_eur_mwh"]).sort_values("datetime")

            st.write("### Cleaned data preview:")
            st.dataframe(df.head())

            # Convert to €/kWh
            df["price_eur_kwh"] = df["price_eur_mwh"] / 1000.0

            # Resample hourly
            df_hourly = df.set_index("datetime")["price_eur_kwh"].resample("1H").mean()
            df_hourly = df_hourly.dropna()

            st.write("### Hourly-resampled data (€/kWh):")
            st.line_chart(df_hourly)

            # Build 24h profile
            hourly_24 = df_hourly.groupby(df_hourly.index.hour).mean()

            st.write("### 24-hour average profile (€/kWh):")
            st.bar_chart(hourly_24)

            year_to_save = st.number_input(
                "Select year to assign this dataset to",
                min_value=2000,
                max_value=2100,
                value=2023,
                step=1
            )

            if st.button("💾 Save cleaned data to /data folder"):
                save_name = "da" if upload_type.startswith("Day") else "id"
                out_file = DATA_DIR / f"{save_name}_{year_to_save}.csv"

                df_clean = df[["datetime", "price_eur_mwh"]].copy()
                df_clean.to_csv(out_file, index=False)

                st.success(f"Saved cleaned file as {out_file}")

    st.subheader("3. Existing Historical Price Files in /data")
    price_files = list(DATA_DIR.glob("*.csv"))
    if not price_files:
        st.info("No historical price files found yet.")
    else:
        for f in price_files:
            st.write(f"- {f.name}")

    st.subheader("4. Tips to get high-quality data")
    st.markdown("""
    - Best source: **ENTSO-E Transparency Platform**  
      https://transparency.entsoe.eu  

    - Download **Day-Ahead Prices** and **Intraday Prices**  
      for area: **DE-LU** (Germany).

    - Resolution can be 60min or 15min.  
      The Price Manager auto-resamples to hourly.

    - Units: ENTSO-E gives **€/MWh**.  
      We convert to **€/kWh** automatically.

    - Once saved into `/data`, the EV Optimizer can use these prices
      via the **Built-in historical (data/…csv)** source.
    """)
