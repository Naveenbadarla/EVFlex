import numpy as np
import pandas as pd
import streamlit as st


# ---------- Synthetic daily price model (DA hourly + ID 15-min) ----------

def get_synthetic_daily_price_profiles():
    """
    Returns synthetic daily profiles:
        da_hourly_eur_mwh: np.array shape (24,)
        id_quarter_eur_mwh: np.array shape (96,)
    DA is hourly, ID is 15-minute.
    """
    hours = np.arange(24)

    # Synthetic DA: cheap at night, more expensive in morning/evening
    base = 70
    morning_peak = 15 * np.sin((hours - 7) / 24 * 2 * np.pi) ** 2
    evening_peak = 35 * np.sin((hours - 18) / 24 * 2 * np.pi) ** 4
    da_hourly = base + morning_peak + evening_peak  # €/MWh

    # Synthetic ID: DA +/- some intra-hour noise on 15min resolution
    rng = np.random.default_rng(42)
    da_quarter = np.repeat(da_hourly, 4)
    noise = rng.normal(0, 6, size=96)  # +/- 6 €/MWh
    id_quarter = da_quarter + noise

    return da_hourly, id_quarter


# ---------- Charging frequency system (returns explicit day indices) ----------

DAY_NAME_TO_IDX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def compute_charging_days(pattern, num_days, custom_weekdays=None, sessions_per_week=None):
    """
    Return a list of day indices [0 .. num_days-1] on which charging occurs.
    Assume day 0 is a Monday (for weekday/weekend logic).
    """
    if custom_weekdays is None:
        custom_weekdays = []

    if pattern == "Every day":
        return list(range(num_days))

    if pattern == "Every other day":
        return list(range(0, num_days, 2))

    if pattern == "Weekdays only (Mon–Fri)":
        days = []
        for d in range(num_days):
            dow = d % 7
            if dow in {0, 1, 2, 3, 4}:
                days.append(d)
        return days

    if pattern == "Weekends only (Sat–Sun)":
        days = []
        for d in range(num_days):
            dow = d % 7
            if dow in {5, 6}:
                days.append(d)
        return days

    if pattern == "Custom weekdays":
        include_dows = {DAY_NAME_TO_IDX[name] for name in custom_weekdays}
        days = []
        for d in range(num_days):
            dow = d % 7
            if dow in include_dows:
                days.append(d)
        return days

    if pattern == "Custom: X sessions per week":
        if sessions_per_week is None or sessions_per_week <= 0:
            return []
        sessions_per_week = int(sessions_per_week)
        days = []
        num_weeks = num_days // 7
        for w in range(num_weeks):
            # Spread sessions evenly over the 7 days of each week
            for s in range(sessions_per_week):
                day_in_week = int(np.floor(s * 7 / sessions_per_week))
                d = w * 7 + day_in_week
                if d < num_days:
                    days.append(d)
        # If num_days is not multiple of 7, ignore remainder for simplicity
        return days

    # Fallback: every day
    return list(range(num_days))


# ---------- Time window helpers (15-min granularity) ----------

def build_available_quarters(arrival_time_h, departure_time_h):
    """
    Convert arrival & departure times (in hours, 0–24) into a list of
    15-minute slot indices in [0, 95] for a single day.

    Supports cross-midnight:
        e.g. arrival=18, departure=6 -> evening + night window.
    """
    arrival_q = int(round(arrival_time_h * 4)) % 96
    departure_q = int(round(departure_time_h * 4)) % 96

    # arrival == departure -> full-day window
    if arrival_q == departure_q:
        return list(range(96)), arrival_q, departure_q

    if arrival_q < departure_q:
        available = list(range(arrival_q, departure_q))
    else:
        available = list(range(arrival_q, 96)) + list(range(0, departure_q))

    return available, arrival_q, departure_q


# ---------- Tariffs + cost calculations ----------

def apply_tariffs(price_kwh, grid_charges, taxes_levies, vat_percent):
    """
    Applies realistic electricity tariff components:
        (energy_price + grid_charges + taxes_levies) * (1 + VAT)
    """
    pre_vat = price_kwh + grid_charges + taxes_levies
    return pre_vat * (1 + vat_percent / 100)


def compute_da_indexed_baseline_daily(
    energy_kwh,
    max_power_kw,
    available_quarters,
    da_hourly_day_eur_mwh,
    grid_charges,
    taxes_levies,
    vat_percent,
):
    """
    Scenario 2: DA-indexed customer, no optimisation.
    Baseline sequential charging from arrival, with DA prices + tariffs for ONE day.
    """
    da_hourly_kwh = da_hourly_day_eur_mwh / 1000.0  # €/kWh
    da_quarter_kwh = np.repeat(da_hourly_kwh, 4)    # 96 values

    energy_remaining = energy_kwh
    total_cost = 0.0
    max_energy_per_q = max_power_kw * 0.25  # kWh per 15 min

    for q in available_quarters:
        if energy_remaining <= 1e-6:
            break

        e = min(max_energy_per_q, energy_remaining)
        unit_cost = apply_tariffs(da_quarter_kwh[q], grid_charges, taxes_levies, vat_percent)
        total_cost += e * unit_cost
        energy_remaining -= e

    if energy_remaining > 1e-6:
        raise ValueError("Time window too short to deliver requested energy.")

    return total_cost


def compute_optimised_daily_cost(
    energy_kwh,
    max_power_kw,
    available_quarters,
    price_quarter_day_eur_mwh,
    grid_charges,
    taxes_levies,
    vat_percent,
):
    """
    For DA-optimised and DA+ID-optimised for ONE day.
    Optimises within the available quarters using the given price_quarter (€/MWh).
    """
    price_kwh = price_quarter_day_eur_mwh / 1000.0
    max_energy_per_q = max_power_kw * 0.25

    max_deliverable = len(available_quarters) * max_energy_per_q
    if max_deliverable + 1e-6 < energy_kwh:
        raise ValueError(
            f"Charging window too short: max {max_deliverable:.1f} kWh, "
            f"{energy_kwh:.1f} kWh requested."
        )

    sorted_q = sorted(available_quarters, key=lambda q: price_kwh[q])

    energy_remaining = energy_kwh
    total_cost = 0.0

    for q in sorted_q:
        if energy_remaining <= 1e-6:
            break

        e = min(max_energy_per_q, energy_remaining)
        unit_cost = apply_tariffs(price_kwh[q], grid_charges, taxes_levies, vat_percent)
        total_cost += e * unit_cost
        energy_remaining -= e

    return total_cost


# ---------- Helpers: loading yearly price series ----------

def load_price_series_from_csv(uploaded_file, expected_multiple, label):
    """
    Load a 1D price series from a CSV.

    - Tries 'price' column first, otherwise first numeric column.
    - Ensures length is a multiple of expected_multiple (24 for DA, 96 for ID).
    Returns:
        series (np.array), num_days
    """
    df = pd.read_csv(uploaded_file)

    if "price" in df.columns:
        series = df["price"].values
    else:
        num_cols = df.select_dtypes(include="number")
        if num_cols.shape[1] == 0:
            st.error(f"{label}: No numeric columns found in uploaded file.")
            return None, 0
        series = num_cols.iloc[:, 0].values

    if len(series) % expected_multiple != 0:
        st.error(
            f"{label}: Length of series ({len(series)}) is not a multiple of {expected_multiple}. "
            f"Expected full-year data, e.g. 8760 for DA (24*365) or 35040 for ID (96*365)."
        )
        return None, 0

    num_days = len(series) // expected_multiple
    return series.astype(float), num_days


# ---------- Streamlit UI ----------

def main():
    st.title("EV Charging: Flat vs DA vs DA-Optimised vs DA+ID-Optimised (Full-Year DA+ID)")

    st.write(
        """
        This app compares **4 customer types** over a full year of prices:

        1. **Flat retail price** (fixed €/kWh, no market exposure)  
        2. **DA-indexed** (hourly dynamic price, no optimisation)  
        3. **DA-optimised** (smart charging using DA only)  
        4. **DA+ID-optimised** (smart charging using min(DA, ID))  

        You can upload **full-year DA & ID price series** or use **synthetic data**.
        """
    )

    # ---------- Sidebar: EV & usage ----------
    st.sidebar.header("EV & Energy Inputs")

    energy_kwh = st.sidebar.number_input(
        "Energy per charging session (kWh)",
        min_value=1.0,
        max_value=200.0,
        value=40.0,
        step=1.0,
    )

    max_power_kw = st.sidebar.number_input(
        "Max charging power (kW)",
        min_value=1.0,
        max_value=50.0,
        value=11.0,
        step=0.5,
    )

    st.sidebar.header("Arrival & Departure (0–24 h)")

    arrival_time_h = st.sidebar.slider(
        "Arrival time [h]",
        min_value=0.0,
        max_value=24.0,
        value=18.0,
        step=0.25,
        help="E.g. 18.5 = 18:30",
    )

    departure_time_h = st.sidebar.slider(
        "Departure time [h]",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.25,
        help="Can be earlier than arrival to represent next day.",
    )

    # ---------- Sidebar: Charging frequency ----------
    st.sidebar.header("Charging Frequency")

    frequency_mode = st.sidebar.selectbox(
        "Charging pattern",
        (
            "Every day",
            "Every other day",
            "Weekdays only (Mon–Fri)",
            "Weekends only (Sat–Sun)",
            "Custom weekdays",
            "Custom: X sessions per week",
        ),
    )

    custom_weekdays = []
    sessions_per_week = None

    if frequency_mode == "Custom weekdays":
        custom_weekdays = st.sidebar.multiselect(
            "Select weekdays",
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            default=["Mon", "Wed", "Fri"],
        )

    if frequency_mode == "Custom: X sessions per week":
        sessions_per_week = st.sidebar.number_input(
            "Sessions per week",
            min_value=1,
            max_value=14,
            value=3,
            step=1,
        )

    # ---------- Sidebar: Tariffs ----------
    st.sidebar.header("Tariffs")

    st.sidebar.subheader("1) Flat Retail Tariff (Scenario 1)")

    flat_retail_price = st.sidebar.number_input(
        "Flat all-in retail price (€/kWh)",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.01,
        help="Final €/kWh paid by flat-price customer (energy + grid + taxes + VAT).",
    )

    st.sidebar.subheader("2) Dynamic Tariff Components (Scenarios 2–4)")

    grid_charges = st.sidebar.number_input(
        "Grid network charges (€/kWh)",
        min_value=0.0,
        max_value=1.0,
        value=0.11,
        step=0.01,
        help="DSO + TSO variable network fees (Germany example ≈ 0.09–0.13 €/kWh).",
    )

    taxes_levies = st.sidebar.number_input(
        "Taxes & levies (€/kWh)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        help="Electricity tax + other surcharges (Germany example ≈ 0.04–0.06 €/kWh).",
    )

    vat_percent = st.sidebar.number_input(
        "VAT (%)",
        min_value=0.0,
        max_value=30.0,
        value=19.0,
        step=1.0,
        help="Standard VAT (Germany: 19%).",
    )

    # ---------- Sidebar: Upload market data ----------
    st.sidebar.header("Upload Market Data (Optional)")

    da_file = st.sidebar.file_uploader(
        "Upload DA prices (hourly, CSV, full year: 24 * days rows)",
        type=["csv"],
        key="da_file",
    )

    id_file = st.sidebar.file_uploader(
        "Upload ID prices (15-min, CSV, full year: 96 * days rows)",
        type=["csv"],
        key="id_file",
    )

    # ---------- Build yearly DA & ID arrays ----------
    # Step 1: synthetic daily patterns
    da_daily_syn, id_daily_syn = get_synthetic_daily_price_profiles()

    # Step 2: if upload is present, load; otherwise use synthetic repeated over 365 days
    da_hourly_year = None
    id_quarter_year = None
    num_days_da = 365
    num_days_id = 365

    if da_file is not None:
        da_series, num_days_da = load_price_series_from_csv(da_file, 24, "DA prices")
        if da_series is not None:
            da_hourly_year = da_series
    else:
        da_hourly_year = np.tile(da_daily_syn, 365)
        num_days_da = 365

    if id_file is not None:
        id_series, num_days_id = load_price_series_from_csv(id_file, 96, "ID prices")
        if id_series is not None:
            id_quarter_year = id_series
    else:
        id_quarter_year = np.tile(id_daily_syn, 365)
        num_days_id = 365

    # Determine how many full days we have consistently
    num_days_year = min(num_days_da, num_days_id)
    if num_days_year == 0:
        st.error("No valid price data loaded. Check DA/ID uploads.")
        return

    st.sidebar.info(f"Using **{num_days_year} days** of market data.")

    # Truncate to common number of days
    da_hourly_year = da_hourly_year[: num_days_year * 24]
    id_quarter_year = id_quarter_year[: num_days_year * 96]

    # ---------- Charging days over the year ----------
    charging_days = compute_charging_days(
        frequency_mode,
        num_days_year,
        custom_weekdays=custom_weekdays,
        sessions_per_week=sessions_per_week,
    )
    days_per_year = len(charging_days)
    st.sidebar.info(f"Charging occurs on **{days_per_year}** days (pattern + data length).")

    # ---------- Inspect price profiles for the first day ----------
    with st.expander("Show DA & ID price profiles for Day 1 (index 0)"):
        da_day0 = da_hourly_year[0:24]
        id_day0 = id_quarter_year[0:96]
        da_quarter0 = np.repeat(da_day0, 4)
        effective0 = np.minimum(da_quarter0, id_day0)

        times_str = [f"{int(i//4):02d}:{int((i%4)*15):02d}" for i in range(96)]
        df_prices0 = pd.DataFrame(
            {
                "DA (€/MWh)": da_quarter0,
                "ID (€/MWh)": id_day0,
                "Effective min(DA, ID) (€/MWh)": effective0,
            },
            index=times_str,
        )
        df_prices0.index.name = "Time"
        st.dataframe(df_prices0.style.format("{:.1f}"))
        st.line_chart(df_prices0)

    # ---------- Compute costs over the selected days ----------
    try:
        # Single-day time window (same pattern used for all days)
        available_quarters, arrival_q, departure_q = build_available_quarters(
            arrival_time_h, departure_time_h
        )

        if len(available_quarters) == 0:
            st.error("Selected charging window is empty.")
            return

        # Scenario 1: Flat retail price customer (no market exposure)
        flat_daily_cost = energy_kwh * flat_retail_price

        flat_annual = flat_daily_cost * days_per_year

        # Initialise sums for scenarios 2–4
        da_indexed_annual = 0.0
        da_optimised_annual = 0.0
        da_id_optimised_annual = 0.0

        for day in charging_days:
            # Extract this day's price slices
            da_day = da_hourly_year[day * 24 : (day + 1) * 24]
            id_day = id_quarter_year[day * 96 : (day + 1) * 96]
            da_quarter_day = np.repeat(da_day, 4)
            effective_day = np.minimum(da_quarter_day, id_day)

            # Scenario 2: DA-indexed baseline (no optimisation)
            daily_da_indexed = compute_da_indexed_baseline_daily(
                energy_kwh,
                max_power_kw,
                available_quarters,
                da_day,
                grid_charges,
                taxes_levies,
                vat_percent,
            )
            da_indexed_annual += daily_da_indexed

            # Scenario 3: DA-optimised (smart, DA only)
            daily_da_opt = compute_optimised_daily_cost(
                energy_kwh,
                max_power_kw,
                available_quarters,
                da_quarter_day,
                grid_charges,
                taxes_levies,
                vat_percent,
            )
            da_optimised_annual += daily_da_opt

            # Scenario 4: DA+ID-optimised (smart, DA+ID)
            daily_da_id_opt = compute_optimised_daily_cost(
                energy_kwh,
                max_power_kw,
                available_quarters,
                effective_day,
                grid_charges,
                taxes_levies,
                vat_percent,
            )
            da_id_optimised_annual += daily_da_id_opt

        # ---------- Display results ----------
        st.subheader("Annual Cost by Customer Type (using yearly DA & ID)")

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        col1.metric("1) Flat retail (fixed €/kWh)", f"{flat_annual:,.0f} €")
        col2.metric("2) DA-indexed (no optimisation)", f"{da_indexed_annual:,.0f} €")
        col3.metric("3) DA-optimised (smart)", f"{da_optimised_annual:,.0f} €")
        col4.metric("4) DA+ID-optimised (smart+)", f"{da_id_optimised_annual:,.0f} €")

        # ---------- Savings vs flat retail ----------
        st.markdown("---")
        st.subheader("Savings vs Flat Retail Customer")

        def savings(new_cost, ref_cost):
            abs_s = ref_cost - new_cost
            pct_s = 0.0 if ref_cost <= 0 else 100.0 * (1.0 - new_cost / ref_cost)
            return abs_s, pct_s

        s2_abs, s2_pct = savings(da_indexed_annual, flat_annual)
        s3_abs, s3_pct = savings(da_optimised_annual, flat_annual)
        s4_abs, s4_pct = savings(da_id_optimised_annual, flat_annual)

        c1, c2, c3 = st.columns(3)
        c1.metric("DA-indexed vs flat", f"{s2_abs:,.0f} € / yr", f"{s2_pct:.1f} %")
        c2.metric("DA-optimised vs flat", f"{s3_abs:,.0f} € / yr", f"{s3_pct:.1f} %")
        c3.metric("DA+ID-optimised vs flat", f"{s4_abs:,.0f} € / yr", f"{s4_pct:.1f} %")

        # ---------- Bar chart ----------
        st.markdown("---")
        st.subheader("Cost Comparison (Annual)")

        df_bar = pd.DataFrame(
            {
                "Scenario": [
                    "Flat retail",
                    "DA-indexed",
                    "DA-optimised",
                    "DA+ID-optimised",
                ],
                "Annual cost (€)": [
                    flat_annual,
                    da_indexed_annual,
                    da_optimised_annual,
                    da_id_optimised_annual,
                ],
            }
        ).set_index("Scenario")

        st.bar_chart(df_bar)

        # ---------- Assumptions ----------
        st.markdown("---")
        st.subheader("Key Assumptions")

        st.write(
            f"""
            - Optimisation resolution: **15 minutes (96 slots/day)**  
            - DA: hourly prices; ID: 15-min prices  
            - DA+ID effective price = **min(DA, ID)** per 15-min slot  
            - If you upload price files:
                - DA file must have **24 × N** rows (hourly)  
                - ID file must have **96 × N** rows (15-min)  
                - The model uses the first **N = min(N_DA_days, N_ID_days)** days  
            - If no files are uploaded, synthetic daily profiles are repeated for 365 days.  
            - Flat retail customer pays: **flat_retail_price (€/kWh)** all-in (no extra VAT applied in the model).  
            - Dynamic customers (2–4) pay:  
              **(wholesale price + grid_charges + taxes_levies) × (1 + VAT)**  
            - Arrival & departure times are the same every charging day.  
            - Charging days are selected according to your pattern over the **{num_days_year}** days of price data, resulting in **{days_per_year}** charging sessions per year.  
            """
        )

    except ValueError as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
