import numpy as np
import pandas as pd
import streamlit as st


# ---------- Synthetic price model (DA hourly + ID 15-min) ----------

def get_synthetic_price_profiles():
    """
    Returns:
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


# ---------- Charging frequency system ----------

def compute_days_per_year_from_pattern(frequency_mode, custom_weekdays=None, sessions_per_week=None):
    """
    Convert user selection to the effective number of charging days per year.
    Assumes 365 days/year.
    """
    if frequency_mode == "Every day":
        return 365

    if frequency_mode == "Every other day":
        return 365 // 2

    if frequency_mode == "Weekdays only (Mon–Fri)":
        return 5 * 52  # approx 260

    if frequency_mode == "Weekends only (Sat–Sun)":
        return 2 * 52  # approx 104

    if frequency_mode == "Custom weekdays":
        return len(custom_weekdays) * 52

    if frequency_mode == "Custom: X sessions per week":
        return int(sessions_per_week * 52)

    return 365


# ---------- Time window helpers (15-min granularity) ----------

def build_available_quarters(arrival_time_h, departure_time_h):
    """
    Convert arrival & departure times (in hours, 0–24) into a list of
    15-minute slot indices in [0, 95].

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
    da_hourly_eur_mwh,
    grid_charges,
    taxes_levies,
    vat_percent,
):
    """
    Scenario 2: DA-indexed customer, no optimisation.
    Baseline sequential charging from arrival, with DA prices + tariffs.
    """
    da_hourly_kwh = da_hourly_eur_mwh / 1000.0  # €/kWh
    da_quarter_kwh = np.repeat(da_hourly_kwh, 4)

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
    price_quarter_eur_mwh,
    grid_charges,
    taxes_levies,
    vat_percent,
):
    """
    For DA-optimised and DA+ID-optimised.
    Optimises within the available quarters using the given price_quarter (€/MWh).
    """
    price_kwh = price_quarter_eur_mwh / 1000.0
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


# ---------- Streamlit UI ----------

def main():
    st.title("EV Charging: Flat Retail vs DA vs DA-Optimised vs DA+ID-Optimised")

    st.write(
        """
        This app compares **4 customer archetypes** for EV charging:

        1. **Flat retail price** (one fixed €/kWh, no market exposure)  
        2. **DA-indexed** (dynamic hourly price, no optimisation)  
        3. **DA-optimised smart charging** (cheapest 15-min slots using DA only)  
        4. **DA+ID-optimised smart charging** (cheapest 15-min slots using min(DA, ID))  

        All with the **same EV**, **same driving/charging pattern**, and **same tariffs**.
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

    days_per_year = compute_days_per_year_from_pattern(
        frequency_mode, custom_weekdays, sessions_per_week
    )
    st.sidebar.info(f"Estimated charging days per year: **{days_per_year}**")

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

    st.sidebar.subheader("2) Dynamic DA / DA+ID Tariff Components (Scenarios 2–4)")

    grid_charges = st.sidebar.number_input(
        "Grid network charges (€/kWh)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
        help="DSO + TSO variable network fees",
    )

    taxes_levies = st.sidebar.number_input(
        "Taxes & levies (€/kWh)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        help="Energy duty, renewable surcharges, etc.",
    )

    vat_percent = st.sidebar.number_input(
        "VAT (%)",
        min_value=0.0,
        max_value=30.0,
        value=20.0,
        step=1.0,
        help="Applied on top of energy + grid + taxes for dynamic tariffs.",
    )

    # ---------- Price profiles ----------
    da_hourly, id_quarter = get_synthetic_price_profiles()
    da_quarter = np.repeat(da_hourly, 4)              # 96 values
    effective_price = np.minimum(da_quarter, id_quarter)

    with st.expander("Show synthetic DA & ID price profiles"):
        times_str = [f"{int(i//4):02d}:{int((i%4)*15):02d}" for i in range(96)]
        df_prices = pd.DataFrame(
            {
                "DA (€/MWh)": da_quarter,
                "ID (€/MWh)": id_quarter,
                "Effective min(DA, ID) (€/MWh)": effective_price,
            },
            index=times_str,
        )
        df_prices.index.name = "Time"
        st.dataframe(df_prices.style.format("{:.1f}"))
        st.line_chart(df_prices)

    # ---------- Compute costs ----------
    try:
        # Time window (15-min slots)
        available_quarters, arrival_q, departure_q = build_available_quarters(
            arrival_time_h, departure_time_h
        )

        if len(available_quarters) == 0:
            raise ValueError("Selected charging window is empty.")

        # Scenario 1: Flat retail price customer (no need for time structure)
        flat_daily = energy_kwh * flat_retail_price
        flat_annual = flat_daily * days_per_year

        # Scenario 2: DA-indexed (baseline, no optimisation)
        da_indexed_daily = compute_da_indexed_baseline_daily(
            energy_kwh,
            max_power_kw,
            available_quarters,
            da_hourly,
            grid_charges,
            taxes_levies,
            vat_percent,
        )
        da_indexed_annual = da_indexed_daily * days_per_year

        # Scenario 3: DA-optimised (smart charging, DA only)
        da_optimised_daily = compute_optimised_daily_cost(
            energy_kwh,
            max_power_kw,
            available_quarters,
            da_quarter,
            grid_charges,
            taxes_levies,
            vat_percent,
        )
        da_optimised_annual = da_optimised_daily * days_per_year

        # Scenario 4: DA+ID-optimised (smart charging, DA+ID)
        da_id_optimised_daily = compute_optimised_daily_cost(
            energy_kwh,
            max_power_kw,
            available_quarters,
            effective_price,
            grid_charges,
            taxes_levies,
            vat_percent,
        )
        da_id_optimised_annual = da_id_optimised_daily * days_per_year

        # ---------- Display results ----------
        st.subheader("Annual Cost by Customer Type")

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

        # ---------- Optional comparison chart ----------
        st.markdown("---")
        st.subheader("Cost Comparison (bar chart)")

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
            - Synthetic **DA hourly** and **ID 15-min** price profiles  
            - DA+ID effective price = **min(DA, ID)** per 15-min slot  
            - Flat retail customer pays a **fixed all-in €/kWh** (no extra VAT in the model)  
            - Dynamic customers (2–4) pay:  
              **(wholesale price + grid_charges + taxes_levies) × (1 + VAT)**  
            - Same EV, energy per session = **{energy_kwh:.1f} kWh**, max power = **{max_power_kw:.1f} kW**  
            - Charging days per year based on your selection: **{days_per_year}**  
            """
        )

    except ValueError as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
