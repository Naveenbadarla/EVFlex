import math
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
    da_quarter = np.repeat(da_hourly, 4)  # expand to 96 quarters
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


# ---------- Time window helpers (15-min resolution) ----------

def build_available_quarters(arrival_time_h, departure_time_h):
    """
    Convert arrival & departure times (in hours, 0–24) into a list of
    15-minute slot indices in [0, 95].

    Supports cross-midnight:
        e.g. arrival=18, departure=6 -> evening + night window.
    """
    # Convert hours → quarter index
    arrival_q = int(round(arrival_time_h * 4)) % 96
    departure_q = int(round(departure_time_h * 4)) % 96

    # Interpret arrival == departure as full-day window
    if arrival_q == departure_q:
        return list(range(96)), arrival_q, departure_q

    # Normal daytime window
    if arrival_q < departure_q:
        available = list(range(arrival_q, departure_q))
    else:
        # Cross-midnight window
        available = list(range(arrival_q, 96)) + list(range(0, departure_q))

    return available, arrival_q, departure_q


# ---------- Cost calculation (15-min granularity) ----------

def compute_baseline_cost_quarter(
    energy_kwh,
    max_power_kw,
    available_quarters,
    da_hourly_eur_mwh,
    grid_fee_eur_per_kwh,
    taxes_eur_per_kwh,
):
    """
    Baseline: charge at max power starting at arrival, using DA prices only.
    Uses 15-min resolution, DA prices constant within each hour.
    """
    da_hourly_kwh = da_hourly_eur_mwh / 1000.0  # €/kWh
    da_quarter_kwh = np.repeat(da_hourly_kwh, 4)  # 96 values

    energy_remaining = energy_kwh
    total_cost = 0.0
    max_energy_per_quarter = max_power_kw * 0.25  # kWh per 15 min

    for q in available_quarters:
        if energy_remaining <= 1e-6:
            break
        e = min(max_energy_per_quarter, energy_remaining)
        unit_cost = da_quarter_kwh[q] + grid_fee_eur_per_kwh + taxes_eur_per_kwh
        total_cost += e * unit_cost
        energy_remaining -= e

    if energy_remaining > 1e-6:
        raise ValueError(
            f"Not enough time in the selected window to deliver {energy_kwh:.1f} kWh "
            f"at {max_power_kw:.1f} kW."
        )

    return total_cost


def compute_optimised_cost_quarter(
    energy_kwh,
    max_power_kw,
    available_quarters,
    price_quarter_eur_mwh,
    grid_fee_eur_per_kwh,
    taxes_eur_per_kwh,
):
    """
    Optimised cost: within the available quarters, charge in the cheapest
    15-min slots according to the given price_quarter (€/MWh).
    """
    price_kwh = price_quarter_eur_mwh / 1000.0  # €/kWh
    max_energy_per_quarter = max_power_kw * 0.25  # kWh per 15 min

    max_deliverable = len(available_quarters) * max_energy_per_quarter
    if max_deliverable + 1e-6 < energy_kwh:
        raise ValueError(
            f"Time window too short: max deliverable {max_deliverable:.1f} kWh "
            f"but {energy_kwh:.1f} kWh required."
        )

    # Sort quarters by increasing price
    sorted_quarters = sorted(available_quarters, key=lambda q: price_kwh[q])

    energy_remaining = energy_kwh
    total_cost = 0.0

    for q in sorted_quarters:
        if energy_remaining <= 1e-6:
            break
        e = min(max_energy_per_quarter, energy_remaining)
        unit_cost = price_kwh[q] + grid_fee_eur_per_kwh + taxes_eur_per_kwh
        total_cost += e * unit_cost
        energy_remaining -= e

    return total_cost


# ---------- Streamlit UI ----------

def main():
    st.title("EV Charging Cost Optimisation: DA Hourly + ID 15-min")

    st.write(
        """
        Compare EV charging costs using:

        **1. Baseline** (DA only, sequential)  
        **2. DA-optimised** (cheapest 15-min slots using hourly DA)  
        **3. DA+ID-optimised** (cheapest 15-min slots using min(DA, ID))  

        Fully supports **cross-midnight** arrival/departure times.
        """
    )

    # ---------- Sidebar inputs ----------
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

    st.sidebar.header("Arrival & Departure (hours, 0–24)")

    arrival_time_h = st.sidebar.slider(
        "Arrival time [h]",
        min_value=0.0,
        max_value=24.0,
        value=18.0,
        step=0.25,
    )

    departure_time_h = st.sidebar.slider(
        "Departure time [h]",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.25,
    )

    # ---------- Charging frequency ----------
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
        )
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
        frequency_mode,
        custom_weekdays=custom_weekdays,
        sessions_per_week=sessions_per_week,
    )

    st.sidebar.info(f"Estimated charging days per year: **{days_per_year} days**")

    # ---------- Grid fees ----------
    st.sidebar.header("Grid Fees, Taxes & Levies")

    grid_fee = st.sidebar.number_input(
        "Grid fee (€/kWh)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
    )

    taxes_levies = st.sidebar.number_input(
        "Taxes & levies (€/kWh)",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
    )

    # ---------- Prices ----------
    st.sidebar.header("Price Profiles")
    da_hourly, id_quarter = get_synthetic_price_profiles()
    da_quarter = np.repeat(da_hourly, 4)
    effective_price = np.minimum(da_quarter, id_quarter)

    # ---------- Display price tables ----------
    with st.expander("Show synthetic price profiles"):
        times_str = [f"{int(i//4):02d}:{int((i%4)*15):02d}" for i in range(96)]
        df_prices = pd.DataFrame(
            {
                "DA (€/MWh)": da_quarter,
                "ID (€/MWh)": id_quarter,
                "Effective min(DA, ID)": effective_price,
            },
            index=times_str,
        )
        df_prices.index.name = "Time"
        st.dataframe(df_prices.style.format("{:.1f}"))
        st.line_chart(df_prices)

    # ---------- Build time window ----------
    try:
        available_quarters, arrival_q, departure_q = build_available_quarters(
            arrival_time_h, departure_time_h
        )

        if len(available_quarters) == 0:
            raise ValueError("Selected charging window is empty.")

        # Daily baseline
        baseline_daily = compute_baseline_cost_quarter(
            energy_kwh,
            max_power_kw,
            available_quarters,
            da_hourly,
            grid_fee,
            taxes_levies,
        )

        # DA-optimised
        da_optimised_daily = compute_optimised_cost_quarter(
            energy_kwh,
            max_power_kw,
            available_quarters,
            da_quarter,
            grid_fee,
            taxes_levies,
        )

        # DA+ID-optimised
        da_id_optimised_daily = compute_optimised_cost_quarter(
            energy_kwh,
            max_power_kw,
            available_quarters,
            effective_price,
            grid_fee,
            taxes_levies,
        )

        # Annual totals
        baseline_annual = baseline_daily * days_per_year
        da_annual = da_optimised_daily * days_per_year
        da_id_annual = da_id_optimised_daily * days_per_year

        # ---------- Results ----------
        st.subheader("Annual Cost Comparison")

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline (DA only)", f"{baseline_annual:,.0f} €")
        col2.metric("DA-optimised", f"{da_annual:,.0f} €")
        col3.metric("DA+ID-optimised", f"{da_id_annual:,.0f} €")

        st.markdown("---")
        st.subheader("Savings vs Baseline")

        def pct_savings(new, base):
            return 100 * (1 - new / base)

        colA, colB = st.columns(2)
        colA.metric(
            "Savings (DA-only optimisation)",
            f"{baseline_annual - da_annual:,.0f} €",
            f"{pct_savings(da_annual, baseline_annual):.1f} %",
        )
        colB.metric(
            "Savings (DA+ID optimisation)",
            f"{baseline_annual - da_id_annual:,.0f} €",
            f"{pct_savings(da_id_annual, baseline_annual):.1f} %",
        )

        st.markdown("---")
        st.subheader("Model Assumptions")

        st.write(
            """
            - Optimisation granularity: **15 minutes (96 slots/day)**  
            - DA prices: **hourly**  
            - ID prices: **15-minute**  
            - DA+ID effective price = **min(DA, ID)** per 15 minutes  
            - Cross-midnight windows supported  
            - Charging frequency selector determines annual multipliers  
            - Synthetic prices → plug in real market DA/ID easily  
            """
        )

    except ValueError as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
