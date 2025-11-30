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

    if arrival_q == departure_q:
        return list(range(96)), arrival_q, departure_q

    if arrival_q < departure_q:
        available = list(range(arrival_q, departure_q))
    else:
        available = list(range(arrival_q, 96)) + list(range(0, departure_q))

    return available, arrival_q, departure_q


# ---------- Cost calculation (15-min resolution) ----------

def apply_tariffs(price_kwh, grid_charges, taxes_levies, vat_percent):
    """
    Applies realistic electricity tariff components:
        (energy_price + grid_charges + taxes_levies) * (1 + VAT)
    """
    pre_vat = price_kwh + grid_charges + taxes_levies
    return pre_vat * (1 + vat_percent / 100)


def compute_baseline_cost_quarter(
    energy_kwh,
    max_power_kw,
    available_quarters,
    da_hourly_eur_mwh,
    grid_charges,
    taxes_levies,
    vat_percent
):
    da_hourly_kwh = da_hourly_eur_mwh / 1000.0
    da_quarter_kwh = np.repeat(da_hourly_kwh, 4)

    energy_remaining = energy_kwh
    total_cost = 0.0
    max_energy_per_q = max_power_kw * 0.25

    for q in available_quarters:
        if energy_remaining <= 1e-6:
            break

        e = min(max_energy_per_q, energy_remaining)
        unit_cost = apply_tariffs(da_quarter_kwh[q], grid_charges, taxes_levies, vat_percent)
        total_cost += e * unit_cost
        energy_remaining -= e

    if energy_remaining > 1e-6:
        raise ValueError("Time window too short to deliver energy.")

    return total_cost


def compute_optimised_cost_quarter(
    energy_kwh,
    max_power_kw,
    available_quarters,
    price_quarter_eur_mwh,
    grid_charges,
    taxes_levies,
    vat_percent
):
    price_kwh = price_quarter_eur_mwh / 1000.0
    max_energy_per_q = max_power_kw * 0.25

    max_deliverable = len(available_quarters) * max_energy_per_q
    if max_deliverable + 1e-6 < energy_kwh:
        raise ValueError("Charging window too short.")

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
    st.title("EV Charging Optimisation: DA Hourly + ID 15-Min with Real Tariffs")

    st.write(
        """
        This model compares EV charging costs using:
        - **Baseline** (no optimisation, DA-only)
        - **DA-optimised** (cheapest 15-min slots using DA price)
        - **DA+ID-optimised** (cheapest 15-min slots using min(DA, ID))

        ✔ Cross-midnight supported  
        ✔ Realistic tariff model: grid charges, taxes, VAT  
        ✔ Flexible charging frequency  
        """
    )

    # ---------- EV Inputs ----------
    st.sidebar.header("EV & Energy Inputs")

    energy_kwh = st.sidebar.number_input("Energy per session (kWh)", 1.0, 200.0, 40.0, 1.0)
    max_power_kw = st.sidebar.number_input("Max charging power (kW)", 1.0, 50.0, 11.0, 0.5)

    # ---------- Arrival/Departure ----------
    st.sidebar.header("Arrival & Departure (0–24h)")

    arrival_time_h = st.sidebar.slider("Arrival time (h)", 0.0, 24.0, 18.0, 0.25)
    departure_time_h = st.sidebar.slider("Departure time (h)", 0.0, 24.0, 7.0, 0.25)

    # ---------- Charging Frequency ----------
    st.sidebar.header("Charging Frequency")

    frequency_mode = st.sidebar.selectbox(
        "Charging pattern",
        [
            "Every day",
            "Every other day",
            "Weekdays only (Mon–Fri)",
            "Weekends only (Sat–Sun)",
            "Custom weekdays",
            "Custom: X sessions per week",
        ]
    )

    custom_weekdays = []
    sessions_per_week = None

    if frequency_mode == "Custom weekdays":
        custom_weekdays = st.sidebar.multiselect(
            "Select days",
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            ["Mon", "Wed", "Fri"]
        )

    if frequency_mode == "Custom: X sessions per week":
        sessions_per_week = st.sidebar.number_input("Sessions per week", 1, 14, 3)

    days_per_year = compute_days_per_year_from_pattern(
        frequency_mode, custom_weekdays, sessions_per_week
    )
    st.sidebar.info(f"Estimated annual charging days: **{days_per_year}**")

    # ---------- Tariffs ----------
    st.sidebar.header("Tariff Components")

    grid_charges = st.sidebar.number_input(
        "Grid network charges (€/kWh)", 0.0, 1.0, 0.10, 0.01,
        help="DSO + TSO variable network fees"
    )

    taxes_levies = st.sidebar.number_input(
        "Taxes & levies (€/kWh)", 0.0, 1.0, 0.05, 0.01,
        help="Energy duty, renewable surcharges, etc."
    )

    vat_percent = st.sidebar.number_input(
        "VAT (%)", 0.0, 30.0, 20.0, 1.0,
        help="Applied to total energy + fees + taxes"
    )

    # ---------- Price profiles ----------
    da_hourly, id_quarter = get_synthetic_price_profiles()
    da_quarter = np.repeat(da_hourly, 4)
    effective_price = np.minimum(da_quarter, id_quarter)

    # ---------- Time window ----------
    try:
        available_quarters, arrival_q, departure_q = build_available_quarters(
            arrival_time_h, departure_time_h
        )

        # Daily baseline
        baseline_daily = compute_baseline_cost_quarter(
            energy_kwh, max_power_kw, available_quarters,
            da_hourly, grid_charges, taxes_levies, vat_percent
        )

        # DA-only optimisation
        da_optimised_daily = compute_optimised_cost_quarter(
            energy_kwh, max_power_kw, available_quarters,
            da_quarter, grid_charges, taxes_levies, vat_percent
        )

        # DA+ID optimisation
        da_id_optimised_daily = compute_optimised_cost_quarter(
            energy_kwh, max_power_kw, available_quarters,
            effective_price, grid_charges, taxes_levies, vat_percent
        )

        # Annual totals
        baseline_annual = baseline_daily * days_per_year
        da_annual = da_optimised_daily * days_per_year
        da_id_annual = da_id_optimised_daily * days_per_year

        # ---------- Output ----------
        st.subheader("Annual Cost Comparison")
        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline (DA only)", f"{baseline_annual:,.0f} €")
        col2.metric("DA Optimised", f"{da_annual:,.0f} €")
        col3.metric("DA + ID Optimised", f"{da_id_annual:,.0f} €")

        # Savings
        st.subheader("Savings vs Baseline")
        def pct(new, base): return 100 * (1 - new/base)

        c1, c2 = st.columns(2)
        c1.metric("DA Savings", f"{baseline_annual - da_annual:,.0f} €", f"{pct(da_annual, baseline_annual):.1f}%")
        c2.metric("DA+ID Savings", f"{baseline_annual - da_id_annual:,.0f} €", f"{pct(da_id_annual, baseline_annual):.1f}%")

    except ValueError as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
