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


# ---------- Time window helpers (15-min resolution) ----------

def build_available_quarters(arrival_time_h, departure_time_h):
    """
    Convert arrival & departure times (in hours, 0–24) into a list of
    15-minute slot indices in [0, 95].

    Supports cross-midnight:
      e.g. arrival=18, departure=6 -> evening + night window.
    """
    # Map hours to quarter indices (0..96) then to [0..95] with modulo
    arrival_q = int(round(arrival_time_h * 4)) % 96
    departure_q = int(round(departure_time_h * 4)) % 96

    if arrival_q == departure_q:
        # Interpret as full 24-hour window
        available = list(range(96))
    elif arrival_q < departure_q:
        # Same-day window
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

    # Sequentially from arrival through available quarters
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
    price_quarter_kwh = price_quarter_eur_mwh / 1000.0  # €/kWh
    max_energy_per_quarter = max_power_kw * 0.25  # kWh per 15 min

    # Check maximum possible energy
    max_deliverable = len(available_quarters) * max_energy_per_quarter
    if max_deliverable + 1e-6 < energy_kwh:
        raise ValueError(
            f"Not enough available time: max {max_deliverable:.1f} kWh deliverable, "
            f"but {energy_kwh:.1f} kWh requested."
        )

    # Sort available quarters by increasing price
    sorted_quarters = sorted(available_quarters, key=lambda q: price_quarter_kwh[q])

    energy_remaining = energy_kwh
    total_cost = 0.0

    for q in sorted_quarters:
        if energy_remaining <= 1e-6:
            break
        e = min(max_energy_per_quarter, energy_remaining)
        unit_cost = price_quarter_kwh[q] + grid_fee_eur_per_kwh + taxes_eur_per_kwh
        total_cost += e * unit_cost
        energy_remaining -= e

    return total_cost


# ---------- Streamlit UI ----------

def main():
    st.title("EV Charging Cost Optimisation: DA (hourly) vs DA+ID (15-min)")

    st.write(
        """
        This app compares **annual EV charging costs** under three strategies:
        
        1. **Baseline** – charge immediately at arrival (no optimisation), using DA prices.
        2. **DA-optimised** – choose cheapest 15-min slots based on **DA hourly** prices
           (DA price is constant inside each hour).
        3. **DA+ID-optimised** – choose cheapest 15-min slots using the minimum of
           **DA hourly** and **ID 15-min** prices.
        
        Time resolution for the optimisation is **15 minutes**, and **cross-midnight
        charging windows are supported**.
        """
    )

    # ---------- Sidebar inputs ----------
    st.sidebar.header("EV & Usage Inputs")

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

    st.sidebar.markdown("### Arrival & Departure (hours, 0–24)")

    arrival_time_h = st.sidebar.slider(
        "Arrival time [h]",
        min_value=0.0,
        max_value=24.0,
        value=18.0,
        step=0.25,
        help="In hours from midnight (e.g. 18.5 = 18:30).",
    )

    departure_time_h = st.sidebar.slider(
        "Departure time [h]",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.25,
        help="Can be earlier than arrival to represent cross-midnight (e.g. 7:00 next day).",
    )

    days_per_year = st.sidebar.number_input(
        "Charging days per year",
        min_value=1,
        max_value=365,
        value=250,
        step=1,
    )

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

    # ---------- Price profiles ----------
    st.sidebar.header("Price Profiles")
    st.sidebar.write(
        "This demo uses **synthetic** DA hourly and ID 15-min prices.\n\n"
        "In production, replace `get_synthetic_price_profiles()` with your own data source."
    )

    da_hourly, id_quarter = get_synthetic_price_profiles()
    da_quarter = np.repeat(da_hourly, 4)
    effective_da_id_quarter = np.minimum(da_quarter, id_quarter)

    # Build a human-readable table at 15-min resolution
    quarter_indices = np.arange(96)
    times_str = [
        f"{int(q // 4):02d}:{int((q % 4) * 15):02d}" for q in quarter_indices
    ]

    with st.expander("Show synthetic price profiles"):
        st.markdown("#### DA (hourly) prices (€/MWh)")
        df_da_hourly = pd.DataFrame(
            {"DA (€/MWh)": da_hourly},
            index=[f"{h:02d}:00" for h in range(24)],
        )
        df_da_hourly.index.name = "Hour"
        st.dataframe(df_da_hourly.style.format("{:.1f}"))
        st.line_chart(df_da_hourly)

        st.markdown("#### DA vs ID vs DA+ID effective (15-min, €/MWh)")
        df_quarter = pd.DataFrame(
            {
                "DA (€/MWh)": da_quarter,
                "ID (€/MWh)": id_quarter,
                "Effective min(DA, ID) (€/MWh)": effective_da_id_quarter,
            },
            index=times_str,
        )
        df_quarter.index.name = "Time"
        st.dataframe(df_quarter.style.format("{:.1f}"))
        st.line_chart(df_quarter)

    # ---------- Build time window ----------
    try:
        available_quarters, arrival_q, departure_q = build_available_quarters(
            arrival_time_h, departure_time_h
        )

        if len(available_quarters) == 0:
            raise ValueError("Selected window has zero length. Adjust arrival/departure times.")

        max_energy_per_quarter = max_power_kw * 0.25
        max_deliverable = len(available_quarters) * max_energy_per_quarter
        if max_deliverable + 1e-6 < energy_kwh:
            raise ValueError(
                f"Not enough time in the selected window to deliver {energy_kwh:.1f} kWh "
                f"at {max_power_kw:.1f} kW. Maximum possible: {max_deliverable:.1f} kWh."
            )

        # ---------- Compute daily costs ----------
        baseline_daily = compute_baseline_cost_quarter(
            energy_kwh,
            max_power_kw,
            available_quarters,
            da_hourly,
            grid_fee,
            taxes_levies,
        )

        # DA-optimised (15-min, DA price only)
        da_optimised_daily = compute_optimised_cost_quarter(
            energy_kwh,
            max_power_kw,
            available_quarters,
            da_quarter,  # €/MWh
            grid_fee,
            taxes_levies,
        )

        # DA+ID-optimised (15-min, min(DA, ID))
        da_id_optimised_daily = compute_optimised_cost_quarter(
            energy_kwh,
            max_power_kw,
            available_quarters,
            effective_da_id_quarter,  # €/MWh
            grid_fee,
            taxes_levies,
        )

        # Annual costs
        baseline_annual = baseline_daily * days_per_year
        da_annual = da_optimised_daily * days_per_year
        da_id_annual = da_id_optimised_daily * days_per_year

        # ---------- Results ----------
        st.subheader("Annual Cost Comparison")

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline (DA, no optimisation)", f"{baseline_annual:,.0f} €")
        col2.metric("DA-optimised (15-min, DA only)", f"{da_annual:,.0f} €")
        col3.metric("DA+ID-optimised (15-min)", f"{da_id_annual:,.0f} €")

        st.markdown("---")

        st.subheader("Savings vs Baseline")

        def pct_savings(new, base):
            if base <= 0:
                return 0.0
            return 100.0 * (1.0 - new / base)

        da_savings_abs = baseline_annual - da_annual
        da_savings_pct = pct_savings(da_annual, baseline_annual)

        da_id_savings_abs = baseline_annual - da_id_annual
        da_id_savings_pct = pct_savings(da_id_annual, baseline_annual)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Savings with DA optimisation",
                f"{da_savings_abs:,.0f} € / year",
                f"{da_savings_pct:.1f} %",
            )
        with col_b:
            st.metric(
                "Savings with DA+ID optimisation",
                f"{da_id_savings_abs:,.0f} € / year",
                f"{da_id_savings_pct:.1f} %",
            )

        st.markdown("---")
        st.subheader("Model assumptions")

        st.write(
            f"""
            - Time resolution: **15 minutes** (96 slots per day).
            - DA prices are **hourly**; the same DA price applies to the 4 quarter-hours inside each hour.
            - ID prices are **15-minute** synthetic series.
            - DA+ID effective price per 15 minutes = **min(DA_hour, ID_15min)**.
            - Baseline: charge sequentially at max power starting from arrival, using DA prices only.
            - DA-optimised: pick cheapest **15-min slots** using DA prices only.
            - DA+ID-optimised: pick cheapest **15-min slots** using min(DA, ID).
            - Arrival and departure times can span **midnight** (e.g. 18:00 → 06:00).
            - If arrival = departure, the window is interpreted as **full 24h**.
            - A single day's price pattern is repeated for all **{days_per_year}** charging days per year.
            """
        )

    except ValueError as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
