from __future__ import annotations

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

EXCEL_MAX_ROWS = 1_048_575  # Leave 1 row for header


# --- Simulation Core Logic ---
def simulate(draws: int, seed: int) -> tuple[np.ndarray, list[dict[str, float]], pd.DataFrame]:
    if draws <= 0:
        raise ValueError("draws must be a positive integer")

    rng = np.random.default_rng(seed)

    sigma = rng.uniform(0.55, 1.15, size=draws)
    b = rng.lognormal(mean=0.0, sigma=0.15, size=draws)
    L = rng.lognormal(mean=np.log(3.0), sigma=0.35, size=draws)

    N = 100
    n = 1.0
    horizon = 12.0
    access_grid = np.array(
        [0.0050, 0.0075, 0.0100, 0.0125, 0.0150, 0.0175, 0.0200],
        dtype=float,
    )

    # Base draw-level dataframe for the full simulation sheet
    draw_data: dict[str, np.ndarray] = {
        "draw_id": np.arange(1, draws + 1),
        "sigma_supply_ratio": sigma,
        "b_parallel_price": b,
        "L_liquid_wealth": L,
    }

    results: list[dict[str, float]] = []
    for p_i in access_grid:
        other_weight = (1.0 - p_i) / (N - 1)
        if other_weight < 0.0:
            raise ValueError("access share exceeds the feasible range")

        allocation = np.where(sigma >= 1.0, n, sigma * N * n * p_i)
        shortfall = np.maximum(0.0, n - allocation)

        runway = np.full(draws, np.inf, dtype=float)
        short = shortfall > 0.0
        runway[short] = L[short] / (b[short] * shortfall[short])

        # Record draw-level survival (>= 12 months) for each access share
        draw_data[f"survives_12m_p_{p_i:.4f}"] = (runway >= horizon).astype(int)

        results.append(
            {
                "access_share": float(p_i),
                "other_household_access_share": float(other_weight),
                "rswi_12_month": float(np.mean(runway >= horizon)),
                "mean_shortfall_fraction_of_need": float(np.mean(shortfall / n)),
                "probability_of_positive_shortfall": float(np.mean(short)),
            }
        )

    full_sim_df = pd.DataFrame(draw_data)
    return access_grid, results, full_sim_df


# --- Streamlit Web Interface ---
st.set_page_config(page_title="Household RSWI Sensitivity Demo", layout="centered")

st.title("Household RSWI Sensitivity Simulation")
st.caption(
    "Reproducible illustrative Monte Carlo for the household RSWI paper. "
    "This is a sensitivity illustration, not an empirical calibration."
)

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
draws = st.sidebar.number_input(
    "Monte Carlo draws",
    min_value=1_000,
    max_value=1_000_000,
    value=200_000,
    step=10_000,
)
seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=999_999_999,
    value=20261112,
    step=1,
)


@st.cache_data
def run_cached_simulation(num_draws: int, rng_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, rows, full_df = simulate(num_draws, rng_seed)
    return pd.DataFrame(rows), full_df


df, full_sim_df = run_cached_simulation(int(draws), int(seed))

# --- Sensitivity Plot ---
st.subheader("RSWI Sensitivity to Relative Access")
fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
ax.plot(df["access_share"], df["rswi_12_month"], marker="o", linewidth=1.8)
ax.set_xlabel("Focal household access share, $p_i$")
ax.set_ylabel("Estimated 12-month RSWI")
ax.set_ylim(-0.02, 1.02)
ax.set_xlim(0.004, 0.021)
ax.grid(True, alpha=0.3)
ax.set_title("Illustrative RSWI sensitivity to relative access")
st.pyplot(fig)
plt.close(fig)

# --- Results Table ---
st.subheader("Simulation Summary Results")
st.dataframe(
    df.style.format(
        {
            "access_share": "{:.4f}",
            "other_household_access_share": "{:.6f}",
            "rswi_12_month": "{:.6f}",
            "mean_shortfall_fraction_of_need": "{:.6f}",
            "probability_of_positive_shortfall": "{:.6f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# --- Export / Download Section ---
st.subheader("Download Simulation Report")
col1, col2 = st.columns(2)

# CSV buffer (Summary)
csv_bytes = df.to_csv(index=False).encode("utf-8")
with col1:
    st.download_button(
        label="Download Summary as CSV",
        data=csv_bytes,
        file_name="rswi_simulation_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Excel buffer: Sheet 1 = Full Simulation, Sheet 2 = Summary, Sheet 3 = Parameters
@st.cache_data
def build_excel_report(
    full_df: pd.DataFrame, summary_df: pd.DataFrame, num_draws: int, rng_seed: int
) -> bytes:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        # First worksheet: Full Simulation
        full_df.iloc[:EXCEL_MAX_ROWS].to_excel(
            writer, index=False, sheet_name="Full_Simulation"
        )
        # Second worksheet: Aggregated Summary
        summary_df.to_excel(writer, index=False, sheet_name="Summary_Results")
        # Third worksheet: Parameters
        meta_df = pd.DataFrame(
            {
                "Parameter": [
                    "Monte Carlo Draws",
                    "Random Seed",
                    "Households (N)",
                    "Horizon (Months)",
                ],
                "Value": [num_draws, rng_seed, 100, 12.0],
            }
        )
        meta_df.to_excel(writer, index=False, sheet_name="Parameters")
    return excel_buffer.getvalue()


with col2:
    st.download_button(
        label="Download Full Report (.xlsx)",
        data=build_excel_report(full_sim_df, df, int(draws), int(seed)),
        file_name="rswi_full_simulation_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )