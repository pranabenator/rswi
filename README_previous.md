# Relative Survival Wealth Index (RSWI) — Interactive Laboratory

An interactive companion to the paper **“When Does Relative Position Matter? A Rationing–Runway Model of Household Survival in Contracting Economies.”**

The laboratory lets readers change model assumptions, inspect the equations represented by each panel, run two Monte Carlo experiments, and download the resulting data.

> **Important:** All probability distributions and parameter values are illustrative assumptions, not empirical estimates. The application demonstrates implications of the stated model; it does not produce forecasts, causal estimates, or portfolio advice.

## Contents

- [What the laboratory covers](#what-the-laboratory-covers)
- [Model notation](#model-notation)
- [Application guide](#application-guide)
- [Experiments](#experiments)
- [Colour coding](#colour-coding)
- [Installation](#installation)
- [Run locally](#run-locally)
- [Reproducibility and downloads](#reproducibility-and-downloads)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Repository files](#repository-files)

## What the laboratory covers

The application is organised into ten tabs. The tabs follow the model's sequence from access and allocation, through liquidity and runway, to the RSWI and the numerical experiments.

| Tab | Subject | Main equation or claim |
|---|---|---|
| Start here | Model foundations and user guide | \(p_i=q_i/\sum_jq_j\), \(S=\sigma Nn\), \(d_i=\max(0,n-x_i)\) |
| 1 · Access threshold | Relative access under scarcity | \(p^*(\sigma)=1/(N\sigma)\) |
| 2 · Allocation institutions | R1–R4 allocation rules | Proportional, capped-proportional, market, and egalitarian allocation |
| 3 · Liquidity & runway | Liquidity finances residual shortfall | \(\tau_i=L_i/(bd_i)\) when \(d_i>0\) |
| 4 · RSWI probability | RSWI as a probability | \(\Pr[\tau_i\geq T^*]\) |
| 5 · Liquidity-aware rationing | Runway-maximin and headcount objectives | \(r_i(T)=\max(0,n-L_i/(bT))\) |
| 6 · Contracting supply | Access-failure date under declining supply | \(\sigma_t=\sigma_0e^{-gt}\) |
| 7 · Experiment 1 | Focal-household access-share sensitivity | Monte Carlo RSWI and midpoint-quadrature benchmark |
| 8 · Experiment 2 | Allocation institutions in a heterogeneous population | Comparison of R1, R2, R4, R5, and headcount-optimal allocation |
| 9 · Empirical & welfare limits | Proposed empirical specification, hypotheses, and limitations | Censored shortfall specification and normative qualifications |

The application displays the equations and explanatory text in the relevant tabs. The operator controls in the sidebar feed the analytical panels and the Monte Carlo experiments.

## Model notation

| Symbol | Meaning |
|---|---|
| \(N\) | Number of households |
| \(n\) | Basic need per household, in basket units per month |
| \(q_i\) | Effective access claim of household \(i\) |
| \(p_i\) | Normalised access share, \(q_i/\sum_jq_j\) |
| \(\sigma\) | Aggregate supply-to-needs ratio |
| \(S\) | Aggregate supply, \(\sigma Nn\) |
| \(x_i\) | Allocation to household \(i\) |
| \(d_i\) | Basic-needs shortfall, \(\max(0,n-x_i)\) |
| \(L_i\) | Accessible real liquidity |
| \(b\) | Real price of the basic-needs basket |
| \(\tau_i\) | Funded runway, in months |
| \(T^*\) | Target protection horizon |
| \(T^\dagger\) | Runway-maximin horizon |

The baseline proportional-claims rule with an abundance guarantee is:

\[
x_i=
\begin{cases}
n, & \sigma\geq 1,\\
\sigma Nnp_i, & 0\leq\sigma<1.
\end{cases}
\]

The household's shortfall is:

\[
d_i=\max(0,n-x_i).
\]

When the shortfall is positive, the funded runway is:

\[
\tau_i=\frac{L_i}{bd_i}.
\]

If \(d_i=0\), the application represents runway as infinity. The target-horizon funding condition is:

\[
L_i\geq bT^*d_i.
\]

## Application guide

### Sidebar operator controls

The sidebar provides the common parameters used by the analytical tabs and experiments:

- Number of households \(N\)
- Basic need \(n\)
- Target horizon \(T^*\)
- Random seed
- Monte Carlo draw count
- Lower and upper bounds of the uniform supply-ratio distribution
- Log-mean and log-standard-deviation of the real-price distribution
- Median and log-standard-deviation of accessible liquidity
- Case A access-weight log-standard-deviation
- Experiment 2 batch size

Individual tabs also provide local controls for the focal household's access share, supply ratio, liquidity, real price, protection horizon, and Experiment 1 access-share grid.

### Allocation institutions

The application presents the following rules:

- **R1 — Proportional claims with abundance guarantee:** in scarcity, allocation is proportional to access shares; in abundance, every household receives its basic need.
- **R2 — Capped proportional:** \(x_i=\min(n,\lambda p_i)\), with \(\lambda\) chosen so that allocations sum to available supply.
- **R3 — Market:** \(x_i=Sp_i\) at every supply level. It is shown as a diagnostic comparison in the institution tab.
- **R4 — Egalitarian:** \(x_i=\min(n,\sigma n)\).
- **R5 — Runway-maximin:** allocation is chosen to maximise the minimum funded runway.
- **Headcount-optimal:** allocates the minimum amount required to bring as many households as possible to the selected target horizon.

Experiment 2 reports R1, R2, R4, R5, and headcount-optimal results. It restricts the institutional comparison to shortage draws (\(\sigma<1\)). R3 is excluded from the reported numerical comparison because it coincides with R1 in shortage draws; its distinguishing implication concerns its behaviour in abundance.

## Experiments

### Experiment 1 — Focal-household access-share sensitivity

Experiment 1 varies a focal household's access share \(p_i\) over a user-selected grid. The same scenario draws are used at every grid point, so the comparison is paired: it isolates the effect of changing the access share while holding the simulated scenarios common.

For each grid value, the application reports:

- Monte Carlo estimate of RSWI
- Midpoint-quadrature evaluation of the paper's closed-form expression under the stated distributional assumptions
- Mean shortfall relative to basic need
- Probability of a positive shortfall

The chart compares the Monte Carlo result with the quadrature benchmark. The quadrature calculation is a numerical benchmark under the model's independence and lognormal assumptions; it is not an empirical estimate.

Downloads:

- `rswi_experiment1_summary.csv`
- `rswi_experiment1_draws.csv`

### Experiment 2 — Allocation institutions in a heterogeneous population

Experiment 2 compares allocation institutions across two access-weight cases, using only shortage draws:

- **Case A — Independent access weights:** access weights are lognormally distributed and independent of household liquidity.
- **Case B — Access weights equal liquidity:** each household's access weight is set equal to its accessible liquidity \(L_i\).

For each case and institution, the application reports the average share of households protected, the probability that all households are protected, the median minimum runway where applicable, the median \(T^\dagger\) for R5, and mean protected headcount for the headcount-optimal rule.

The experiment is launched explicitly with **Run Experiment 2 now**. It is batched to manage intermediate computation and exposes the batch size as a sidebar control.

Downloads:

- `rswi_experiment2_summary.csv`
- `rswi_experiment2_draws.csv`
- `rswi_experiment2_settings.json`

## Colour coding

The runway status uses three colours relative to the selected target horizon \(T^*\):

| Colour | Condition | Meaning |
|---|---|---|
| **Green** | \(\tau_i\geq T^*\) | Target horizon funded |
| **Yellow** | \(0<\tau_i<T^*\) | Positive runway, but below target |
| **Red** | \(\tau_i=0\) | No funded runway against a positive shortfall |

A zero shortfall is represented by infinite runway and is therefore green. These colours classify the modelled funding condition; they do not represent health status, welfare, or a prediction of biological survival.

## Installation

Use Python 3.11 or another Python version compatible with the pinned packages and your Streamlit deployment environment.

Clone the repository, or place the application and dependency file in the same project directory.

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The supplied `requirements.txt` pins:

```text
streamlit==1.67.0
numpy==2.4.6
pandas==3.1.1
matplotlib==3.11.2
```

## Run locally

From the directory containing the application:

```bash
python -m streamlit run rswi_interactive_laboratory_v3_2.py
```

Streamlit will display a local URL in the terminal and normally open the application in a browser.

To stop the local server, press `Ctrl+C` in the terminal.

## Reproducibility and downloads

- The random seed is exposed in the sidebar.
- Experiment 1 uses common scenario draws across its access-share grid.
- Experiment 2 uses separate random streams for its simulated inputs and a batch-invariant design.
- The experiment tabs provide CSV downloads; Experiment 2 also provides a JSON file containing run settings.
- Results depend on the selected assumptions, parameter values, draw count, and seed. Changing these settings can change the numerical output.

For a reproducible comparison, record the sidebar settings and use the same application and dependency versions.

## Interpretation and limitations

This application is a computational companion to the model, not an empirical calibration.

In particular:

- The supply, price, liquidity, and access-weight distributions are illustrative assumptions, not fitted empirical distributions.
- Monte Carlo output demonstrates internal implications of the model under the selected assumptions. It is not a causal estimate or forecast.
- The proposed empirical shortfall specification and hypotheses are not estimated in this application.
- A higher individual RSWI is not, by itself, evidence of higher aggregate welfare.
- The runway-maximin and headcount-optimal rules pursue different objectives and can distribute protection differently.
- The liquidity-aware rules presume that the planner observes household liquidity. Private information would change the implementable allocation problem.
- The model does not represent health, mortality, informal transfers, violence, endogenous prices, or general-equilibrium feedback.
- The contracting-supply panel displays the access-failure date. It does not calculate the funds-exhaustion date, which would require integrating the cost of the evolving shortfall over time.

## Repository files

The minimal application repository needs:

```text
.
├── rswi_interactive_laboratory_v3_2.py
├── requirements.txt
└── README.md
```

The Streamlit deployment's configured **Main file path** must point to `rswi_interactive_laboratory_v3_2.py`. If an existing Streamlit app is configured to use a different filename, either retain that configured filename in the repository or update the Main file path in the app settings. The public app URL does not need to change merely because the code is updated.
