# Relative Survival Wealth Index (RSWI) — Interactive Laboratory

An interactive computational companion to the paper **“When Does Relative Position Matter? A Rationing–Runway Model of Household Survival in Contracting Economies.”**

The laboratory illustrates how an allocation institution, a household's access position, and accessible real liquidity jointly determine its **funded runway** and its **Relative Survival Wealth Index (RSWI)** under explicitly specified scenarios.

> **Research-use notice.** Parameter values and probability distributions used in the numerical experiments are illustrative assumptions, not empirical estimates. The application demonstrates model implications under selected assumptions. It does not provide forecasts, causal estimates, empirically calibrated survival probabilities, or investment advice.

## 1. What the laboratory is for

The paper separates three stages:

1. **Allocation:** an institution maps aggregate supply and household access claims into a physical allocation.
2. **Residual need and funded runway:** any shortfall against basic need may be financed using accessible real liquidity at the scenario-specific real basket price.
3. **Scenario-based protection:** RSWI evaluates the probability that funded runway reaches a specified target horizon over an explicit scenario distribution.

The central distinction is between physical allocation and the ability to finance the residual shortfall. Relative position matters through the allocation rule; liquidity matters through the household's capacity to fund what the allocation does not provide.

The application is a computational illustration of this structure. It should be read alongside the paper for formal definitions, assumptions, propositions, and qualifications.

## 2. Core model

Let:

- \(N\) denote the number of households;
- \(n\) denote basic need per household per period;
- \(q_i\) denote household \(i\)'s effective access claim;
- \(p_i=q_i/\sum_j q_j\) denote its normalised access share;
- \(S\) denote aggregate supply;
- \(\sigma=S/(Nn)\) denote the supply-to-needs ratio;
- \(x_i\) denote the physical allocation to household \(i\);
- \(d_i=\max(0,n-x_i)\) denote the residual basic-needs shortfall;
- \(L_i(\omega)\) denote accessible real liquidity in scenario \(\omega\);
- \(b(\omega)\) denote the scenario-specific real price of the essential basket;
- \(\tau_i(\omega)\) denote funded runway.

When \(d_i>0\), funded runway is

\[
\tau_i(\omega)=\frac{L_i(\omega)}{b(\omega)d_i}.
\]

When \(d_i=0\), the household has no residual shortfall to finance and its funded runway is represented as infinite. For a target horizon \(T>0\), the funding condition is equivalently

\[
L_i(\omega)\geq b(\omega)T d_i.
\]

RSWI is the probability, under the specified scenario distribution, that funded runway reaches the selected target horizon:

\[
\operatorname{RSWI}_i(T)
=\Pr_{\omega}\!\left[\tau_i(\omega)\geq T\right].
\]

It is a **scenario-based funded-runway probability**. It is not a biological survival probability, a mortality estimate, or a general portfolio-loss measure.

## 3. Allocation institutions

The paper distinguishes the following allocation rules:

- **R1 — Proportional claims with abundance guarantee.** In scarcity, supply is allocated in proportion to access shares. In abundance, each household receives its basic need.
- **R2 — Capped proportional allocation.** Allocations are proportional to access shares subject to the basic-needs cap: \(x_i=\min(n,\lambda p_i)\), with \(\lambda\) chosen to exhaust available supply when the cap-constrained allocation is feasible.
- **R3 — Market allocation.** Allocation follows the access-share distribution at every supply level. Its behaviour in abundance differs from R1 because R3 does not impose R1's abundance guarantee.
- **R4 — Egalitarian allocation.** Households receive equal allocations, subject to the basic-needs cap.
- **R5 — Liquidity-aware runway-maximin allocation.** The allocation maximises the minimum funded runway, subject to the model's resource and non-negativity constraints.

R5 has the constrained-equal-losses form

\[
x_i=\max(0,n-\lambda L_i),
\]

with the multiplier determined by the resource constraint in the relevant shortage case. The paper's contribution is the funded-runway maximin characterisation and its implications, not a claim to have invented the general constrained-equal-losses form.

The paper also distinguishes runway-maximin allocation from **headcount-maximising triage**. These objectives need not select the same allocation: maximising the minimum funded runway protects the worst-positioned household, whereas headcount triage seeks to bring as many households as possible to a specified target horizon. Neither objective should be casually equated with aggregate utilitarian welfare.

**Information qualification:** the liquidity-aware benchmark presumes that the planner observes household liquidity. The model does not establish an incentive-compatible or private-information implementation mechanism.

## 4. Runway-maximin benchmark

For a candidate common protection horizon \(T>0\), the minimum physical allocation required to bring household \(i\) to that horizon is

\[
r_i(T)=\left[n-\frac{L_i}{bT}\right]_+,
\]

where \([z]_+=\max(0,z)\). A common horizon is feasible when

\[
\sum_{i=1}^{N}r_i(T)\leq S.
\]

The runway-maximin horizon \(T^\dagger\) is the greatest feasible common horizon. In the binding case it satisfies

\[
\sum_{i=1}^{N}r_i(T^\dagger)=S.
\]

This characterisation is a full-information benchmark under the paper's stated assumptions. It is not, by itself, a mechanism-design result.

## 5. Experiments and application guide

The existing application guide describes the laboratory as a sequence of analytical panels followed by numerical experiments. The exact visible tab labels and controls should be checked against the v1.9.3 application source before release.

The numerical experiments are intended to illustrate:

- how the focal household's access share affects its shortfall and RSWI under common scenario draws;
- how alternative allocation institutions distribute funded protection in a heterogeneous population;
- how the runway-maximin and headcount objectives can differ;
- how results depend on the selected scenario distributions, parameters, and random seed.

Where a Monte Carlo estimate is compared with a numerical quadrature benchmark, the latter is a numerical evaluation under the stated distributional assumptions—not an empirical estimate.

### Scenario and reproducibility discipline

For any reported run, retain the settings that define the scenario distribution, including the random seed, draw count, supply-ratio distribution, real-price distribution, liquidity distribution, access-weight assumptions, and any experiment-specific settings. Comparisons are meaningful only when the relevant scenario assumptions and computational settings are made explicit.

If the application provides downloadable CSV or JSON outputs, retain those files together with the settings used to generate them. The exact download names and experiment controls should be confirmed against the v1.9.3 source.

## 6. Installation

Use a supported Python version compatible with the pinned packages and the Streamlit runtime. The dependency file supplied with this release pins:

```text
streamlit==1.67.0
numpy==2.4.6
pandas==3.1.1
matplotlib==3.11.2
```

Clone the repository, or place the application and `requirements.txt` in the same directory. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## 7. Run locally

From the directory containing the application:

```bash
python -m streamlit run rswi_interactive_laboratory_v1_9_3.py
```

Streamlit prints a local URL in the terminal and normally opens the application in a browser. Stop the local server with `Ctrl+C`.

## 8. Interpretation and limitations

The laboratory is a computational companion to a theoretical model, not an empirical calibration. In particular:

- Supply, access, liquidity, and price distributions are illustrative unless a particular run explicitly documents otherwise.
- Simulated probabilities are conditional on the chosen scenario distribution and parameterisation. They are not forecasts or causal estimates.
- RSWI measures the probability of meeting a funded-runway horizon under the model; it does not estimate biological survival.
- A higher individual RSWI does not, by itself, establish higher aggregate welfare.
- Runway-maximin and headcount-maximising triage pursue different objectives and can distribute protection differently.
- Liquidity-aware allocation presumes planner observability of household liquidity; private information and incentive compatibility are not solved by this benchmark.
- The framework is partial-equilibrium: access weights, aggregate supply, and the scenario price process are specified exogenously unless a particular extension explicitly states otherwise.
- The model does not, without additional structure, capture health and mortality processes, informal transfers, endogenous prices, replenishment, or general-equilibrium feedback.
- A declining-supply or access-failure calculation should not be described as a funds-exhaustion date unless the evolving shortfall-financing costs are explicitly integrated through time.

These qualifications are part of the model's interpretation, not defects that the numerical application can resolve.

## 9. Repository contents

The minimal release for the v1.9.3 interactive laboratory is:

```text
rswi/
├── rswi_interactive_laboratory_v1_9_3.py
├── requirements.txt
└── README_v9.md
```

The paper source, bibliography, generated PDF, and auxiliary research materials may be maintained separately or added as distinct, clearly documented repository components. Do not imply that they are included in this minimal application release unless they are actually present.

For a hosted Streamlit deployment, set the deployment's **Main file path** to:

```text
rswi_interactive_laboratory_v1_9_3.py
```

If an existing deployment is configured for another filename, update that setting when changing the deployed entry point.

## 10. Citation and relationship to the paper

Cite the paper as the primary source for the model's formal results. The application is an interactive illustration and should not be cited as an independent proof or empirical study.

**Paper:** “When Does Relative Position Matter? A Rationing–Runway Model of Household Survival in Contracting Economies.”

Before publication or archival release, replace this descriptive citation with the paper's final bibliographic details and persistent identifier, if assigned.
