from __future__ import annotations

import io
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# RSWI Interactive Laboratory
# Equations and terminology follow rswi_revised_1_2.tex (Propositions 4.1--6.1).
# ---------------------------------------------------------------------------

GREEN = "#16803c"   # runway >= target horizon
YELLOW = "#d69e00"  # positive runway, but below target
RED = "#c53030"     # no liquidity runway against a positive shortfall
BLUE = "#2563eb"


def runway(allocation: np.ndarray, wealth: np.ndarray, price: np.ndarray, need: float):
    shortfall = np.maximum(0.0, need - allocation)
    tau = np.full(np.broadcast_shapes(np.shape(shortfall), np.shape(wealth)), np.inf)
    shortfall_b = np.broadcast_to(shortfall, tau.shape)
    wealth_b = np.broadcast_to(wealth, tau.shape)
    price_b = np.broadcast_to(price, tau.shape)
    mask = shortfall_b > 0
    tau[mask] = wealth_b[mask] / (price_b[mask] * shortfall_b[mask])
    return shortfall_b, tau


def status_label(tau: float, target: float) -> str:
    if tau >= target:
        return "GREEN — target horizon funded"
    if tau > 0:
        return "YELLOW — positive runway, below target"
    return "RED — no funded runway"


def status_color(tau: float, target: float) -> str:
    if tau >= target:
        return GREEN
    if tau > 0:
        return YELLOW
    return RED


def exact_rswi_midpoint(
    p: float, n_households: int, need: float, horizon: float,
    sigma_low: float, sigma_high: float, b_logmean: float, b_logsd: float,
    wealth_median: float, wealth_logsd: float, nodes: int = 6000,
) -> float:
    """Closed-form conditioning on sigma, evaluated by midpoint quadrature."""
    if sigma_high <= sigma_low or sigma_low < 0:
        raise ValueError("Supply-ratio bounds must satisfy 0 <= low < high.")
    sigma = sigma_low + (np.arange(nodes) + 0.5) * (sigma_high - sigma_low) / nodes
    # Rho=L/b is lognormal when L and b are independent lognormals.
    rho_mean = math.log(wealth_median) - b_logmean
    rho_sd = math.sqrt(wealth_logsd**2 + b_logsd**2)
    required = horizon * need * np.maximum(0.0, 1.0 - sigma*n_households*p)
    survival = np.ones(nodes)
    positive = required > 0
    z = (np.log(required[positive]) - rho_mean) / rho_sd
    survival[positive] = np.array([0.5*math.erfc(float(v)/math.sqrt(2)) for v in z])
    return float(survival.mean())


def exp1_simulation(
    draws: int, seed: int, n_households: int, need: float, horizon: float,
    sigma_low: float, sigma_high: float, b_logmean: float, b_logsd: float,
    wealth_median: float, wealth_logsd: float, p_values: np.ndarray,
):
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(sigma_low, sigma_high, draws)
    b = rng.lognormal(b_logmean, b_logsd, draws)
    L = rng.lognormal(math.log(wealth_median), wealth_logsd, draws)
    rows = []
    draw_columns = {
        "draw_id": np.arange(1, draws+1),
        "sigma": sigma, "b": b, "L": L,
    }
    for p in p_values:
        alloc = np.where(sigma >= 1, need, sigma*n_households*need*p)
        short = np.maximum(0, need-alloc)
        tau = np.full(draws, np.inf)
        mask = short > 0
        tau[mask] = L[mask]/(b[mask]*short[mask])
        protected = tau >= horizon
        rows.append({
            "p_i": float(p),
            "threshold_at_mean_sigma": (1/(n_households*((sigma_low+sigma_high)/2))
                                        if (sigma_low+sigma_high)/2 > 0 else np.nan),
            "RSWI Monte Carlo": float(protected.mean()),
            "RSWI quadrature": exact_rswi_midpoint(
                float(p), n_households, need, horizon, sigma_low, sigma_high,
                b_logmean, b_logsd, wealth_median, wealth_logsd),
            "Mean shortfall / need": float(short.mean()/need),
            "Probability of positive shortfall": float(mask.mean()),
        })
        draw_columns[f"protected_p_{p:.5f}"] = protected.astype(np.int8)
    return pd.DataFrame(rows), pd.DataFrame(draw_columns)


def capped_proportional(shares: np.ndarray, supply: float, need: float) -> np.ndarray:
    """Solve x_i=min(n, lambda p_i), with sum x_i=supply."""
    lo, hi = 0.0, need/max(float(np.min(shares)), 1e-15)
    for _ in range(70):
        mid = (lo+hi)/2
        total = np.minimum(need, mid*shares).sum()
        if total < supply:
            lo = mid
        else:
            hi = mid
    return np.minimum(need, ((lo+hi)/2)*shares)


def maximin_rule(L: np.ndarray, b: float, supply: float, need: float):
    """x_i=max(0,n-L_i/(b*Tdagger)); solve sum x_i=supply."""
    lo = 0.0
    hi = max(need*float(np.max(L))/(b*max(need-supply/len(L), 1e-9)), 1.0)
    # Monotone increasing in T: sum max(0,n-L/(bT)).
    while np.maximum(0, need-L/(b*hi)).sum() < supply:
        hi *= 2
        if hi > 1e12:
            raise RuntimeError("Could not bracket T-dagger.")
    for _ in range(90):
        mid = (lo+hi)/2
        total = np.maximum(0, need-L/(b*mid)).sum()
        if total < supply:
            lo = mid
        else:
            hi = mid
    Tdag = (lo+hi)/2
    alloc = np.maximum(0, need-L/(b*Tdag))
    return alloc, Tdag


def headcount_rule(L: np.ndarray, b: float, supply: float, need: float, target: float):
    req = np.maximum(0, need-L/(b*target))
    order = np.argsort(req)
    alloc = np.zeros_like(L)
    remaining = supply
    count = 0
    for i in order:
        if req[i] <= remaining + 1e-12:
            alloc[i] = req[i]
            remaining -= req[i]
            count += 1
        else:
            break
    return alloc, count


def rule_allocations(shares, L, b, sigma, n_households, need):
    supply = sigma*n_households*need
    r1 = supply*shares
    r2 = capped_proportional(shares, supply, need)
    r3 = supply*shares  # market rule applies at all sigma; only compare in scarcity
    r4 = np.full(n_households, min(need, sigma*need))
    r5, tdag = maximin_rule(L, b, supply, need)
    return {"R1 — proportional claims": r1,
            "R2 — capped proportional": r2,
            "R3 — market": r3,
            "R4 — egalitarian": r4,
            "R5 — runway-maximin": r5}, tdag


def _capped_proportional_batch(shares: np.ndarray, supply: np.ndarray, need: float):
    """Vectorized active-set solution for x_i=min(n, lambda p_i)."""
    rows, households = shares.shape
    order = np.argsort(-shares, axis=1)
    sorted_w = np.take_along_axis(shares, order, axis=1)
    cumulative = np.cumsum(sorted_w, axis=1)
    count = np.arange(households, dtype=float)
    remaining = 1.0 - np.concatenate(
        [np.zeros((rows, 1)), cumulative[:, :-1]], axis=1
    )
    numerator = supply[:, None] - count[None, :] * need
    lam = np.divide(numerator, remaining, out=np.full_like(numerator, np.inf),
                    where=remaining > 1e-14)
    cap = need / sorted_w
    lower = np.concatenate([np.zeros((rows, 1)), cap[:, :-1]], axis=1)
    valid = ((lam >= lower - 1e-10) & (lam <= cap + 1e-10)
             & (numerator >= -1e-10) & (remaining > 1e-14))
    if not np.all(valid.any(axis=1)):
        raise RuntimeError("Capped-proportional batch solution failed.")
    chosen = np.argmax(valid, axis=1)
    selected = lam[np.arange(rows), chosen]
    sorted_alloc = np.minimum(need, selected[:, None] * sorted_w)
    alloc = np.empty_like(sorted_alloc)
    alloc[np.arange(rows)[:, None], order] = sorted_alloc
    return alloc


def _maximin_batch(L: np.ndarray, b: np.ndarray, supply: np.ndarray, need: float):
    """Vectorized exact active-set solution for R5 and T-dagger."""
    ratio = L / b[:, None]
    sorted_ratio = np.sort(ratio, axis=1)
    cumulative = np.cumsum(sorted_ratio, axis=1)
    rows, households = ratio.shape
    k = np.arange(1, households + 1, dtype=float)
    denom = k[None, :] * need - supply[:, None]
    T_candidates = np.divide(cumulative, denom, out=np.full_like(cumulative, np.inf),
                             where=denom > 1e-14)
    boundary = need * T_candidates
    valid = (sorted_ratio <= boundary + 1e-10) & (denom > 1e-14)
    valid[:, :-1] &= boundary[:, :-1] <= sorted_ratio[:, 1:] + 1e-10
    valid &= np.isfinite(T_candidates)
    if not np.all(valid.any(axis=1)):
        raise RuntimeError("Runway-maximin active-set solution failed.")
    idx = np.argmax(valid, axis=1)
    Tdag = T_candidates[np.arange(rows), idx]
    alloc = np.maximum(0.0, need - ratio / Tdag[:, None])
    return alloc, Tdag


def _headcount_batch(L: np.ndarray, b: np.ndarray, supply: np.ndarray,
                     need: float, target: float):
    req = np.maximum(0.0, need - L/(b[:, None]*target))
    ordered = np.sort(req, axis=1)
    cumulative = np.cumsum(ordered, axis=1)
    K = np.sum(cumulative <= supply[:, None] + 1e-10, axis=1)
    # For the headcount-optimal allocation, allocate each selected household
    # exactly its requirement; all others receive zero.
    order = np.argsort(req, axis=1)
    alloc = np.zeros_like(req)
    selected = np.arange(req.shape[1])[None, :] < K[:, None]
    alloc[np.arange(req.shape[0])[:, None], order] = np.where(selected, 
        np.take_along_axis(req, order, axis=1), 0.0)
    return alloc, K


def institution_experiment(
    draws, seed, batch_size, n_households, need, horizon, sigma_low,
    sigma_high, b_logmean, b_logsd, wealth_median, wealth_logsd, weight_logsd,
):
    """Vectorized, batched Experiment 2. Independent streams are batch-invariant."""
    streams = np.random.SeedSequence(seed).spawn(4)
    rs, rb, rL, rw = [np.random.default_rng(s) for s in streams]
    rows_out = []
    shortage_count = 0
    institution_names = ("R1 — proportional claims", "R2 — capped proportional",
                         "R4 — egalitarian", "R5 — runway-maximin",
                         "Headcount-optimal")
    for start in range(0, draws, batch_size):
        m = min(batch_size, draws-start)
        sig_all = rs.uniform(sigma_low, sigma_high, m)
        b_all = rb.lognormal(b_logmean, b_logsd, m)
        L_all = rL.lognormal(math.log(wealth_median), wealth_logsd,
                             (m, n_households))
        w_all = rw.lognormal(0.0, weight_logsd, (m, n_households))
        take = sig_all < 1.0
        if not take.any():
            continue
        shortage_count += int(take.sum())
        sig, bvec, L, w = sig_all[take], b_all[take], L_all[take], w_all[take]
        supply = sig*n_households*need
        shares_a = w/w.sum(axis=1, keepdims=True)
        shares_b = L/L.sum(axis=1, keepdims=True)

        for case, shares in (("A_independent", shares_a), ("B_weights_equal_L", shares_b)):
            r1 = supply[:, None]*shares
            r2 = _capped_proportional_batch(shares, supply, need)
            r4 = np.broadcast_to((sig*need)[:, None], shares.shape).copy()
            r5, tdag = _maximin_batch(L, bvec, supply, need)
            xhc, K = _headcount_batch(L, bvec, supply, need, horizon)
            allocations = (r1, r2, r4, r5, xhc)
            for name, alloc in zip(institution_names, allocations):
                d = np.maximum(0.0, need-alloc)
                tau = np.full_like(d, np.inf)
                mask = d > 0
                tau[mask] = np.broadcast_to(L/bvec[:, None], d.shape)[mask]/d[mask]
                protected = tau >= horizon-1e-10
                minrun = np.min(tau, axis=1)
                if name == "Headcount-optimal":
                    minrun = np.full(len(sig), np.nan)
                block = pd.DataFrame({
                    "draw_id": np.arange(start+1, start+m+1)[take],
                    "case": case,
                    "institution": name,
                    "share_protected": protected.mean(axis=1),
                    "all_protected": protected.all(axis=1).astype(np.int8),
                    "minimum_runway_months": minrun,
                    "T_dagger": tdag if name.startswith("R5") else np.nan,
                    "headcount_K": K if name == "Headcount-optimal" else np.nan,
                })
                rows_out.append(block)

    detail = pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()
    summary = (detail.groupby(["case", "institution"], as_index=False)
               .agg(share_protected=("share_protected", "mean"),
                    probability_all_protected=("all_protected", "mean"),
                    median_minimum_runway_months=("minimum_runway_months", "median"),
                    median_T_dagger=("T_dagger", "median"),
                    mean_headcount_K=("headcount_K", "mean")))
    meta = {"draws": draws, "shortage_draws": shortage_count,
            "seed": seed, "batch_size": batch_size,
            "households": n_households, "need": need, "horizon": horizon}
    return summary, detail, meta

def make_fig():
    return plt.subplots(figsize=(8, 4.6), constrained_layout=True)


# ---------------------------------------------------------------------------
# Streamlit layout and operator controls
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RSWI Interactive Laboratory", layout="wide")
st.title("Relative Survival Wealth Index (RSWI) — Interactive Laboratory")
st.markdown(
    "An interactive companion to the paper. Change the assumptions, run the "
    "experiments, and inspect the equation each result represents. "
    "**All distributions are illustrative assumptions, not empirical estimates.**"
)

with st.sidebar:
    st.header("Operator controls")
    st.caption("These settings feed the analytical tabs and both Monte Carlo experiments.")
    N = int(st.number_input("Households, N", 2, 500, 100, 1))
    n = float(st.number_input("Basic need per household, n (basket units/month)",
                              0.1, 100.0, 1.0, 0.1))
    T = float(st.number_input("Target horizon, T* (months)", 0.5, 120.0, 12.0, 0.5))
    seed = int(st.number_input("Random seed", 0, 2_147_483_647, 20261112, 1))
    draws = int(st.select_slider("Monte Carlo draws",
                                 options=[1000, 5000, 10000, 25000, 50000, 100000, 200000],
                                 value=200000))
    st.divider()
    st.subheader("Scenario distributions")
    sigma_low = float(st.number_input("Supply ratio lower bound", 0.0, 2.0, 0.55, 0.05))
    sigma_high = float(st.number_input("Supply ratio upper bound", 0.01, 3.0, 1.15, 0.05))
    b_logmean = float(st.number_input("log-mean of real price b", -3.0, 3.0, 0.0, 0.05))
    b_logsd = float(st.number_input("log-SD of real price b", 0.01, 2.0, 0.15, 0.01))
    wealth_median = float(st.number_input("Median accessible liquidity L", 0.01, 1000.0, 3.0, 0.1))
    wealth_logsd = float(st.number_input("log-SD of liquidity L", 0.01, 2.0, 0.35, 0.01))
    weight_logsd = float(st.number_input("Case A access-weight log-SD", 0.01, 2.0, 0.5, 0.01))
    batch = int(st.select_slider("Experiment 2 batch size",
                                 options=[100, 250, 500, 1000, 2000],
                                 value=500))
    if sigma_high <= sigma_low:
        st.error("Upper supply bound must exceed lower supply bound.")

st.markdown(
    f"""
    <div style="padding:12px;border-radius:8px;background:#f1f5f9;color:#172554">
    <b>Current run:</b> N={N}; n={n:g}; target={T:g} months; draws={draws:,};
    seed={seed}; supply σ~U[{sigma_low:g},{sigma_high:g}];
    price lognormal(log-mean={b_logmean:g}, log-SD={b_logsd:g});
    liquidity median={wealth_median:g}, log-SD={wealth_logsd:g}.
    </div>
    """, unsafe_allow_html=True
)

tabs = st.tabs([
    "Start here",
    "1 · Access threshold",
    "2 · Allocation institutions",
    "3 · Liquidity & runway",
    "4 · RSWI probability",
    "5 · Liquidity-aware rationing",
    "6 · Contracting supply",
    "7 · Experiment 1",
    "8 · Experiment 2",
    "9 · Empirical & welfare limits",
])

with tabs[0]:
    st.header("How to use this laboratory")
    st.write("This application follows the paper's model in the same order as its major claims.")
    st.subheader("Model foundations")
    st.latex(r"p_i=\frac{q_i}{\sum_{j=1}^{N}q_j},\qquad S=\sigma Nn,\qquad d_i=\max(0,n-x_i)")
    st.write("qᵢ is the effective access claim; pᵢ its normalized share; S aggregate supply; "
             "n basic need; xᵢ allocation; and dᵢ the physical shortfall.")
    st.markdown("""
    1. **Set assumptions** in the sidebar. Every analytical tab uses the current values.
    2. **Explore the propositions** in Tabs 1–6. Each tab shows the governing equation,
       explains its terms, and lets you change the inputs.
    3. **Run Experiment 1** to vary one household's access share while holding the
       scenario draws common across the access-share grid.
    4. **Run Experiment 2** to compare allocation institutions for a population of
       heterogeneous households in two access-weight cases.
    5. **Download the results** from the experiment tabs as CSV.
    """)
    st.subheader("Colour key — funded runway against the target horizon")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div style='background:{GREEN};color:white;padding:12px;border-radius:8px'><b>GREEN</b><br>τ ≥ T*: target funded</div>", unsafe_allow_html=True)
    c2.markdown(f"<div style='background:{YELLOW};color:#111;padding:12px;border-radius:8px'><b>YELLOW</b><br>0 &lt; τ &lt; T*: positive, insufficient runway</div>", unsafe_allow_html=True)
    c3.markdown(f"<div style='background:{RED};color:white;padding:12px;border-radius:8px'><b>RED</b><br>τ = 0: no liquid funds against a positive shortfall</div>", unsafe_allow_html=True)
    st.warning("A zero shortfall is shown as an infinite runway and therefore GREEN. "
               "The colours classify the modelled funding condition, not health, welfare, "
               "or a prediction of biological survival.")
    st.subheader("Model map")
    st.dataframe(pd.DataFrame([
        ["Access threshold", "Proposition 4.1 & Corollary 4.2; equations (6)--(8)"],
        ["Institution comparison", "Rules R1–R4 & Proposition 4.4"],
        ["Funded runway", "Proposition 4.5; equations (9)--(13)"],
        ["RSWI", "Proposition 5.1; equations (15)--(18)"],
        ["Liquidity-aware allocation", "Proposition 4.7 & Corollaries 4.8--4.9; equation (14)"],
        ["Contracting supply", "Proposition 4.11; equation (15)"],
        ["Experiment 1", "Section 6.1--6.2, Table 1, & Proposition 6.1"],
        ["Experiment 2", "Section 6.3, Table 2, & Proposition 4.7"],
    ], columns=["Tab", "Paper claim / equation"]), hide_index=True, use_container_width=True)


with tabs[1]:
    st.header("Claim 1 — Relative access matters under scarcity")
    st.markdown("**Paper:** Section 4.1, Proposition 4.1; equations for the no-shortfall threshold and shortfall slope.")
    p = st.slider("Household access share, pᵢ", 0.0, 1.0, min(0.01, 1/N), 0.0005, format="%.4f")
    sigma = st.slider("Supply-to-needs ratio, σ", 0.0, 2.0, 0.80, 0.01)
    threshold = (1/(N*sigma)) if sigma > 0 else math.inf
    allocation = n if sigma >= 1 else sigma*N*n*p
    d = max(0.0, n-allocation)
    st.latex(r"x_i=\begin{cases}n,&\sigma\geq1\\\sigma Nnp_i,&0\leq\sigma<1\end{cases}")
    st.latex(r"p^*(\sigma)=\frac{1}{N\sigma},\qquad d_i=n\max(0,1-\sigma Np_i)")
    a,b,c = st.columns(3)
    a.metric("No-shortfall threshold p*", "∞" if not math.isfinite(threshold) else f"{threshold:.5f}")
    b.metric("Allocation xᵢ", f"{allocation:.4f}")
    c.metric("Shortfall dᵢ", f"{d:.4f}")
    if sigma >= 1:
        st.success("ABUNDANCE: under R1, every household receives its basic need. Relative access does not affect this basic-needs allocation.")
    elif p >= threshold:
        st.success("GREEN: this household meets its basic need under the proportional-claims baseline.")
    else:
        st.warning("SHORTAGE: this household is below the access threshold and has a positive shortfall.")
        st.write(f"Below the threshold, ∂dᵢ/∂pᵢ = −σNn = **{-sigma*N*n:.4f}**.")
    sigmas = np.linspace(max(0.01, sigma_low), max(0.02, sigma_high), 250)
    fig, ax = make_fig()
    ax.plot(sigmas, 1/(N*sigmas), label="Required access share p*(σ)")
    ax.axvline(1, linestyle="--", label="Abundance boundary σ=1")
    ax.axhline(p, linestyle=":", label="Selected household pᵢ")
    ax.set_ylim(0, min(1.05, max(0.1, np.nanpercentile(1/(N*sigmas), 95)*1.1)))
    ax.set_xlabel("Supply-to-needs ratio, σ")
    ax.set_ylabel("Access share")
    ax.set_title("The no-shortfall threshold rises as supply contracts")
    ax.grid(alpha=.25); ax.legend()
    st.pyplot(fig); plt.close(fig)
    st.info(f"Feasibility corollary: at σ={sigma:.2f}, no more than "
            f"{math.floor(N*sigma)} of {N} households can receive at least n "
            "when σ<1, regardless of the allocation rule." if sigma < 1 else
            "When σ≥1, aggregate supply is sufficient for all N basic needs.")


with tabs[2]:
    st.header("Claim 2 — Results depend on the allocation institution")
    st.markdown("**Paper:** Section 4.2, Rules R1–R4 and Proposition 4.4. The same access share can have different consequences under different institutions.")
    p = st.slider("Focal household share pᵢ", 0.0001, 1.0, min(0.01, 1/N), 0.0005, key="inst_p")
    sigma = st.slider("Supply ratio σ", 0.05, 1.50, 0.80, 0.01, key="inst_sigma")
    shares = np.full(N, (1-p)/max(N-1,1)); shares[0] = p
    # Correct shares if N=2 etc; all entries sum to one.
    if N > 1:
        shares[1:] = (1-p)/(N-1)
    S = sigma*N*n
    x1 = n if sigma >= 1 else S*p
    x2 = capped_proportional(shares, S, n) [0] if sigma < 1 else n
    x3 = S*p  # market applies at all sigma
    x4 = min(n, sigma*n)
    comp = pd.DataFrame([
        ["R1 · Proportional claims with abundance guarantee", x1, max(0,n-x1)],
        ["R2 · Capped proportional", x2, max(0,n-x2)],
        ["R3 · Market allocation", x3, max(0,n-x3)],
        ["R4 · Egalitarian", x4, max(0,n-x4)],
    ], columns=["Institution", "Allocation to focal household", "Basic-needs shortfall"])
    st.latex(r"\text{R2: }x_i=\min\{n,\lambda p_i\},\quad\sum_i x_i=S")
    st.latex(r"\text{R3: }x_i=Sp_i\text{ at every supply level};\qquad"
             r"\text{R4: }x_i=\min\{n,\sigma n\}")
    st.dataframe(comp.style.format({"Allocation to focal household":"{:.4f}",
                                    "Basic-needs shortfall":"{:.4f}"}),
                 hide_index=True, use_container_width=True)
    st.warning("R3 is the market rule: unlike R1/R2/R4, it does not guarantee basic needs when σ≥1. "
               "The paper's threshold claim is therefore institution-specific, not universal.")
    fig, ax = make_fig()
    ax.bar(comp["Institution"], comp["Allocation to focal household"])
    ax.axhline(n, color=GREEN, linestyle="--", label="Basic need n")
    ax.set_ylabel("Allocation (basket units/month)")
    ax.set_title("Institutional allocation for the selected household")
    ax.tick_params(axis="x", rotation=18); ax.legend(); ax.grid(axis="y", alpha=.2)
    st.pyplot(fig); plt.close(fig)


with tabs[3]:
    st.header("Claim 3 — Liquidity finances the residual shortfall")
    st.markdown("**Paper:** Section 4.3, Proposition 4.5; funded runway and horizon funding equations.")
    c1,c2,c3 = st.columns(3)
    p = c1.slider("Access share pᵢ", 0.0, 1.0, min(0.01,1/N), 0.0005, key="run_p")
    sigma = c2.slider("Supply ratio σ", 0.01, 1.5, 0.80, 0.01, key="run_sigma")
    L = c3.number_input("Accessible real liquidity Lᵢ", 0.0, 10000.0, wealth_median, 0.1)
    bprice = st.number_input("Real price of basket b", 0.01, 1000.0, float(math.exp(b_logmean)), 0.01)
    x = n if sigma >= 1 else sigma*N*n*p
    d = max(0.0,n-x)
    tau = math.inf if d == 0 else L/(bprice*d)
    st.latex(r"\tau_i=\begin{cases}\infty,&d_i=0\\L_i/(b d_i),&d_i>0\end{cases}")
    st.latex(r"L_i\geq bT^*n\max(0,1-\sigma Np_i)")
    st.metric("Funded runway τᵢ", "∞ months" if math.isinf(tau) else f"{tau:.3f} months")
    st.markdown(f"<div style='background:{status_color(tau,T)};color:{'#111' if 0<tau<T else 'white'};padding:12px;border-radius:8px;font-weight:700'>{status_label(tau,T)}</div>", unsafe_allow_html=True)
    needed = bprice*T*d
    st.write(f"Liquidity required to fund the current shortfall for {T:g} months: **{needed:.4f}**.")
    st.write(f"Liquidity surplus / (gap): **{L-needed:+.4f}**.")
    if sigma < 1 and L < bprice*n*T:
        funded_threshold = (1/(N*sigma))*(1-L/(bprice*n*T))
        st.write(f"Equivalent access threshold for this liquidity level: **pᵢ ≥ {funded_threshold:.5f}**.")
    else:
        st.write("At this liquidity level, the funding condition is met even for the maximum possible shortfall." if L >= bprice*n*T else "In abundance, the basic-needs shortfall is zero.")


with tabs[4]:
    st.header("Claim 4 — RSWI is a probability, not a weighted score")
    st.markdown("**Paper:** Section 5.1, RSWI definition and finite-scenario estimator.")
    st.latex(r"\mathrm{RSWI}_i(T^*\mid\mathcal F_0)"
             r"=\Pr_{\omega\mid\mathcal F_0}[\tau_i(\omega)\geq T^*]")
    st.latex(r"\widehat{\mathrm{RSWI}}_i(T^*)="
             r"\sum_{k=1}^{K}\pi_k\,\mathbf 1\{\tau_i(\omega_k)\geq T^*\}")
    st.write("For a finite scenario set, each scenario contributes its probability only when the household's runway reaches the stated horizon.")
    p = st.slider("Access share pᵢ", 0.0, 1.0, min(0.01,1/N), 0.0005, key="rswi_p")
    local_draws = st.slider("Illustrative scenario draws", 1000, 50000, 10000, 1000)
    rng = np.random.default_rng(seed+71)
    sig = rng.uniform(sigma_low,sigma_high,local_draws)
    bp = rng.lognormal(b_logmean,b_logsd,local_draws)
    liq = rng.lognormal(math.log(wealth_median),wealth_logsd,local_draws)
    alloc = np.where(sig>=1,n,sig*N*n*p)
    sh = np.maximum(0,n-alloc)
    tau = np.full(local_draws,np.inf)
    mask=sh>0
    tau[mask]=liq[mask]/(bp[mask]*sh[mask])
    protected=tau>=T
    rswi=float(protected.mean())
    st.metric("Estimated RSWI",f"{rswi:.4f}")
    st.write(f"Protected scenarios: **{int(protected.sum()):,} / {local_draws:,}**.")
    st.latex(r"q_i'=cq_i\;(c>0)\quad\Longrightarrow\quad p_i'=p_i")
    st.caption("Common rescaling of nominal claims leaves normalized access shares unchanged. A pure redenomination also leaves real Lᵢ, b and runway unchanged; an inflation shock need not.")
    st.caption("The scenario distribution is a declared assumption. Changing its parameters changes the probability; it does not change the definition of the index.")
    st.download_button("Download scenario-level RSWI calculation",
        pd.DataFrame({"sigma":sig,"b":bp,"L":liq,"allocation":alloc,
                      "shortfall":sh,"runway_months":tau,
                      "protected":protected.astype(int)}).to_csv(index=False),
        "rswi_scenario_calculation.csv","text/csv")


with tabs[5]:
    st.header("Claim 5 — Liquidity-aware rationing has two different objectives")
    st.markdown("**Paper:** Section 4.4, Proposition 4.7; Corollaries 4.8 and 4.9.")
    st.latex(r"r_i(T)=\max\left(0,n-\frac{L_i}{bT}\right)")
    st.write("This is the minimum allocation needed for household i to finance its remaining shortfall for T months.")
    c1,c2,c3=st.columns(3)
    sigma= c1.slider("Shortage supply ratio σ",0.01,0.99,0.80,0.01,key="liq_sigma")
    bprice=c2.number_input("Real price b",0.01,100.0,float(math.exp(b_logmean)),0.01,key="liq_b")
    target=c3.number_input("Protection horizon T (months)",0.5,120.0,T,0.5,key="liq_T")
    rng=np.random.default_rng(seed+202)
    Lvec=rng.lognormal(math.log(wealth_median),wealth_logsd,N)
    supply=sigma*N*n
    req=np.maximum(0,n-Lvec/(bprice*target))
    reqsum=float(req.sum())
    x5,tdag=maximin_rule(Lvec,bprice,supply,n)
    xhc,k= headcount_rule(Lvec,bprice,supply,n,target)
    st.metric("Supply available S=σNn",f"{supply:.3f}")
    st.metric("Total requirement Σrᵢ(T)",f"{reqsum:.3f}")
    if reqsum<=supply:
        st.success("The target horizon is feasible for every household: Σrᵢ(T) ≤ S.")
    else:
        st.warning("The target horizon is infeasible for everyone under this scenario: Σrᵢ(T) > S.")
    st.metric("Maximin horizon T†",f"{tdag:.3f} months")
    st.metric("Headcount-optimal households protected",f"{k} / {N}")
    st.latex(r"T^\dagger=\frac{\sum_{i\in\mathcal A}L_i}{b(|\mathcal A|n-S)}")
    st.write("Within a fixed active set, the maximin horizon rises with supply and active-household liquidity, and falls with the real price b.")
    st.latex(r"\Pr(\text{all protected})\leq\Pr(\sigma\geq1)+\Pr(\sigma<1,\sum_i r_i(T^*)\leq S)")
    xcols=st.columns(2)
    xcols[0].markdown("**R5 — runway-maximin**")
    xcols[0].latex(r"x_i^\dagger=\max(0,n-L_i/(bT^\dagger))")
    xcols[1].markdown("**Headcount-optimal triage**")
    xcols[1].latex(r"K^*(T)=\max\{K:\sum_{j=1}^{K}r_{(j)}(T)\leq S\}")
    d5,t5=runway(x5,Lvec,bprice,n)
    dh,th=runway(xhc,Lvec,bprice,n)
    show=pd.DataFrame({"Household":np.arange(1,N+1),"Liquidity Lᵢ":Lvec,
                       "R5 allocation":x5,"R5 runway":t5,
                       "Headcount allocation":xhc,"Headcount runway":th})
    status_r5=[status_label(float(v),target) for v in t5]
    status_hc=[status_label(float(v),target) for v in th]
    show["R5 status"]=status_r5
    show["Headcount status"]=status_hc
    def runway_style(value):
        if value.startswith("GREEN"):
            return f"background-color: {GREEN}; color: white"
        if value.startswith("YELLOW"):
            return f"background-color: {YELLOW}; color: #111111"
        return f"background-color: {RED}; color: white"
    styled=show.style.format(precision=3).map(runway_style,subset=["R5 status","Headcount status"])
    st.dataframe(styled,use_container_width=True,hide_index=True)
    st.caption("The generated household portfolio is illustrative and reproducible from the current seed. "
               "R5 maximizes the minimum funded runway; headcount-optimal triage maximizes the number "
               "reaching the selected target and may leave other households unprotected.")
    st.download_button("Download household-level allocation comparison",
                       show.to_csv(index=False),"liquidity_aware_households.csv","text/csv")


with tabs[6]:
    st.header("Claim 6 — Contracting supply produces an access-failure date")
    st.markdown("**Paper:** Section 4.5, Proposition 4.11; access-failure time equation.")
    st.latex(r"\sigma_t=\sigma_0e^{-gt},\qquad"
             r"t_i=\frac{1}{g}\ln\left(\sigma_0\max\{1,Np_i\}\right)")
    c1,c2,c3=st.columns(3)
    sigma0=c1.number_input("Initial supply ratio σ₀",0.01,5.0,1.5,0.05)
    g=c2.number_input("Contraction rate g (per month)",0.001,2.0,0.05,0.005)
    p=c3.slider("Household access share pᵢ",0.0,1.0,min(0.01,1/N),0.0005,key="dyn_p")
    m=sigma0*max(1,N*p)
    ti=(math.log(m)/g) if m>1 else 0.0
    st.metric("Access-failure time tᵢ",f"{ti:.3f} months")
    times=np.linspace(0,max(24,ti*1.5+1),300)
    sigt=sigma0*np.exp(-g*times)
    xt=np.where(sigt>=1,n,sigt*N*n*p)
    dt=np.maximum(0,n-xt)
    fig,ax=make_fig()
    ax.plot(times,sigt,label="Supply ratio σₜ")
    ax.axhline(1,color="black",linestyle="--",label="Abundance boundary")
    ax2=ax.twinx()
    ax2.plot(times,dt,color=RED,label="Household shortfall")
    ax.axvline(ti,color=YELLOW,linestyle=":",label="Access failure tᵢ")
    ax.set_xlabel("Months"); ax.set_ylabel("Supply ratio"); ax2.set_ylabel("Shortfall")
    ax.set_title("Supply contraction and household access")
    ax.grid(alpha=.2)
    handles,labels=ax.get_legend_handles_labels()
    h2,l2=ax2.get_legend_handles_labels()
    ax.legend(handles+h2,labels+l2,loc="best")
    st.pyplot(fig);plt.close(fig)
    st.caption("If mᵢ≤1, the household has a positive shortfall for every t>0 and the failure date is set to zero. "
               "This panel shows the access-failure date; the paper's funds-exhaustion time additionally "
               "requires integrating b·dᵢ(t) over time.")


with tabs[7]:
    st.header("Experiment 1 — Focal-household access-share sensitivity")
    st.markdown("**Paper:** Section 6.2, Table 1, and closed-form Proposition 6.1.")
    st.latex(r"\widehat{\mathrm{RSWI}}_i(T^*)="
             r"\frac{1}{M}\sum_{m=1}^{M}\mathbf 1\{\tau_i(\omega_m)\geq T^*\}")
    st.write("The same scenario draws (σ, b, L) are used for every access-share value. "
             "This paired design isolates the effect of changing pᵢ.")
    pmin=st.number_input("Grid minimum pᵢ",0.0001,0.5,0.005,0.0005)
    pmax=st.number_input("Grid maximum pᵢ",0.0002,1.0,0.020,0.0005)
    points=st.slider("Number of access-share grid points",3,25,7)
    if pmax<=pmin:
        st.error("Grid maximum must exceed grid minimum.")
    else:
        pgrid=np.linspace(pmin,pmax,points)
        with st.spinner("Running Experiment 1 with the current sidebar parameters..."):
            @st.cache_data(show_spinner=False)
            def cached_exp1(*args):
                return exp1_simulation(*args)
            summary,detail=cached_exp1(draws,seed,N,n,T,sigma_low,sigma_high,
                                      b_logmean,b_logsd,wealth_median,wealth_logsd,pgrid)
        st.dataframe(summary.style.format(precision=4),use_container_width=True,hide_index=True)
        fig,ax=make_fig()
        ax.plot(summary["p_i"],summary["RSWI Monte Carlo"],marker="o",label="Monte Carlo")
        ax.plot(summary["p_i"],summary["RSWI quadrature"],linestyle="--",label="Quadrature")
        ax.set_xlabel("Access share pᵢ");ax.set_ylabel("RSWI")
        ax.set_ylim(-.02,1.02);ax.set_title("Access-share sensitivity of funded-runway probability")
        ax.grid(alpha=.25);ax.legend()
        st.pyplot(fig);plt.close(fig)
        st.download_button("Download Experiment 1 summary CSV",summary.to_csv(index=False),
                           "rswi_experiment1_summary.csv","text/csv")
        st.download_button("Download Experiment 1 draw-level CSV",detail.to_csv(index=False),
                           "rswi_experiment1_draws.csv","text/csv")
        st.info("The quadrature line evaluates the paper's closed-form expression under the "
                "independence and lognormal assumptions shown in the sidebar. It is a numerical "
                "benchmark, not a separate empirical estimate.")


with tabs[8]:
    st.header("Experiment 2 — Allocation institutions in a heterogeneous population")
    st.markdown("**Paper:** Section 6.3, Table 2; Proposition 4.7.")
    st.write("Only shortage draws (σ<1) enter the institution comparison. Supply and price are common "
             "within a draw; each household has its own liquid wealth.")
    st.latex(r"\text{Case A: }w_i\sim\mathrm{LogNormal}(0,s_w^2)"
             r"\text{ independent of }L_i;\quad"
             r"\text{Case B: }w_i=L_i")
    st.caption("R3 is shown as a diagnostic rule, but the paper's reported numerical table excludes R3. "
               "It coincides with R1 in shortage draws; its distinguishing implication is its abundance-regime behavior.")
    if st.button("Run Experiment 2 now",type="primary"):
        st.session_state["run_exp2"]=True
    if st.session_state.get("run_exp2",False):
        with st.spinner("Running batched household simulations. This can take some time..."):
            @st.cache_data(show_spinner=False)
            def cached_exp2(*args):
                return institution_experiment(*args)
            summary2,detail2,meta2=cached_exp2(draws,seed,batch,N,n,T,sigma_low,
                sigma_high,b_logmean,b_logsd,wealth_median,wealth_logsd,weight_logsd)
        st.write(f"Shortage draws: **{meta2['shortage_draws']:,} / {draws:,}**.")
        st.dataframe(summary2.style.format(precision=4),use_container_width=True,hide_index=True)
        for case in summary2["case"].unique():
            subset=summary2[summary2["case"]==case]
            fig,ax=make_fig()
            ax.bar(subset["institution"],subset["share_protected"])
            ax.set_ylim(0,1);ax.set_ylabel("Average share of households protected")
            ax.set_title(f"{case}: institution comparison")
            ax.tick_params(axis="x",rotation=20);ax.grid(axis="y",alpha=.2)
            st.pyplot(fig);plt.close(fig)
        st.download_button("Download Experiment 2 summary CSV",summary2.to_csv(index=False),
                           "rswi_experiment2_summary.csv","text/csv")
        st.download_button("Download Experiment 2 draw-level metrics CSV",detail2.to_csv(index=False),
                           "rswi_experiment2_draws.csv","text/csv")
        st.download_button("Download Experiment 2 settings JSON",
                           pd.Series(meta2).to_json(indent=2),"rswi_experiment2_settings.json",
                           "application/json")
    else:
        st.info("Press **Run Experiment 2 now** to calculate the comparison using the current settings. "
                "The application does not silently run the larger household simulation on page load.")


with tabs[9]:
    st.header("Claim 7 — Empirical and normative limits")
    st.markdown("**Paper:** Sections 7–9: empirical propositions, distributional interpretation, and limitations.")
    st.subheader("Empirical shortfall specification")
    st.latex(r"d^*_{ict}=\mathcal R_{ct}(\theta_0-\theta_1\widehat{\sigma}_{ct}N_{ct}p_{ic})+\mathbf X_{ict}'\psi+\varepsilon_{ict}")
    st.latex(r"d_{ict}=\max(0,d^*_{ict}),\qquad \mathcal R_{ct}=\mathbf 1\{\widehat{\sigma}_{ct}<1\}")
    st.write("The structural benchmark is θ₀ = θ₁ = n. The scarcity indicator switches off the access-share mechanism in abundance under the abundance-guarantee baseline. This is a proposed test specification, not an estimated result.")
    st.subheader("Three testable hypotheses")
    st.markdown("""
    - **H1 — Real-liquidity effect:** conditional on shortfall and local real prices,
      greater accessible real liquidity should predict longer *observed* duration before
      forced consumption cuts or distress asset sales. Do not test this using the constructed
      runway alone; that would be true by definition.
    - **H2 — Scarcity interaction:** under abundance-guarantee institutions, access shares
      should predict shortfalls in rationing periods but not in abundance periods. Under market
      allocation, the relationship can persist in abundance.
    - **H3 — Balance-sheet transmission:** asset and liability composition matters only
      insofar as it changes scenario-specific accessible real liquidity.
    """)
    st.subheader("Normative limits")
    st.warning("A higher individual RSWI is not, by itself, evidence of higher aggregate welfare.")
    st.markdown("""
    - Access shares are zero-sum at the proportional-allocation stage.
    - R5 maximizes the minimum funded runway and prioritizes less-liquid households.
    - Headcount-optimal triage maximizes the number reaching the target horizon and may
      leave less-liquid households unprotected.
    - The liquidity-aware rules presume the planner observes household liquidity. With
      private information, the implementable allocation problem is different.
    - The model does not represent health, mortality, informal transfers, violence,
      endogenous prices, or general-equilibrium feedback.
    """)
    st.subheader("What this application does not claim")
    st.write("The distributions are not fitted to historical data. The experiments demonstrate "
             "internal implications of the stated model, not causal effects, forecasts, or portfolio advice.")
