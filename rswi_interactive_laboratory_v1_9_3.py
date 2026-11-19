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
# Equations and terminology follow rswi_revised_1_9.tex.
# Proposition and section references are aligned to the v1.9 manuscript; the
# application is an illustrative companion, not an independent empirical model.
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
    # R1's abundance guarantee sets the shortfall to zero for sigma >= 1.
    required = np.where(sigma >= 1.0, 0.0,
                        horizon * need * np.maximum(0.0, 1.0 - sigma*n_households*p))
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

def make_fig(width: float = 6.4, height: float = 3.5):
    return plt.subplots(figsize=(width, height), constrained_layout=True)


GLOSSARY = {
    "q_i": "Effective access claim (qᵢ): The strength of household i's recognized entitlement, ration card, priority status, or spendable claim under the active distribution rule.",
    "p_i": "Normalized access share (pᵢ = qᵢ / Σqⱼ): Household i's fraction of all claims in the community. Always between 0 and 1, and all households' shares sum to 1.",
    "S": "Aggregate supply (S): Total physical units of the essential consumption basket available across the entire community in one period.",
    "sigma": "Supply-to-needs ratio (σ = S / Nn): Total supply divided by total community basic needs. σ ≥ 1 means enough for everyone; σ < 1 means aggregate scarcity.",
    "n": "Basic need per household (n): The fixed minimum number of standardized essential-goods baskets (food, fuel, medicine) one household requires per month.",
    "basket": "Essential-goods basket: A standardized bundle of basic survival goods (e.g., staple food, water, fuel, basic medicine) used as the physical unit of measurement.",
    "x_i": "Allocation (xᵢ): The physical quantity of essential baskets assigned to household i by the distribution rule. Indexed by i because different households can receive different amounts.",
    "d_i": "Physical shortfall (dᵢ = max(0, n − xᵢ)): How many essential baskets household i is still missing after receiving its rationed allocation xᵢ.",
    "threshold": "No-shortfall access threshold (p* = 1 / Nσ): The minimum access share a household needs under Rule R1 so that its rationed share covers its full basic need n.",
    "scarcity": "Scarcity regime (σ < 1): Aggregate supply S is smaller than total community needs Nn, so not everyone can receive a full basket unless surplus is redistributed.",
    "abundance": "Abundance regime (σ ≥ 1): Total community supply S is at least Nn—enough physical baskets exist to give every household its full basic need n.",
    "institution": "Allocation institution (rule): The social, legal, or market mechanism (e.g., ration cards, capped needs-first sharing, open market, or equal split) deciding who gets how much of S.",
    "institutions_plural": "Across institutions: Comparing how the exact same household fares under four alternative distribution systems (R1, R2, R3, R4) facing the same supply level.",
    "focal_share": "Focal share (pᵢ): The access share of the single household you are inspecting, while the remaining N−1 households split the leftover share 1 − pᵢ.",
    "R1": "Rule R1 (Proportional + abundance guarantee): Gives everyone n when σ ≥ 1, and splits S proportionally to pᵢ when σ < 1 (allowing high-claim households to receive above n).",
    "R2": "Rule R2 (Capped proportional): Gives everyone n when σ ≥ 1; in scarcity, allocates proportionally to pᵢ but caps every household at n so surplus goes to needier households.",
    "R3": "Rule R3 (Market allocation): Allocates purely by purchasing-power share pᵢ at every supply level—offering no basic-needs guarantee even when aggregate supply is abundant (σ ≥ 1).",
    "R4": "Rule R4 (Egalitarian): Splits available supply equally across all N households (xᵢ = min(n, σn)), ignoring differences in claims or wealth.",
    "R5": "Rule R5 (Runway-maximin): A full-information planner rule that directs scarce rations to the least-liquid households so the shortest survival runway in the community is as long as possible.",
    "real_liquidity": "Accessible real liquid wealth (Lᵢ): Cash and quickly sellable assets minus short-term debts, measured in essential-basket units ('real' means adjusted for inflation and liquidation discounts, not raw paper currency).",
    "real_price": "Real parallel-market price (b): The inflation-adjusted cost of buying one missing essential basket on the open/parallel market, expressed in the same real unit as Lᵢ.",
    "runway": "Funded runway (τᵢ = Lᵢ / b dᵢ): How many months household i's liquid savings Lᵢ can buy its missing baskets dᵢ at price b. Infinite (∞) if shortfall dᵢ = 0.",
    "horizon": "Target planning horizon (T*): The benchmark crisis duration (e.g., 12 months) over which we test whether a household's liquid savings can cover its shortfall.",
    "rswi": "Relative Survival Wealth Index (RSWI): The probability (from 0 to 1) across uncertain crisis scenarios that household i's funded runway τᵢ is at least as long as the target horizon T*.",
    "protected": "Protected household / scenario: A scenario outcome where the household avoids deprivation for the full horizon (τᵢ ≥ T*), either via sufficient rationed access or sufficient liquid savings.",
    "headcount": "Headcount-optimal triage: A planner rule that maximizes the total number of households reaching horizon T* by serving households with the smallest funding gaps first—leaving the poorest unserved when supply is tight.",
    "tdagger": "Maximin horizon (T†): The longest uniform survival runway that a planner can guarantee to every single household simultaneously using available supply S.",
    "contraction_rate": "Contraction rate (g / month): The monthly exponential speed at which aggregate supply shrinks (σₜ = σ₀ e⁻ᵍᵗ). For example, g = 0.05 means roughly 5% supply loss per month.",
    "access_failure": "Access-failure date (tᵢ): The exact month when shrinking supply crosses below household i's no-shortfall threshold, causing its physical shortfall dᵢ(t) to turn positive.",
    "funds_exhausted": "Funds-exhaustion date (Tᵢ): The month when household i's initial liquid savings Lᵢ are completely spent paying for its widening monthly shortfall after access failure.",
    "heterogeneous": "Heterogeneous population: Simulating N = 100 distinct households with different liquid wealth levels Lᵢ and different access weights wᵢ rather than identical clones.",
}


def term(key: str, label: str) -> str:
    """Render an inline term with a styled hover tooltip."""
    tip = GLOSSARY.get(key, "").replace('"', "'")
    return f'<span class="rswi-term" title="{tip}">{label}</span>'


def economic_callout(title: str, plain_english: str, regime_color: str = BLUE):
    """Render a live plain-English economic interpretation card."""
    html = (
        f'<div style="border-left:5px solid {regime_color};background:#f8fafc;'
        f'padding:10px 14px;border-radius:6px;margin:8px 0;color:#0f172a;">'
        f'<div style="font-weight:700;font-size:0.95rem;margin-bottom:3px;">{title}</div>'
        f'<div style="font-size:0.88rem;line-height:1.45;">{plain_english}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streamlit layout and operator controls
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RSWI Interactive Laboratory",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    "<style>"
    ".block-container { padding-top: 1.1rem; padding-bottom: 1.0rem; padding-left: 1.5rem; padding-right: 1.5rem; max-width: 100%; }"
    "h1 { font-size: clamp(1.35rem, 2.1vw, 1.95rem) !important; margin-bottom: 0.2rem !important; }"
    "h2 { font-size: clamp(1.10rem, 1.6vw, 1.45rem) !important; margin-top: 0.3rem !important; }"
    "h3 { font-size: clamp(0.98rem, 1.3vw, 1.20rem) !important; }"
    "p, li, label, .stMarkdown { font-size: clamp(0.84rem, 0.95vw, 0.96rem) !important; }"
    ".rswi-term { border-bottom: 1.5px dotted #2563eb; cursor: help; font-weight: 600; color: #1e3a8a; padding: 0 1px; }"
    ".rswi-term:hover { background-color: #dbeafe; border-radius: 3px; }"
    "</style>",
    unsafe_allow_html=True,
)
st.title("Relative Survival Wealth Index (RSWI) — Interactive Laboratory")
st.markdown(
    "<div style='background:#eff6ff;border:1px solid #bfdbfe;color:#1e3a8a;"
    "padding:6px 12px;border-radius:6px;font-size:0.86rem;margin-bottom:6px;'>"
    "💡 <b>Reader guide:</b> Hover your mouse over any <span class='rswi-term' "
    "title='Like this! Every dotted-underlined term explains its plain-English meaning and formula role.'>"
    "dotted-underlined term</span> to see its plain-English definition. Click the top-left <b>></b> arrow "
    "anytime to open or hide global simulation settings.</div>",
    unsafe_allow_html=True,
)
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
    st.subheader("Model foundations (hover over any underlined term)")
    st.latex(r"p_i=\frac{q_i}{\sum_{j=1}^{N}q_j},\qquad S=\sigma Nn,\qquad d_i=\max(0,n-x_i)")
    st.markdown(
        f"• {term('q_i', 'Effective access claim (qᵢ)')} and {term('p_i', 'Normalized access share (pᵢ)')}: "
        f"If a town has 100 households with equal ration points, each holds pᵢ = 1/100 = 0.01.<br>"
        f"• {term('S', 'Aggregate supply (S)')} and {term('sigma', 'Supply-to-needs ratio (σ)')}: "
        f"Notice σ comes first in S = σNn because σ is the economy-wide scarcity multiplier: if N = 100 households each need "
        f"{term('n', 'n = 1 essential basket')} per month (total need Nn = 100 baskets), then σ = 0.80 means S = 80 baskets are available.<br>"
        f"• {term('x_i', 'Allocation (xᵢ)')} and {term('d_i', 'Physical shortfall (dᵢ = max(0, n − xᵢ))')}: "
        f"Both xᵢ and dᵢ carry the subscript i because each household receives its own ration xᵢ; if xᵢ = 0.60 baskets against need n = 1.00, "
        f"its unmet gap is dᵢ = 0.40 baskets.",
        unsafe_allow_html=True,
    )
    economic_callout(
        "The core intuition in one sentence",
        "Surviving a severe economic contraction requires two distinct things: <b>Access</b> (getting physical essentials through the distribution system) "
        "and <b>Endurance</b> (having enough inflation-adjusted liquid savings to buy whatever essentials the ration system failed to deliver).",
        BLUE,
    )
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
        ["Access threshold", "Section 4.1; Proposition 4.1; Corollary 4.2"],
        ["Institution comparison", "Section 4.2; Rules R1–R4; Proposition 4.3; R2 multiplier"],
        ["Funded runway", "Section 4.3; Proposition 4.4; horizon condition"],
        ["RSWI", "Section 5; conditional scenario-probability definition"],
        ["Liquidity-aware allocation", "Section 4.4; Proposition 4.5; Corollaries 4.6–4.7; Remark 4.8"],
        ["Contracting supply", "Section 4.5; Proposition 4.9"],
        ["Experiment 1", "Section 6.1–6.2 & 6.4; Table 1; Proposition 6.1 closed-form benchmark"],
        ["Experiment 2", "Section 6.3; Table 2; R1/R2/R4/R5 and headcount comparison"],
    ], columns=["Tab", "Paper claim / equation"]), hide_index=True, use_container_width=True)


with tabs[1]:
    st.header("Claim 1 — Relative access matters under scarcity")
    st.markdown(
        f"**Paper:** Section 4.1, Proposition 4.1 and Corollary 4.2 · Key terms: "
        f"{term('threshold', 'Access threshold p*')} · {term('scarcity', 'Scarcity (σ < 1)')} · "
        f"{term('abundance', 'Abundance (σ ≥ 1)')} · {term('focal_share', 'Access share pᵢ')} · "
        f"{term('x_i', 'Allocation xᵢ')} · {term('d_i', 'Shortfall dᵢ')} · "
        f"{term('real_liquidity', 'Real liquidity Lᵢ')} · {term('real_price', 'Real price b')} · "
        f"{term('runway', 'Runway τᵢ')}",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.05, 1.35], gap="medium")
    with left:
        p_col, s_col = st.columns(2)
        p = p_col.slider("Access share pᵢ", 0.0, 0.05, min(0.01, 1/N), 0.0005, format="%.4f")
        sigma = s_col.slider("Supply ratio σ", 0.0, 2.0, 0.80, 0.01)
        l_col, b_col = st.columns(2)
        L1 = l_col.number_input("Liquidity Lᵢ", 0.0, 100.0, wealth_median, 0.2, key="t1_L")
        b1 = b_col.number_input("Real price b", 0.05, 50.0, float(math.exp(b_logmean)), 0.05, key="t1_b")
        threshold = (1/(N*sigma)) if sigma > 0 else math.inf
        allocation = n if sigma >= 1 else sigma*N*n*p
        d = max(0.0, n-allocation)
        tau1 = math.inf if d == 0 else L1/(b1*d)
        a, b_m, c, d_m = st.columns(4)
        a.metric("Threshold p*", "∞" if not math.isfinite(threshold) else f"{threshold:.4f}")
        b_m.metric("Allocation xᵢ", f"{allocation:.3f}")
        c.metric("Shortfall dᵢ", f"{d:.3f}")
        d_m.metric("Runway τᵢ", "∞" if math.isinf(tau1) else f"{tau1:.1f}m")
        st.markdown(
            f"<div style='background:{status_color(tau1,T)};color:{'#111' if 0<tau1<T else 'white'};"
            f"padding:8px 12px;border-radius:6px;font-weight:700;font-size:0.88rem'>{status_label(tau1,T)}</div>",
            unsafe_allow_html=True,
        )
        if sigma >= 1:
            msg = (f"Because aggregate supply covers all needs (σ = {sigma:.2f} ≥ 1), R1 guarantees every household "
                   f"its full basket ({n:g}) regardless of relative access share pᵢ.")
        elif d == 0:
            msg = (f"Supply is rationed (σ = {sigma:.2f} < 1), requiring p* = {threshold:.4f}. Because pᵢ = {p:.4f} ≥ p*, "
                   f"the household obtains its full basket with zero shortfall.")
        elif tau1 >= T:
            msg = (f"In scarcity (σ = {sigma:.2f}), pᵢ = {p:.4f} < p* = {threshold:.4f} leaves a monthly shortfall of "
                   f"{d:.3f} units, but liquid wealth Lᵢ = {L1:g} covers {tau1:.1f} months (≥ T* = {T:g}).")
        elif tau1 > 0:
            msg = (f"Access share pᵢ = {p:.4f} is below p* = {threshold:.4f}, creating a shortfall of {d:.3f} units/month. "
                   f"Liquid wealth Lᵢ = {L1:g} buys only {tau1:.1f} months of endurance (< T* = {T:g}).")
        else:
            msg = (f"Access share pᵢ = {p:.4f} falls short of p* = {threshold:.4f} (shortfall {d:.3f}/month), and with "
                   f"zero liquid wealth (Lᵢ = 0), the household has zero funded runway.")
        economic_callout("What is happening economically", msg, status_color(tau1, T))
    with right:
        st.latex(r"x_i=\begin{cases}n,&\sigma\geq1\\\sigma Nnp_i,&0\leq\sigma<1\end{cases}\qquad p^*(\sigma)=\frac{1}{N\sigma}")
        sigmas = np.linspace(max(0.01, sigma_low), max(0.02, sigma_high), 250)
        fig, ax = make_fig(6.2, 3.2)
        ax.plot(sigmas, 1/(N*sigmas), label="Required access share p*(σ)")
        ax.axvline(1, linestyle="--", label="Abundance boundary σ=1")
        ax.axhline(p, linestyle=":", label="Selected household pᵢ")
        ax.set_ylim(0, min(0.10, max(0.03, np.nanpercentile(1/(N*sigmas), 95)*1.1)))
        ax.set_xlabel("Supply-to-needs ratio, σ")
        ax.set_ylabel("Access share")
        ax.set_title("No-shortfall threshold rises as supply contracts")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.caption(f"Corollary 4.2 resource bound: at σ={sigma:.2f}, at most {min(N, math.floor(N*sigma))} of {N} households can avoid a shortfall.")


with tabs[2]:
    st.header("Claim 2 — Results depend on the allocation institution")
    st.markdown(
        f"**Paper:** Section 4.2, Rules R1–R4 and Proposition 4.3 · Key terms: "
        f"{term('institution', 'Allocation institution')} · {term('institutions_plural', 'Across institutions')} · "
        f"{term('focal_share', 'Focal share pᵢ')} · {term('basket', 'Basket units/month')} · "
        f"{term('R1', 'R1 (Proportional + abundance)')} · {term('R2', 'R2 (Capped proportional)')} · "
        f"{term('R3', 'R3 (Market)')} · {term('R4', 'R4 (Egalitarian)')}",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.15, 1.25], gap="medium")
    with left:
        c1, c2 = st.columns(2)
        p = c1.slider("Focal share pᵢ", 0.0001, 0.05, min(0.01, 1/N), 0.0005, key="inst_p", format="%.4f")
        sigma = c2.slider("Supply ratio σ", 0.05, 1.50, 0.80, 0.01, key="inst_sigma")
        c3, c4 = st.columns(2)
        L2 = c3.number_input("Liquidity Lᵢ", 0.0, 100.0, wealth_median, 0.2, key="t2_L")
        b2 = c4.number_input("Real price b", 0.05, 50.0, float(math.exp(b_logmean)), 0.05, key="t2_b")
        shares = np.full(N, (1-p)/max(N-1,1)); shares[0] = p
        if N > 1:
            shares[1:] = (1-p)/(N-1)
        S = sigma*N*n
        x1 = n if sigma >= 1 else S*p
        x2 = capped_proportional(shares, S, n)[0] if sigma < 1 else n
        x3 = S*p
        x4 = min(n, sigma*n)
        rows2 = []
        for label_i, xi in [("R1 · Proportional + abundance", x1), ("R2 · Capped proportional", x2),
                            ("R3 · Market allocation", x3), ("R4 · Egalitarian", x4)]:
            di = max(0.0, n - xi)
            ti = math.inf if di == 0 else L2/(b2*di)
            reg = "GREEN" if ti >= T else ("YELLOW" if ti > 0 else "RED")
            rows2.append([label_i, xi, di, "∞" if math.isinf(ti) else f"{ti:.1f}m", reg])
        comp = pd.DataFrame(rows2, columns=["Institution", "Allocation", "Shortfall", "Runway", "Status"])
        def _st_style(val):
            if val == "GREEN": return f"background-color:{GREEN};color:white;font-weight:700"
            if val == "YELLOW": return f"background-color:{YELLOW};color:#111;font-weight:700"
            return f"background-color:{RED};color:white;font-weight:700"
        st.dataframe(
            comp.style.format({"Allocation": "{:.3f}", "Shortfall": "{:.3f}"}).map(_st_style, subset=["Status"]),
            hide_index=True, use_container_width=True,
        )
        if sigma >= 1 and p < 1/N:
            msg2 = (f"Even though aggregate supply is abundant (σ = {sigma:.2f} ≥ 1), R1, R2, and R4 deliver the full "
                    f"basket ({n:g}), whereas R3 (market) allocates only {x3:.3f} because pᵢ = {p:.4f} < 1/N.")
            col2 = YELLOW if L2 > 0 else RED
        elif sigma < 1:
            msg2 = (f"In scarcity (σ = {sigma:.2f}), R2 caps wealthy/high-claim households at n and redistributes "
                    f"surplus to others (x₂ = {x2:.3f} vs. x₁ = {x1:.3f}), changing whether Lᵢ = {L2:g} reaches T* = {T:g}.")
            col2 = BLUE
        else:
            msg2 = f"In abundance (σ = {sigma:.2f} ≥ 1) with pᵢ = {p:.4f} ≥ 1/N, all four rules meet the household's basic need."
            col2 = GREEN
        economic_callout("Institutional comparison in plain English", msg2, col2)
    with right:
        st.latex(r"\text{R2: }x_i=\min\{n,\lambda p_i\};\quad \text{R3: }x_i=Sp_i;\quad \text{R4: }x_i=\min\{n,\sigma n\}")
        fig, ax = make_fig(6.2, 3.3)
        bar_colors = [GREEN if r == "GREEN" else (YELLOW if r == "YELLOW" else RED) for r in comp["Status"]]
        ax.bar(["R1", "R2", "R3", "R4"], comp["Allocation"], color=bar_colors)
        ax.axhline(n, color=GREEN, linestyle="--", label="Basic need n")
        ax.set_ylabel("Allocation (basket units/month)")
        ax.set_title("Allocation & runway status across institutions")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
        st.pyplot(fig, use_container_width=True); plt.close(fig)


with tabs[3]:
    st.header("Claim 3 — Liquidity finances the residual shortfall")
    st.markdown(
        f"**Paper:** Section 4.3, Proposition 4.4 · Hover terms: "
        f"{term('real_liquidity', 'Accessible real liquidity Lᵢ')} · {term('real_price', 'Real price of basket b')} · "
        f"{term('runway', 'Funded runway τᵢ')} · {term('horizon', 'Horizon funding T*')} · "
        f"{term('focal_share', 'Access share pᵢ')} · {term('sigma', 'Supply ratio σ')}",
        unsafe_allow_html=True,
    )
    p_btns = st.columns(3)
    if p_btns[0].button("Preset: RED (Zero liquidity)", use_container_width=True):
        st.session_state["run_L_val"] = 0.0
    if p_btns[1].button("Preset: YELLOW (Partial runway)", use_container_width=True):
        st.session_state["run_L_val"] = 1.0
    if p_btns[2].button("Preset: GREEN (Full horizon funded)", use_container_width=True):
        st.session_state["run_L_val"] = 3.5
    left, right = st.columns([1.15, 1.25], gap="medium")
    with left:
        c1, c2 = st.columns(2)
        p = c1.slider("Access share pᵢ", 0.0, 0.05, min(0.01,1/N), 0.0005, key="run_p", format="%.4f")
        sigma = c2.slider("Supply ratio σ", 0.01, 1.5, 0.80, 0.01, key="run_sigma")
        c3, c4 = st.columns(2)
        L = c3.number_input("Accessible real liquidity Lᵢ", 0.0, 10000.0,
                            float(st.session_state.get("run_L_val", wealth_median)), 0.1)
        bprice = c4.number_input("Real price of basket b", 0.01, 1000.0, float(math.exp(b_logmean)), 0.01)
        x = n if sigma >= 1 else sigma*N*n*p
        d = max(0.0, n-x)
        tau = math.inf if d == 0 else L/(bprice*d)
        needed = bprice*T*d
        m1, m2, m3 = st.columns(3)
        m1.metric("Funded runway τᵢ", "∞ months" if math.isinf(tau) else f"{tau:.2f} mo")
        m2.metric("Liquidity needed", f"{needed:.3f}")
        m3.metric("Surplus / (gap)", f"{L-needed:+.3f}")
        st.markdown(
            f"<div style='background:{status_color(tau,T)};color:{'#111' if 0<tau<T else 'white'};"
            f"padding:9px 12px;border-radius:6px;font-weight:700'>{status_label(tau,T)}</div>",
            unsafe_allow_html=True,
        )
        if d == 0:
            msg3 = "The household receives its full basic basket through the allocation rule, so no liquidity is spent."
        elif tau >= T:
            msg3 = (f"Although rationing leaves a shortfall of {d:.3f} units/month, real liquidity Lᵢ = {L:.2f} "
                    f"buys {tau:.1f} months of parallel-market replacement at price b = {bprice:.2f} (exceeding T* = {T:g}).")
        elif tau > 0:
            msg3 = (f"Liquid wealth Lᵢ = {L:.2f} bridges the {d:.3f}/month physical gap for {tau:.1f} months, "
                    f"leaving {T - tau:.1f} months unfunded before the {T:g}-month target horizon ends.")
        else:
            msg3 = (f"The household faces an unmet physical gap of {d:.3f} units/month and holds Lᵢ = 0 liquid wealth, "
                    f"resulting in immediate essential-consumption deprivation (0 months runway).")
        economic_callout("Access vs. endurance diagnosis", msg3, status_color(tau, T))
    with right:
        st.latex(r"\tau_i=\begin{cases}\infty,&d_i=0\\L_i/(b d_i),&d_i>0\end{cases}\qquad L_i\geq bT^*n\max(0,1-\sigma Np_i)")
        L_grid = np.linspace(0, max(5.0, needed * 1.4, L * 1.2), 200)
        tau_grid = np.where(d == 0, T * 1.5, L_grid / max(bprice * d, 1e-12))
        fig, ax = make_fig(6.2, 3.2)
        ax.plot(L_grid, tau_grid, label="Funded runway τᵢ(Lᵢ)")
        ax.axhline(T, color=GREEN, linestyle="--", label=f"Target horizon T* = {T:g}m")
        ax.axvline(L, color=BLUE, linestyle=":", label=f"Current Lᵢ = {L:.2f}")
        ax.set_xlabel("Accessible real liquid wealth Lᵢ")
        ax.set_ylabel("Funded runway (months)")
        ax.set_title("Liquid wealth converts a physical shortfall into endurance")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close(fig)


with tabs[4]:
    st.header("Claim 4 — RSWI is a probability, not a weighted score")
    st.markdown(
        f"**Paper:** Section 5 · Hover terms: {term('rswi', 'RSWI')} · {term('protected', 'Protected scenarios')} · "
        f"{term('horizon', 'Horizon T*')} · {term('focal_share', 'Access share pᵢ')}",
        unsafe_allow_html=True,
    )
    st.latex(r"\mathrm{RSWI}_i(T^*\mid\mathcal F_0)"
             r"=\Pr_{\omega\mid\mathcal F_0}[\tau_i(\omega)\geq T^*]")
    st.latex(r"\widehat{\mathrm{RSWI}}_i(T^*)="
             r"\sum_{k=1}^{K}\pi_k\,\mathbf 1\{\tau_i(\omega_k)\geq T^*\}")
    economic_callout(
        "Plain-English meaning of both equations & the bottom rescaling rule",
        "• <b>Equation 1 (Definition):</b> Imagine 10,000 plausible future crisis scenarios ω (varying in supply shortages and parallel-market prices). "
        "RSWI is simply the fraction of those futures in which your liquid savings last at least T* = 12 months (τᵢ ≥ T*).<br>"
        "• <b>Equation 2 (Finite-scenario sum):</b> If Scenario A (40% chance) leaves you funded for 15 months (indicator <b>1</b> = protected) "
        "and Scenario B (60% chance) leaves you funded for 5 months (indicator <b>0</b> = unprotected), your RSWI is 0.40×1 + 0.60×0 = <b>0.40</b>.<br>"
        "• <b>Bottom rescaling rule (qᵢ' = cqᵢ ⇒ pᵢ' = pᵢ):</b> If the government multiplies everyone's ration points or currency notes by c = 1,000 "
        "in a currency redenomination, your relative share pᵢ is completely unchanged—only real purchasing power relative to basket prices matters.",
        BLUE,
    )
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
    st.markdown(
        f"**Paper:** Section 4.4, Proposition 4.5, Corollaries 4.6–4.7, and Remark 4.8 · Hover terms: "
        f"{term('R5', 'R5 (Runway-maximin)')} · {term('tdagger', 'Maximin horizon T†')} · "
        f"{term('headcount', 'Headcount-optimal triage')} · {term('sigma', 'Shortage supply ratio σ')} · "
        f"{term('S', 'Supply available S = σNn')}",
        unsafe_allow_html=True,
    )
    st.latex(r"r_i(T)=\max\left(0,n-\frac{L_i}{bT}\right)")
    economic_callout(
        "Everyday example: Rescue rations when the planner can see household savings",
        "Suppose a household needs n = 1.0 basket/month for T = 12 months at price b = 1. If it holds Lᵢ = 6 units of savings, "
        "its cash can buy 6/(1×12) = 0.5 baskets/month on the market—so its minimum rescue ration requirement is rᵢ(12) = 1.0 − 0.5 = <b>0.5 baskets/month</b>.<br>"
        "• <b>R5 (Runway-maximin):</b> Gives rations to cash-poor households first so that <i>everyone's</i> survival runway reaches at least T† months.<br>"
        "• <b>Headcount-optimal triage:</b> Prioritizes households with the smallest top-up requirements rᵢ(T) to maximize the number reaching the target horizon—leaving cash-poor families unserved when supply is tight.",
        BLUE,
    )
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
    st.markdown("**Paper:** Section 4.5, Proposition 4.9; contracting-supply access-failure and funds-exhaustion result.")
    left, right = st.columns([1.05, 1.35], gap="medium")
    with left:
        c1, c2 = st.columns(2)
        sigma0 = c1.number_input("Initial supply σ₀", 0.01, 5.0, 1.5, 0.05)
        g = c2.number_input("Contraction rate g/mo", 0.001, 2.0, 0.05, 0.005)
        c3, c4 = st.columns(2)
        p = c3.slider("Access share pᵢ", 0.0, 0.05, min(0.01, 1/N), 0.0005, key="dyn_p", format="%.4f")
        L6 = c4.number_input("Initial liquidity Lᵢ", 0.0, 100.0, wealth_median, 0.2, key="t6_L")
        b6 = float(math.exp(b_logmean))
        m = sigma0*max(1, N*p)
        ti = (math.log(m)/g) if m > 1 else 0.0
        # Numerical cumulative cost to find funds-exhaustion time T_i
        t_grid = np.linspace(0, max(180.0, ti + T*2 + 24.0), 4000)
        sig_g = sigma0*np.exp(-g*t_grid)
        d_g = np.where(sig_g >= 1.0, 0.0, np.maximum(0.0, n - sig_g*N*n*p))
        cum_cost = np.cumsum(b6 * d_g) * (t_grid[1] - t_grid[0])
        if L6 <= 0 and ti == 0.0:
            Ti_exh = 0.0
        elif cum_cost[-1] <= L6:
            Ti_exh = math.inf
        else:
            Ti_exh = float(t_grid[np.searchsorted(cum_cost, L6)])
        m1, m2 = st.columns(2)
        m1.metric("Access failure tᵢ", f"{ti:.2f} mo")
        m2.metric("Funds exhausted Tᵢ", "∞" if math.isinf(Ti_exh) else f"{Ti_exh:.2f} mo")
        st.markdown(
            f"<div style='background:{status_color(Ti_exh,T)};color:{'#111' if 0<Ti_exh<T else 'white'};"
            f"padding:8px 12px;border-radius:6px;font-weight:700;font-size:0.88rem'>{status_label(Ti_exh,T)}</div>",
            unsafe_allow_html=True,
        )
        msg6 = (f"Think of σₜ = σ₀e⁻ᵍᵗ as a shrinking water reservoir losing roughly {g*100:.1f}% of its supply each month. "
                f"While supply is ample, the household gets its full basket; at month <b>tᵢ = {ti:.1f}</b> (the <i>access-failure date</i>), "
                f"rationing begins to pinch and a shortfall opens up. Initial liquid wealth Lᵢ = {L6:g} covers that widening gap until "
                f"month <b>Tᵢ = {Ti_exh:.1f}</b> (the <i>funds-exhaustion date</i>).<br>"
                f"<b>Try this:</b> Raise <i>Access share pᵢ</i> to 0.015 to watch tᵢ shift right (later rationing), or raise <i>Initial liquidity Lᵢ</i> "
                f"to watch Tᵢ push past the {T:g}-month target into GREEN.")
        economic_callout("Dynamic trajectory & how to test it", msg6, status_color(Ti_exh, T))
    with right:
        st.latex(r"\sigma_t=\sigma_0e^{-gt},\qquad t_i=\frac{1}{g}\ln\left(\sigma_0\max\{1,Np_i\}\right),\qquad L_i=\int_{t_i}^{T_i}b\,d_i(s)\,ds")
        times = np.linspace(0, max(24.0, (Ti_exh if math.isfinite(Ti_exh) else ti)*1.35 + 2.0), 300)
        sigt = sigma0*np.exp(-g*times)
        xt = np.where(sigt >= 1, n, sigt*N*n*p)
        dt = np.maximum(0, n-xt)
        fig, ax = make_fig(6.2, 3.2)
        ax.plot(times, sigt, label="Supply ratio σₜ")
        ax.axhline(1, color="black", linestyle="--", label="Abundance boundary")
        ax2 = ax.twinx()
        ax2.plot(times, dt, color=RED, label="Shortfall dᵢ(t)")
        ax.axvline(ti, color=YELLOW, linestyle=":", label=f"Access failure tᵢ={ti:.1f}m")
        if math.isfinite(Ti_exh) and Ti_exh > 0:
            ax.axvline(Ti_exh, color=GREEN, linestyle="-.", label=f"Funds exhausted Tᵢ={Ti_exh:.1f}m")
        ax.set_xlabel("Months"); ax.set_ylabel("Supply ratio"); ax2.set_ylabel("Shortfall")
        ax.set_title("Supply contraction, access failure, and liquidity exhaustion")
        ax.grid(alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(handles+h2, labels+l2, loc="best", fontsize=7.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)


with tabs[7]:
    st.header("Experiment 1 — Focal-household access-share sensitivity")
    st.markdown("**Paper:** Sections 6.1–6.2 & 6.4, Table 1, and Proposition 6.1 (closed-form R1 RSWI).")
    st.latex(r"\widehat{\mathrm{RSWI}}_i(T^*)="
             r"\frac{1}{M}\sum_{m=1}^{M}\mathbf 1\{\tau_i(\omega_m)\geq T^*\}")
    economic_callout(
        "What Experiment 1 demonstrates in plain English",
        "We generate 200,000 random crisis futures (varying supply σ, market price b, and household liquid wealth L) and test one focal household "
        "as its access share pᵢ climbs from 0.005 (half the community average) to 0.020 (twice the average). "
        "The solid curve counts the fraction of futures where the household survives ≥ 12 months; the dashed curve checks that exact same number "
        "using the paper's analytical integral formula (Proposition 6.1)—proving the simulation and calculus match to four decimal places.",
        BLUE,
    )
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
    st.markdown("**Paper:** Section 6, Table 2; liquidity-aware allocation and headcount comparison.")
    economic_callout(
        "What Experiment 2 demonstrates in plain English",
        f"Instead of testing one household, we simulate a full community of {term('heterogeneous', '100 heterogeneous households')} "
        "with unequal savings Lᵢ across ~150,000 shortage futures (σ < 1), comparing five distribution rules side by side:<br>"
        "• <b>Case A:</b> Ration claims are independent of wealth (e.g., universal ration cards).<br>"
        "• <b>Case B:</b> Ration claims follow liquid wealth (wᵢ = Lᵢ, e.g., money-based access).<br>"
        "Click <b>Run Experiment 2 now</b> to see why R5 protects the entire community far more often than proportional or egalitarian rules, "
        "while Headcount-optimal triage maximizes the average share protected by leaving the poorest unserved.",
        BLUE,
    )
    st.latex(r"\text{Case A: }w_i\sim\mathrm{LogNormal}(0,s_w^2)"
             r"\text{ independent of }L_i;\quad"
             r"\text{Case B: }w_i=L_i")
    st.caption("R3 is omitted from the shortage-only Monte Carlo table because its allocation equals R1 whenever σ<1. "
               "R3 is distinguished in the institutional tab by its no-needs-guarantee behaviour when σ≥1.")
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
    economic_callout(
        "Plain-English guide to the empirical regression equation above",
        "To test the theory on real household survey data, we measure each family's observed food/essential shortfall dᵢcₜ:<br>"
        "• <b>Scarcity switch ℛcₜ = 1{σ̂cₜ < 1}:</b> Equals 1 during an actual shortage and 0 in normal times—testing whether access share pᵢc "
        "only predicts deprivation when supplies are tight.<br>"
        "• <b>Structural parameters θ₀, θ₁:</b> Under the baseline model, both equal basic need n.<br>"
        "• <b>Controls Xᵢcₜ'ψ and noise εᵢcₜ:</b> Household size, location, and unmeasured shocks.",
        BLUE,
    )
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
