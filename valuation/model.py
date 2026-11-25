"""Valuation engine: forecast, WACC, DCF, comps, LBO returns, sensitivity, and Monte Carlo.

Pure functions so every number can be tested and reused outside the app.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    revenue0: float = 500.0          # last fiscal year revenue, $M
    growth: tuple = (0.12, 0.11, 0.10, 0.08, 0.06)
    ebitda_margin: tuple = (0.22, 0.23, 0.24, 0.25, 0.25)
    da_pct: float = 0.04             # D&A as % of revenue
    capex_pct: float = 0.05
    nwc_pct: float = 0.10            # net working capital as % of revenue
    tax_rate: float = 0.25
    # capital structure and WACC
    risk_free: float = 0.043
    beta: float = 1.15
    erp: float = 0.055
    pre_tax_cost_debt: float = 0.065
    debt: float = 300.0
    cash: float = 80.0
    shares: float = 50.0             # millions
    price: float = 19.5              # current share price
    # terminal value
    terminal_growth: float = 0.025
    exit_multiple: float = 11.0      # EV / EBITDA


def forecast(a: Assumptions) -> pd.DataFrame:
    rows, rev = [], a.revenue0
    nwc_prev = a.revenue0 * a.nwc_pct
    for i, (g, m) in enumerate(zip(a.growth, a.ebitda_margin), start=1):
        rev = rev * (1 + g)
        ebitda = rev * m
        da = rev * a.da_pct
        ebit = ebitda - da
        nopat = ebit * (1 - a.tax_rate)
        capex = rev * a.capex_pct
        nwc = rev * a.nwc_pct
        d_nwc = nwc - nwc_prev
        nwc_prev = nwc
        fcf = nopat + da - capex - d_nwc
        rows.append(dict(Year=i, Revenue=rev, Growth=g, EBITDA=ebitda, Margin=m, EBIT=ebit,
                         NOPAT=nopat, DA=da, Capex=capex, dNWC=d_nwc, FCF=fcf))
    return pd.DataFrame(rows)


def wacc(a: Assumptions) -> dict:
    equity_value = a.price * a.shares
    cost_equity = a.risk_free + a.beta * a.erp
    after_tax_debt = a.pre_tax_cost_debt * (1 - a.tax_rate)
    total = equity_value + a.debt
    we, wd = equity_value / total, a.debt / total
    return dict(cost_equity=cost_equity, after_tax_debt=after_tax_debt, we=we, wd=wd,
                wacc=we * cost_equity + wd * after_tax_debt)


def dcf(a: Assumptions, discount: float | None = None, tg: float | None = None, method: str = "gordon") -> dict:
    f = forecast(a)
    r = discount if discount is not None else wacc(a)["wacc"]
    g = tg if tg is not None else a.terminal_growth
    years = f["Year"].to_numpy()
    # mid year convention: cash arrives on average halfway through each year
    disc = 1 / (1 + r) ** (years - 0.5)
    pv_fcf = float((f["FCF"].to_numpy() * disc).sum())
    last = f.iloc[-1]
    if method == "gordon":
        if r <= g:
            raise ValueError("Discount rate must exceed terminal growth")
        tv = last["FCF"] * (1 + g) / (r - g)
    else:
        tv = last["EBITDA"] * a.exit_multiple
    pv_tv = tv / (1 + r) ** years[-1]
    ev = pv_fcf + pv_tv
    equity = ev - a.debt + a.cash
    per_share = equity / a.shares
    return dict(pv_fcf=pv_fcf, tv=tv, pv_tv=pv_tv, ev=ev, equity=equity, per_share=per_share,
                tv_share=pv_tv / ev, implied_exit=tv / last["EBITDA"], wacc=r)


def sensitivity(a: Assumptions, waccs, growths) -> pd.DataFrame:
    grid = {}
    for g in growths:
        col = []
        for w in waccs:
            col.append(dcf(a, discount=w, tg=g)["per_share"] if w > g else np.nan)
        grid[f"{g:.1%}"] = col
    return pd.DataFrame(grid, index=[f"{w:.1%}" for w in waccs])


def comps_value(a: Assumptions, peers: pd.DataFrame) -> dict:
    """Implied share price from peer EV/EBITDA and EV/Revenue (next year metrics)."""
    f = forecast(a).iloc[0]
    out = {}
    for col, metric in (("EV/EBITDA", f["EBITDA"]), ("EV/Revenue", f["Revenue"])):
        vals = peers[col].dropna()
        lo, med, hi = np.percentile(vals, 25), np.median(vals), np.percentile(vals, 75)
        to_px = lambda m: (m * metric - a.debt + a.cash) / a.shares
        out[col] = dict(low=to_px(lo), mid=to_px(med), high=to_px(hi), multiple=med)
    return out


def lbo(a: Assumptions, entry_multiple: float, leverage: float, rate: float, exit_multiple: float,
        sweep: float = 1.0) -> dict:
    """Simple 5 year LBO: buy at entry multiple, sweep free cash flow to pay down debt, exit at multiple."""
    f = forecast(a)
    ebitda0 = a.revenue0 * a.ebitda_margin[0]
    entry_ev = entry_multiple * ebitda0
    debt = leverage * ebitda0
    equity_in = entry_ev - debt
    if equity_in <= 0:
        raise ValueError("Leverage too high for entry price")
    sched = []
    for _, r in f.iterrows():
        interest = debt * rate
        fcf_levered = r["FCF"] - interest * (1 - a.tax_rate)
        paydown = max(0.0, min(debt, fcf_levered * sweep))
        debt -= paydown
        sched.append(dict(Year=int(r["Year"]), EBITDA=r["EBITDA"], **{"Interest": interest, "Levered FCF": fcf_levered, "Debt paydown": paydown, "Ending debt": debt}))
    exit_ev = exit_multiple * f.iloc[-1]["EBITDA"]
    equity_out = exit_ev - debt
    moic = equity_out / equity_in
    irr = moic ** (1 / len(f)) - 1
    return dict(entry_ev=entry_ev, equity_in=equity_in, exit_ev=exit_ev, equity_out=equity_out,
                moic=moic, irr=irr, schedule=pd.DataFrame(sched))


def monte_carlo(a: Assumptions, n: int = 5000, growth_sd: float = 0.03, margin_sd: float = 0.02,
                wacc_sd: float = 0.01, tg_sd: float = 0.005, seed: int = 11) -> np.ndarray:
    """Distribution of DCF value per share when the key drivers are uncertain."""
    rng = np.random.default_rng(seed)
    base_w = wacc(a)["wacc"]
    out = np.empty(n)
    for i in range(n):
        dg, dm = rng.normal(0, growth_sd), rng.normal(0, margin_sd)
        w = max(0.04, rng.normal(base_w, wacc_sd))
        g = min(w - 0.01, rng.normal(a.terminal_growth, tg_sd))
        trial = replace(a, growth=tuple(x + dg for x in a.growth),
                        ebitda_margin=tuple(max(0.01, x + dm) for x in a.ebitda_margin))
        out[i] = dcf(trial, discount=w, tg=g)["per_share"]
    return out


def recommendation(value: float, price: float, prob_up: float) -> tuple[str, str]:
    upside = value / price - 1
    if upside >= 0.15 and prob_up >= 0.6:
        return "BUY", f"{upside:+.0%} upside to blended value, {prob_up:.0%} of simulations above today's price"
    if upside <= -0.15 and prob_up <= 0.4:
        return "SELL", f"{upside:+.0%} downside to blended value, only {prob_up:.0%} of simulations above price"
    return "HOLD", f"{upside:+.0%} to blended value, {prob_up:.0%} of simulations above price"


SAMPLE_PEERS = pd.DataFrame({
    "Peer": ["Peer A", "Peer B", "Peer C", "Peer D", "Peer E", "Peer F"],
    "EV/EBITDA": [10.5, 12.0, 9.2, 13.4, 11.1, 8.7],
    "EV/Revenue": [2.4, 3.1, 1.9, 3.6, 2.7, 1.8],
})
