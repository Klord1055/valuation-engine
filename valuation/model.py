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
    # Gordon value is built from mid year cash flows, so it is discounted on the same mid year basis;
    # an exit multiple is a price paid at the end of year N, so it is discounted a full N years.
    pv_tv = tv / (1 + r) ** (years[-1] - 0.5) if method == "gordon" else tv / (1 + r) ** years[-1]
    ev = pv_fcf + pv_tv
    equity = ev - a.debt + a.cash
    per_share = equity / a.shares
    return dict(pv_fcf=pv_fcf, tv=tv, pv_tv=pv_tv, ev=ev, equity=equity, per_share=per_share,
                tv_share=pv_tv / ev if ev else float("nan"),
                implied_exit=(pv_tv * (1 + r) ** years[-1]) / last["EBITDA"] if last["EBITDA"] else float("nan"), wacc=r)


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
        vals = pd.to_numeric(peers[col], errors="coerce").dropna() if col in peers else pd.Series(dtype=float)
        vals = vals[vals > 0]
        if vals.empty:
            out[col] = dict(low=np.nan, mid=np.nan, high=np.nan, multiple=np.nan)
            continue
        lo, med, hi = np.percentile(vals, 25), np.median(vals), np.percentile(vals, 75)
        to_px = lambda m: (m * metric - a.debt + a.cash) / a.shares
        out[col] = dict(low=to_px(lo), mid=to_px(med), high=to_px(hi), multiple=med)
    return out


def lbo(a: Assumptions, entry_multiple: float, leverage: float, rate: float, exit_multiple: float,
        sweep: float = 1.0) -> dict:
    """5 year LBO. Entry and leverage multiples apply to next year EBITDA (the same basis as comps).
    Free cash flow after interest pays down debt; once debt is gone it builds cash. A cash shortfall
    is funded by borrowing more (a revolver). Equity at exit = exit EV - debt + cash."""
    f = forecast(a)
    ebitda0 = float(f.iloc[0]["EBITDA"])
    entry_ev = entry_multiple * ebitda0
    debt = leverage * ebitda0
    equity_in = entry_ev - debt
    if ebitda0 <= 0:
        raise ValueError("A buyout needs positive EBITDA")
    if equity_in <= 0:
        raise ValueError("Leverage too high for entry price")
    cash, sched = 0.0, []
    for _, r in f.iterrows():
        interest = debt * rate
        fcf_levered = r["FCF"] - interest * (1 - a.tax_rate)
        if fcf_levered >= 0:
            paydown = min(debt, fcf_levered * sweep)
            cash += fcf_levered - paydown
        else:
            paydown = fcf_levered  # negative: new borrowing to cover the shortfall
        debt -= paydown
        sched.append(dict(Year=int(r["Year"]), EBITDA=r["EBITDA"], **{"Interest": interest, "Levered FCF": fcf_levered,
                                                                    "Debt paydown": paydown, "Ending debt": debt, "Cash built": cash}))
    exit_ev = exit_multiple * f.iloc[-1]["EBITDA"]
    equity_out = exit_ev - debt + cash
    moic = equity_out / equity_in
    irr = moic ** (1 / len(f)) - 1 if moic > 0 else -1.0  # equity wiped out means a total loss
    return dict(entry_ev=entry_ev, equity_in=equity_in, exit_ev=exit_ev, equity_out=equity_out,
                moic=moic, irr=irr, schedule=pd.DataFrame(sched))


def blended_dcf_vec(a: Assumptions, dg, dm, w, g) -> np.ndarray:
    """Vectorized blended DCF (average of long run growth and exit multiple methods) for arrays of
    growth shifts, margin shifts, discount rates and long run growth rates. Matches dcf() exactly."""
    dg, dm, w, g = (np.asarray(x, dtype=float)[:, None] for x in (dg, dm, w, g))
    growth = np.asarray(a.growth)[None, :] + dg
    margin = np.clip(np.asarray(a.ebitda_margin)[None, :] + dm, 0.0, 0.95)
    rev = a.revenue0 * np.cumprod(1 + growth, axis=1)
    prev = np.concatenate([np.full((rev.shape[0], 1), a.revenue0), rev[:, :-1]], axis=1)
    ebitda = rev * margin
    da = rev * a.da_pct
    fcf = (ebitda - da) * (1 - a.tax_rate) + da - rev * a.capex_pct - a.nwc_pct * (rev - prev)
    years = np.arange(1, rev.shape[1] + 1)[None, :]
    pv_fcf = (fcf / (1 + w) ** (years - 0.5)).sum(axis=1)
    N = years[0, -1]
    w1, g1 = w[:, 0], g[:, 0]
    tv_g = fcf[:, -1] * (1 + g1) / (w1 - g1)
    tv_x = ebitda[:, -1] * a.exit_multiple
    ps = lambda pv_tv: (pv_fcf + pv_tv - a.debt + a.cash) / a.shares
    return (ps(tv_g / (1 + w1) ** (N - 0.5)) + ps(tv_x / (1 + w1) ** N)) / 2


def monte_carlo(a: Assumptions, n: int = 5000, growth_sd: float = 0.03, margin_sd: float = 0.02,
                wacc_sd: float = 0.01, tg_sd: float = 0.005, seed: int = 11) -> np.ndarray:
    """Distribution of blended DCF value per share when growth, margin, WACC and long run growth are
    uncertain. Each draw keeps WACC at least 1 point above long run growth."""
    rng = np.random.default_rng(seed)
    base_w = wacc(a)["wacc"]
    dg, dm = rng.normal(0, growth_sd, n), rng.normal(0, margin_sd, n)
    w = np.maximum(0.02, rng.normal(base_w, wacc_sd, n))
    g = np.minimum(w - 0.01, rng.normal(a.terminal_growth, tg_sd, n))
    return blended_dcf_vec(a, dg, dm, w, g)


def max_entry_multiple(a: Assumptions, leverage: float, rate: float, exit_multiple: float,
                       target_irr: float) -> float:
    """Highest entry EV/EBITDA that still earns the target IRR, by bisection. NaN if no price works."""
    lo, hi = leverage + 0.01, 40.0
    try:
        if lbo(a, lo, leverage, rate, exit_multiple)["irr"] < target_irr:
            return float("nan")
    except ValueError:
        return float("nan")
    for _ in range(50):
        mid = (lo + hi) / 2
        try:
            ok = lbo(a, mid, leverage, rate, exit_multiple)["irr"] >= target_irr
        except ValueError:
            ok = False
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


def recommendation(value: float, price: float, prob_up: float) -> tuple[str, str]:
    upside = value / price - 1
    if not np.isfinite(prob_up):
        move = f"{upside:.0%} upside" if upside >= 0 else f"{abs(upside):.0%} downside"
        return "HOLD", f"{move} to blended value. Simulations were unavailable, so no BUY or SELL is given"
    if upside >= 0.15 and prob_up >= 0.6:
        return "BUY", f"{upside:.0%} upside to blended value, {prob_up:.0%} of simulations above today's price"
    if upside <= -0.15 and prob_up <= 0.4:
        return "SELL", f"{abs(upside):.0%} downside to blended value, only {prob_up:.0%} of simulations above price"
    move = f"{upside:.0%} upside" if upside >= 0 else f"{abs(upside):.0%} downside"
    return "HOLD", f"{move} to blended value, {prob_up:.0%} of simulations above price. Not enough edge for a BUY or SELL"


SAMPLE_PEERS = pd.DataFrame({
    "Peer": ["Peer A", "Peer B", "Peer C", "Peer D", "Peer E", "Peer F"],
    "EV/EBITDA": [10.5, 12.0, 9.2, 13.4, 11.1, 8.7],
    "EV/Revenue": [2.4, 3.1, 1.9, 3.6, 2.7, 1.8],
})


def _shift(a: Assumptions, dg: float = 0.0, dm: float = 0.0) -> Assumptions:
    return replace(a, growth=tuple(x + dg for x in a.growth),
                   ebitda_margin=tuple(min(0.95, max(0.0, x + dm)) for x in a.ebitda_margin))


def blended_dcf(a: Assumptions, discount: float | None = None) -> float:
    """Average of the Gordon growth and exit multiple DCF values per share."""
    return (dcf(a, discount=discount)["per_share"] + dcf(a, discount=discount, method="exit")["per_share"]) / 2


def solve_shift(a: Assumptions, target: float, driver: str = "growth", lo: float = -0.5, hi: float = 0.8) -> float | None:
    """Reverse DCF: the uniform shift to every year's growth (or margin) that makes the blended
    DCF value equal the target price. Returns None if no shift in range gets there."""
    f = (lambda s: blended_dcf(_shift(a, dg=s)) - target) if driver == "growth" else \
        (lambda s: blended_dcf(_shift(a, dm=s)) - target)
    try:
        flo, fhi = f(lo), f(hi)
    except ValueError:
        return None
    if flo * fhi > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm > 0) == (fhi > 0):
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def tornado(a: Assumptions) -> pd.DataFrame:
    """Blended DCF value when each driver moves down and up by a standard step. Sorted by swing."""
    base_w = wacc(a)["wacc"]
    base = blended_dcf(a)
    tests = [
        ("Revenue growth, every year (±2 pts)", lambda s: blended_dcf(_shift(a, dg=0.02 * s))),
        ("EBITDA margin, every year (±2 pts)", lambda s: blended_dcf(_shift(a, dm=0.02 * s))),
        ("WACC (±1 pt)", lambda s: blended_dcf(a, discount=base_w - 0.01 * s)),
        ("Terminal growth (±0.5 pt)", lambda s: blended_dcf(replace(a, terminal_growth=a.terminal_growth + 0.005 * s))),
        ("Exit multiple (±1.0x)", lambda s: blended_dcf(replace(a, exit_multiple=a.exit_multiple + 1.0 * s))),
        ("Capex % of revenue (±1 pt)", lambda s: blended_dcf(replace(a, capex_pct=max(0.0, a.capex_pct - 0.01 * s)))),
    ]
    rows = []
    for name, fn in tests:
        try:
            down, up = fn(-1), fn(1)
        except ValueError:
            continue
        rows.append(dict(Driver=name, Downside=down, Upside=up, Swing=abs(up - down), Base=base))
    return pd.DataFrame(rows).sort_values("Swing", ascending=True).reset_index(drop=True)
