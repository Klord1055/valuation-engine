"""Valuation & Deal Engine: DCF, trading comps, LBO returns, sensitivity and Monte Carlo in one app."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from valuation.model import (SAMPLE_PEERS, Assumptions, comps_value, dcf, forecast, lbo, monte_carlo,
                             recommendation, sensitivity, wacc)

st.set_page_config(page_title="Valuation & Deal Engine", page_icon="📈", layout="wide")
NAVY, GOLD, GREEN, RED = "#1F2A44", "#C9A227", "#2E8B57", "#C0392B"

st.title("Valuation & Deal Engine")
st.caption("Values a company four ways (DCF with Gordon growth, DCF with an exit multiple, trading comps, "
           "and a leveraged buyout), stress tests the DCF with Monte Carlo, and turns it all into a BUY, HOLD "
           "or SELL call. Defaults describe a fictional mid cap company; every input is editable.")

d = Assumptions()
with st.sidebar:
    st.header("Company")
    name = st.text_input("Company", "Sample Co")
    price = st.number_input("Share price ($)", 0.5, 5000.0, d.price, 0.5)
    shares = st.number_input("Diluted shares (M)", 1.0, 20000.0, d.shares, 1.0)
    rev0 = st.number_input("Last year revenue ($M)", 1.0, 1_000_000.0, d.revenue0, 10.0)
    debt = st.number_input("Total debt ($M)", 0.0, 1_000_000.0, d.debt, 10.0)
    cash = st.number_input("Cash ($M)", 0.0, 1_000_000.0, d.cash, 10.0)
    st.header("Operating drivers")
    g1 = st.slider("Revenue growth, year 1", -0.20, 0.60, d.growth[0], 0.005, format="%.3f")
    g5 = st.slider("Revenue growth, year 5", -0.10, 0.40, d.growth[-1], 0.005, format="%.3f")
    m1 = st.slider("EBITDA margin, year 1", 0.0, 0.70, d.ebitda_margin[0], 0.005, format="%.3f")
    m5 = st.slider("EBITDA margin, year 5", 0.0, 0.70, d.ebitda_margin[-1], 0.005, format="%.3f")
    da = st.slider("D&A % of revenue", 0.0, 0.20, d.da_pct, 0.005, format="%.3f")
    capex = st.slider("Capex % of revenue", 0.0, 0.30, d.capex_pct, 0.005, format="%.3f")
    nwc = st.slider("Working capital % of revenue", -0.10, 0.40, d.nwc_pct, 0.005, format="%.3f")
    tax = st.slider("Tax rate", 0.0, 0.40, d.tax_rate, 0.01)
    st.header("Cost of capital")
    rf = st.slider("Risk free rate", 0.0, 0.08, d.risk_free, 0.001, format="%.3f")
    beta = st.slider("Beta", 0.2, 2.5, d.beta, 0.05)
    erp = st.slider("Equity risk premium", 0.03, 0.09, d.erp, 0.0025, format="%.4f")
    kd = st.slider("Pre tax cost of debt", 0.02, 0.15, d.pre_tax_cost_debt, 0.0025, format="%.4f")
    st.header("Terminal value")
    tg = st.slider("Terminal growth", 0.0, 0.05, d.terminal_growth, 0.0025, format="%.4f")
    exit_mult = st.slider("Exit EV/EBITDA", 4.0, 25.0, d.exit_multiple, 0.5)

lerp = lambda a, b: tuple(a + (b - a) * i / 4 for i in range(5))
a = replace(d, revenue0=rev0, growth=lerp(g1, g5), ebitda_margin=lerp(m1, m5), da_pct=da, capex_pct=capex,
            nwc_pct=nwc, tax_rate=tax, risk_free=rf, beta=beta, erp=erp, pre_tax_cost_debt=kd, debt=debt,
            cash=cash, shares=shares, price=price, terminal_growth=tg, exit_multiple=exit_mult)

w = wacc(a)
if w["wacc"] <= tg:
    st.error("WACC must be above terminal growth. Lower terminal growth or raise the cost of capital.")
    st.stop()
f = forecast(a)
g_dcf, x_dcf = dcf(a), dcf(a, method="exit")

tabs = st.tabs(["Summary", "DCF", "Comps", "LBO", "Monte Carlo", "Methodology"])

# Comps inputs live in their tab but feed the summary, so read them from session state
peers = st.session_state.get("peers", SAMPLE_PEERS)
comps = comps_value(a, peers)
lbo_inputs = st.session_state.get("lbo_inputs", dict(entry=10.0, lev=4.5, rate=0.08, exit=10.5, target=0.20))

with tabs[4]:
    st.subheader("Monte Carlo on DCF value")
    c1, c2, c3, c4, c5 = st.columns(5)
    n = c1.select_slider("Simulations", [1000, 2500, 5000, 10000], 2500)
    gsd = c2.slider("Growth uncertainty (sd)", 0.0, 0.10, 0.03, 0.005, format="%.3f")
    msd = c3.slider("Margin uncertainty (sd)", 0.0, 0.08, 0.02, 0.005, format="%.3f")
    wsd = c4.slider("WACC uncertainty (sd)", 0.0, 0.03, 0.01, 0.0025, format="%.4f")
    tsd = c5.slider("Terminal growth sd", 0.0, 0.015, 0.005, 0.001, format="%.3f")
    sims = monte_carlo(a, n, gsd, msd, wsd, tsd)
    prob_up = float((sims > price).mean())
    p5, p50, p95 = np.percentile(sims, [5, 50, 95])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Median value", f"${p50:,.2f}")
    k2.metric("5th to 95th percentile", f"${p5:,.0f} to ${p95:,.0f}")
    k3.metric("Chance value beats price", f"{prob_up:.0%}")
    k4.metric("Value at risk (5th pct vs price)", f"{p5 / price - 1:+.0%}")
    h = px.histogram(x=sims, nbins=70, color_discrete_sequence=[NAVY], labels={"x": "DCF value per share ($)"})
    h.add_vline(x=price, line_color=RED, annotation_text=f"Price ${price:,.2f}")
    h.add_vline(x=p50, line_color=GOLD, line_dash="dash", annotation_text="Median")
    h.update_layout(height=360, margin=dict(t=30, b=10), yaxis_title="Simulations")
    st.plotly_chart(h, use_container_width=True)

# Summary
with tabs[0]:
    lb = lbo(a, lbo_inputs["entry"], lbo_inputs["lev"], lbo_inputs["rate"], lbo_inputs["exit"])
    # price a sponsor could pay and still hit the target IRR, found by bisection on entry multiple
    lo, hi = 2.0, 30.0
    for _ in range(40):
        mid = (lo + hi) / 2
        try:
            ok = lbo(a, mid, lbo_inputs["lev"], lbo_inputs["rate"], lbo_inputs["exit"])["irr"] >= lbo_inputs["target"]
        except ValueError:
            ok = False
        lo, hi = (mid, hi) if ok else (lo, mid)
    ebitda0 = a.revenue0 * a.ebitda_margin[0]
    lbo_px = (lo * ebitda0 - a.debt + a.cash) / a.shares
    methods = pd.DataFrame([
        ("DCF, Gordon growth", g_dcf["per_share"] * 0.9, g_dcf["per_share"], g_dcf["per_share"] * 1.1),
        ("DCF, exit multiple", x_dcf["per_share"] * 0.9, x_dcf["per_share"], x_dcf["per_share"] * 1.1),
        ("Comps, EV/EBITDA", comps["EV/EBITDA"]["low"], comps["EV/EBITDA"]["mid"], comps["EV/EBITDA"]["high"]),
        ("Comps, EV/Revenue", comps["EV/Revenue"]["low"], comps["EV/Revenue"]["mid"], comps["EV/Revenue"]["high"]),
        ("Monte Carlo DCF (5th to 95th)", p5, p50, p95),
        (f"LBO at {lbo_inputs['target']:.0%} IRR", lbo_px * 0.9, lbo_px, lbo_px * 1.1),
    ], columns=["Method", "Low", "Mid", "High"])
    blended = float(methods["Mid"].iloc[:4].mean())
    call, why = recommendation(blended, price, prob_up)
    color = {"BUY": GREEN, "SELL": RED, "HOLD": GOLD}[call]
    st.markdown(f"<div style='padding:14px 18px;border-radius:8px;background:{color}18;border-left:6px solid {color}'>"
                f"<span style='font-size:28px;font-weight:700;color:{color}'>{call}</span>"
                f"<span style='font-size:17px'> &nbsp;{name}: blended value ${blended:,.2f} vs price ${price:,.2f}. {why}.</span></div>",
                unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("WACC", f"{w['wacc']:.2%}")
    s2.metric("DCF value (Gordon)", f"${g_dcf['per_share']:,.2f}", f"{g_dcf['per_share'] / price - 1:+.0%}")
    s3.metric("Comps value (EV/EBITDA)", f"${comps['EV/EBITDA']['mid']:,.2f}", f"{comps['EV/EBITDA']['mid'] / price - 1:+.0%}")
    s4.metric("Sponsor LBO IRR", f"{lb['irr']:.1%}", f"{lb['moic']:.1f}x MOIC", delta_color="off")
    ff = go.Figure()
    for _, r in methods.iloc[::-1].iterrows():
        ff.add_trace(go.Bar(y=[r["Method"]], x=[r["High"] - r["Low"]], base=[r["Low"]], orientation="h",
                            marker_color=NAVY, showlegend=False,
                            hovertemplate=f"{r['Method']}<br>${r['Low']:,.2f} to ${r['High']:,.2f}<extra></extra>"))
        ff.add_trace(go.Scatter(y=[r["Method"]], x=[r["Mid"]], mode="markers", marker=dict(color=GOLD, size=11, symbol="diamond"),
                                showlegend=False, hovertemplate=f"Mid ${r['Mid']:,.2f}<extra></extra>"))
    ff.add_vline(x=price, line_color=RED, annotation_text=f"Price ${price:,.2f}")
    ff.add_vline(x=blended, line_color=GREEN, line_dash="dash", annotation_text=f"Blended ${blended:,.2f}")
    ff.update_layout(title="Football field: value per share by method", height=380, margin=dict(t=40, b=10), xaxis_title="$ per share")
    st.plotly_chart(ff, use_container_width=True)
    st.caption("Blended value averages the two DCFs and the two comps methods. DCF bars show plus or minus 10%; "
               "comps bars show the 25th to 75th percentile peer multiple.")

# DCF
with tabs[1]:
    st.subheader("Five year forecast and free cash flow")
    show = f.set_index("Year")[["Revenue", "Growth", "EBITDA", "Margin", "NOPAT", "DA", "Capex", "dNWC", "FCF"]].T
    show.columns = [f"Year {c}" for c in show.columns]
    show.index = ["Revenue", "Growth", "EBITDA", "EBITDA margin", "NOPAT", "D&A", "Capex", "Change in working capital", "Free cash flow"]
    pct_rows = ["Growth", "Margin"]
    st.dataframe(show.style.format(lambda v: f"{v:.1%}" if abs(v) < 1.5 and v != 0 else f"{v:,.1f}"), use_container_width=True)
    bars = go.Figure()
    bars.add_bar(x=f["Year"], y=f["Revenue"], name="Revenue", marker_color="#B4C3DD")
    bars.add_bar(x=f["Year"], y=f["EBITDA"], name="EBITDA", marker_color=NAVY)
    bars.add_scatter(x=f["Year"], y=f["FCF"], name="Free cash flow", mode="lines+markers", line=dict(color=GOLD, width=3))
    bars.update_layout(barmode="overlay", height=320, margin=dict(t=10, b=10), yaxis_title="$M", xaxis_title="Year")
    st.plotly_chart(bars, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**WACC build**")
        st.dataframe(pd.DataFrame({"Item": ["Cost of equity (CAPM)", "After tax cost of debt", "Equity weight", "Debt weight", "WACC"],
                                   "Value": [f"{w['cost_equity']:.2%}", f"{w['after_tax_debt']:.2%}", f"{w['we']:.1%}", f"{w['wd']:.1%}", f"{w['wacc']:.2%}"]}),
                     hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Enterprise to equity bridge (Gordon growth)**")
        st.dataframe(pd.DataFrame({"Item": ["PV of forecast FCF", "PV of terminal value", "Enterprise value", "Less debt", "Plus cash", "Equity value", "Value per share"],
                                   "$M": [g_dcf["pv_fcf"], g_dcf["pv_tv"], g_dcf["ev"], -a.debt, a.cash, g_dcf["equity"], g_dcf["per_share"]]})
                     .style.format({"$M": "{:,.2f}"}), hide_index=True, use_container_width=True)
        st.caption(f"Terminal value is {g_dcf['tv_share']:.0%} of enterprise value and implies {g_dcf['implied_exit']:.1f}x EBITDA "
                   f"(vs your {exit_mult:.1f}x exit multiple).")
    st.markdown("**Sensitivity: value per share by WACC (rows) and terminal growth (columns)**")
    base = w["wacc"]
    sens = sensitivity(a, [base + x for x in (-0.02, -0.01, 0, 0.01, 0.02)], [tg + x for x in (-0.01, -0.005, 0, 0.005, 0.01)])
    hm = px.imshow(sens.values, x=sens.columns, y=sens.index, text_auto=".2f", aspect="auto",
                   color_continuous_scale=[[0, "#F3C6C1"], [0.5, "#FFFFFF"], [1, "#C6E4D0"]],
                   color_continuous_midpoint=price, labels=dict(x="Terminal growth", y="WACC", color="$/share"))
    hm.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(hm, use_container_width=True)
    st.caption("Green cells are above today's price, red cells below.")

# Comps
with tabs[2]:
    st.subheader("Trading comparables")
    st.write("Edit the peer set. Implied value uses next year's forecast EBITDA and revenue.")
    edited = st.data_editor(peers, num_rows="dynamic", use_container_width=True, hide_index=True, key="peer_editor")
    if not edited.equals(peers):
        st.session_state["peers"] = edited
        st.rerun()
    rows = [{"Method": k, "Median multiple": v["multiple"], "Low ($)": v["low"], "Mid ($)": v["mid"], "High ($)": v["high"]} for k, v in comps.items()]
    st.dataframe(pd.DataFrame(rows).style.format({"Median multiple": "{:.1f}x", "Low ($)": "{:,.2f}", "Mid ($)": "{:,.2f}", "High ($)": "{:,.2f}"}),
                 hide_index=True, use_container_width=True)
    sc = px.scatter(edited, x="EV/Revenue", y="EV/EBITDA", text="Peer", color_discrete_sequence=[NAVY])
    sc.update_traces(textposition="top center")
    sc.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(sc, use_container_width=True)

# LBO
with tabs[3]:
    st.subheader("Leveraged buyout returns")
    c1, c2, c3, c4, c5 = st.columns(5)
    entry = c1.slider("Entry EV/EBITDA", 4.0, 20.0, lbo_inputs["entry"], 0.5)
    lev = c2.slider("Debt / EBITDA", 0.0, 7.0, lbo_inputs["lev"], 0.25)
    rate = c3.slider("Interest rate", 0.03, 0.14, lbo_inputs["rate"], 0.0025, format="%.4f")
    exitm = c4.slider("Exit EV/EBITDA", 4.0, 20.0, lbo_inputs["exit"], 0.5)
    target = c5.slider("Target IRR", 0.10, 0.35, lbo_inputs["target"], 0.01)
    new = dict(entry=entry, lev=lev, rate=rate, exit=exitm, target=target)
    if new != lbo_inputs:
        st.session_state["lbo_inputs"] = new
        st.rerun()
    try:
        r = lbo(a, entry, lev, rate, exitm)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Equity invested", f"${r['equity_in']:,.0f}M")
        k2.metric("Equity at exit", f"${r['equity_out']:,.0f}M")
        k3.metric("IRR", f"{r['irr']:.1%}", f"{r['irr'] - target:+.1%} vs target")
        k4.metric("MOIC", f"{r['moic']:.2f}x")
        st.dataframe(r["schedule"].style.format({c: "{:,.1f}" for c in r["schedule"].columns if c != "Year"}), hide_index=True, use_container_width=True)
        grid = pd.DataFrame({f"{x:.1f}x exit": [lbo(a, e, lev, rate, x)["irr"] for e in (entry - 1, entry, entry + 1)] for x in (exitm - 1, exitm, exitm + 1)},
                            index=[f"{e:.1f}x entry" for e in (entry - 1, entry, entry + 1)])
        st.markdown("**IRR by entry and exit multiple**")
        st.dataframe(grid.style.format("{:.1%}"), use_container_width=True)
    except ValueError as e:
        st.error(str(e))

with tabs[5]:
    st.markdown((Path(__file__).parent / "HOW_IT_WORKS.md").read_text())
