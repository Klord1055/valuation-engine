"""Valuation & Deal Engine: DCF, comps, LBO and Monte Carlo in one place, plus a reverse DCF that
shows what the share price already assumes and which drivers could flip the call."""
import html
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import importlib
import ui
importlib.reload(ui)  # pick up ui.py changes on hot redeploy
from ui import intro, note, GOLD, GREEN, GREY, NAVY, RED, setup, takeaway, wrap
from valuation.model import (SAMPLE_PEERS, Assumptions, blended_dcf, comps_value, dcf, forecast, lbo,
                             max_entry_multiple, monte_carlo, recommendation, sensitivity, solve_shift,
                             tornado, wacc)

st.set_page_config(page_title="Valuation & Deal Engine", page_icon="📈", layout="wide")
setup()
W = "stretch"
D = Assumptions()
DEFAULT_LBO = dict(entry=10.0, lev=4.5, rate=8.0, exit=10.5, target=20.0)


def reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def pct(label, lo, hi, default, step, help_text, key):
    """Slider shown in plain percent, returned as a decimal."""
    return st.slider(label, lo, hi, round(default * 100, 2), step, format="%.1f%%", help=help_text, key=key) / 100


# Sidebar inputs, grouped the way an analyst builds a model
with st.sidebar:
    st.button("Reset to defaults", on_click=reset, width=W)
    st.header("1. The company")
    name = st.text_input("Company name", "Sample Co", key="name")
    price = st.number_input("Share price today ($)", 0.5, 5000.0, D.price, 0.5, key="price",
                            help="The current market price per share. Every call is judged against this.")
    shares = st.number_input("Diluted shares (millions)", 1.0, 20000.0, D.shares, 1.0, key="shares",
                             help="Shares outstanding including options and convertibles.")
    rev0 = st.number_input("Last year revenue ($M)", 1.0, 1_000_000.0, D.revenue0, 10.0, key="rev0")
    debt = st.number_input("Total debt ($M)", 0.0, 1_000_000.0, D.debt, 10.0, key="debt")
    cash = st.number_input("Cash ($M)", 0.0, 1_000_000.0, D.cash, 10.0, key="cash")
    st.header("2. Growth and profit")
    g1 = pct("Revenue growth next year", -20.0, 60.0, D.growth[0], 0.5,
             "Growth glides in a straight line from next year's rate to the year 5 rate.", "g1")
    g5 = pct("Revenue growth in year 5", -10.0, 40.0, D.growth[-1], 0.5, "Where growth settles by year 5.", "g5")
    m1 = pct("EBITDA margin next year", 0.0, 70.0, D.ebitda_margin[0], 0.5,
             "EBITDA as a share of revenue. Moves in a straight line to the year 5 margin.", "m1")
    m5 = pct("EBITDA margin in year 5", 0.0, 70.0, D.ebitda_margin[-1], 0.5, "Margin the business reaches by year 5.", "m5")
    with st.expander("Cash flow details"):
        da = pct("D&A, % of revenue", 0.0, 20.0, D.da_pct, 0.5, "Depreciation and amortization.", "da")
        capex = pct("Capex, % of revenue", 0.0, 30.0, D.capex_pct, 0.5, "Capital spending to sustain and grow.", "capex")
        nwc = pct("Working capital, % of revenue", -10.0, 40.0, D.nwc_pct, 0.5,
                  "Cash tied up in receivables and inventory, net of payables.", "nwc")
        tax = pct("Tax rate", 0.0, 40.0, D.tax_rate, 1.0, "Cash tax rate on operating profit.", "tax")
    st.header("3. Cost of capital")
    rf = pct("Risk free rate", 0.0, 8.0, D.risk_free, 0.1, "Usually the 10 year Treasury yield.", "rf")
    beta = st.slider("Beta", 0.2, 2.5, D.beta, 0.05, key="beta",
                     help="How much the stock moves with the market. 1.0 moves in line with it.")
    erp = pct("Equity risk premium", 3.0, 9.0, D.erp, 0.25, "Extra return investors want for owning stocks.", "erp")
    kd = pct("Pre tax cost of debt", 2.0, 15.0, D.pre_tax_cost_debt, 0.25, "Interest rate the company pays on new debt.", "kd")
    st.header("4. Long run value")
    tg = pct("Long run growth after year 5", 0.0, 5.0, D.terminal_growth, 0.25,
             "Growth forever after the forecast. Usually near inflation.", "tg")
    exit_mult = st.slider("Exit EV/EBITDA multiple", 4.0, 25.0, D.exit_multiple, 0.5, key="exitm",
                          help="What a buyer would pay for the business in year 5, as a multiple of EBITDA.")

lerp = lambda x, y: tuple(x + (y - x) * i / 4 for i in range(5))
a = replace(D, revenue0=rev0, growth=lerp(g1, g5), ebitda_margin=lerp(m1, m5), da_pct=da, capex_pct=capex,
            nwc_pct=nwc, tax_rate=tax, risk_free=rf, beta=beta, erp=erp, pre_tax_cost_debt=kd, debt=debt,
            cash=cash, shares=shares, price=price, terminal_growth=tg, exit_multiple=exit_mult)

st.title("Valuation & Deal Engine")
intro("Four standard valuation methods side by side, plus a reverse DCF that shows what today's price "
           "already assumes. Defaults describe a fictional mid cap company. Change anything in the left panel.")

w = wacc(a)
safe_name = "".join(ch if ch.isalnum() else "_" for ch in name) or "company"
if w["wacc"] <= tg + 0.005:
    st.error(f"Your cost of capital ({w['wacc']:.1%}) needs to be at least half a point above long run growth "
             f"({tg:.1%}) for the model to work. Lower long run growth or raise the cost of capital in the left panel.")
    st.stop()

if w["wacc"] - tg < 0.02:
    st.warning(f"Cost of capital ({w['wacc']:.1%}) is within 2 points of long run growth ({tg:.1%}), so the long run value "
               "dominates and small input changes swing the answer a lot. Treat the DCF with caution.")
f = forecast(a)
ebitda_next = float(f.iloc[0]["EBITDA"])
g_dcf, x_dcf = dcf(a), dcf(a, method="exit")
peers = st.session_state.get("peers", SAMPLE_PEERS)
comps = comps_value(a, peers)
li = st.session_state.get("lbo_inputs", DEFAULT_LBO)

# Monte Carlo inputs are set in their tab but the headline needs the result, so read them from state
mc = st.session_state.get("mc", dict(n=2500, gsd=3.0, msd=2.0, wsd=1.0, tsd=0.5))
sims = monte_carlo(a, mc["n"], mc["gsd"] / 100, mc["msd"] / 100, mc["wsd"] / 100, mc["tsd"] / 100)
sims = sims[np.isfinite(sims)]
prob_up = float((sims > price).mean()) if len(sims) else float("nan")
p5, p50, p95 = np.percentile(sims, [5, 50, 95]) if len(sims) else (np.nan,) * 3

mid_values = [g_dcf["per_share"], x_dcf["per_share"], comps["EV/EBITDA"]["mid"], comps["EV/Revenue"]["mid"]]
blended = float(np.nanmean(mid_values))
call, why = recommendation(blended, price, prob_up)
color = {"BUY": GREEN, "SELL": RED, "HOLD": GOLD}[call]

# Reverse DCF: what today's price assumes
need_dg = solve_shift(a, price, "growth")
need_dm = solve_shift(a, price, "margin")
avg_g = float(np.mean(a.growth))
avg_m = float(np.mean(a.ebitda_margin))

# Headline: the answer first
st.markdown(
    f"<div style='padding:16px 20px;border-radius:10px;background:#FFFFFF;border:2px solid {color};'>"
    f"<div style='font-size:15px;color:#2B2F36;font-weight:600'>THE CALL</div>"
    f"<div style='font-size:34px;font-weight:800;color:{color};line-height:1.2'>{call}</div>"
    f"<div style='font-size:18px;color:#111'>{html.escape(name)}: blended value <b>${blended:,.2f}</b> vs price <b>${price:,.2f}</b>. {why}.</div>"
    f"</div>", unsafe_allow_html=True)

h1, h2, h3, h4 = st.columns(4)
h1.metric("Blended value per share", f"${blended:,.2f}", f"{blended / price - 1:+.0%} vs price")
h2.metric("Chance value beats price", f"{prob_up:.0%}" if np.isfinite(prob_up) else "n/a",
          help="Share of Monte Carlo simulations where DCF value lands above today's price.")
if need_dg is not None:
    h3.metric("Growth the price assumes", f"{avg_g + need_dg:.1%} a year",
              help="Reverse DCF: the average annual revenue growth that makes the DCF value exactly equal today's price.")
    note(h3, f"{abs(need_dg) * 100:.1f} pts {'below' if need_dg < 0 else 'above'} your {avg_g:.1%} forecast")
else:
    h3.metric("Growth the price assumes", "no realistic answer",
              help="No growth path within 50 points below to 80 points above your forecast makes the DCF equal today's price.")
h4.metric("Cost of capital (WACC)", f"{w['wacc']:.2%}")

tabs = st.tabs(["Summary", "What the price assumes", "DCF", "Comps", "LBO", "Monte Carlo", "Methodology"])

# Summary
with tabs[0]:
    lbo_price = lambda irr: (max_entry_multiple(a, min(li["lev"], 30.0), li["rate"] / 100, li["exit"], irr) * ebitda_next
                             - a.debt + a.cash) / a.shares
    try:
        lb = lbo(a, li["entry"], li["lev"], li["rate"] / 100, li["exit"])
        lbo_px, lbo_hi, lbo_lo = lbo_price(li["target"] / 100), lbo_price(li["target"] / 100 - 0.05), lbo_price(li["target"] / 100 + 0.05)
    except ValueError:
        lb, lbo_px, lbo_hi, lbo_lo = None, np.nan, np.nan, np.nan
    dw = [w["wacc"] + 0.01, w["wacc"] - 0.01]
    g_rng = [dcf(a, discount=x)["per_share"] if x > tg else np.nan for x in dw]
    x_rng = [dcf(a, discount=x, method="exit")["per_share"] for x in dw]
    methods = pd.DataFrame([
        ("DCF, long run growth", g_rng[0], g_dcf["per_share"], g_rng[1]),
        ("DCF, exit multiple", x_rng[0], x_dcf["per_share"], x_rng[1]),
        ("Comps, EV/EBITDA", comps["EV/EBITDA"]["low"], comps["EV/EBITDA"]["mid"], comps["EV/EBITDA"]["high"]),
        ("Comps, EV/Revenue", comps["EV/Revenue"]["low"], comps["EV/Revenue"]["mid"], comps["EV/Revenue"]["high"]),
        ("Monte Carlo DCF, 5th to 95th", p5, p50, p95),
        (f"Max buyout price at {li['target']:.0f}% IRR", lbo_lo, lbo_px, lbo_hi),
    ], columns=["Method", "Low", "Mid", "High"]).dropna(subset=["Mid"])
    methods["Low"] = methods["Low"].fillna(methods["Mid"])
    methods["High"] = methods["High"].fillna(methods["Mid"])
    ff = go.Figure()
    for _, r in methods.iloc[::-1].iterrows():
        ff.add_trace(go.Bar(y=[r["Method"]], x=[r["High"] - r["Low"]], base=[r["Low"]], orientation="h",
                            marker_color=NAVY, showlegend=False,
                            hovertemplate=f"{r['Method']}<br>${r['Low']:,.2f} to ${r['High']:,.2f}<extra></extra>"))
        ff.add_trace(go.Scatter(y=[r["Method"]], x=[r["Mid"]], mode="markers", showlegend=False,
                                marker=dict(color=GOLD, size=13, symbol="diamond", line=dict(color="#111", width=1)),
                                hovertemplate=f"Midpoint ${r['Mid']:,.2f}<extra></extra>"))
    ff.add_vline(x=price, line_color=RED, line_width=3, annotation_text=f"Price ${price:,.2f}",
                 annotation_position="top left", annotation_yshift=18, annotation_font=dict(size=14, color=RED))
    ff.add_vline(x=blended, line_color=GREEN, line_dash="dash", line_width=3, annotation_text=f"Blended ${blended:,.2f}",
                 annotation_position="bottom right", annotation_font=dict(size=14, color=GREEN))
    ff.update_layout(title="Football field: value per share by method", height=400, margin=dict(t=80, b=10, l=10),
                     xaxis_title="$ per share")
    ff.update_yaxes(tickvals=list(methods["Method"]), ticktext=[wrap(m) for m in methods["Method"]])
    st.plotly_chart(ff, width=W, theme=None)
    above = int((methods["Mid"] > price).sum())
    n_comps = int(sum(np.isfinite(comps[k]["mid"]) for k in comps))
    blend_desc = "the two DCFs and the two comps methods" if n_comps == 2 else (
        "the two DCFs and one comps method" if n_comps == 1 else "the two DCFs only, because no valid peers are entered")
    takeaway(f"<b>{above} of {len(methods)}</b> methods put value above today's price. Blended value averages {blend_desc}. "
             f"DCF bars show WACC one point higher and lower; comps bars show the 25th to 75th percentile peer multiple; "
             f"the buyout bar shows a required return 5 points higher and lower.")
    brief = pd.DataFrame({
        "Item": ["Company", "Call", "Price", "Blended value", "Upside", "Chance value beats price", "WACC",
                 "DCF, long run growth", "DCF, exit multiple", "Comps EV/EBITDA", "Comps EV/Revenue",
                 "Growth the price assumes", "Sponsor IRR"],
        "Value": [name, call, f"${price:,.2f}", f"${blended:,.2f}", f"{blended / price - 1:+.1%}", f"{prob_up:.0%}" if np.isfinite(prob_up) else "n/a",
                  f"{w['wacc']:.2%}", f"${g_dcf['per_share']:,.2f}", f"${x_dcf['per_share']:,.2f}",
                  f"${comps['EV/EBITDA']['mid']:,.2f}", f"${comps['EV/Revenue']['mid']:,.2f}",
                  f"{avg_g + need_dg:.1%}" if need_dg is not None else "out of range",
                  f"{lb['irr']:.1%}" if lb else "n/a"]})
    st.download_button("Download one page summary (CSV)", brief.to_csv(index=False), f"{safe_name}_valuation_summary.csv")

# What the price assumes
with tabs[1]:
    st.subheader("What you have to believe")
    c1, c2 = st.columns(2)
    with c1:
        if need_dg is not None:
            st.metric("Average revenue growth the price assumes", f"{avg_g + need_dg:.1%}")
            note(st, f"{abs(need_dg) * 100:.1f} pts {'below' if need_dg < 0 else 'above'} your {avg_g:.1%} forecast")
        else:
            st.metric("Average revenue growth the price assumes", "out of range")
    with c2:
        if need_dm is not None:
            st.metric("Average EBITDA margin the price assumes", f"{avg_m + need_dm:.1%}")
            note(st, f"{abs(need_dm) * 100:.1f} pts {'below' if need_dm < 0 else 'above'} your {avg_m:.1%} forecast")
        else:
            st.metric("Average EBITDA margin the price assumes", "out of range")
    if need_dg is None:
        takeaway("No realistic growth path makes the DCF equal today's price. The price is far from what the business can "
                 "justify on these inputs, or an input such as share count or debt needs checking.")
    else:
        if need_dg < 0:
            takeaway(f"The market price only needs <b>{avg_g + need_dg:.1%}</b> growth a year, below your <b>{avg_g:.1%}</b>. "
                     f"If you trust your forecast, the stock has a cushion of <b>{-need_dg * 100:.1f} points</b> of growth.")
        else:
            takeaway(f"The market price needs <b>{avg_g + need_dg:.1%}</b> growth a year, above your <b>{avg_g:.1%}</b>. "
                     f"The market is more optimistic than you by <b>{need_dg * 100:.1f} points</b> of growth.")
    tor = tornado(a)
    st.subheader("What moves value the most")
    tf = go.Figure()
    tf.add_bar(y=tor["Driver"], x=tor["Downside"] - tor["Base"], base=tor["Base"], orientation="h", name="Driver gets worse",
               marker_color=RED, customdata=tor["Downside"], hovertemplate="%{y}<br>Value $%{customdata:,.2f}<extra></extra>")
    tf.add_bar(y=tor["Driver"], x=tor["Upside"] - tor["Base"], base=tor["Base"], orientation="h", name="Driver gets better",
               marker_color=GREEN, customdata=tor["Upside"], hovertemplate="%{y}<br>Value $%{customdata:,.2f}<extra></extra>")
    tf.add_vline(x=price, line_color="#111", line_width=2, line_dash="dot", annotation_text=f"Price ${price:,.2f}",
                  annotation_position="top right", annotation_yshift=18)
    tf.update_layout(barmode="overlay", height=380, margin=dict(t=55, b=10), xaxis_title="Blended DCF value per share ($)",
                     legend=dict(orientation="h", y=-0.2))
    tf.update_yaxes(tickvals=list(tor["Driver"]), ticktext=[wrap(m) for m in tor["Driver"]])
    st.plotly_chart(tf, width=W, theme=None)
    top = tor.iloc[-1]
    crosses = tor[(tor["Downside"] < price) & (tor["Upside"] > price) | (tor["Downside"] > price) & (tor["Upside"] < price)]
    short = lambda d: d.split(" (")[0].replace(", every year", "")
    names = [short(d) for d in crosses["Driver"]][::-1]
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1] if names else ""
    msg = (f"<b>{short(top['Driver'])}</b> matters most: moving it from one step worse to one step better swings value by "
           f"<b>${top['Swing']:,.2f}</b> a share. ")
    msg += (f"One step in <b>{joined}</b> is enough to push value across today's price." if names
            else "No single step in any driver pushes value across today's price, so the call is not fragile.")
    takeaway(msg)

# DCF
with tabs[2]:
    st.subheader("Five year forecast and free cash flow ($M)")
    show = f.set_index("Year")[["Revenue", "Growth", "EBITDA", "Margin", "NOPAT", "DA", "Capex", "dNWC", "FCF"]].T
    show.columns = [f"Year {c}" for c in show.columns]
    labels = ["Revenue", "Revenue growth", "EBITDA", "EBITDA margin", "After tax operating profit", "D&A", "Capex",
              "Change in working capital", "Free cash flow"]
    show.index = labels
    fmt = show.copy().astype(object)
    for r in labels:
        fmt.loc[r] = [f"{v:.1%}" if r in ("Revenue growth", "EBITDA margin") else f"{v:,.1f}" for v in show.loc[r]]
    st.dataframe(fmt, width=W)
    bars = go.Figure()
    bars.add_bar(x=f["Year"], y=f["Revenue"], name="Revenue", marker_color="#8EA3CF")
    bars.add_bar(x=f["Year"], y=f["EBITDA"], name="EBITDA", marker_color=NAVY)
    bars.add_scatter(x=f["Year"], y=f["FCF"], name="Free cash flow", mode="lines+markers", line=dict(color=GOLD, width=4))
    bars.update_layout(barmode="overlay", height=330, margin=dict(t=10, b=10), yaxis_title="$M", xaxis_title="Year",
                       legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(bars, width=W, theme=None)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cost of capital build**")
        st.dataframe(pd.DataFrame({"Item": ["Cost of equity (CAPM)", "After tax cost of debt", "Equity weight", "Debt weight", "WACC"],
                                   "Value": [f"{w['cost_equity']:.2%}", f"{w['after_tax_debt']:.2%}", f"{w['we']:.1%}",
                                             f"{w['wd']:.1%}", f"{w['wacc']:.2%}"]}), hide_index=True, width=W)
    with c2:
        st.markdown("**From enterprise value to value per share**")
        st.dataframe(pd.DataFrame({"Item": ["Present value of 5 year cash flow", "Present value of long run value", "Enterprise value",
                                            "Less debt", "Plus cash", "Equity value", "Value per share ($)"],
                                   "$M": [g_dcf["pv_fcf"], g_dcf["pv_tv"], g_dcf["ev"], -a.debt, a.cash, g_dcf["equity"],
                                          g_dcf["per_share"]]}).style.format({"$M": "{:,.2f}"}), hide_index=True, width=W)
    if np.isfinite(g_dcf["tv_share"]):
        implied = (f"it implies <b>{g_dcf['implied_exit']:.1f}x</b> EBITDA versus your {exit_mult:.1f}x exit multiple. A big gap "
                   "between those two means one of the long run inputs needs a second look." if np.isfinite(g_dcf["implied_exit"])
                   else "year 5 EBITDA is zero, so no exit multiple can be implied.")
        takeaway(f"<b>{g_dcf['tv_share']:.0%}</b> of enterprise value comes after year 5, and {implied}")
    st.markdown("**Value per share by WACC (rows) and long run growth (columns)**")
    sens = sensitivity(a, [w["wacc"] + x for x in (-0.02, -0.01, 0, 0.01, 0.02)], [tg + x for x in (-0.01, -0.005, 0, 0.005, 0.01)])
    hm = px.imshow(sens.values, x=list(sens.columns), y=list(sens.index), text_auto=".2f", aspect="auto",
                   color_continuous_scale=[[0, "#F4B4AE"], [0.5, "#FFFFFF"], [1, "#9ED3B0"]], color_continuous_midpoint=price,
                   labels=dict(x="Long run growth", y="WACC", color="$/share"))
    hm.update_traces(textfont=dict(size=15, color="#111"))
    hm.update_layout(height=340, margin=dict(t=10, b=10))
    st.plotly_chart(hm, width=W, theme=None)
    takeaway(f"Green cells are above today's ${price:,.2f} price, red cells are below. "
             f"<b>{int((sens.values > price).sum())} of {int(np.isfinite(sens.values).sum())}</b> combinations beat the price.")

# Comps
with tabs[3]:
    st.subheader("Trading comparables")
    edited = st.data_editor(peers, num_rows="dynamic", width=W, hide_index=True, key="peer_editor",
                            column_config={"EV/EBITDA": st.column_config.NumberColumn(min_value=0.0, format="%.1fx"),
                                           "EV/Revenue": st.column_config.NumberColumn(min_value=0.0, format="%.2fx")})
    if not edited.equals(peers):
        st.session_state["peers"] = edited
        st.rerun()
    if all(np.isnan(v["mid"]) for v in comps.values()):
        st.warning("Add at least one peer with a positive multiple to value the company on comps.")
    else:
        rows = [{"Method": k, "Median peer multiple": v["multiple"], "Low ($)": v["low"], "Mid ($)": v["mid"], "High ($)": v["high"]}
                for k, v in comps.items()]
        st.dataframe(pd.DataFrame(rows).style.format({"Median peer multiple": "{:.1f}x", "Low ($)": "{:,.2f}", "Mid ($)": "{:,.2f}",
                                                      "High ($)": "{:,.2f}"}, na_rep="n/a"), hide_index=True, width=W)
        valid = edited.dropna(subset=["EV/EBITDA", "EV/Revenue"])
        if len(valid):
            sc = px.scatter(valid, x="EV/Revenue", y="EV/EBITDA", text="Peer", color_discrete_sequence=[NAVY])
            sc.update_traces(textposition="top center", marker=dict(size=13), textfont=dict(size=14))
            sc.update_layout(height=330, margin=dict(t=10, b=10))
            st.plotly_chart(sc, width=W, theme=None)
        own = (a.price * a.shares + a.debt - a.cash) / ebitda_next if ebitda_next > 0 else np.nan
        med = comps["EV/EBITDA"]["multiple"]
        if np.isfinite(own) and np.isfinite(med):
            takeaway(f"{html.escape(name)} trades at <b>{own:.1f}x</b> next year EBITDA against a peer median of <b>{med:.1f}x</b>, "
                     f"a <b>{abs(own / med - 1):.0%} {'premium' if own > med else 'discount'}</b>. Edit, add or delete peers in the table.")

# LBO
with tabs[4]:
    st.subheader("Could a private equity buyer make money here?")
    st.markdown("<div class='note' style='margin-top:0'>Multiples below are on next year EBITDA, the same basis as the comps tab.</div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    entry = c1.slider("Purchase price, EV/EBITDA", 4.0, 20.0, li["entry"], 0.5, help="What the buyer pays, as a multiple of EBITDA.")
    lev = c2.slider("Debt, x EBITDA", 0.0, 7.0, li["lev"], 0.25, help="How much of the purchase is funded with debt.")
    rate = c3.slider("Interest rate", 3.0, 14.0, li["rate"], 0.25, format="%.2f%%")
    exitm = c4.slider("Sale price in year 5, EV/EBITDA", 4.0, 20.0, li["exit"], 0.5)
    target = c5.slider("Required return (IRR)", 10.0, 35.0, li["target"], 1.0, format="%.0f%%")
    new = dict(entry=entry, lev=lev, rate=rate, exit=exitm, target=target)
    if new != li:
        st.session_state["lbo_inputs"] = new
        st.rerun()
    if ebitda_next <= 0:
        st.warning("A buyout needs positive EBITDA to borrow against. Raise next year's EBITDA margin in the left panel.")
    elif lev >= entry:
        st.warning(f"Debt of {lev:.2f}x EBITDA covers the whole {entry:.1f}x purchase price, so the buyer puts in no equity. "
                   "Lower the debt or raise the purchase price.")
    else:
        r = lbo(a, entry, lev, rate / 100, exitm)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Buyer's equity in", f"${r['equity_in']:,.0f}M")
        k2.metric("Equity value at exit", f"${r['equity_out']:,.0f}M")
        k3.metric("IRR", f"{r['irr']:.1%}" if r["irr"] > -1 else "Total loss", f"{(r['irr'] * 100 - target):+.1f} pts vs target")
        k4.metric("Money multiple (MOIC)", f"{r['moic']:.2f}x")
        max_m = max_entry_multiple(a, lev, rate / 100, exitm, target / 100)
        if not np.isfinite(max_m):
            takeaway(f"No purchase price earns a {target:.0f}% return with this debt, rate and sale price, so a buyout does not support the stock.")
        else:
            max_px = (max_m * ebitda_next - a.debt + a.cash) / a.shares
            gap = max_px / price - 1
            verdict = ("That is well above today's price, so takeover interest could support the stock." if gap > 0.15 else
                       "That is above today's price, so buyout interest offers some support." if gap > 0.05 else
                       "That is close to today's price, so buyout interest offers only modest support." if gap > -0.05 else
                       "That is below today's price, so buyout interest does not support the stock.")
            takeaway(f"A buyer needing a {target:.0f}% return could pay up to <b>{max_m:.1f}x</b> next year EBITDA, about "
                     f"<b>${max_px:,.2f}</b> a share versus ${price:,.2f} today. {verdict}")
        sched = r["schedule"].copy()
        st.dataframe(sched.style.format({c: "{:,.1f}" for c in sched.columns if c != "Year"}), hide_index=True, width=W)
        ents, exs = (entry - 1, entry, entry + 1), (exitm - 1, exitm, exitm + 1)
        grid = pd.DataFrame({f"Sell at {x:.1f}x": [lbo(a, e, min(lev, e - 0.01), rate / 100, x)["irr"] for e in ents] for x in exs},
                            index=[f"Buy at {e:.1f}x" for e in ents])
        st.markdown("**IRR by purchase and sale multiple**")
        st.dataframe(grid.style.format(lambda v: f"{v:.1%}" if v > -1 else "Total loss"), width=W)

# Monte Carlo
with tabs[5]:
    st.subheader("How sure can we be?")
    c1, c2, c3, c4, c5 = st.columns(5)
    n = c1.select_slider("Simulations", [1000, 2500, 5000, 10000], mc["n"])
    gsd = c2.slider("Growth uncertainty (±pts)", 0.0, 10.0, mc["gsd"], 0.5, format="%.1f", help="Standard deviation of growth, in percentage points.")
    msd = c3.slider("Margin uncertainty (±pts)", 0.0, 8.0, mc["msd"], 0.5, format="%.1f")
    wsd = c4.slider("WACC uncertainty (±pts)", 0.0, 3.0, mc["wsd"], 0.25, format="%.2f")
    tsd = c5.slider("Long run growth uncertainty (±pts)", 0.0, 1.5, mc["tsd"], 0.1, format="%.1f")
    newmc = dict(n=n, gsd=gsd, msd=msd, wsd=wsd, tsd=tsd)
    if newmc != mc:
        st.session_state["mc"] = newmc
        st.rerun()
    if len(sims):
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median value", f"${p50:,.2f}")
        k2.metric("Likely range (5th to 95th)", f"${p5:,.0f} to ${p95:,.0f}")
        k3.metric("Chance value beats price", f"{prob_up:.0%}")
        k4.metric("Bad case vs price (5th pct)", f"{p5 / price - 1:+.0%}")
        h = px.histogram(x=sims, nbins=70, color_discrete_sequence=[NAVY], labels={"x": "DCF value per share ($)"})
        h.add_vline(x=price, line_color=RED, line_width=3, annotation_text=f"Price ${price:,.2f}", annotation_font=dict(color=RED, size=14),
                    annotation_position="top left", annotation_yshift=18)
        h.add_vline(x=p50, line_color=GOLD, line_width=3, line_dash="dash", annotation_text="Median",
                    annotation_position="top right", annotation_yshift=18)
        h.update_layout(height=370, margin=dict(t=55, b=10), yaxis_title="Simulations")
        st.plotly_chart(h, width=W, theme=None)
        takeaway(f"In <b>{prob_up:.0%}</b> of {len(sims):,} simulated futures the company is worth more than today's price. "
                 f"In a bad case (5th percentile) value is <b>${p5:,.2f}</b>, {abs(p5 / price - 1):.0%} {'below' if p5 < price else 'above'} today's price.")

with tabs[6]:
    st.markdown((Path(__file__).parent / "HOW_IT_WORKS.md").read_text())
