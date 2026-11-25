import numpy as np
import pytest

from valuation.model import (SAMPLE_PEERS, Assumptions, comps_value, dcf, forecast, lbo, monte_carlo,
                             recommendation, sensitivity, wacc)

A = Assumptions()


def test_forecast_length_and_growth():
    f = forecast(A)
    assert len(f) == 5
    assert np.isclose(f["Revenue"].iloc[0], A.revenue0 * 1.12)


def test_wacc_between_debt_and_equity_costs():
    w = wacc(A)
    assert w["after_tax_debt"] < w["wacc"] < w["cost_equity"]
    assert np.isclose(w["we"] + w["wd"], 1)


def test_dcf_bridge():
    d = dcf(A)
    assert np.isclose(d["ev"], d["pv_fcf"] + d["pv_tv"])
    assert np.isclose(d["equity"], d["ev"] - A.debt + A.cash)


def test_higher_wacc_lowers_value():
    assert dcf(A, discount=0.12)["per_share"] < dcf(A, discount=0.08)["per_share"]


def test_gordon_rejects_bad_inputs():
    with pytest.raises(ValueError):
        dcf(A, discount=0.02, tg=0.03)


def test_sensitivity_shape_and_order():
    s = sensitivity(A, [0.08, 0.09, 0.10], [0.02, 0.03])
    assert s.shape == (3, 2)
    assert s.iloc[0, 0] > s.iloc[2, 0]


def test_lbo_math():
    r = lbo(A, 10, 4, 0.08, 10)
    assert np.isclose(r["moic"], r["equity_out"] / r["equity_in"])
    assert r["schedule"]["Ending debt"].is_monotonic_decreasing


def test_comps_ordering():
    c = comps_value(A, SAMPLE_PEERS)["EV/EBITDA"]
    assert c["low"] <= c["mid"] <= c["high"]


def test_monte_carlo_centers_near_base():
    sims = monte_carlo(A, n=800)
    assert abs(np.median(sims) / dcf(A)["per_share"] - 1) < 0.25


def test_recommendation_labels():
    assert recommendation(60, 40, 0.8)[0] == "BUY"
    assert recommendation(30, 40, 0.2)[0] == "SELL"
    assert recommendation(42, 40, 0.5)[0] == "HOLD"


from valuation.model import blended_dcf, blended_dcf_vec, max_entry_multiple, solve_shift, tornado  # noqa: E402


def test_vectorized_dcf_matches_exact():
    from dataclasses import replace as r
    a = Assumptions()
    w = wacc(a)["wacc"]
    v = blended_dcf_vec(a, [0.02], [-0.01], [w + 0.01], [0.02])[0]
    b = r(a, growth=tuple(x + 0.02 for x in a.growth), ebitda_margin=tuple(x - 0.01 for x in a.ebitda_margin), terminal_growth=0.02)
    assert abs(v - blended_dcf(b, discount=w + 0.01)) < 1e-9


def test_reverse_dcf_hits_price():
    a = Assumptions()
    from dataclasses import replace as r
    dg = solve_shift(a, a.price, "growth")
    assert abs(blended_dcf(r(a, growth=tuple(x + dg for x in a.growth))) - a.price) < 1e-6
    assert solve_shift(a, 1e6, "growth") is None


def test_gordon_tv_uses_mid_year_discount():
    a = Assumptions()
    d = dcf(a)
    assert abs(d["pv_tv"] - d["tv"] / (1 + d["wacc"]) ** 4.5) < 1e-9


def test_lbo_keeps_surplus_cash_and_funds_shortfalls():
    a = Assumptions()
    no_debt = lbo(a, 10, 0, 0.08, 10)
    assert no_debt["schedule"]["Cash built"].iloc[-1] > 0
    from dataclasses import replace as r
    burn = lbo(r(a, capex_pct=0.25), 10, 4, 0.08, 10)
    assert burn["schedule"]["Ending debt"].iloc[-1] > burn["schedule"]["Ending debt"].iloc[0] - 1e-9


def test_max_entry_multiple_nan_when_impossible():
    import math
    assert math.isnan(max_entry_multiple(Assumptions(), 6.5, 0.14, 3.0, 0.20))


def test_tornado_has_all_drivers():
    assert len(tornado(Assumptions())) == 6
