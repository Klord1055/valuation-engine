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
