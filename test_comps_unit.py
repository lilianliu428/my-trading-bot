"""
test_comps_unit.py

Unit tests for Layer 2 comps pure-math functions. No network calls except
the mocked feature-computation test. Run directly: python3 test_comps_unit.py
"""

import math
from unittest.mock import patch

import numpy as np
import pandas as pd

from valuation.comps.multiples import winsorize
from valuation.comps.features import CompanyFeatures, compute_features_for_ticker
from valuation.comps.regression import MultipleObservation, fit_regression, predict_multiple
from valuation.comps.similarity import find_nearest_peers

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_winsorize():
    print("\n--- winsorize() ---")
    values = np.arange(1, 101, dtype=float)  # 1..100
    result = winsorize(values, lower=0.01, upper=0.99)

    lo_expected = np.percentile(values, 1)
    hi_expected = np.percentile(values, 99)

    check("no value below the 1st percentile", result.min() >= lo_expected - 1e-9)
    check("no value above the 99th percentile", result.max() <= hi_expected + 1e-9)
    check("extreme low value (1) got clipped upward", result[0] == lo_expected)
    check("extreme high value (100) got clipped downward", result[-1] == hi_expected)
    check("middle values unchanged", np.array_equal(result[40:60], values[40:60]))
    check("output length matches input", len(result) == len(values))


def test_predict_multiple_reverses_log_transform():
    print("\n--- predict_multiple() reverses log transform ---")
    # Construct a noiseless log-linear relationship:
    #   log(multiple) = 2.0 + 3.0 * trailing_growth  (other coefficients ~0)
    # so OLS should recover it near-exactly and predict_multiple should
    # return exp(predicted_log), matching the known ground truth.
    intercept, coef = 2.0, 3.0
    observations = []
    for i in range(12):
        g = -0.2 + 0.05 * i  # varies trailing_growth, everything else fixed
        feats = CompanyFeatures(
            ticker=f"SYN{i}", trailing_growth=g, operating_margin=0.20,
            roic=0.15, log_revenue=20.0, revenue_ttm=1e9, ebitda_ttm=2e8, data_flags=[],
        )
        multiple_value = math.exp(intercept + coef * g)
        observations.append(MultipleObservation(f"SYN{i}", "ev_sales", multiple_value, feats))

    reg = fit_regression(observations, "ev_sales")
    check("regression fit successfully", reg.r_squared is not None)
    if reg.r_squared is not None:
        check("R² near 1.0 on noiseless data", reg.r_squared > 0.99, f"got {reg.r_squared}")

    target = CompanyFeatures(
        ticker="TARGET", trailing_growth=0.10, operating_margin=0.20,
        roic=0.15, log_revenue=20.0, revenue_ttm=1e9, ebitda_ttm=2e8, data_flags=[],
    )
    expected_multiple = math.exp(intercept + coef * 0.10)
    predicted, se_log = predict_multiple(reg, target)

    check("predicted_multiple is not None", predicted is not None)
    if predicted is not None:
        check(
            "predicted_multiple matches exp(known log-linear ground truth)",
            abs(predicted - expected_multiple) / expected_multiple < 0.05,
            f"predicted={predicted}, expected={expected_multiple}",
        )
    check("prediction_std_error is a positive float", se_log is not None and se_log >= 0)

    incomplete = CompanyFeatures("INCOMPLETE", None, 0.2, 0.15, 20.0, 1e9, 2e8, [])
    pred_none, se_none = predict_multiple(reg, incomplete)
    check("incomplete features -> (None, None)", pred_none is None and se_none is None)


def test_find_nearest_peers_ordering():
    print("\n--- find_nearest_peers() ordering ---")
    # Only log_revenue varies; other 3 features constant across the universe
    # (also exercises the divide-by-zero-std guard).
    target = CompanyFeatures("TARGET", 0.10, 0.20, 0.15, 20.0, 1e9, 2e8, [])
    universe = [
        CompanyFeatures("CLOSE", 0.10, 0.20, 0.15, 20.5, 1e9, 2e8, []),    # nearest
        CompanyFeatures("MEDIUM", 0.10, 0.20, 0.15, 22.0, 1e9, 2e8, []),
        CompanyFeatures("FAR", 0.10, 0.20, 0.15, 30.0, 1e9, 2e8, []),      # farthest
        CompanyFeatures("TARGET", 0.10, 0.20, 0.15, 20.0, 1e9, 2e8, []),   # self, must be excluded
        CompanyFeatures("INCOMPLETE", None, 0.20, 0.15, 21.0, 1e9, 2e8, []),  # must be dropped
    ]

    peers = find_nearest_peers(target, universe, n=5)
    tickers = [p[0] for p in peers]

    check("target excluded from results", "TARGET" not in tickers)
    check("incomplete-feature company excluded", "INCOMPLETE" not in tickers)
    check("returns 3 candidates (5 requested, only 3 eligible)", len(peers) == 3, f"got {len(peers)}")
    check("sorted ascending by distance", tickers == ["CLOSE", "MEDIUM", "FAR"], f"got {tickers}")
    distances = [p[1] for p in peers]
    check("distances are non-negative and non-decreasing", all(d >= 0 for d in distances) and distances == sorted(distances))


def test_feature_computation_synthetic_ticker():
    print("\n--- compute_features_for_ticker() on synthetic mocked data ---")
    dates = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"])
    income_stmt = pd.DataFrame(
        {dates[0]: {"Total Revenue": 1_000_000_000.0, "Operating Income": 200_000_000.0},
         dates[1]: {"Total Revenue": 900_000_000.0, "Operating Income": 150_000_000.0},
         dates[2]: {"Total Revenue": 850_000_000.0, "Operating Income": 120_000_000.0},
         dates[3]: {"Total Revenue": 800_000_000.0, "Operating Income": 100_000_000.0}},
    )
    cash_flow = pd.DataFrame(
        {dates[0]: {"Depreciation And Amortization": 50_000_000.0},
         dates[1]: {"Depreciation And Amortization": 45_000_000.0}},
    )

    class FakeTicker:
        def __init__(self, ticker):
            self.info = {"financialCurrency": "USD"}
            self.income_stmt = income_stmt
            self.cashflow = cash_flow

    with patch("valuation.comps.features.yf.Ticker", FakeTicker), \
         patch("valuation.comps.features.compute_fundamental_growth", return_value={"roic": 0.15, "data_flags": []}):
        feats = compute_features_for_ticker("SYNTH")

    expected_revenue_ttm = 1_000_000_000.0
    expected_margin = 200_000_000.0 / 1_000_000_000.0
    expected_growth = (1_000_000_000.0 / 800_000_000.0) ** (1 / 3) - 1
    expected_log_revenue = math.log(1_000_000_000.0)
    expected_ebitda = 200_000_000.0 + 50_000_000.0

    check("revenue_ttm matches latest column", feats.revenue_ttm == expected_revenue_ttm)
    check("operating_margin matches hand-computed value", abs(feats.operating_margin - expected_margin) < 1e-9)
    check("trailing_growth matches 3-year CAGR formula", abs(feats.trailing_growth - expected_growth) < 1e-9)
    check("log_revenue matches math.log(revenue_ttm)", abs(feats.log_revenue - expected_log_revenue) < 1e-9)
    check("ebitda_ttm = operating_income + D&A", feats.ebitda_ttm == expected_ebitda)
    check("roic passed through from compute_fundamental_growth", feats.roic == 0.15)
    check("feature vector reports complete", feats.is_complete())


if __name__ == "__main__":
    test_winsorize()
    test_predict_multiple_reverses_log_transform()
    test_find_nearest_peers_ordering()
    test_feature_computation_synthetic_ticker()

    print(f"\n{'=' * 50}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) — {FAILURES}")
        raise SystemExit(1)
    print("ALL UNIT TESTS PASSED")
