"""
test_comps_backtest_sanity.py

Backtest sanity check per LAYER_2_COMPS_SPEC.md "Testing Requirements" #3:
run on a 5-year window with 10 tickers, verify no NaN, verify correlations
in [-1, 1] and hit rates in [0, 1]. Hits yfinance for real data.

Note: yfinance's annual statements only go back ~5 years, so the requested
2020-2025 window gets clipped to whatever's actually available — see
BacktestResult.data_flags for the clip. Run directly:
python3 test_comps_backtest_sanity.py
"""

import math

from valuation.comps.backtest.framework import run_backtest
from valuation.comps.universe import get_universe

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def is_nan(x):
    return isinstance(x, float) and math.isnan(x)


if __name__ == "__main__":
    tickers = get_universe("semiconductors")[:10]
    print(f"Running 5-year backtest sanity check on: {tickers}")

    result = run_backtest(tickers, "2020-01-01", "2025-12-31", forward_windows=[6, 12])

    print(f"\nperiod: {result.period_start} -> {result.period_end}")
    print(f"n_rebalancing_dates: {result.n_rebalancing_dates}  n_predictions: {result.n_predictions}")

    check("backtest ran without exception and returned a BacktestResult", result is not None)
    check("n_predictions > 0 (non-degenerate)", result.n_predictions > 0, f"got {result.n_predictions}")

    numeric_fields = {
        "correlation_6m": result.correlation_6m, "correlation_12m": result.correlation_12m,
        "decile_spread_6m": result.decile_spread_6m, "decile_spread_12m": result.decile_spread_12m,
        "sharpe_6m": result.sharpe_6m, "sharpe_12m": result.sharpe_12m,
        "hit_rate_6m": result.hit_rate_6m, "hit_rate_12m": result.hit_rate_12m,
        "r_squared_6m": result.r_squared_6m, "r_squared_12m": result.r_squared_12m,
    }
    for name, value in numeric_fields.items():
        check(f"{name} is not NaN", not is_nan(value), f"got {value}")
        print(f"  {name} = {value}")

    if result.correlation_6m is not None:
        check("correlation_6m in [-1, 1]", -1 <= result.correlation_6m <= 1)
    if result.correlation_12m is not None:
        check("correlation_12m in [-1, 1]", -1 <= result.correlation_12m <= 1)
    if result.hit_rate_6m is not None:
        check("hit_rate_6m in [0, 1]", 0 <= result.hit_rate_6m <= 1)
    if result.hit_rate_12m is not None:
        check("hit_rate_12m in [0, 1]", 0 <= result.hit_rate_12m <= 1)

    print(f"\nflags ({len(result.data_flags)} total, last 5):")
    for f in result.data_flags[-5:]:
        print(f"  - {f}")

    print(f"\n{'=' * 50}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) — {FAILURES}")
        raise SystemExit(1)
    print("ALL BACKTEST SANITY CHECKS PASSED")
