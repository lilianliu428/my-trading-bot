"""
test_comps_integration.py

Integration test on real tickers (AMD, MSFT) per LAYER_2_COMPS_SPEC.md
"Testing Requirements" #2. Hits yfinance for real data. Run directly:
python3 test_comps_integration.py
"""

from valuation.comps.orchestrator import compute_comps_valuation

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def run_for(ticker, bucket):
    print(f"\n--- {ticker} ({bucket}) ---")
    result = compute_comps_valuation(ticker, bucket=bucket)

    check(f"{ticker}: returns a CompsValuationResult", result is not None)
    check(f"{ticker}: market_price populated", result.market_price is not None and result.market_price > 0)
    check(f"{ticker}: ev_sales R² is reasonable (> 0.1)", result.ev_sales_r2 is not None and result.ev_sales_r2 > 0.1,
          f"R²={result.ev_sales_r2}")

    if result.fair_value_from_ev_sales is not None and result.market_price:
        ratio = result.fair_value_from_ev_sales / result.market_price
        check(f"{ticker}: EV/Sales fair value within 3x of market price", 1 / 3 <= ratio <= 3,
              f"fair_value={result.fair_value_from_ev_sales:.2f}, market_price={result.market_price:.2f}, ratio={ratio:.2f}")
    else:
        check(f"{ticker}: EV/Sales fair value computed", False, "fair_value_from_ev_sales is None")

    check(f"{ticker}: nearest_peers has entries", len(result.nearest_peers) > 0)
    check(f"{ticker}: nearest_peers and peer_distances same length", len(result.nearest_peers) == len(result.peer_distances))
    if result.peer_distances:
        check(f"{ticker}: peer distances are finite and non-negative",
              all(d >= 0 for d in result.peer_distances))
        check(f"{ticker}: peer distances sorted ascending",
              result.peer_distances == sorted(result.peer_distances))

    print(f"  market_price=${result.market_price:.2f}")
    print(f"  predicted_ev_sales={result.predicted_ev_sales:.2f}x  fair_value_from_ev_sales=${result.fair_value_from_ev_sales:.2f}" if result.predicted_ev_sales else "  ev_sales prediction unavailable")
    print(f"  peers: {list(zip(result.nearest_peers, [round(d, 2) for d in result.peer_distances]))}")


if __name__ == "__main__":
    run_for("AMD", "semiconductors")
    run_for("MSFT", "mature_tech")

    print(f"\n{'=' * 50}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) — {FAILURES}")
        raise SystemExit(1)
    print("ALL INTEGRATION TESTS PASSED")
