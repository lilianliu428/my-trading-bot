"""
test_full_valuation_comparison.py

LAYER_2_COMPS_SPEC.md "Testing Requirements" #4: compute both Layer 1 DCF
and Layer 2 Comps for 5 tickers, verify divergence numbers are computed
correctly, no exceptions during full compute_full_valuation execution.

Layer 1 DCF depends on valuation/inputs/beta.py, which reads a
`price_history` table that isn't populated in this local environment (the
table isn't even created by data_pipeline/database.py's migrations here —
it's presumably populated on the EC2 deployment via a scraper not present
in this checkout). That's a pre-existing local-environment gap unrelated
to Layer 2. This test stubs Layer 1's DCF output so it can still verify
what Layer 2 owns: the composition logic, signal_summary math, and that
compute_full_valuation raises no exceptions end-to-end. Layer 2's own
pieces (compute_layer_2_comps — the finalized, per-bucket-routed engine)
run for real against yfinance.

Updated for LAYER_2_FINAL_SPEC's routing: mature_tech tickers now only
ship an ev_ebitda_regression (ev_sales_regression is always None for that
bucket, not just when data happens to be missing), so this test no longer
assumes both regressions are present for every ticker.

Run directly: python3 test_full_valuation_comparison.py
"""

from unittest.mock import patch

import valuation.composite as composite

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


# Plausible stand-in fair values, distinct per ticker so divergence isn't
# trivially zero everywhere.
STUB_DCF_FAIR_VALUE = {
    "AMD": 400.0, "NVDA": 150.0, "MSFT": 420.0, "AAPL": 200.0, "GOOGL": 190.0,
}


def make_stub_dcf(ticker, bucket):
    def _stub(t, b):
        fair_value = STUB_DCF_FAIR_VALUE[t]
        # market_price comes from a real yfinance call inside compute_comps_valuation;
        # here we just need *a* plausible current_price for upside math, so reuse it later.
        return {
            "ticker": t, "bucket": b, "per_share_value": fair_value,
            "current_price": None,  # composite falls back to comps' market_price
            "upside_downside": None,
            "data_flags": [f"Stubbed Layer 1 for {t} — local price_history table unavailable"],
        }
    return _stub


if __name__ == "__main__":
    tickers = [("AMD", "semiconductors"), ("NVDA", "semiconductors"), ("MSFT", "mature_tech"),
               ("AAPL", "mature_tech"), ("GOOGL", "mature_tech")]

    for ticker, bucket in tickers:
        print(f"\n--- {ticker} ({bucket}) ---")
        try:
            with patch.object(composite, "_run_layer_1_dcf", side_effect=make_stub_dcf(ticker, bucket)):
                result = composite.compute_full_valuation(ticker, bucket=bucket)
        except Exception as e:
            check(f"{ticker}: compute_full_valuation raises no exception", False, f"{type(e).__name__}: {e}")
            continue
        check(f"{ticker}: compute_full_valuation raises no exception", True)

        check(f"{ticker}: top-level shape matches spec",
              set(result.keys()) == {"ticker", "bucket", "market_price", "layer_1_dcf", "layer_2_comps", "signal_summary", "data_flags"})

        dcf_value = result["layer_1_dcf"]["fair_value"]
        signal = result["signal_summary"]
        check(f"{ticker}: signal_summary.dcf_value matches layer_1_dcf.fair_value", signal["dcf_value"] == dcf_value)

        comps = result["layer_2_comps"]
        evs = comps["ev_sales_regression"]
        eve = comps["ev_ebitda_regression"]
        comps_values = [v for v in ((evs or {}).get("fair_value"), (eve or {}).get("fair_value")) if v is not None]

        check(f"{ticker}: peer_display always present regardless of regression availability", comps["peer_display"] is not None)
        if bucket == "mature_tech":
            check(f"{ticker}: mature_tech ev_sales_regression is None (not shipped)", evs is None)
            if eve is not None:
                check(f"{ticker}: mature_tech ev_ebitda_regression confidence is MEDIUM", eve["confidence"] == "MEDIUM")
        elif bucket == "semiconductors":
            if evs is not None:
                check(f"{ticker}: semiconductors ev_sales_regression confidence is HIGH", evs["confidence"] == "HIGH")
            if eve is not None:
                check(f"{ticker}: semiconductors ev_ebitda_regression confidence is HIGH", eve["confidence"] == "HIGH")

        if comps_values and dcf_value is not None:
            expected_median = sorted(comps_values)[len(comps_values) // 2] if len(comps_values) % 2 else \
                sum(sorted(comps_values)[len(comps_values) // 2 - 1:len(comps_values) // 2 + 1]) / 2
            expected_divergence = expected_median - dcf_value
            check(f"{ticker}: divergence = comps_median - dcf_value",
                  abs(signal["divergence"] - expected_divergence) < 1e-6,
                  f"got {signal['divergence']}, expected {expected_divergence}")
            check(f"{ticker}: comps_range = (min, max) of available comps fair values",
                  signal["comps_range"] == (min(comps_values), max(comps_values)))
        else:
            check(f"{ticker}: comps fair values available for divergence check", False, "no comps values computed")

        print(f"  DCF fair value (stubbed): ${dcf_value:.2f}" if dcf_value else "  DCF fair value: N/A")
        print(f"  Comps range: {signal['comps_range']}")
        print(f"  Divergence: {signal['divergence']}")
        print(f"  Interpretation: {signal['interpretation']}")

    print(f"\n{'=' * 50}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) — {FAILURES}")
        raise SystemExit(1)
    print("ALL DCF-COMPARISON TESTS PASSED")
