"""
test_layer2_final_engine.py

Tests for the finalized, per-bucket-routed Layer 2 engine (LAYER_2_FINAL_SPEC
— semi_forest for semiconductors, mature_relative for mature_tech EV/EBITDA,
peer_display universally). Covers:
    1. Unit tests on synthetic data: semi_forest fit/predict, mature_relative
       reverses the relative-multiple transform correctly, peer_display
       returns 5 nearest peers in correct order.
    2. Integration test per bucket: AMD (semis) -> both regressions + peer
       display; MSFT (mature_tech) -> EV/EBITDA only + peer display;
       CMCSA (communication, forced) -> peer display only, no regression.
    3. End-to-end validation: compute_full_valuation on AMD, NVDA, AVGO,
       MSFT, GOOGL, AAPL — no exceptions, correct routing, confidence tiers.
    4. Confidence disclosure test.

Hits yfinance for real data in sections 2-3. Run directly:
python3 test_layer2_final_engine.py
"""

from unittest.mock import patch

import numpy as np

import valuation.composite as composite
from valuation.comps.features import CompanyFeatures
from valuation.comps.regression import MultipleObservation
from valuation.comps.models.semi_forest import fit_semi_forest, predict_semi_forest
from valuation.comps.models.mature_relative import fit_mature_relative, predict_mature_relative
from valuation.comps.models.peer_display import compute_peer_display
from valuation.comps.final_engine import compute_layer_2_comps

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def make_features(ticker, **overrides):
    base = dict(
        ticker=ticker, trailing_growth=0.1, operating_margin=0.2, roic=0.15, log_revenue=22.0,
        revenue_ttm=1e9, ebitda_ttm=3e8, data_flags=[],
        trailing_growth_5y=None, growth_stability=0.5, gross_margin=0.4, ebitda_margin=0.3,
        r_and_d_intensity=0.1, fcf_conversion=1.0,
    )
    base.update(overrides)
    return CompanyFeatures(**base)


# ════════════════════════════════════════════════════════════════════════
# 1. Unit tests (synthetic data, no network)
# ════════════════════════════════════════════════════════════════════════

def test_semi_forest_unit():
    print("\n--- semi_forest: fit/predict on synthetic data ---")
    np.random.seed(10)
    observations = []
    for i in range(30):
        g = -0.2 + 0.02 * i
        f = make_features(f"S{i}", trailing_growth=g)
        mult = float(np.exp(1.0 + 2.0 * g + np.random.randn() * 0.05))
        observations.append(MultipleObservation(f"S{i}", "ev_sales", mult, f))

    model, flags = fit_semi_forest(observations, "ev_sales")
    check("model fits successfully", model is not None)
    if model is None:
        return

    target = make_features("TARGET", trailing_growth=0.1)
    pred, missing, _flags = predict_semi_forest(model, target)
    check("complete target produces a prediction", pred is not None and not missing)

    target_high_g = make_features("HIGH", trailing_growth=0.24)
    target_low_g = make_features("LOW", trailing_growth=-0.18)
    pred_high, _, _ = predict_semi_forest(model, target_high_g)
    pred_low, _, _ = predict_semi_forest(model, target_low_g)
    check("model learned the direction of the known relationship (higher growth -> higher multiple)",
          pred_high is not None and pred_low is not None and pred_high > pred_low,
          f"pred_high={pred_high}, pred_low={pred_low}")

    target_2missing = make_features("PARTIAL", growth_stability=None, ebitda_margin=None)
    pred2, missing2, _ = predict_semi_forest(model, target_2missing)
    check("2-of-8 missing features -> still predicts via imputation", pred2 is not None and len(missing2) == 2)

    target_3missing = make_features("INSUFFICIENT", growth_stability=None, ebitda_margin=None, roic=None)
    pred3, missing3, _ = predict_semi_forest(model, target_3missing)
    check("3-of-8 missing features -> refuses (insufficient data)", pred3 is None and len(missing3) == 3)


def test_mature_relative_unit():
    print("\n--- mature_relative: reverses the relative-multiple transform correctly ---")
    # Known ground truth: relative = 0.2*fcf + 0.1*gross_margin (noiseless),
    # fcf/gross_margin centered on 0 so relative (and hence the resulting
    # multiples) are symmetric around bucket_median=20.0 — otherwise the
    # TRUE median of 20*(1+relative) isn't 20 just because we picked 20 as
    # a base, since relative wouldn't itself be centered on 0.
    observations = []
    for i in range(30):
        fcf = -0.3 + 0.02 * i    # -0.3 .. 0.28, centered near 0
        gm = -0.15 + 0.01 * i    # -0.15 .. 0.14, centered near 0
        f = make_features(f"M{i}", fcf_conversion=fcf, gross_margin=gm)
        relative = 0.2 * fcf + 0.1 * gm
        mult = 20.0 * (1 + relative)
        observations.append(MultipleObservation(f"M{i}", "ev_ebitda", mult, f))

    model, flags = fit_mature_relative(observations)
    check("model fits successfully", model is not None)
    if model is None:
        return
    check("recovers the known bucket_median", abs(model.bucket_median - 20.0) < 0.5, f"got {model.bucket_median}")

    target = make_features("TARGET", fcf_conversion=0.5, gross_margin=0.3)
    expected_relative = 0.2 * 0.5 + 0.1 * 0.3
    expected_multiple = 20.0 * (1 + expected_relative)
    pred, missing, _flags = predict_mature_relative(model, target)
    check("predicted multiple close to known ground truth (transform correctly reversed)",
          pred is not None and abs(pred - expected_multiple) / expected_multiple < 0.15,
          f"got {pred}, expected ~{expected_multiple}")

    target_missing = make_features("MISSING", fcf_conversion=None)
    pred_missing, missing_list, _ = predict_mature_relative(model, target_missing)
    check("missing required feature -> refuses", pred_missing is None and missing_list == ["fcf_conversion"])


def test_peer_display_unit():
    print("\n--- peer_display: 5 nearest peers in correct order ---")
    target = make_features("TARGET", trailing_growth=0.10, log_revenue=20.0)
    universe = [
        make_features("CLOSEST", trailing_growth=0.10, log_revenue=20.2),
        make_features("CLOSE", trailing_growth=0.10, log_revenue=20.5),
        make_features("MEDIUM", trailing_growth=0.10, log_revenue=22.0),
        make_features("FAR", trailing_growth=0.10, log_revenue=26.0),
        make_features("FARTHEST", trailing_growth=0.10, log_revenue=30.0),
        make_features("EXTRA", trailing_growth=0.10, log_revenue=35.0),  # 6th candidate, shouldn't appear in top 5
    ]
    target_multiples = {"ev_sales": 10.0, "ev_ebitda": 20.0}
    universe_multiples = {f.ticker: {"ev_sales": 5.0 + i, "ev_ebitda": 15.0 + i} for i, f in enumerate(universe)}

    result = compute_peer_display(target, universe, target_multiples, universe_multiples)
    check("returns exactly 5 peers", len(result.nearest_peers) == 5, f"got {result.nearest_peers}")
    check("peers ordered nearest-first",
          result.nearest_peers == ["CLOSEST", "CLOSE", "MEDIUM", "FAR", "FARTHEST"],
          f"got {result.nearest_peers}")
    check("6th candidate (EXTRA) excluded", "EXTRA" not in result.nearest_peers)
    check("distances non-decreasing", result.peer_distances == sorted(result.peer_distances))


# ════════════════════════════════════════════════════════════════════════
# 2. Integration tests per bucket (real yfinance data)
# ════════════════════════════════════════════════════════════════════════

def test_integration_semiconductors():
    print("\n--- Integration: AMD (semiconductors) -> both regressions + peer display ---")
    r = compute_layer_2_comps("AMD", bucket="semiconductors")
    check("AMD: ev_sales_regression present", r.ev_sales_regression is not None)
    check("AMD: ev_ebitda_regression present", r.ev_ebitda_regression is not None)
    if r.ev_sales_regression:
        check("AMD: ev_sales confidence is HIGH", r.ev_sales_regression.confidence == "HIGH")
        check("AMD: ev_sales model_type is RandomForest", "RandomForest" in r.ev_sales_regression.model_type)
    if r.ev_ebitda_regression:
        check("AMD: ev_ebitda confidence is HIGH", r.ev_ebitda_regression.confidence == "HIGH")
    check("AMD: peer_display present", r.peer_display is not None)
    check("AMD: peer_display has nearest_peers", r.peer_display is not None and len(r.peer_display.nearest_peers) > 0)


def test_integration_mature_tech():
    print("\n--- Integration: MSFT (mature_tech) -> EV/EBITDA only + peer display ---")
    r = compute_layer_2_comps("MSFT", bucket="mature_tech")
    check("MSFT: ev_sales_regression is None (not shipped)", r.ev_sales_regression is None)
    check("MSFT: ev_ebitda_regression present", r.ev_ebitda_regression is not None)
    if r.ev_ebitda_regression:
        check("MSFT: ev_ebitda confidence is MEDIUM", r.ev_ebitda_regression.confidence == "MEDIUM")
        check("MSFT: ev_ebitda model_type is OLS relative", r.ev_ebitda_regression.model_type == "OLS relative 2-feature")
        check("MSFT: ev_ebitda formulation is relative", r.ev_ebitda_regression.formulation == "relative")
    check("MSFT: peer_display present", r.peer_display is not None)


def test_integration_communication():
    print("\n--- Integration: CMCSA (communication, forced) -> peer display only ---")
    r = compute_layer_2_comps("CMCSA", bucket="communication")
    check("CMCSA: ev_sales_regression is None (bucket not shipped)", r.ev_sales_regression is None)
    check("CMCSA: ev_ebitda_regression is None (bucket not shipped)", r.ev_ebitda_regression is None)
    check("CMCSA: peer_display present even with no regression", r.peer_display is not None)


# ════════════════════════════════════════════════════════════════════════
# 3. End-to-end validation + 4. confidence disclosure test
# ════════════════════════════════════════════════════════════════════════

STUB_DCF_FAIR_VALUE = {"AMD": 400.0, "NVDA": 150.0, "AVGO": 300.0, "MSFT": 420.0, "GOOGL": 190.0, "AAPL": 200.0}


def make_stub_dcf(ticker):
    def _stub(t, b):
        return {
            "ticker": t, "bucket": b, "per_share_value": STUB_DCF_FAIR_VALUE[t],
            "current_price": None, "upside_downside": None,
            "data_flags": [f"Stubbed Layer 1 for {t}"],
        }
    return _stub


def test_end_to_end_six_tickers():
    print("\n--- End-to-end: compute_full_valuation on 6 tickers ---")
    tickers = [
        ("AMD", "semiconductors"), ("NVDA", "semiconductors"), ("AVGO", "semiconductors"),
        ("MSFT", "mature_tech"), ("GOOGL", "mature_tech"), ("AAPL", "mature_tech"),
    ]
    for ticker, bucket in tickers:
        try:
            with patch.object(composite, "_run_layer_1_dcf", side_effect=make_stub_dcf(ticker)):
                result = composite.compute_full_valuation(ticker, bucket=bucket)
        except Exception as e:
            check(f"{ticker}: compute_full_valuation raises no exception", False, f"{type(e).__name__}: {e}")
            continue
        check(f"{ticker}: compute_full_valuation raises no exception", True)

        l2 = result["layer_2_comps"]
        check(f"{ticker}: layer_2_comps schema has all 3 keys",
              set(l2.keys()) == {"ev_sales_regression", "ev_ebitda_regression", "peer_display"})
        check(f"{ticker}: peer_display always present", l2["peer_display"] is not None)

        if bucket == "semiconductors":
            for key in ("ev_sales_regression", "ev_ebitda_regression"):
                if l2[key] is not None:
                    check(f"{ticker}: {key} confidence == HIGH", l2[key]["confidence"] == "HIGH")
        elif bucket == "mature_tech":
            check(f"{ticker}: ev_sales_regression is None (user cannot see it)", l2["ev_sales_regression"] is None)
            if l2["ev_ebitda_regression"] is not None:
                check(f"{ticker}: ev_ebitda_regression confidence == MEDIUM", l2["ev_ebitda_regression"]["confidence"] == "MEDIUM")

        print(f"  {ticker}: ev_sales={'shipped' if l2['ev_sales_regression'] else 'not shipped'}, "
              f"ev_ebitda={'shipped' if l2['ev_ebitda_regression'] else 'not shipped'}, "
              f"peers={l2['peer_display']['nearest_peers'][:3]}...")


if __name__ == "__main__":
    test_semi_forest_unit()
    test_mature_relative_unit()
    test_peer_display_unit()
    test_integration_semiconductors()
    test_integration_mature_tech()
    test_integration_communication()
    test_end_to_end_six_tickers()

    print(f"\n{'=' * 50}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) — {FAILURES}")
        raise SystemExit(1)
    print("ALL LAYER 2 FINAL ENGINE TESTS PASSED")
