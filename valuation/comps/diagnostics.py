"""
Diagnostic: does segmenting the semiconductors universe into sub-segments
(AI infra, memory, analog, equipment, foundry) improve regression fit vs.
the single flat semiconductors regression?

Read-only diagnostic — does not modify universe.py or any other existing
module, and does not feed back into compute_comps_valuation. Run directly:
python3 -m valuation.comps.diagnostics
"""

import numpy as np

from valuation.comps.universe import COMPS_UNIVERSE
from valuation.comps.features import compute_features_for_ticker
from valuation.comps.multiples import compute_multiples_for_ticker
from valuation.comps.regression import MultipleObservation, fit_regression, predict_multiple, FEATURE_ORDER, MIN_OBSERVATIONS

MIN_SEGMENT_TICKERS = 5

# Sub-segments of the semiconductors bucket. Only tickers already present
# in COMPS_UNIVERSE["semiconductors"] are kept — anything else is skipped
# (this diagnostic doesn't expand the universe, just partitions it).
SEMI_SEGMENTS_RAW = {
    "ai_infra": ["NVDA", "AMD", "AVGO", "MRVL"],
    "memory": ["MU", "WDC", "STX"],
    "analog": ["TXN", "ADI", "NXPI", "MCHP", "ON", "SWKS", "MPWR", "CRUS", "SLAB", "LSCC"],
    "equipment": ["AMAT", "LRCX", "KLAC"],
    "foundry": ["TSM", "ASML"],
}


def build_segments():
    """
    Filter SEMI_SEGMENTS_RAW down to tickers actually in the semiconductors
    universe. Returns (segments, skipped) where skipped maps segment name
    to the list of tickers dropped because they weren't in the universe.
    """
    semi_universe = set(COMPS_UNIVERSE.get("semiconductors", []))
    segments = {}
    skipped = {}
    for name, tickers in SEMI_SEGMENTS_RAW.items():
        kept = [t for t in tickers if t in semi_universe]
        dropped = [t for t in tickers if t not in semi_universe]
        segments[name] = kept
        if dropped:
            skipped[name] = dropped
    return segments, skipped


def fetch_observations(tickers):
    """
    Fetch features + multiples for a ticker list, returning
    (ev_sales_observations, ev_ebitda_observations).
    """
    ev_sales_obs = []
    ev_ebitda_obs = []
    for t in tickers:
        feats = compute_features_for_ticker(t)
        mults = compute_multiples_for_ticker(t)
        if mults["ev_sales"] is not None:
            ev_sales_obs.append(MultipleObservation(t, "ev_sales", mults["ev_sales"], feats))
        if mults["ev_ebitda"] is not None:
            ev_ebitda_obs.append(MultipleObservation(t, "ev_ebitda", mults["ev_ebitda"], feats))
    return ev_sales_obs, ev_ebitda_obs


def run_segment_regression(name, tickers):
    """
    Fetch data and fit both regressions for one segment (or the baseline).
    Returns a result dict for the table, never raises.
    """
    if len(tickers) < MIN_SEGMENT_TICKERS:
        return {
            "name": name, "n_tickers": len(tickers),
            "ev_sales_n": None, "ev_sales_r2": None, "ev_sales_coef": None,
            "ev_ebitda_n": None, "ev_ebitda_r2": None, "ev_ebitda_coef": None,
            "note": f"skipped — only {len(tickers)} tickers (< {MIN_SEGMENT_TICKERS} minimum)",
        }

    print(f"\nFetching data for segment '{name}' ({len(tickers)} tickers: {tickers})...")
    ev_sales_obs, ev_ebitda_obs = fetch_observations(tickers)

    ev_sales_reg = fit_regression(ev_sales_obs, "ev_sales")
    ev_ebitda_reg = fit_regression(ev_ebitda_obs, "ev_ebitda")

    note = ""
    if ev_sales_reg.r_squared is None:
        note += f"ev_sales: insufficient usable observations ({ev_sales_reg.n_observations} < {MIN_OBSERVATIONS}); "
    if ev_ebitda_reg.r_squared is None:
        note += f"ev_ebitda: insufficient usable observations ({ev_ebitda_reg.n_observations} < {MIN_OBSERVATIONS})"

    return {
        "name": name, "n_tickers": len(tickers),
        "ev_sales_n": ev_sales_reg.n_observations, "ev_sales_r2": ev_sales_reg.r_squared,
        "ev_sales_coef": ev_sales_reg.coefficients,
        "ev_ebitda_n": ev_ebitda_reg.n_observations, "ev_ebitda_r2": ev_ebitda_reg.r_squared,
        "ev_ebitda_coef": ev_ebitda_reg.coefficients,
        "note": note.strip(),
    }


def _fmt_r2(r2):
    return f"{r2:.3f}" if r2 is not None else "N/A"


def _fmt_coef(coef):
    if not coef:
        return "-"
    return ", ".join(f"{k}={v:+.2f}" for k, v in coef.items())


def print_table(baseline, segment_results):
    rows = [baseline] + segment_results

    print("\n" + "=" * 118)
    print("SEMICONDUCTOR SEGMENTATION DIAGNOSTIC")
    print("=" * 118)
    header = f"{'Segment':<16} {'#Tickers':>8} {'EV/Sales n':>11} {'EV/Sales R²':>12} {'EV/EBITDA n':>12} {'EV/EBITDA R²':>13}  Note"
    print(header)
    print("-" * 118)
    for r in rows:
        print(
            f"{r['name']:<16} {r['n_tickers']:>8} "
            f"{(r['ev_sales_n'] if r['ev_sales_n'] is not None else '-'):>11} {_fmt_r2(r['ev_sales_r2']):>12} "
            f"{(r['ev_ebitda_n'] if r['ev_ebitda_n'] is not None else '-'):>12} {_fmt_r2(r['ev_ebitda_r2']):>13}  {r['note']}"
        )
    print("-" * 118)

    print("\nCoefficients (EV/Sales):")
    for r in rows:
        if r["ev_sales_coef"]:
            print(f"  {r['name']:<16} {_fmt_coef(r['ev_sales_coef'])}")
    print("\nCoefficients (EV/EBITDA):")
    for r in rows:
        if r["ev_ebitda_coef"]:
            print(f"  {r['name']:<16} {_fmt_coef(r['ev_ebitda_coef'])}")

    print("\nBaseline vs. segments (EV/Sales R² delta from full-universe baseline):")
    baseline_r2 = baseline["ev_sales_r2"]
    for r in segment_results:
        if r["ev_sales_r2"] is not None and baseline_r2 is not None:
            delta = r["ev_sales_r2"] - baseline_r2
            print(f"  {r['name']:<16} {delta:+.3f}  ({'improves' if delta > 0 else 'worse than'} baseline)")
        else:
            print(f"  {r['name']:<16} n/a  ({r['note']})")


def run_segment_diagnostics():
    segments, skipped = build_segments()
    if skipped:
        print("Tickers dropped (not in semiconductors universe):")
        for seg, dropped in skipped.items():
            print(f"  {seg}: {dropped}")

    baseline_tickers = COMPS_UNIVERSE.get("semiconductors", [])
    print(f"\nFetching data for baseline 'all_semiconductors' ({len(baseline_tickers)} tickers)...")
    baseline = run_segment_regression("all_semiconductors (baseline)", baseline_tickers)

    segment_results = [run_segment_regression(name, tickers) for name, tickers in segments.items()]

    print_table(baseline, segment_results)
    return baseline, segment_results


# ════════════════════════════════════════════════════════════════════════
# Cross-validation of the analog segment (follow-up diagnostic)
# ════════════════════════════════════════════════════════════════════════

def print_coefficient_report(ev_sales_reg, ev_ebitda_reg):
    """
    Print fitted coefficients + standard errors + p-values for the analog
    segment's full-sample (in-sample) regressions. Standard errors come
    straight from the statsmodels OLSResults stored internally in
    CompsRegressionResult (._model_result) — not recomputed here.
    """
    print("\n" + "=" * 90)
    print("ANALOG SEGMENT — FITTED COEFFICIENTS AND STANDARD ERRORS (full-sample, in-sample)")
    print("=" * 90)
    for label, reg in (("EV/Sales", ev_sales_reg), ("EV/EBITDA", ev_ebitda_reg)):
        print(f"\n{label}  (n={reg.n_observations}, R²={_fmt_r2(reg.r_squared)})")
        if reg._model_result is None:
            print("  regression not fit — insufficient data")
            continue
        bse = reg._model_result.bse
        pvalues = reg._model_result.pvalues
        print(f"  {'term':<20}{'coef':>10}{'std err':>10}{'p-value':>10}")
        print(f"  {'intercept':<20}{reg.intercept:>10.3f}{bse[0]:>10.3f}{pvalues[0]:>10.3f}")
        for i, name in enumerate(FEATURE_ORDER):
            print(f"  {name:<20}{reg.coefficients[name]:>10.3f}{bse[i + 1]:>10.3f}{pvalues[i + 1]:>10.3f}")


def loo_cross_validate(obs_list, multiple_type, feature_names=None, fit_fn=None, predict_fn=None):
    """
    Leave-one-out cross-validation over an already-fetched observation list:
    for each ticker, refit on the other N-1, predict the held-out ticker's
    multiple, compare to its actual observed multiple.

    feature_names: passed through to fit_fn() — defaults to the original 4
    (FEATURE_ORDER) if not given.

    fit_fn/predict_fn: default to regression.py's fit_regression/
    predict_multiple (the log-linear absolute-multiple model). Pass
    alternatives with a matching signature — fit_fn(observations,
    multiple_type, feature_names=...) -> a result object; predict_fn(
    result, features) -> (predicted, se) — to reuse this same LOO loop for
    a different model form (e.g. relative_regression.py's linear model on
    sector-relative multiples).

    Returns (results, cv_r2). results is a list of
    {ticker, actual, predicted} dicts (predicted is None if the refit had
    too few observations or the held-out ticker's features were
    incomplete). cv_r2 = corr(predicted, actual)**2 over tickers with a
    valid prediction; None if fewer than 2 such pairs.
    """
    fit_fn = fit_fn or fit_regression
    predict_fn = predict_fn or predict_multiple
    results = []
    for obs in obs_list:
        held_out_ticker = obs.ticker
        train_obs = [o for o in obs_list if o.ticker != held_out_ticker]
        reg = fit_fn(train_obs, multiple_type, feature_names=feature_names)
        predicted, _se = predict_fn(reg, obs.features)
        results.append({"ticker": held_out_ticker, "actual": obs.multiple_value, "predicted": predicted})

    valid = [(r["predicted"], r["actual"]) for r in results if r["predicted"] is not None]
    cv_r2 = None
    if len(valid) >= 2:
        preds = np.array([v[0] for v in valid])
        actuals = np.array([v[1] for v in valid])
        if np.std(preds) > 0 and np.std(actuals) > 0:
            corr = np.corrcoef(preds, actuals)[0, 1]
            cv_r2 = float(corr ** 2)
    return results, cv_r2


def _fmt_num(x, decimals=3):
    return f"{x:.{decimals}f}" if x is not None else "N/A"


def print_loo_table(multiple_type, results, cv_r2):
    print(f"\n{'-' * 60}")
    print(f"LEAVE-ONE-OUT CROSS-VALIDATION — analog / {multiple_type}")
    print(f"{'-' * 60}")
    print(f"  {'ticker':<10}{'actual':>12}{'predicted':>12}{'error':>12}")
    for r in results:
        error = (r["predicted"] - r["actual"]) if r["predicted"] is not None else None
        print(f"  {r['ticker']:<10}{_fmt_num(r['actual']):>12}{_fmt_num(r['predicted']):>12}{_fmt_num(error):>12}")

    print(f"\n  Cross-validated R² (corr(predicted, actual)²): {_fmt_num(cv_r2)}")
    if cv_r2 is None:
        print("  Not enough valid held-out predictions to compute CV R².")
    elif cv_r2 > 0.4:
        print(f"  -> ROBUST: CV R² ({cv_r2:.3f}) > 0.4 — the in-sample fit generalizes to held-out tickers.")
    elif cv_r2 < 0.3:
        print(f"  -> OVERFITTING: CV R² ({cv_r2:.3f}) < 0.3 — the in-sample R² does not hold up out-of-sample.")
    else:
        print(f"  -> AMBIGUOUS: CV R² ({cv_r2:.3f}) is between 0.3 and 0.4 — inconclusive by the stated thresholds.")


def run_analog_cross_validation_diagnostic():
    """
    Cross-validate the analog segment's regression fit (previous session's
    diagnostic found in-sample R²=0.621 for EV/Sales, 0.677 for EV/EBITDA —
    this checks whether that holds up out-of-sample or is overfitting on
    10 tickers with 4 features + intercept).
    """
    tickers = SEMI_SEGMENTS_RAW["analog"]
    print(f"\nFetching data for analog segment cross-validation ({len(tickers)} tickers: {tickers})...")
    ev_sales_obs, ev_ebitda_obs = fetch_observations(tickers)

    ev_sales_reg = fit_regression(ev_sales_obs, "ev_sales")
    ev_ebitda_reg = fit_regression(ev_ebitda_obs, "ev_ebitda")
    print_coefficient_report(ev_sales_reg, ev_ebitda_reg)

    ev_sales_loo, ev_sales_cv_r2 = loo_cross_validate(ev_sales_obs, "ev_sales")
    print_loo_table("ev_sales", ev_sales_loo, ev_sales_cv_r2)

    ev_ebitda_loo, ev_ebitda_cv_r2 = loo_cross_validate(ev_ebitda_obs, "ev_ebitda")
    print_loo_table("ev_ebitda", ev_ebitda_loo, ev_ebitda_cv_r2)

    print(f"\n{'=' * 60}")
    print("SUMMARY: in-sample vs. cross-validated R²")
    print(f"{'=' * 60}")
    print(f"  EV/Sales:   in-sample R²={_fmt_r2(ev_sales_reg.r_squared)}   CV R²={_fmt_num(ev_sales_cv_r2)}")
    print(f"  EV/EBITDA:  in-sample R²={_fmt_r2(ev_ebitda_reg.r_squared)}   CV R²={_fmt_num(ev_ebitda_cv_r2)}")

    return {
        "ev_sales": {"in_sample_r2": ev_sales_reg.r_squared, "cv_r2": ev_sales_cv_r2, "loo_results": ev_sales_loo},
        "ev_ebitda": {"in_sample_r2": ev_ebitda_reg.r_squared, "cv_r2": ev_ebitda_cv_r2, "loo_results": ev_ebitda_loo},
    }


# ════════════════════════════════════════════════════════════════════════
# Universe expansion diagnostic (follow-up to valuation/comps/universe_expansion.py)
# ════════════════════════════════════════════════════════════════════════

# Pre-expansion ticker lists, frozen here for a fair before/after comparison
# — universe.py's COMPS_UNIVERSE has since been updated in place with the
# expansion, so this is the only remaining record of the "before" state.
PRE_EXPANSION_UNIVERSE = {
    "semiconductors": [
        "NVDA", "AMD", "AVGO", "MRVL", "INTC", "QCOM", "MU", "WDC", "STX",
        "TXN", "ADI", "NXPI", "MCHP", "ON", "AMAT", "LRCX", "KLAC",
        "TSM", "ASML", "SWKS", "MPWR", "CRUS", "SLAB", "LSCC",
        "ARM", "COHR", "IPGP", "MTSI", "OLED",
    ],
    "mature_tech": [
        "MSFT", "GOOGL", "AAPL", "META", "AMZN", "ORCL", "CRM", "ADBE", "IBM", "CSCO",
        "NOW", "INTU", "WDAY", "TEAM", "DDOG", "SNOW", "PANW",
        "NFLX", "DIS", "TMUS", "VZ", "T",
    ],
}


def _fit_and_cv(bucket, ev_sales_obs, ev_ebitda_obs, label, n_tickers):
    """Fit both regressions and run LOO CV on already-fetched observation lists."""
    ev_sales_reg = fit_regression(ev_sales_obs, "ev_sales")
    ev_ebitda_reg = fit_regression(ev_ebitda_obs, "ev_ebitda")
    _ev_sales_loo, ev_sales_cv_r2 = loo_cross_validate(ev_sales_obs, "ev_sales")
    _ev_ebitda_loo, ev_ebitda_cv_r2 = loo_cross_validate(ev_ebitda_obs, "ev_ebitda")
    return {
        "label": label, "bucket": bucket, "n_tickers": n_tickers,
        "ev_sales_n": ev_sales_reg.n_observations, "ev_sales_in_sample_r2": ev_sales_reg.r_squared,
        "ev_sales_cv_r2": ev_sales_cv_r2,
        "ev_ebitda_n": ev_ebitda_reg.n_observations, "ev_ebitda_in_sample_r2": ev_ebitda_reg.r_squared,
        "ev_ebitda_cv_r2": ev_ebitda_cv_r2,
    }


def print_before_after_table(results):
    print("\n" + "=" * 112)
    print("UNIVERSE EXPANSION — BEFORE vs AFTER (baseline in-sample regression + leave-one-out CV)")
    print("=" * 112)
    header = f"{'Universe':<30}{'#Tick':>7}{'EVS n':>7}{'EVS R²':>9}{'EVS CV R²':>11}{'EVE n':>7}{'EVE R²':>9}{'EVE CV R²':>11}"
    print(header)
    print("-" * 112)
    for r in results:
        print(
            f"{r['label']:<30}{r['n_tickers']:>7}"
            f"{r['ev_sales_n']:>7}{_fmt_r2(r['ev_sales_in_sample_r2']):>9}{_fmt_num(r['ev_sales_cv_r2']):>11}"
            f"{r['ev_ebitda_n']:>7}{_fmt_r2(r['ev_ebitda_in_sample_r2']):>9}{_fmt_num(r['ev_ebitda_cv_r2']):>11}"
        )
    print("-" * 112)

    print("\nInterpretation (per task thresholds: CV R² > 0.4 = robust, < 0.3 = overfitting):")
    for r in results:
        cv = r["ev_sales_cv_r2"]
        if cv is None:
            verdict = "n/a"
        elif cv > 0.4:
            verdict = "ROBUST"
        elif cv < 0.3:
            verdict = "OVERFITTING / not robust"
        else:
            verdict = "AMBIGUOUS"
        print(f"  {r['label']:<30} EV/Sales CV R²={_fmt_num(cv):<8} -> {verdict}")


def run_universe_expansion_diagnostic():
    """
    Fetch each bucket's full (post-expansion) observation set once, then
    fit/CV both the pre-expansion subset ("before") and the full expanded
    set ("after") from that single fetch — avoids re-fetching the ~50
    tickers common to both.
    """
    from valuation.comps.universe import COMPS_UNIVERSE as CURRENT_UNIVERSE

    results = []
    for bucket in ("semiconductors", "mature_tech"):
        after_tickers = CURRENT_UNIVERSE[bucket]
        before_tickers = set(PRE_EXPANSION_UNIVERSE[bucket])

        print(f"\nFetching data for {bucket} full expanded universe ({len(after_tickers)} tickers)...")
        ev_sales_obs_all, ev_ebitda_obs_all = fetch_observations(after_tickers)

        ev_sales_obs_before = [o for o in ev_sales_obs_all if o.ticker in before_tickers]
        ev_ebitda_obs_before = [o for o in ev_ebitda_obs_all if o.ticker in before_tickers]

        results.append(_fit_and_cv(
            bucket, ev_sales_obs_before, ev_ebitda_obs_before,
            f"{bucket} (BEFORE)", len(before_tickers),
        ))
        results.append(_fit_and_cv(
            bucket, ev_sales_obs_all, ev_ebitda_obs_all,
            f"{bucket} (AFTER)", len(after_tickers),
        ))

    print_before_after_table(results)
    return results


if __name__ == "__main__":
    run_segment_diagnostics()
    run_analog_cross_validation_diagnostic()
    run_universe_expansion_diagnostic()
