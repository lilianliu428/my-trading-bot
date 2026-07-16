"""
Feature-selection diagnostic: extend the comps feature pool from 4 to 10
candidates and run greedy forward selection (ranked by leave-one-out CV R²)
to find the best-performing subset per bucket and multiple type.

Read-only diagnostic — does not change the live 4-feature regression path
(orchestrator.py, backtest/*, compute_features_for_ticker()). Reuses
regression.py's generalized fit_regression/predict_multiple (feature_names
parameter) and diagnostics.py's loo_cross_validate(), both extended in this
session to accept an arbitrary feature subset while defaulting to the
original 4 for every existing caller.

compute_extended_features_for_ticker() intentionally does ONE yfinance
fetch per ticker and derives all 10 features from it (duplicating a little
arithmetic from features.py rather than re-deriving the original 4 via a
second black-box call to compute_features_for_ticker(), which would double
the network cost — this diagnostic already touches 64+95 tickers).

Run directly: python3 -m valuation.comps.feature_selection
"""

import math

import numpy as np
import yfinance as yf

from valuation.comps.features import CompanyFeatures, _get_field
from valuation.inputs.growth import compute_fundamental_growth
from valuation.comps.regression import MultipleObservation, fit_regression, predict_multiple, MIN_OBSERVATIONS
from valuation.comps.diagnostics import loo_cross_validate

ALL_CANDIDATE_FEATURES = [
    "trailing_growth", "trailing_growth_5y", "growth_stability",
    "operating_margin", "gross_margin", "ebitda_margin",
    "roic", "r_and_d_intensity", "fcf_conversion",
    "log_revenue",
]

MAX_FEATURES = 6            # task constraint: stop growing past this count
MIN_IMPROVEMENT = 0.02      # stop if best next-round CV R² doesn't beat previous round by more than this
MAX_MISSING_FRACTION = 0.20 # drop a candidate feature entirely if unavailable for >20% of the bucket


def compute_extended_features_for_ticker(ticker):
    """
    Compute all 10 candidate features for one ticker from a single
    yfinance fetch. Never raises; missing pieces come back as None with an
    explanatory data_flag, same convention as compute_features_for_ticker().
    """
    data_flags = []
    yf_ticker = yf.Ticker(ticker)

    try:
        currency = yf_ticker.info.get("financialCurrency", "USD")
    except Exception:
        currency = "USD"
    if currency != "USD":
        data_flags.append(f"Non-USD reporting currency ({currency}) — skipping feature computation")
        return CompanyFeatures(ticker, None, None, None, None, None, None, data_flags)

    income_stmt = yf_ticker.income_stmt
    cash_flow = yf_ticker.cashflow

    if income_stmt is None or income_stmt.empty:
        data_flags.append("No income statement data")
        return CompanyFeatures(ticker, None, None, None, None, None, None, data_flags)

    n_years = income_stmt.shape[1]
    latest = income_stmt.iloc[:, 0]

    revenue_ttm = _get_field(latest, ["Total Revenue", "Revenue", "Operating Revenue"])
    operating_income_ttm = _get_field(latest, ["Operating Income", "EBIT"])

    if revenue_ttm is None or revenue_ttm <= 0:
        data_flags.append("No usable TTM revenue")
        return CompanyFeatures(ticker, None, None, None, None, None, None, data_flags)

    log_revenue = math.log(revenue_ttm)
    operating_margin = operating_income_ttm / revenue_ttm if operating_income_ttm is not None else None
    if operating_margin is None:
        data_flags.append("No usable TTM operating income — operating_margin unavailable")

    revenue_series = [_get_field(income_stmt.iloc[:, i], ["Total Revenue", "Revenue", "Operating Revenue"]) for i in range(n_years)]

    # --- trailing_growth: strict 3-year CAGR (same rule as features.py) ---
    trailing_growth = None
    if n_years >= 4 and revenue_series[3] is not None and revenue_series[3] > 0:
        trailing_growth = (revenue_ttm / revenue_series[3]) ** (1 / 3) - 1
    else:
        data_flags.append(f"Only {n_years} years of income statement history (<4) — trailing_growth unavailable")

    # --- trailing_growth_5y: CAGR over the longest available window up to
    # 5 years back. yfinance typically provides ~4-5 annual columns, so
    # this is usually a 4-year window in practice, not a literal 5y —
    # deliberately more lenient than trailing_growth so it's computable at
    # all, and flagged whenever the window is short of 5.
    trailing_growth_5y = None
    back = min(5, n_years - 1)
    if back >= 1 and revenue_series[back] is not None and revenue_series[back] > 0:
        trailing_growth_5y = (revenue_ttm / revenue_series[back]) ** (1 / back) - 1
        if back < 5:
            data_flags.append(f"trailing_growth_5y uses a {back}yr window (yfinance depth limit), not a literal 5yr")
    else:
        data_flags.append("trailing_growth_5y unavailable (insufficient revenue history)")

    # --- growth_stability = 1 - std(yoy growth) / mean(yoy growth), over
    # however many consecutive-year pairs are available (literal task
    # formula, no sign adjustment for shrinking companies).
    growth_stability = None
    yoy_growth = []
    for i in range(n_years - 1):
        r_new, r_old = revenue_series[i], revenue_series[i + 1]
        if r_new is not None and r_old is not None and r_old > 0:
            yoy_growth.append(r_new / r_old - 1)
    if len(yoy_growth) >= 2:
        mean_g = float(np.mean(yoy_growth))
        std_g = float(np.std(yoy_growth))
        if mean_g != 0:
            growth_stability = 1 - std_g / mean_g
        else:
            data_flags.append("growth_stability undefined (mean yoy growth = 0)")
    else:
        data_flags.append(f"Only {len(yoy_growth)} yoy growth observation(s) — growth_stability unavailable")

    # --- gross_margin, r_and_d_intensity (TTM = column 0) ---
    gross_profit = _get_field(latest, ["Gross Profit"])
    gross_margin = gross_profit / revenue_ttm if gross_profit is not None else None
    if gross_margin is None:
        data_flags.append("No Gross Profit field — gross_margin unavailable")

    rnd = _get_field(latest, ["Research And Development"])
    r_and_d_intensity = rnd / revenue_ttm if rnd is not None else None
    if r_and_d_intensity is None:
        data_flags.append("No Research And Development field — r_and_d_intensity unavailable (may mean genuinely no R&D line item, not necessarily zero spend)")

    # --- ebitda_ttm / ebitda_margin ---
    ebitda_ttm = None
    if operating_income_ttm is not None and cash_flow is not None and not cash_flow.empty:
        da_ttm = _get_field(cash_flow.iloc[:, 0], ["Depreciation And Amortization", "Depreciation"])
        if da_ttm is not None and (operating_income_ttm + da_ttm) > 0:
            ebitda_ttm = operating_income_ttm + da_ttm
    ebitda_margin = ebitda_ttm / revenue_ttm if ebitda_ttm is not None else None
    if ebitda_margin is None:
        data_flags.append("ebitda_ttm unavailable — ebitda_margin unavailable")

    # --- fcf_conversion = FCF / net income, capped [0, 3] ---
    fcf_conversion = None
    net_income = _get_field(latest, ["Net Income"])
    fcf = _get_field(cash_flow.iloc[:, 0], ["Free Cash Flow"]) if cash_flow is not None and not cash_flow.empty else None
    if net_income is not None and net_income > 0 and fcf is not None:
        fcf_conversion = max(0.0, min(3.0, fcf / net_income))
    elif net_income is not None and net_income <= 0:
        data_flags.append(f"Net income non-positive ({net_income:,.0f}) — fcf_conversion undefined")
    elif fcf is None:
        data_flags.append("No Free Cash Flow field — fcf_conversion unavailable")

    # --- roic: reuse Layer 1 growth module for consistency (same as features.py) ---
    try:
        fund_growth = compute_fundamental_growth(ticker)
        roic = fund_growth.get("roic")
        data_flags.extend(fund_growth.get("data_flags", []))
        if roic is None:
            data_flags.append("compute_fundamental_growth returned no ROIC")
        elif isinstance(roic, float) and (math.isnan(roic) or math.isinf(roic)):
            data_flags.append(f"compute_fundamental_growth returned non-finite ROIC ({roic}) — treating as unavailable")
            roic = None
    except Exception as e:
        data_flags.append(f"compute_fundamental_growth failed: {type(e).__name__}: {e}")
        roic = None

    return CompanyFeatures(
        ticker=ticker,
        trailing_growth=trailing_growth,
        operating_margin=operating_margin,
        roic=roic,
        log_revenue=log_revenue,
        revenue_ttm=revenue_ttm,
        ebitda_ttm=ebitda_ttm,
        data_flags=data_flags,
        trailing_growth_5y=trailing_growth_5y,
        growth_stability=growth_stability,
        gross_margin=gross_margin,
        ebitda_margin=ebitda_margin,
        r_and_d_intensity=r_and_d_intensity,
        fcf_conversion=fcf_conversion,
    )


def _is_finite(value):
    return value is not None and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))


def filter_available_features(features_list, candidate_features=ALL_CANDIDATE_FEATURES, max_missing_fraction=MAX_MISSING_FRACTION):
    """
    Drop any candidate feature that's unavailable for more than
    max_missing_fraction of features_list. Returns (kept, dropped) where
    dropped maps feature name -> missing fraction.
    """
    n = len(features_list)
    kept, dropped = [], {}
    for name in candidate_features:
        missing = sum(1 for f in features_list if not _is_finite(getattr(f, name, None)))
        missing_fraction = missing / n if n else 1.0
        if missing_fraction > max_missing_fraction:
            dropped[name] = missing_fraction
        else:
            kept.append(name)
    return kept, dropped


def build_observations(features_list, multiples_by_ticker):
    """Build MultipleObservation lists for both multiple types from already-computed extended features."""
    ev_sales_obs, ev_ebitda_obs = [], []
    for f in features_list:
        mults = multiples_by_ticker.get(f.ticker)
        if not mults:
            continue
        if mults["ev_sales"] is not None:
            ev_sales_obs.append(MultipleObservation(f.ticker, "ev_sales", mults["ev_sales"], f))
        if mults["ev_ebitda"] is not None:
            ev_ebitda_obs.append(MultipleObservation(f.ticker, "ev_ebitda", mults["ev_ebitda"], f))
    return ev_sales_obs, ev_ebitda_obs


def forward_selection(obs_list, multiple_type, candidate_features, max_features=MAX_FEATURES, min_improvement=MIN_IMPROVEMENT,
                       fit_fn=None, predict_fn=None):
    """
    Greedy forward feature selection, ranked by leave-one-out CV R² at
    every step (not in-sample R² — the whole point is avoiding the
    overfitting the prior 4-feature/small-universe diagnostics exposed).

    Stops when either:
      - the best candidate this round doesn't beat the previous round's
        CV R² by more than min_improvement, or
      - the selected set reaches max_features.
    Round 1 always adds its best single feature unconditionally (there is
    no "previous round" to compare against).

    fit_fn/predict_fn: default to regression.py's fit_regression/
    predict_multiple (log-linear absolute-multiple model). Pass
    alternatives (matching fit_regression's/predict_multiple's signature)
    to run this identical search procedure against a different model form
    — e.g. relative_regression.py's linear model on sector-relative
    multiples — without duplicating the search loop.

    Returns dict: selected_features, history (per-round detail, including
    every candidate tested), final_cv_r2, final_in_sample_r2, n_observations.
    """
    fit_fn = fit_fn or fit_regression
    predict_fn = predict_fn or predict_multiple
    remaining = list(candidate_features)
    selected = []
    history = []
    best_cv_r2_so_far = -float("inf")

    while remaining and len(selected) < max_features:
        round_results = []
        for candidate in remaining:
            trial_features = selected + [candidate]
            reg = fit_fn(obs_list, multiple_type, feature_names=trial_features)
            if reg.r_squared is None:
                round_results.append({"feature": candidate, "cv_r2": None, "in_sample_r2": None})
                continue
            _loo, cv_r2 = loo_cross_validate(obs_list, multiple_type, feature_names=trial_features, fit_fn=fit_fn, predict_fn=predict_fn)
            round_results.append({"feature": candidate, "cv_r2": cv_r2, "in_sample_r2": reg.r_squared})

        valid = [r for r in round_results if r["cv_r2"] is not None]
        if not valid:
            break

        valid.sort(key=lambda r: r["cv_r2"], reverse=True)
        best = valid[0]

        if selected and (best["cv_r2"] - best_cv_r2_so_far) <= min_improvement:
            history.append({
                "round": len(selected) + 1, "candidates_tested": round_results,
                "best_candidate": best["feature"], "best_cv_r2": best["cv_r2"],
                "stopped": True,
                "reason": f"improvement {best['cv_r2'] - best_cv_r2_so_far:+.3f} <= {min_improvement} threshold",
            })
            break

        history.append({
            "round": len(selected) + 1, "candidates_tested": round_results,
            "best_candidate": best["feature"], "best_cv_r2": best["cv_r2"],
            "stopped": False, "reason": None,
        })
        selected.append(best["feature"])
        remaining.remove(best["feature"])
        best_cv_r2_so_far = best["cv_r2"]

    if selected:
        final_reg = fit_fn(obs_list, multiple_type, feature_names=selected)
        _final_loo, final_cv_r2 = loo_cross_validate(obs_list, multiple_type, feature_names=selected, fit_fn=fit_fn, predict_fn=predict_fn)
    else:
        final_reg, final_cv_r2 = None, None

    return {
        "selected_features": selected,
        "history": history,
        "final_cv_r2": final_cv_r2,
        "final_in_sample_r2": final_reg.r_squared if final_reg else None,
        "n_observations": final_reg.n_observations if final_reg else 0,
    }


def print_round_history(bucket, multiple_type, result):
    print(f"\n{'-' * 90}")
    print(f"Forward selection: {bucket} / {multiple_type}")
    print(f"{'-' * 90}")
    for h in result["history"]:
        marker = "  (stopped here — not added)" if h["stopped"] else "  -> ADDED"
        print(f"  Round {h['round']}: best candidate = {h['best_candidate']:<20} CV R²={h['best_cv_r2']:.3f}{marker}")
        ranked = sorted(h["candidates_tested"], key=lambda r: (r["cv_r2"] if r["cv_r2"] is not None else -999), reverse=True)
        for r in ranked:
            cv = f"{r['cv_r2']:.3f}" if r["cv_r2"] is not None else "N/A"
            print(f"      {r['feature']:<20} CV R²={cv}")
    print(f"\n  FINAL feature set: {result['selected_features']}")
    print(f"  n_observations={result['n_observations']}  in-sample R²={result['final_in_sample_r2']}  CV R²={result['final_cv_r2']}")


def run_feature_selection_for_bucket(bucket, tickers):
    from valuation.comps.multiples import compute_multiples_for_ticker

    print(f"\n{'=' * 90}\nFEATURE SELECTION: {bucket} ({len(tickers)} tickers)\n{'=' * 90}")
    print("Computing extended (10-feature) data for all tickers...")
    features_list = [compute_extended_features_for_ticker(t) for t in tickers]
    multiples_by_ticker = {t: compute_multiples_for_ticker(t) for t in tickers}

    kept_features, dropped_features = filter_available_features(features_list)
    print(f"\nCandidate feature availability (threshold: drop if >{MAX_MISSING_FRACTION * 100:.0f}% missing):")
    for name in ALL_CANDIDATE_FEATURES:
        if name in dropped_features:
            print(f"  {name:<20} SKIPPED — {dropped_features[name] * 100:.0f}% missing")
        else:
            missing = sum(1 for f in features_list if not _is_finite(getattr(f, name, None)))
            print(f"  {name:<20} kept — {missing}/{len(features_list)} missing ({missing / len(features_list) * 100:.0f}%)")

    ev_sales_obs, ev_ebitda_obs = build_observations(features_list, multiples_by_ticker)

    results = {}
    for multiple_type, obs_list in (("ev_sales", ev_sales_obs), ("ev_ebitda", ev_ebitda_obs)):
        result = forward_selection(obs_list, multiple_type, kept_features)
        print_round_history(bucket, multiple_type, result)
        results[multiple_type] = result

    return {"bucket": bucket, "dropped_features": dropped_features, "results": results}


def print_final_summary(all_results):
    print(f"\n{'=' * 100}")
    print("FEATURE SELECTION — FINAL SUMMARY")
    print(f"{'=' * 100}")
    header = f"{'Bucket':<16}{'Multiple':<12}{'#Feat':>6}{'In-sample R²':>14}{'CV R²':>10}  Feature set"
    print(header)
    print("-" * 100)
    best_models = {}
    for bucket_result in all_results:
        bucket = bucket_result["bucket"]
        for multiple_type, r in bucket_result["results"].items():
            n = len(r["selected_features"])
            in_r2 = f"{r['final_in_sample_r2']:.3f}" if r["final_in_sample_r2"] is not None else "N/A"
            cv_r2 = f"{r['final_cv_r2']:.3f}" if r["final_cv_r2"] is not None else "N/A"
            print(f"{bucket:<16}{multiple_type:<12}{n:>6}{in_r2:>14}{cv_r2:>10}  {r['selected_features']}")
            best_models[f"{bucket}/{multiple_type}"] = r["selected_features"]
    print("-" * 100)

    # Robust features: appear in 3+ of the 4 best models
    from collections import Counter
    counts = Counter(feat for feats in best_models.values() for feat in feats)
    robust = sorted([f for f, c in counts.items() if c >= 3], key=lambda f: -counts[f])
    print("\nFeature frequency across the 4 best models (semi EV/Sales, semi EV/EBITDA, mature_tech EV/Sales, mature_tech EV/EBITDA):")
    for f, c in counts.most_common():
        marker = "  <- ROBUST (appears in 3+)" if c >= 3 else ""
        print(f"  {f:<20} {c}/4{marker}")

    return {"best_models": best_models, "robust_features": robust}


def run_feature_selection_diagnostic():
    from valuation.comps.universe import COMPS_UNIVERSE

    all_results = []
    for bucket in ("semiconductors", "mature_tech"):
        all_results.append(run_feature_selection_for_bucket(bucket, COMPS_UNIVERSE[bucket]))

    summary = print_final_summary(all_results)
    return all_results, summary


if __name__ == "__main__":
    run_feature_selection_diagnostic()
