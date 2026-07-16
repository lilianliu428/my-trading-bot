"""
Per-bucket Layer 2 routing — the finalized engine after 6 diagnostic
sessions (segmentation, feature selection, sector-relative multiples,
Ridge, tree regression). Routes each (ticker, bucket, multiple_type)
combination to whichever model was actually validated for it, or to
peer-display-only if none was.

Routing table (LAYER_2_FINAL_SPEC):
    semiconductors  ev_sales   -> semi_forest       (HIGH confidence)
    semiconductors  ev_ebitda  -> semi_forest       (HIGH confidence)
    mature_tech     ev_sales   -> NOT SHIPPED        (peer display only)
    mature_tech     ev_ebitda  -> mature_relative    (MEDIUM confidence)
    everything else            -> NOT SHIPPED        (peer display only)

Peer display is always computed, for every ticker in every bucket,
regardless of whether a regression model is available. This module does
not replace valuation/comps/orchestrator.py's compute_comps_valuation()
(the general absolute-4-feature-OLS engine) — that stays in place
unchanged, still backing the diagnostic modules built during validation.
orchestrator.py re-exports compute_layer_2_comps from here as the new
Layer 2 entry point composite.py now calls.
"""

from valuation.comps.universe import get_universe
from valuation.comps.multiples import compute_multiples_for_ticker, compute_ev_components
from valuation.comps.feature_selection import compute_extended_features_for_ticker
from valuation.comps.regression import MultipleObservation
from valuation.comps.output_schema import Layer2CompsResult, Layer2RegressionOutput
from valuation.comps.models.semi_forest import (
    fit_semi_forest, predict_semi_forest, VALIDATED_HOLDOUT_R2 as SEMI_HOLDOUT_R2,
)
from valuation.comps.models.mature_relative import (
    fit_mature_relative, predict_mature_relative, VALIDATED_HOLDOUT_R2 as MATURE_HOLDOUT_R2,
)
from valuation.comps.models.peer_display import compute_peer_display

# Fixed per the validated diagnostic results, not recomputed live — see
# LAYER_2_FINAL_SPEC's Confidence tiers section. Note: mature_tech/
# ev_ebitda's hold-out R²=0.437 technically clears the doc's own numeric
# HIGH bar (>0.30, gap<0.15), but the spec's results table explicitly
# assigns MEDIUM — honored here as the deliberate, more conservative call,
# given that model rests on only 2 features and a frozen-median
# relative-target formulation used nowhere else in the shipped engine.
CONFIDENCE_TIER = {
    ("semiconductors", "ev_sales"): "HIGH",
    ("semiconductors", "ev_ebitda"): "HIGH",
    ("mature_tech", "ev_ebitda"): "MEDIUM",
}


def _build_observations(tickers, features_by_ticker, multiples_by_ticker, multiple_type):
    obs = []
    for t in tickers:
        f = features_by_ticker.get(t)
        m = multiples_by_ticker.get(t)
        if f is None or m is None or m.get(multiple_type) is None:
            continue
        obs.append(MultipleObservation(t, multiple_type, m[multiple_type], f))
    return obs


def _build_regression_output(multiple_type, predicted_multiple, target_features, shares_outstanding, total_debt, cash,
                              model_type, confidence, holdout_r2, formulation, features_used, features_missing):
    if predicted_multiple is None or not shares_outstanding:
        return None
    basis = target_features.revenue_ttm if multiple_type == "ev_sales" else target_features.ebitda_ttm
    if not basis:
        return None

    implied_ev = predicted_multiple * basis
    equity_value = implied_ev - total_debt + cash
    fair_value_per_share = equity_value / shares_outstanding

    return Layer2RegressionOutput(
        multiple_type=multiple_type, predicted_multiple=predicted_multiple,
        fair_value_per_share=fair_value_per_share, confidence=confidence, model_type=model_type,
        holdout_r2=holdout_r2, formulation=formulation, features_used=features_used, features_missing=features_missing,
    )


def compute_layer_2_comps(ticker, bucket=None):
    """
    Full finalized Layer 2 output for a single ticker: routed regression
    prediction(s) (where validated for this bucket) + universal peer
    display. Never raises for ordinary data-availability issues.
    """
    from valuation.comps.orchestrator import resolve_bucket

    ticker = ticker.upper()
    data_flags = []
    bucket, bucket_flags = resolve_bucket(ticker, bucket)
    data_flags.extend(bucket_flags)

    universe_tickers = get_universe(bucket, exclude_ticker=ticker)
    if not universe_tickers:
        data_flags.append(f"No comps universe defined for bucket '{bucket}' — regression and peer display both unavailable")
        return Layer2CompsResult(ticker=ticker, bucket=bucket, market_price=None, data_flags=data_flags)

    target_features = compute_extended_features_for_ticker(ticker)
    data_flags.extend(target_features.data_flags)
    universe_features = [compute_extended_features_for_ticker(t) for t in universe_tickers]
    features_by_ticker = {t: f for t, f in zip(universe_tickers, universe_features)}

    target_multiples = compute_multiples_for_ticker(ticker)
    data_flags.extend(target_multiples["data_flags"])
    universe_multiples = {t: compute_multiples_for_ticker(t) for t in universe_tickers}

    ev_components = compute_ev_components(ticker)
    data_flags.extend(ev_components["data_flags"])
    market_price = ev_components["current_price"]
    total_debt = ev_components["total_debt"] or 0.0
    cash = ev_components["cash"] or 0.0
    shares_outstanding = ev_components["shares_outstanding"]

    ev_sales_output, ev_ebitda_output = None, None

    if bucket == "semiconductors":
        for multiple_type in ("ev_sales", "ev_ebitda"):
            train_obs = _build_observations(universe_tickers, features_by_ticker, universe_multiples, multiple_type)
            model_bundle, fit_flags = fit_semi_forest(train_obs, multiple_type)
            data_flags.extend(f"[{multiple_type}] {flag}" for flag in fit_flags)
            predicted_multiple, missing, predict_flags = predict_semi_forest(model_bundle, target_features)
            data_flags.extend(f"[{multiple_type}] {flag}" for flag in predict_flags)

            output = _build_regression_output(
                multiple_type, predicted_multiple, target_features, shares_outstanding, total_debt, cash,
                model_type=f"RandomForest {len(model_bundle.feature_names)}-feature" if model_bundle else "RandomForest (not fit)",
                confidence=CONFIDENCE_TIER.get((bucket, multiple_type)),
                holdout_r2=SEMI_HOLDOUT_R2.get(multiple_type),
                formulation="absolute",
                features_used=model_bundle.feature_names if model_bundle else [],
                features_missing=missing,
            )
            if multiple_type == "ev_sales":
                ev_sales_output = output
            else:
                ev_ebitda_output = output

    elif bucket == "mature_tech":
        multiple_type = "ev_ebitda"
        train_obs = _build_observations(universe_tickers, features_by_ticker, universe_multiples, multiple_type)
        model_bundle, fit_flags = fit_mature_relative(train_obs)
        data_flags.extend(f"[{multiple_type}] {flag}" for flag in fit_flags)
        predicted_multiple, missing, predict_flags = predict_mature_relative(model_bundle, target_features)
        data_flags.extend(f"[{multiple_type}] {flag}" for flag in predict_flags)

        ev_ebitda_output = _build_regression_output(
            multiple_type, predicted_multiple, target_features, shares_outstanding, total_debt, cash,
            model_type="OLS relative 2-feature",
            confidence=CONFIDENCE_TIER.get((bucket, multiple_type)),
            holdout_r2=MATURE_HOLDOUT_R2,
            formulation="relative",
            features_used=model_bundle.feature_names if model_bundle else [],
            features_missing=missing,
        )
        data_flags.append("mature_tech EV/Sales not shipped — no model form validated across 6 diagnostic sessions; peer display only")

    else:
        data_flags.append(f"Bucket '{bucket}' has no validated Layer 2 regression model — peer display only")

    peer_display = compute_peer_display(target_features, universe_features, target_multiples, universe_multiples)

    return Layer2CompsResult(
        ticker=ticker, bucket=bucket, market_price=market_price,
        ev_sales_regression=ev_sales_output, ev_ebitda_regression=ev_ebitda_output,
        peer_display=peer_display, data_flags=data_flags,
    )
