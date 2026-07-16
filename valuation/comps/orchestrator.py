"""
Layer 2 entry point — full comps valuation for a single ticker.

compute_comps_valuation(ticker, bucket=None) orchestrates: universe lookup,
feature/multiple computation across the universe, regression fitting,
prediction for the target, conversion to fair value per share, and peer
similarity for interpretability.
"""

from dataclasses import dataclass, field

from valuation.comps.universe import get_universe
from valuation.comps.features import compute_features_for_ticker
from valuation.comps.multiples import compute_multiples_for_ticker, compute_ev_components
from valuation.comps.regression import MultipleObservation, fit_regression, predict_multiple, check_extrapolation
from valuation.comps.similarity import find_nearest_peers


@dataclass
class CompsValuationResult:
    ticker: str
    market_price: float | None

    # EV/Sales prediction
    predicted_ev_sales: float | None
    fair_value_from_ev_sales: float | None
    ev_sales_std_error: float | None
    ev_sales_r2: float | None

    # EV/EBITDA prediction (None if EBITDA-based unavailable)
    predicted_ev_ebitda: float | None
    fair_value_from_ev_ebitda: float | None
    ev_ebitda_std_error: float | None
    ev_ebitda_r2: float | None

    # Peer interpretability
    nearest_peers: list          # ticker symbols of top 5 nearest neighbors
    peer_distances: list         # standardized distances

    # Diagnostics
    data_flags: list = field(default_factory=list)


def resolve_bucket(ticker, bucket):
    """
    Explicit bucket always wins. If not given, best-effort auto-derive via
    the existing classify() lookup (scoring/sectors/classifier.py), using
    fresh yfinance data — mirrors the pattern api/routers/valuation.py notes
    it will eventually replace its hardcoded ticker map with.

    Returns (bucket, data_flags).
    """
    if bucket is not None:
        return bucket, []

    import yfinance as yf
    from scoring.sectors.classifier import classify

    info = yf.Ticker(ticker).info
    row = {
        "ticker": ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "revenue_growth": info.get("revenueGrowth"),
        "profit_margin": info.get("profitMargins"),
        "total_revenue": info.get("totalRevenue"),
    }
    derived = classify(row)
    return derived, [f"No bucket given — auto-derived '{derived}' via classify() from yfinance sector/industry"]


def compute_comps_valuation(ticker, bucket=None):
    """
    Full Layer 2 comps valuation for a single ticker. See docstring flow in
    LAYER_2_COMPS_SPEC.md. Never raises for ordinary data-availability
    issues — degrades to None fields with explanatory data_flags instead.
    """
    ticker = ticker.upper()
    bucket, data_flags = resolve_bucket(ticker, bucket)

    universe_tickers = get_universe(bucket, exclude_ticker=ticker)
    if not universe_tickers:
        data_flags.append(f"No comps universe defined for bucket '{bucket}'")
        return CompsValuationResult(
            ticker=ticker, market_price=None,
            predicted_ev_sales=None, fair_value_from_ev_sales=None, ev_sales_std_error=None, ev_sales_r2=None,
            predicted_ev_ebitda=None, fair_value_from_ev_ebitda=None, ev_ebitda_std_error=None, ev_ebitda_r2=None,
            nearest_peers=[], peer_distances=[], data_flags=data_flags,
        )

    # Step 2-3: features for target + universe
    target_features = compute_features_for_ticker(ticker)
    data_flags.extend(target_features.data_flags)
    universe_features = [compute_features_for_ticker(t) for t in universe_tickers]

    if not target_features.is_complete():
        data_flags.append(f"Insufficient data for {ticker} — missing regression features, no fair value estimate possible")

    # Step 4: multiples for universe
    universe_multiples = {t: compute_multiples_for_ticker(t) for t in universe_tickers}
    for t, m in universe_multiples.items():
        data_flags.extend(f"[{t}] {flag}" for flag in m["data_flags"])

    # Step 5-6: build observations, fit EV/Sales regression (winsorization happens inside fit_regression)
    features_by_ticker = {f.ticker: f for f in universe_features}
    ev_sales_obs = []
    ev_ebitda_obs = []
    for t in universe_tickers:
        feats = features_by_ticker[t]
        mults = universe_multiples[t]
        if mults["ev_sales"] is not None:
            ev_sales_obs.append(MultipleObservation(t, "ev_sales", mults["ev_sales"], feats))
        if mults["ev_ebitda"] is not None:
            ev_ebitda_obs.append(MultipleObservation(t, "ev_ebitda", mults["ev_ebitda"], feats))

    ev_sales_reg = fit_regression(ev_sales_obs, "ev_sales")
    data_flags.extend(f"[ev_sales regression] {flag}" for flag in ev_sales_reg.data_flags)

    # Step 7: EV/EBITDA regression, if enough data
    ev_ebitda_reg = fit_regression(ev_ebitda_obs, "ev_ebitda")
    data_flags.extend(f"[ev_ebitda regression] {flag}" for flag in ev_ebitda_reg.data_flags)

    # Step 8: predict multiples for target
    predicted_ev_sales, ev_sales_se_log = predict_multiple(ev_sales_reg, target_features)
    predicted_ev_ebitda, ev_ebitda_se_log = predict_multiple(ev_ebitda_reg, target_features)
    data_flags.extend(check_extrapolation(ev_sales_reg, target_features))
    data_flags.extend(check_extrapolation(ev_ebitda_reg, target_features))

    # Step 9: convert to fair value per share
    ev_components = compute_ev_components(ticker)
    data_flags.extend(ev_components["data_flags"])
    total_debt = ev_components["total_debt"] or 0.0
    cash = ev_components["cash"] or 0.0
    shares_outstanding = ev_components["shares_outstanding"]
    market_price = ev_components["current_price"]

    fair_value_from_ev_sales = None
    ev_sales_std_error = None
    if predicted_ev_sales is not None and shares_outstanding and target_features.revenue_ttm:
        implied_ev = predicted_ev_sales * target_features.revenue_ttm
        equity_value = implied_ev - total_debt + cash
        fair_value_from_ev_sales = equity_value / shares_outstanding
        # Delta-method approximation: se on the log scale translates to a
        # roughly proportional se on the level (fair value) scale.
        ev_sales_std_error = abs(fair_value_from_ev_sales * ev_sales_se_log) if ev_sales_se_log is not None else None

    fair_value_from_ev_ebitda = None
    ev_ebitda_std_error = None
    if predicted_ev_ebitda is not None and shares_outstanding and target_features.ebitda_ttm:
        implied_ev = predicted_ev_ebitda * target_features.ebitda_ttm
        equity_value = implied_ev - total_debt + cash
        fair_value_from_ev_ebitda = equity_value / shares_outstanding
        ev_ebitda_std_error = abs(fair_value_from_ev_ebitda * ev_ebitda_se_log) if ev_ebitda_se_log is not None else None

    # Step 10: peer similarity — display-only, does not affect fair value
    peers = find_nearest_peers(target_features, universe_features, n=5)
    nearest_peers = [p[0] for p in peers]
    peer_distances = [p[1] for p in peers]

    # Step 11: return
    return CompsValuationResult(
        ticker=ticker,
        market_price=market_price,
        predicted_ev_sales=predicted_ev_sales,
        fair_value_from_ev_sales=fair_value_from_ev_sales,
        ev_sales_std_error=ev_sales_std_error,
        ev_sales_r2=ev_sales_reg.r_squared,
        predicted_ev_ebitda=predicted_ev_ebitda,
        fair_value_from_ev_ebitda=fair_value_from_ev_ebitda,
        ev_ebitda_std_error=ev_ebitda_std_error,
        ev_ebitda_r2=ev_ebitda_reg.r_squared,
        nearest_peers=nearest_peers,
        peer_distances=peer_distances,
        data_flags=data_flags,
    )


# Re-export: the finalized, per-bucket-routed Layer 2 entry point
# (RandomForest for semiconductors, OLS-relative for mature_tech EV/EBITDA,
# peer-display-only elsewhere — see LAYER_2_FINAL_SPEC and final_engine.py).
# compute_comps_valuation() above is UNCHANGED and stays available — it's
# the general absolute-4-feature-OLS engine the diagnostic modules
# (feature_selection.py, holdout_validation.py, etc.) are still built on.
# composite.py now calls compute_layer_2_comps, not compute_comps_valuation.
from valuation.comps.final_engine import compute_layer_2_comps  # noqa: E402,F401


if __name__ == "__main__":
    for ticker, bucket in [("AMD", "semiconductors"), ("NVDA", "semiconductors"), ("MSFT", "mature_tech")]:
        r = compute_comps_valuation(ticker, bucket=bucket)
        print(f"\n{'=' * 60}\n{r.ticker} ({bucket})\n{'=' * 60}")
        print(f"  Market price:            ${r.market_price:.2f}")
        if r.fair_value_from_ev_sales is not None:
            print(f"  Fair value (EV/Sales):   ${r.fair_value_from_ev_sales:.2f}  (predicted {r.predicted_ev_sales:.2f}x, R²={r.ev_sales_r2:.3f}, ±${r.ev_sales_std_error:.2f})")
        if r.fair_value_from_ev_ebitda is not None:
            print(f"  Fair value (EV/EBITDA):  ${r.fair_value_from_ev_ebitda:.2f}  (predicted {r.predicted_ev_ebitda:.2f}x, R²={r.ev_ebitda_r2:.3f}, ±${r.ev_ebitda_std_error:.2f})")
        print(f"  Nearest peers: {list(zip(r.nearest_peers, [round(d, 2) for d in r.peer_distances]))}")
