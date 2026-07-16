"""
Point-in-time backtesting harness for the Layer 2 comps regression.

Fundamentals refresh only as often as annual filings are actually
published (yfinance gives ~5 years of them) — each ticker's fundamentals
are carried forward from its most recent point-in-time snapshot until a
newer one becomes available. Prices are sampled at the requested
`frequency` (monthly by default), so predicted_upside moves with price
between fundamental refreshes even when the regression inputs don't.

Survivorship: a ticker only enters the cross-section at a rebalancing date
once it has at least one point-in-time snapshot as-of that date, and only
if a price is available — no synthetic backfill.
"""

import datetime
from dataclasses import dataclass, field

from valuation.comps.regression import MultipleObservation, fit_regression, predict_multiple, MIN_OBSERVATIONS
from valuation.comps.backtest.historical_data import get_point_in_time_snapshots, fetch_price_series, nearest_price
from valuation.comps.backtest import analysis


@dataclass
class BacktestResult:
    # Meta
    period_start: str
    period_end: str
    n_rebalancing_dates: int
    n_predictions: int

    # Overall metrics (across all predictions)
    correlation_6m: float | None
    correlation_12m: float | None
    decile_spread_6m: float | None
    decile_spread_12m: float | None
    sharpe_6m: float | None
    sharpe_12m: float | None
    hit_rate_6m: float | None
    hit_rate_12m: float | None
    r_squared_6m: float | None
    r_squared_12m: float | None

    # By sub-analysis
    by_bucket: dict = field(default_factory=dict)
    by_ticker_type: dict = field(default_factory=dict)

    # Diagnostic
    data_flags: list = field(default_factory=list)


def _monthly_dates(start, end):
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        month = d.month + 1
        year = d.year + (1 if month > 12 else 0)
        month = month - 12 if month > 12 else month
        day = min(d.day, 28)
        d = datetime.date(year, month, day)
    return dates


def _build_result_from_rows(rows, forward_windows, period_start, period_end, n_rebalancing_dates, data_flags,
                             bucket_by_ticker=None, ticker_type_by_ticker=None):
    metrics = analysis.compute_all_metrics(rows, forward_windows)

    by_bucket = {}
    if bucket_by_ticker:
        buckets_present = {bucket_by_ticker[r["ticker"]] for r in rows if r["ticker"] in bucket_by_ticker}
        for b in buckets_present:
            sub_rows = [r for r in rows if bucket_by_ticker.get(r["ticker"]) == b]
            by_bucket[b] = analysis.compute_all_metrics(sub_rows, forward_windows)

    by_ticker_type = {}
    if ticker_type_by_ticker:
        types_present = {ticker_type_by_ticker[r["ticker"]] for r in rows if r["ticker"] in ticker_type_by_ticker}
        for tt in types_present:
            sub_rows = [r for r in rows if ticker_type_by_ticker.get(r["ticker"]) == tt]
            by_ticker_type[tt] = analysis.compute_all_metrics(sub_rows, forward_windows)

    return BacktestResult(
        period_start=period_start,
        period_end=period_end,
        n_rebalancing_dates=n_rebalancing_dates,
        n_predictions=len(rows),
        correlation_6m=metrics.get("correlation_6m"),
        correlation_12m=metrics.get("correlation_12m"),
        decile_spread_6m=metrics.get("decile_spread_6m"),
        decile_spread_12m=metrics.get("decile_spread_12m"),
        sharpe_6m=metrics.get("sharpe_6m"),
        sharpe_12m=metrics.get("sharpe_12m"),
        hit_rate_6m=metrics.get("hit_rate_6m"),
        hit_rate_12m=metrics.get("hit_rate_12m"),
        r_squared_6m=metrics.get("r_squared_6m"),
        r_squared_12m=metrics.get("r_squared_12m"),
        by_bucket=by_bucket,
        by_ticker_type=by_ticker_type,
        data_flags=data_flags,
    )


def run_backtest(tickers, start_date, end_date, forward_windows=[6, 12], frequency="M",
                  bucket_by_ticker=None, ticker_type_by_ticker=None):
    """
    Run point-in-time backtest of the comps model over a historical period.

    bucket_by_ticker / ticker_type_by_ticker: optional {ticker: label} maps
    for the by_bucket / by_ticker_type sub-analyses (see spec). Left empty
    if not supplied — computing ticker_type requires a Layer 1 DCF fair
    value per ticker, which this module deliberately doesn't compute itself
    (keeps the backtest decoupled from the expensive, non-point-in-time DCF
    pipeline; callers can classify tickers however they've already done so
    and pass the map in).

    Returns BacktestResult. start_date/end_date are clipped to whatever
    point-in-time fundamentals are actually available (yfinance's annual
    statements go back ~5 years) — the clip is recorded in data_flags.
    """
    data_flags = []
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    max_fwd_months = max(forward_windows)

    snapshots_by_ticker = {}
    prices_by_ticker = {}
    for t in tickers:
        snaps, flags = get_point_in_time_snapshots(t)
        data_flags.extend(f"[{t}] {f}" for f in flags)
        snapshots_by_ticker[t] = snaps

        price_start = start - datetime.timedelta(days=14)
        price_end = end + datetime.timedelta(days=32 * max_fwd_months)
        prices_by_ticker[t] = fetch_price_series(t, price_start, price_end)

    all_as_of = [s["as_of_date"] for snaps in snapshots_by_ticker.values() for s in snaps]
    if not all_as_of:
        data_flags.append("No point-in-time snapshots available for any ticker — backtest cannot run")
        return _build_result_from_rows([], forward_windows, start_date, end_date, 0, data_flags)

    available_start = max(start, min(all_as_of))
    if available_start > start:
        data_flags.append(
            f"Requested start {start_date} predates available fundamentals for this ticker set — "
            f"clipped to {available_start.isoformat()} (yfinance annual statements have limited depth)"
        )

    rebal_dates = [d for d in _monthly_dates(available_start, end) if d <= end]
    if not rebal_dates:
        data_flags.append("No rebalancing dates in the clipped range")
        return _build_result_from_rows([], forward_windows, available_start.isoformat(), end_date, 0, data_flags)

    rows = []
    dates_with_cross_section = 0

    for T in rebal_dates:
        cross_section = {}
        for t, snaps in snapshots_by_ticker.items():
            usable = [s for s in snaps if s["as_of_date"] <= T]
            if usable:
                cross_section[t] = usable[-1]  # most recent snapshot as-of T

        if len(cross_section) < MIN_OBSERVATIONS:
            continue

        observations = []
        prices_at_T = {}
        for t, snap in cross_section.items():
            price_T = nearest_price(prices_by_ticker[t], T)
            shares = snap["shares_outstanding"]
            revenue = snap["features"].revenue_ttm
            if price_T is None or not shares or not revenue or revenue <= 0:
                continue
            prices_at_T[t] = price_T
            market_cap = price_T * shares
            ev = market_cap + (snap["total_debt"] or 0) - (snap["cash"] or 0)
            ev_sales = ev / revenue
            if ev_sales > 0:
                observations.append(MultipleObservation(t, "ev_sales", ev_sales, snap["features"]))

        if len(observations) < MIN_OBSERVATIONS:
            continue

        reg = fit_regression(observations, "ev_sales")
        if reg.r_squared is None:
            continue

        dates_with_cross_section += 1

        for t, snap in cross_section.items():
            if t not in prices_at_T:
                continue
            predicted_mult, _se = predict_multiple(reg, snap["features"])
            if predicted_mult is None:
                continue
            revenue = snap["features"].revenue_ttm
            shares = snap["shares_outstanding"]
            implied_ev = predicted_mult * revenue
            equity_value = implied_ev - (snap["total_debt"] or 0) + (snap["cash"] or 0)
            predicted_fair_value = equity_value / shares
            price_T = prices_at_T[t]
            predicted_upside = predicted_fair_value / price_T - 1

            row = {"ticker": t, "date": T, "predicted_upside": predicted_upside}
            for w in forward_windows:
                future_date = T + datetime.timedelta(days=30 * w)
                price_future = nearest_price(prices_by_ticker[t], future_date)
                if price_future is not None:
                    row[f"actual_return_{w}m"] = price_future / price_T - 1
            rows.append(row)

    data_flags.append(
        f"{len(rows)} (ticker, date) predictions across {dates_with_cross_section} of "
        f"{len(rebal_dates)} candidate rebalancing dates (rest skipped — cross-section below "
        f"{MIN_OBSERVATIONS} names with usable fundamentals + price)"
    )

    return _build_result_from_rows(
        rows, forward_windows, available_start.isoformat(), end_date, len(rebal_dates), data_flags,
        bucket_by_ticker=bucket_by_ticker, ticker_type_by_ticker=ticker_type_by_ticker,
    )
