"""
Assembles the (ticker, date, features, forward_returns) panel for the alpha
research layer. Pure reuse of the existing point-in-time backtest machinery
in valuation/comps/backtest/ — no duplicated fetching or date logic:

    - get_point_in_time_snapshots / fetch_price_series / nearest_price
      (historical_data.py) for fundamentals + prices, including the
      Category A features added there for this layer.
    - _monthly_dates (framework.py) for the rebalancing date grid.
    - compute_beta_from_price_series / compute_point_in_time_risk_free_rate /
      compute_point_in_time_wacc / compute_point_in_time_dcf_fair_value
      (dcf_approximation.py) for point-in-time dcf_upside.

No lookahead: at rebalancing date T, only snapshots with as_of_date <= T
are used, beta/WACC/DCF only look at price and rate history up to T, and
forward returns are read from T forward.
"""

import datetime
import time

import pandas as pd

from valuation.comps.backtest.historical_data import get_point_in_time_snapshots, fetch_price_series, nearest_price
from valuation.comps.backtest.framework import _monthly_dates
from valuation.comps.backtest.dcf_approximation import (
    compute_beta_from_price_series,
    compute_point_in_time_risk_free_rate,
    compute_point_in_time_wacc,
    compute_point_in_time_dcf_fair_value,
    BETA_LOOKBACK_DAYS,
)
from valuation.comps.universe import COMPS_UNIVERSE

FORWARD_WINDOWS = (6, 12)

# Category A + Category B feature columns the panel produces per (ticker, date).
FEATURE_COLUMNS = [
    "ex_goodwill_roic", "rd_capitalized_margin", "normalized_reinvestment",
    "sbc_dilution_rate", "goodwill_ratio", "dcf_upside",
    "gross_margin", "trailing_growth", "roic", "ev_ebitda",
]


def build_bucket_map():
    """
    One primary bucket per ticker, from the two universes this layer tests
    (semiconductors, mature_tech — "communication" isn't in the spec's
    scope). STX and WDC appear in BOTH universe.py lists (a pre-existing
    overlap in that file, not introduced here); semiconductors wins as the
    more specific classification for neutralize()'s bucket grouping.
    """
    bucket_map = {}
    for t in COMPS_UNIVERSE.get("mature_tech", []):
        bucket_map[t] = "mature_tech"
    for t in COMPS_UNIVERSE.get("semiconductors", []):
        bucket_map[t] = "semiconductors"
    return bucket_map


def fetch_ticker_data(ticker, start_date, end_date, forward_windows=FORWARD_WINDOWS):
    """
    Network-fetching step for one ticker: point-in-time snapshots + a price
    series wide enough to cover beta lookback and the forward-return
    horizons. Split out from build_panel so fetches can be cached/resumed
    per-ticker (yfinance rate-limits escalate on long unbroken runs).

    Returns (snapshots, prices, data_flags).
    """
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    max_fwd_months = max(forward_windows)

    snaps, flags = get_point_in_time_snapshots(ticker)
    data_flags = [f"[{ticker}] {f}" for f in flags]
    price_start = start - datetime.timedelta(days=BETA_LOOKBACK_DAYS + 30)
    price_end = end + datetime.timedelta(days=32 * max_fwd_months)
    prices = fetch_price_series(ticker, price_start, price_end)
    return snaps, prices, data_flags


def assemble_panel(tickers, start_date, end_date, bucket_by_ticker, snapshots_by_ticker,
                    prices_by_ticker, spy_prices, forward_windows=FORWARD_WINDOWS, verbose=False):
    """
    Pure in-memory assembly step (no network calls): builds the
    (ticker, date, features, forward_returns) panel from already-fetched
    snapshots/prices. See build_panel for the fetch-then-assemble
    convenience wrapper.
    """
    data_flags = []
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)

    if not spy_prices:
        data_flags.append("Could not fetch SPY price series — dcf_upside will be all-NaN")

    all_as_of = [s["as_of_date"] for snaps in snapshots_by_ticker.values() for s in snaps]
    if not all_as_of:
        data_flags.append("No point-in-time snapshots for any ticker — panel is empty")
        return pd.DataFrame(columns=["ticker", "date", "bucket"] + FEATURE_COLUMNS), data_flags

    available_start = max(start, min(all_as_of))
    if available_start > start:
        data_flags.append(
            f"Requested start {start_date} predates available fundamentals — "
            f"clipped to {available_start.isoformat()}"
        )
    rebal_dates = [d for d in _monthly_dates(available_start, end) if d <= end]

    rows = []
    n_skipped_no_snapshot, n_skipped_no_price = 0, 0
    for date_idx, T in enumerate(rebal_dates):
        if verbose:
            print(f"  [date {date_idx + 1}/{len(rebal_dates)}] {T} — {len(rows)} rows so far", flush=True)
        for t in tickers:
            usable = [s for s in snapshots_by_ticker[t] if s["as_of_date"] <= T]
            if not usable:
                n_skipped_no_snapshot += 1
                continue
            snap = usable[-1]

            price_T = nearest_price(prices_by_ticker[t], T)
            if price_T is None:
                n_skipped_no_price += 1
                continue

            features = snap["features"]
            shares = snap["shares_outstanding"]
            debt = snap["total_debt"] or 0.0
            cash = snap["cash"] or 0.0

            ev_ebitda = None
            if shares and features.ebitda_ttm and features.ebitda_ttm > 0:
                ev = price_T * shares + debt - cash
                if ev > 0:
                    ev_ebitda = ev / features.ebitda_ttm

            dcf_upside = None
            if spy_prices:
                beta = compute_beta_from_price_series(prices_by_ticker[t], spy_prices, T)
                rf = compute_point_in_time_risk_free_rate(T)
                if beta is not None and rf is not None:
                    wacc = compute_point_in_time_wacc(snap, price_T, rf, beta)
                    fair_value = compute_point_in_time_dcf_fair_value(snap, wacc, rf)
                    if fair_value is not None and fair_value > 0:
                        dcf_upside = fair_value / price_T - 1

            row = {
                "ticker": t,
                "date": T,
                "bucket": bucket_by_ticker.get(t),
                "ex_goodwill_roic": snap.get("roic_ex_goodwill"),
                "rd_capitalized_margin": snap.get("rd_capitalized_margin"),
                "normalized_reinvestment": snap.get("normalized_reinvestment"),
                "sbc_dilution_rate": snap.get("sbc_dilution_rate"),
                "goodwill_ratio": snap.get("goodwill_ratio"),
                "dcf_upside": dcf_upside,
                "gross_margin": features.gross_margin,
                "trailing_growth": features.trailing_growth,
                "roic": features.roic,
                "ev_ebitda": ev_ebitda,
            }
            for w in forward_windows:
                future_date = T + datetime.timedelta(days=30 * w)
                price_future = nearest_price(prices_by_ticker[t], future_date)
                row[f"fwd_return_{w}m"] = (price_future / price_T - 1) if price_future is not None else None
            rows.append(row)

    panel_df = pd.DataFrame(rows)
    data_flags.append(
        f"Panel: {len(panel_df)} (ticker, date) rows across {len(rebal_dates)} candidate rebalancing "
        f"dates, {len(tickers)} tickers requested (skipped: {n_skipped_no_snapshot} no-snapshot-yet, "
        f"{n_skipped_no_price} no-price)"
    )
    return panel_df, data_flags


def build_panel(tickers, start_date, end_date, bucket_by_ticker, forward_windows=FORWARD_WINDOWS,
                 verbose=False, pace_seconds=0.0):
    """
    Convenience wrapper: fetch_ticker_data for every ticker + SPY, then
    assemble_panel. See those two functions' docstrings for details;
    use them directly instead of this wrapper when fetches need to be
    cached/resumed across process restarts (e.g. under rate limiting).
    """
    data_flags = []
    snapshots_by_ticker, prices_by_ticker = {}, {}
    for idx, t in enumerate(tickers):
        if verbose:
            print(f"  [{idx + 1}/{len(tickers)}] fetching {t}...", flush=True)
        snaps, prices, flags = fetch_ticker_data(t, start_date, end_date, forward_windows)
        snapshots_by_ticker[t] = snaps
        prices_by_ticker[t] = prices
        data_flags.extend(flags)
        if pace_seconds:
            time.sleep(pace_seconds)

    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    max_fwd_months = max(forward_windows)
    spy_prices = fetch_price_series(
        "SPY",
        start - datetime.timedelta(days=BETA_LOOKBACK_DAYS + 30),
        end + datetime.timedelta(days=32 * max_fwd_months),
    )

    panel_df, assemble_flags = assemble_panel(
        tickers, start_date, end_date, bucket_by_ticker,
        snapshots_by_ticker, prices_by_ticker, spy_prices, forward_windows, verbose,
    )
    data_flags.extend(assemble_flags)
    return panel_df, data_flags


def coverage_report(panel_df):
    """
    Per-date cross-sectional width and per-column non-null coverage, for
    the Phase 1 sanity check ("expected shape, no lookahead, reasonable
    coverage per date").
    """
    if panel_df.empty:
        return {"n_dates": 0, "min_width": 0, "max_width": 0, "mean_width": 0.0,
                "tickers_per_date": {}, "column_coverage": {}}

    tickers_per_date = panel_df.groupby("date")["ticker"].nunique().to_dict()
    widths = list(tickers_per_date.values())
    feature_cols = [c for c in panel_df.columns if c not in ("ticker", "date", "bucket")]
    column_coverage = {c: float(panel_df[c].notna().mean()) for c in feature_cols}

    return {
        "n_dates": panel_df["date"].nunique(),
        "min_width": min(widths),
        "max_width": max(widths),
        "mean_width": sum(widths) / len(widths),
        "tickers_per_date": tickers_per_date,
        "column_coverage": column_coverage,
    }
