"""
Layer-1-style DCF APPROXIMATION for backtesting mature_tech — NOT the real
production valuation/tech/mature/mature_dcf.py engine.

Why this exists: compute_tech_intrinsic_value() cannot run in this local
environment at all — its WACC computation depends on valuation/inputs/
beta.py reading a local `price_history` SQL table that was never created
here (confirmed: `sqlite3.OperationalError: no such table: price_history`,
same gap documented back when Layer 2's composite.py tests were built).
Even where it could run, it isn't parameterized by a historical date
anywhere in growth.py/wacc.py — recomputing "what the real engine would
have said as of date T" would require refactoring that machinery
throughout, well beyond backtest-infrastructure work.

This module is a self-contained, explicitly simplified substitute, built
specifically to test the underlying hypothesis (do DCF-style fair-value
signals predict returns for mature tech) without either blocker:

    - Beta: computed directly from yfinance daily price history (stock vs
      SPY) over a trailing window ending at each rebalancing date —
      bypasses the broken local DB entirely.
    - Risk-free rate: fetched from FRED for the date in question (not
      "current", like valuation/data_sources/fred.py's get_risk_free_rate()).
    - Growth model: a plain 2-stage DCF (5-year explicit + Gordon-growth
      terminal), NOT Layer 1's 3-stage/ROIC-regime-detection/ex-goodwill-
      routing/R&D-capitalization/SBC-dilution machinery.

Reuses unchanged: get_point_in_time_snapshots, fetch_price_series,
nearest_price (historical_data.py); _monthly_dates (framework.py);
compute_all_metrics and every metric function (analysis.py) — the metrics
layer doesn't care whether "predicted_upside" came from a comps regression
or a DCF, so none of it needed touching.

Read directly: python3 -m valuation.comps.backtest.dcf_approximation
"""

import datetime

import numpy as np

from valuation.comps.backtest.historical_data import get_point_in_time_snapshots, fetch_price_series, nearest_price
from valuation.comps.backtest.framework import _monthly_dates
from valuation.comps.backtest import analysis
from valuation.inputs.macro import IMPLIED_ERP

TAX_RATE = 0.21
HIGH_GROWTH_YEARS = 5
DEFAULT_TERMINAL_ROIC = 0.15  # matches growth.py's mature_tech industry average
MAX_INITIAL_GROWTH = 0.25
MIN_INITIAL_GROWTH = -0.10
MAX_REINVESTMENT = 0.80
CREDIT_SPREAD = 0.015  # flat spread over risk-free for a simplified cost of debt
BETA_LOOKBACK_DAYS = 730  # ~2 years of daily returns
MIN_RETURN_OBSERVATIONS = 60

_RISK_FREE_RATE_CACHE = {}


def compute_beta_from_price_series(stock_series, market_series, as_of_date, lookback_days=BETA_LOOKBACK_DAYS):
    """
    Beta = Cov(stock_returns, market_returns) / Var(market_returns), from
    daily returns over a trailing window ending at as_of_date, computed
    directly from yfinance price series (bypasses valuation/inputs/beta.py
    entirely, since that reads from the broken local price_history table).
    """
    window_start = as_of_date - datetime.timedelta(days=lookback_days)
    stock_window = sorted((d, p) for d, p in stock_series if window_start <= d <= as_of_date)
    if len(stock_window) < MIN_RETURN_OBSERVATIONS:
        return None
    market_by_date = {d: p for d, p in market_series if window_start <= d <= as_of_date}
    if len(market_by_date) < MIN_RETURN_OBSERVATIONS:
        return None

    stock_returns, market_returns = [], []
    for i in range(1, len(stock_window)):
        d0, p0 = stock_window[i - 1]
        d1, p1 = stock_window[i]
        if d0 not in market_by_date or d1 not in market_by_date or p0 == 0 or market_by_date[d0] == 0:
            continue
        stock_returns.append(p1 / p0 - 1)
        market_returns.append(market_by_date[d1] / market_by_date[d0] - 1)

    if len(stock_returns) < MIN_RETURN_OBSERVATIONS // 2:
        return None
    stock_returns, market_returns = np.array(stock_returns), np.array(market_returns)
    market_var = np.var(market_returns)
    if market_var == 0:
        return None
    return float(np.cov(stock_returns, market_returns)[0, 1] / market_var)


def compute_point_in_time_risk_free_rate(as_of_date):
    """10Y Treasury yield as of as_of_date, from FRED. Cached per-date (many tickers share the same rebalancing dates)."""
    key = as_of_date.isoformat()
    if key in _RISK_FREE_RATE_CACHE:
        return _RISK_FREE_RATE_CACHE[key]
    try:
        from valuation.data_sources.fred import _get_client
        client = _get_client()
        start = as_of_date - datetime.timedelta(days=14)
        series = client.get_series("DGS10", observation_start=start, observation_end=as_of_date)
        series = series.dropna()
        rate = float(series.iloc[-1]) / 100.0 if not series.empty else None
    except Exception:
        rate = None
    _RISK_FREE_RATE_CACHE[key] = rate
    return rate


def compute_point_in_time_wacc(snapshot, price_at_T, risk_free_rate, beta, erp=IMPLIED_ERP):
    """Simplified WACC: CAPM cost of equity + flat-spread after-tax cost of debt, weighted by point-in-time capital structure."""
    shares = snapshot["shares_outstanding"]
    if not shares or price_at_T is None or beta is None or risk_free_rate is None:
        return None
    market_cap = price_at_T * shares
    debt = snapshot["total_debt"] or 0.0
    total_capital = market_cap + debt
    if total_capital <= 0:
        return None
    equity_weight = market_cap / total_capital
    debt_weight = debt / total_capital
    cost_of_equity = risk_free_rate + beta * erp
    cost_of_debt_after_tax = (risk_free_rate + CREDIT_SPREAD) * (1 - TAX_RATE)
    return equity_weight * cost_of_equity + debt_weight * cost_of_debt_after_tax


def compute_point_in_time_dcf_fair_value(snapshot, wacc, risk_free_rate):
    """Simplified 2-stage DCF (5yr explicit + Gordon-growth terminal) from point-in-time fundamentals."""
    features = snapshot["features"]
    revenue, operating_margin = features.revenue_ttm, features.operating_margin
    if revenue is None or operating_margin is None or wacc is None or risk_free_rate is None:
        return None

    roic = features.roic if features.roic and features.roic > 0 else DEFAULT_TERMINAL_ROIC
    trailing_growth = features.trailing_growth if features.trailing_growth is not None else 0.05
    initial_growth = max(MIN_INITIAL_GROWTH, min(MAX_INITIAL_GROWTH, trailing_growth))
    terminal_growth = min(risk_free_rate, 0.03)
    if wacc <= terminal_growth:
        return None

    starting_nopat = operating_margin * revenue * (1 - TAX_RATE)
    if starting_nopat <= 0:
        return None

    reinvestment = min(initial_growth / roic, MAX_REINVESTMENT) if roic > 0 else 0.30
    current_nopat, pv_explicit = starting_nopat, 0.0
    for t in range(1, HIGH_GROWTH_YEARS + 1):
        current_nopat *= (1 + initial_growth)
        fcff = current_nopat * (1 - reinvestment)
        pv_explicit += fcff / ((1 + wacc) ** t)

    terminal_roic = max(DEFAULT_TERMINAL_ROIC, 0.4 * roic)
    terminal_reinvestment = min(terminal_growth / terminal_roic, MAX_REINVESTMENT) if terminal_roic > 0 else 0.30
    terminal_nopat = current_nopat * (1 + terminal_growth)
    terminal_fcff = terminal_nopat * (1 - terminal_reinvestment)
    terminal_value = terminal_fcff / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** HIGH_GROWTH_YEARS)

    firm_value = pv_explicit + pv_terminal
    equity_value = firm_value - (snapshot["total_debt"] or 0.0) + (snapshot["cash"] or 0.0)
    shares = snapshot["shares_outstanding"]
    if not shares:
        return None
    return equity_value / shares


def run_dcf_backtest(tickers, start_date, end_date, forward_windows=(6, 12), ticker_type_by_ticker=None):
    """
    Point-in-time DCF-approximation backtest. Mirrors framework.run_backtest's
    structure but WITHOUT a cross-sectional regression step — each ticker's
    fair value is computed independently at each date, so there's no
    MIN_OBSERVATIONS cross-section requirement the way the comps backtest has.
    """
    data_flags = []
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    max_fwd_months = max(forward_windows)
    price_buffer_start = start - datetime.timedelta(days=BETA_LOOKBACK_DAYS + 30)
    price_end = end + datetime.timedelta(days=32 * max_fwd_months)

    snapshots_by_ticker, prices_by_ticker = {}, {}
    for t in tickers:
        snaps, flags = get_point_in_time_snapshots(t)
        data_flags.extend(f"[{t}] {f}" for f in flags)
        snapshots_by_ticker[t] = snaps
        prices_by_ticker[t] = fetch_price_series(t, price_buffer_start, price_end)

    spy_prices = fetch_price_series("SPY", price_buffer_start, price_end)
    if not spy_prices:
        data_flags.append("Could not fetch SPY price series — beta cannot be computed, backtest cannot run")
        return {"rows": [], "data_flags": data_flags, "n_predictions": 0, "n_rebalancing_dates": 0}

    all_as_of = [s["as_of_date"] for snaps in snapshots_by_ticker.values() for s in snaps]
    if not all_as_of:
        data_flags.append("No point-in-time snapshots available for any ticker — backtest cannot run")
        return {"rows": [], "data_flags": data_flags, "n_predictions": 0, "n_rebalancing_dates": 0}

    available_start = max(start, min(all_as_of))
    if available_start > start:
        data_flags.append(f"Requested start {start_date} predates available fundamentals — clipped to {available_start.isoformat()}")

    rebal_dates = [d for d in _monthly_dates(available_start, end) if d <= end]

    rows = []
    n_skipped_no_beta, n_skipped_no_rf, n_skipped_no_fv = 0, 0, 0
    for T in rebal_dates:
        for t in tickers:
            snaps = snapshots_by_ticker[t]
            usable = [s for s in snaps if s["as_of_date"] <= T]
            if not usable:
                continue
            snapshot = usable[-1]

            price_T = nearest_price(prices_by_ticker[t], T)
            if price_T is None:
                continue

            beta = compute_beta_from_price_series(prices_by_ticker[t], spy_prices, T)
            if beta is None:
                n_skipped_no_beta += 1
                continue
            risk_free_rate = compute_point_in_time_risk_free_rate(T)
            if risk_free_rate is None:
                n_skipped_no_rf += 1
                continue

            wacc = compute_point_in_time_wacc(snapshot, price_T, risk_free_rate, beta)
            fair_value = compute_point_in_time_dcf_fair_value(snapshot, wacc, risk_free_rate)
            if fair_value is None or fair_value <= 0:
                n_skipped_no_fv += 1
                continue

            row = {"ticker": t, "date": T, "predicted_upside": fair_value / price_T - 1}
            for w in forward_windows:
                future_date = T + datetime.timedelta(days=30 * w)
                price_future = nearest_price(prices_by_ticker[t], future_date)
                if price_future is not None:
                    row[f"actual_return_{w}m"] = price_future / price_T - 1
            rows.append(row)

    data_flags.append(
        f"{len(rows)} (ticker, date) DCF predictions across {len(rebal_dates)} rebalancing dates "
        f"(skipped: {n_skipped_no_beta} no-beta, {n_skipped_no_rf} no-risk-free-rate, {n_skipped_no_fv} no-fair-value)"
    )

    overall_metrics = analysis.compute_all_metrics(rows, list(forward_windows))

    by_ticker_type = {}
    if ticker_type_by_ticker:
        types_present = {ticker_type_by_ticker[r["ticker"]] for r in rows if r["ticker"] in ticker_type_by_ticker}
        for tt in types_present:
            sub_rows = [r for r in rows if ticker_type_by_ticker.get(r["ticker"]) == tt]
            by_ticker_type[tt] = analysis.compute_all_metrics(sub_rows, list(forward_windows))
            by_ticker_type[tt]["n_predictions"] = len(sub_rows)

    return {
        "rows": rows, "overall_metrics": overall_metrics, "by_ticker_type": by_ticker_type,
        "n_rebalancing_dates": len(rebal_dates), "n_predictions": len(rows),
        "period_start": available_start.isoformat(), "period_end": end_date,
        "data_flags": data_flags,
    }


# ════════════════════════════════════════════════════════════════════════
# Sub-analysis groups, success criteria, verdict framework, and reporting
# ════════════════════════════════════════════════════════════════════════

PURE_MATURE_COMPOUNDERS = ["MSFT", "GOOGL", "AAPL", "META", "ORCL", "IBM", "CSCO"]
SAAS_GROWTH_ADJACENT = ["CRM", "ADBE", "NOW", "INTU", "WDAY", "TEAM", "DDOG", "SNOW", "PANW"]

TICKER_TYPE_MAP = {t: "pure_mature_compounder" for t in PURE_MATURE_COMPOUNDERS}
TICKER_TYPE_MAP.update({t: "saas_growth_adjacent" for t in SAAS_GROWTH_ADJACENT})


def _annualize_spread(spread, window_months):
    """
    Compound the window's top-minus-bottom spread up to a 12-month
    equivalent — a standard, approximate way to annualize a sub-annual
    return spread (treats it as if the same edge compounded each period).
    """
    if spread is None:
        return None
    if spread <= -1:
        return None
    periods_per_year = 12 / window_months
    return float((1 + spread) ** periods_per_year - 1)


def classify_verdict(correlation, spread_annualized, hit_rate):
    if correlation is None or spread_annualized is None or hit_rate is None:
        return "N/A (insufficient data)"
    if correlation > 0.15 and spread_annualized > 0.10 and hit_rate > 0.60:
        return "Strong signal"
    if 0.10 <= correlation <= 0.15 and 0.05 <= spread_annualized <= 0.10 and 0.55 <= hit_rate <= 0.60:
        return "Moderate signal"
    if correlation < 0.10 and spread_annualized < 0.05 and hit_rate < 0.55:
        return "No signal"
    return "Mixed (does not cleanly match Strong/Moderate/No-signal bands)"


def check_success_criteria(metrics, window):
    corr = metrics.get(f"correlation_{window}m")
    pval = metrics.get(f"correlation_{window}m_pvalue")
    spread = metrics.get(f"decile_spread_{window}m")
    spread_ann = _annualize_spread(spread, window)
    hit = metrics.get(f"hit_rate_{window}m")

    corr_pass = corr is not None and corr > 0.10 and pval is not None and pval < 0.05
    spread_pass = spread_ann is not None and spread_ann > 0.05
    hit_pass = hit is not None and hit > 0.55

    return {
        "correlation": corr, "p_value": pval, "correlation_pass": corr_pass,
        "decile_spread_raw": spread, "decile_spread_annualized": spread_ann, "spread_pass": spread_pass,
        "hit_rate": hit, "hit_pass": hit_pass,
        "all_pass": corr_pass and spread_pass and hit_pass,
        "verdict": classify_verdict(corr, spread_ann, hit),
    }


def print_metrics_report(label, metrics, n_predictions=None):
    print(f"\n{'-' * 90}\n{label}" + (f"  (n={n_predictions})" if n_predictions is not None else "") + f"\n{'-' * 90}")
    for window in (6, 12):
        r = check_success_criteria(metrics, window)
        print(f"  {window}m window:")
        print(f"    Correlation: {r['correlation']}  (p={r['p_value']})  "
              f"{'PASS' if r['correlation_pass'] else 'FAIL'} (need >0.10, p<0.05)")
        print(f"    Decile spread: raw={r['decile_spread_raw']}  annualized={r['decile_spread_annualized']}  "
              f"{'PASS' if r['spread_pass'] else 'FAIL'} (need >5% annualized)")
        print(f"    Hit rate: {r['hit_rate']}  {'PASS' if r['hit_pass'] else 'FAIL'} (need >55%)")
        print(f"    Sharpe: {metrics.get(f'sharpe_{window}m')}   R² (predicted vs actual): {metrics.get(f'r_squared_{window}m')}")
        print(f"    Overall success criteria: {'ALL PASS' if r['all_pass'] else 'NOT ALL PASS'}")
        print(f"    Verdict framework: {r['verdict']}")


def run_full_analysis():
    from valuation.comps.universe import COMPS_UNIVERSE

    tickers = COMPS_UNIVERSE["mature_tech"]
    print(f"Universe: {len(tickers)} mature_tech tickers")
    print(f"Pure mature compounders ({len(PURE_MATURE_COMPOUNDERS)}): {PURE_MATURE_COMPOUNDERS}")
    print(f"SaaS/growth-adjacent ({len(SAAS_GROWTH_ADJACENT)}): {SAAS_GROWTH_ADJACENT}")

    result = run_dcf_backtest(tickers, "2022-01-01", "2025-12-31", forward_windows=[6, 12], ticker_type_by_ticker=TICKER_TYPE_MAP)

    print(f"\nPeriod: {result['period_start']} -> {result['period_end']}")
    print(f"Rebalancing dates: {result['n_rebalancing_dates']}   Total predictions: {result['n_predictions']}")
    print("\nData flags (last 10):")
    for f in result["data_flags"][-10:]:
        print(f"  {f}")

    print_metrics_report("OVERALL — all mature_tech", result["overall_metrics"], result["n_predictions"])

    for tt_label, tt_key in [("PURE MATURE COMPOUNDERS", "pure_mature_compounder"), ("SAAS/GROWTH-ADJACENT", "saas_growth_adjacent")]:
        sub = result["by_ticker_type"].get(tt_key)
        if sub:
            print_metrics_report(tt_label, sub, sub.get("n_predictions"))
        else:
            print(f"\n{tt_label}: no data")

    return result


# ════════════════════════════════════════════════════════════════════════
# Semiconductor universe — same simplified DCF approximation, applied to
# semis to test whether the "DCF for mature, Comps for semis" framing
# holds up (or whether DCF just doesn't predict returns anywhere).
# run_dcf_backtest() above is fully generic (no mature_tech-specific
# logic), so this is pure reuse — no new backtest machinery.
# ════════════════════════════════════════════════════════════════════════

AI_INFRA_SEMIS = ["NVDA", "AMD", "AVGO", "MRVL", "QCOM"]
MEMORY_SEMIS = ["MU", "WDC", "STX"]
ANALOG_SEMIS = ["TXN", "ADI", "NXPI", "MCHP", "ON", "SWKS", "MPWR", "CRUS", "SLAB", "LSCC"]
EQUIPMENT_SEMIS = ["AMAT", "LRCX", "KLAC"]
FOUNDRY_SEMIS = ["TSM", "ASML"]

SEMI_TYPE_MAP = {}
for _t in AI_INFRA_SEMIS:
    SEMI_TYPE_MAP[_t] = "ai_infra"
for _t in MEMORY_SEMIS:
    SEMI_TYPE_MAP[_t] = "memory"
for _t in ANALOG_SEMIS:
    SEMI_TYPE_MAP[_t] = "analog"
for _t in EQUIPMENT_SEMIS:
    SEMI_TYPE_MAP[_t] = "equipment"
for _t in FOUNDRY_SEMIS:
    SEMI_TYPE_MAP[_t] = "foundry"

# This task's verdict framework is explicitly 2D (correlation + hit rate
# only) — no decile-spread threshold, unlike the mature_tech task's 3D
# version (classify_verdict() above). Implemented separately so it matches
# what was actually specified here, while decile spread/Sharpe are still
# reported for context.
def classify_verdict_2d(correlation, hit_rate):
    if correlation is None or hit_rate is None:
        return "N/A (insufficient data)"
    if correlation > 0.15 and hit_rate > 0.60:
        return "Strong signal"
    if 0.10 <= correlation <= 0.15 and 0.55 <= hit_rate <= 0.60:
        return "Moderate signal"
    if correlation < 0.10 and hit_rate < 0.55:
        return "No signal"
    return "Mixed (does not cleanly match Strong/Moderate/No-signal bands)"


def print_metrics_report_2d(label, metrics, n_predictions=None):
    print(f"\n{'-' * 90}\n{label}" + (f"  (n={n_predictions})" if n_predictions is not None else "") + f"\n{'-' * 90}")
    for window in (6, 12):
        corr = metrics.get(f"correlation_{window}m")
        pval = metrics.get(f"correlation_{window}m_pvalue")
        hit = metrics.get(f"hit_rate_{window}m")
        spread = metrics.get(f"decile_spread_{window}m")
        spread_ann = _annualize_spread(spread, window)
        verdict = classify_verdict_2d(corr, hit)
        print(f"  {window}m window:")
        print(f"    Correlation: {corr}  (p={pval})")
        print(f"    Hit rate: {hit}")
        print(f"    [context, not part of this task's verdict] Decile spread: raw={spread}  annualized={spread_ann}  "
              f"Sharpe: {metrics.get(f'sharpe_{window}m')}")
        print(f"    Verdict framework (correlation + hit rate only): {verdict}")


# Actual mature_tech backtest results, as recorded in this session's own
# output (NOT the "0.078 correlation, 62% hit rate" cited when this task
# was requested — that doesn't match what the mature_tech backtest
# actually produced; using the real numbers here rather than silently
# accepting an incorrect benchmark. See the earlier session's report:
# both windows FAILED the same 55%-hit-rate / 0.10-correlation bar).
MATURE_TECH_ACTUAL_RESULTS = {
    "overall": {
        "correlation_6m": -0.0197, "hit_rate_6m": 0.4677,
        "correlation_12m": 0.0316, "hit_rate_12m": 0.4767,
    },
    "pure_mature_compounders": {
        "correlation_6m": 0.1243, "hit_rate_6m": 0.3693,
        "correlation_12m": 0.1304, "hit_rate_12m": 0.2714,
    },
    "saas_growth_adjacent": {
        "correlation_6m": -0.3549, "hit_rate_6m": 0.4829,
        "correlation_12m": -0.4245, "hit_rate_12m": 0.4417,
    },
}


def run_semi_analysis():
    from valuation.comps.universe import get_universe

    tickers = get_universe("semiconductors")
    print(f"Universe: {len(tickers)} semiconductor tickers")
    print(f"  AI infra ({len(AI_INFRA_SEMIS)}): {AI_INFRA_SEMIS}")
    print(f"  Memory ({len(MEMORY_SEMIS)}): {MEMORY_SEMIS}")
    print(f"  Analog ({len(ANALOG_SEMIS)}): {ANALOG_SEMIS}")
    print(f"  Equipment ({len(EQUIPMENT_SEMIS)}): {EQUIPMENT_SEMIS}")
    print(f"  Foundry ({len(FOUNDRY_SEMIS)}): {FOUNDRY_SEMIS}")

    result = run_dcf_backtest(tickers, "2022-01-01", "2025-12-31", forward_windows=[6, 12], ticker_type_by_ticker=SEMI_TYPE_MAP)

    print(f"\nPeriod: {result['period_start']} -> {result['period_end']}")
    print(f"Rebalancing dates: {result['n_rebalancing_dates']}   Total predictions: {result['n_predictions']}")
    print("\nData flags (last 10):")
    for f in result["data_flags"][-10:]:
        print(f"  {f}")

    print_metrics_report_2d("OVERALL — all semiconductors", result["overall_metrics"], result["n_predictions"])

    for tt_label, tt_key in [("AI INFRA", "ai_infra"), ("MEMORY", "memory"), ("ANALOG", "analog"),
                              ("EQUIPMENT", "equipment"), ("FOUNDRY", "foundry")]:
        sub = result["by_ticker_type"].get(tt_key)
        if sub:
            print_metrics_report_2d(tt_label, sub, sub.get("n_predictions"))
        else:
            print(f"\n{tt_label}: no data")

    print(f"\n{'=' * 100}\nCOMPARISON TO MATURE_TECH DCF RESULTS (actual, from this session)\n{'=' * 100}")
    semi_overall = result["overall_metrics"]
    mt = MATURE_TECH_ACTUAL_RESULTS["overall"]
    print(f"  {'':<20}{'Semis 6m':>12}{'MatureTech 6m':>16}{'Semis 12m':>12}{'MatureTech 12m':>16}")
    print(f"  {'Correlation':<20}{semi_overall.get('correlation_6m'):>12.3f}{mt['correlation_6m']:>16.3f}"
          f"{semi_overall.get('correlation_12m'):>12.3f}{mt['correlation_12m']:>16.3f}")
    print(f"  {'Hit rate':<20}{semi_overall.get('hit_rate_6m'):>12.3f}{mt['hit_rate_6m']:>16.3f}"
          f"{semi_overall.get('hit_rate_12m'):>12.3f}{mt['hit_rate_12m']:>16.3f}")

    return result


# ════════════════════════════════════════════════════════════════════════
# Combined Layer 1 (DCF) + Layer 2 (unconditional peer EV/EBITDA premium)
# signal test — does layering add value over either alone? Reuses this
# module's point-in-time DCF machinery unchanged (compute_beta_from_price_series,
# compute_point_in_time_risk_free_rate, compute_point_in_time_wacc,
# compute_point_in_time_dcf_fair_value) plus get_point_in_time_snapshots /
# fetch_price_series / nearest_price / _monthly_dates, same as every other
# backtest in this module. The Layer 2 signal here is a raw peer-premium
# ratio, NOT the mature_relative.py regression model — deliberately
# simpler and untested until now, per the task's own framing.
# ════════════════════════════════════════════════════════════════════════

import statistics as _statistics


def run_combined_signal_backtest(tickers, start_date, end_date, forward_window=12):
    """
    At each rebalancing date T, for each ticker with both signals
    computable:
        Signal A: DCF upside = (dcf_fair_value - price_T) / price_T
        Signal B: -peer_premium, where peer_premium =
                  (ticker_ev_ebitda - bucket_median_ev_ebitda) / bucket_median_ev_ebitda
                  (negated so "trading below peers" reads as positive, same
                  sign convention as Signal A's "upside")
        Signal C: mean(A, B)
        Signal D: A, kept only when sign(A) == sign(B) (peer agreement);
                  None otherwise (excluded from that signal's metrics)

    Point-in-time throughout: bucket_median is recomputed from the T-dated
    cross-section every date, not a fixed/live value.
    """
    data_flags = []
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)

    snapshots_by_ticker, prices_by_ticker = {}, {}
    for t in tickers:
        snaps, flags = get_point_in_time_snapshots(t)
        data_flags.extend(f"[{t}] {f}" for f in flags)
        snapshots_by_ticker[t] = snaps
        price_start = start - datetime.timedelta(days=BETA_LOOKBACK_DAYS + 30)
        price_end = end + datetime.timedelta(days=32 * forward_window)
        prices_by_ticker[t] = fetch_price_series(t, price_start, price_end)

    spy_prices = fetch_price_series("SPY", start - datetime.timedelta(days=BETA_LOOKBACK_DAYS + 30),
                                     end + datetime.timedelta(days=32 * forward_window))
    if not spy_prices:
        data_flags.append("Could not fetch SPY price series — cannot run")
        return {"master_rows": [], "data_flags": data_flags, "n_predictions": 0}

    all_as_of = [s["as_of_date"] for snaps in snapshots_by_ticker.values() for s in snaps]
    if not all_as_of:
        data_flags.append("No point-in-time snapshots available for any ticker")
        return {"master_rows": [], "data_flags": data_flags, "n_predictions": 0}

    available_start = max(start, min(all_as_of))
    if available_start > start:
        data_flags.append(f"Requested start {start_date} predates available fundamentals — clipped to {available_start.isoformat()}")

    rebal_dates = [d for d in _monthly_dates(available_start, end) if d <= end]

    master_rows = []
    n_skipped_no_dcf, n_skipped_no_peer, n_skipped_no_future = 0, 0, 0

    for T in rebal_dates:
        cross_section = {}
        for t in tickers:
            usable = [s for s in snapshots_by_ticker[t] if s["as_of_date"] <= T]
            if usable:
                cross_section[t] = usable[-1]

        # point-in-time EV/EBITDA per ticker, for the peer-premium bucket median
        ev_ebitda_by_ticker, price_by_ticker_at_T = {}, {}
        for t, snap in cross_section.items():
            price_T = nearest_price(prices_by_ticker[t], T)
            shares, ebitda = snap["shares_outstanding"], snap["features"].ebitda_ttm
            if price_T is None or not shares or not ebitda or ebitda <= 0:
                continue
            ev = price_T * shares + (snap["total_debt"] or 0) - (snap["cash"] or 0)
            if ev / ebitda > 0:
                ev_ebitda_by_ticker[t] = ev / ebitda
                price_by_ticker_at_T[t] = price_T

        if len(ev_ebitda_by_ticker) < 2:
            n_skipped_no_peer += len(cross_section)
            continue
        bucket_median = _statistics.median(ev_ebitda_by_ticker.values())
        if bucket_median == 0:
            n_skipped_no_peer += len(cross_section)
            continue

        for t, snap in cross_section.items():
            if t not in ev_ebitda_by_ticker:
                n_skipped_no_peer += 1
                continue
            price_T = price_by_ticker_at_T[t]
            peer_premium = (ev_ebitda_by_ticker[t] - bucket_median) / bucket_median
            signal_b = -peer_premium

            beta = compute_beta_from_price_series(prices_by_ticker[t], spy_prices, T)
            risk_free_rate = compute_point_in_time_risk_free_rate(T)
            dcf_upside = None
            if beta is not None and risk_free_rate is not None:
                wacc = compute_point_in_time_wacc(snap, price_T, risk_free_rate, beta)
                fair_value = compute_point_in_time_dcf_fair_value(snap, wacc, risk_free_rate)
                if fair_value is not None and fair_value > 0:
                    dcf_upside = fair_value / price_T - 1

            if dcf_upside is None:
                n_skipped_no_dcf += 1
                continue

            future_date = T + datetime.timedelta(days=30 * forward_window)
            price_future = nearest_price(prices_by_ticker[t], future_date)
            if price_future is None:
                n_skipped_no_future += 1
                continue
            actual_return = price_future / price_T - 1

            signal_c = (dcf_upside + signal_b) / 2
            agree = (dcf_upside > 0) == (signal_b > 0)
            signal_d = dcf_upside if agree else None

            master_rows.append({
                "ticker": t, "date": T,
                "signal_a": dcf_upside, "signal_b": signal_b, "signal_c": signal_c,
                "signal_d": signal_d, "agree": agree,
                f"actual_return_{forward_window}m": actual_return,
            })

    n_agree = sum(1 for r in master_rows if r["agree"])
    data_flags.append(
        f"{len(master_rows)} (ticker, date) observations with both signals across "
        f"{len(rebal_dates)} candidate rebalancing dates (skipped: {n_skipped_no_peer} no-peer-premium, "
        f"{n_skipped_no_dcf} no-DCF, {n_skipped_no_future} no-forward-price); "
        f"{n_agree} of {len(master_rows)} ({100 * n_agree / len(master_rows):.0f}%) had peer agreement"
    )

    return {
        "master_rows": master_rows, "data_flags": data_flags, "n_predictions": len(master_rows),
        "n_agree": n_agree, "period_start": available_start.isoformat(), "period_end": end_date,
        "forward_window": forward_window,
    }


def _rows_for_signal(master_rows, signal_key, forward_window):
    rows = []
    for r in master_rows:
        v = r.get(signal_key)
        if v is None:
            continue
        actual_key = f"actual_return_{forward_window}m"
        if actual_key not in r:
            continue
        rows.append({"ticker": r["ticker"], "date": r["date"], "predicted_upside": v, actual_key: r[actual_key]})
    return rows


def print_signal_metrics(label, metrics, window, n):
    corr = metrics.get(f"correlation_{window}m")
    pval = metrics.get(f"correlation_{window}m_pvalue")
    hit = metrics.get(f"hit_rate_{window}m")
    spread = metrics.get(f"decile_spread_{window}m")
    sharpe = metrics.get(f"sharpe_{window}m")
    print(f"  {label:<45} n={n:<6} corr={_fmt(corr):>8}  p={_fmt(pval):>8}  hit_rate={_fmt(hit):>8}  "
          f"decile_spread={_fmt(spread):>8}  sharpe={_fmt(sharpe):>8}")
    return corr, hit, spread, sharpe


def _fmt(x):
    return f"{x:.3f}" if x is not None else "N/A"


def run_combined_signal_analysis():
    from valuation.comps.universe import COMPS_UNIVERSE

    tickers = COMPS_UNIVERSE["mature_tech"]
    window = 12
    print(f"Universe: {len(tickers)} mature_tech tickers, forward_window={window}m")

    result = run_combined_signal_backtest(tickers, "2022-01-01", "2025-12-31", forward_window=window)
    print(f"\nPeriod: {result.get('period_start')} -> {result.get('period_end')}")
    print("\nData flags:")
    for f in result["data_flags"]:
        print(f"  {f}")

    master_rows = result["master_rows"]
    if not master_rows:
        print("\nNo observations produced — cannot compute metrics.")
        return result

    print(f"\n{'=' * 130}\nSIGNAL COMPARISON — 12-month forward returns\n{'=' * 130}")
    signal_results = {}
    for label, key in [
        ("Signal A: DCF upside alone (baseline)", "signal_a"),
        ("Signal B: peer premium alone (negated)", "signal_b"),
        ("Signal C: average(A, B)", "signal_c"),
        ("Signal D: A, filtered to peer agreement", "signal_d"),
    ]:
        rows = _rows_for_signal(master_rows, key, window)
        metrics = analysis.compute_all_metrics(rows, [window])
        corr, hit, spread, sharpe = print_signal_metrics(label, metrics, window, len(rows))
        signal_results[key] = {"metrics": metrics, "n": len(rows), "correlation": corr, "hit_rate": hit, "spread": spread, "sharpe": sharpe}

    print(f"\n{'=' * 130}\nVERDICT\n{'=' * 130}")
    a, b, c, d = signal_results["signal_a"], signal_results["signal_b"], signal_results["signal_c"], signal_results["signal_d"]
    SIGNIFICANT_MARGIN = 0.03  # correlation-points; a threshold this task doesn't specify numerically — documented here rather than left implicit

    def _describe(name, other, baseline):
        if other["correlation"] is None or baseline["correlation"] is None:
            return f"{name}: N/A (insufficient data)"
        delta = other["correlation"] - baseline["correlation"]
        if delta > SIGNIFICANT_MARGIN:
            return f"{name} correlation {other['correlation']:.3f} vs baseline {baseline['correlation']:.3f} (Δ={delta:+.3f}) — BEATS baseline by >{SIGNIFICANT_MARGIN}"
        elif delta < -SIGNIFICANT_MARGIN:
            return f"{name} correlation {other['correlation']:.3f} vs baseline {baseline['correlation']:.3f} (Δ={delta:+.3f}) — WORSE than baseline by >{SIGNIFICANT_MARGIN}"
        else:
            return f"{name} correlation {other['correlation']:.3f} vs baseline {baseline['correlation']:.3f} (Δ={delta:+.3f}) — approximately EQUAL (within ±{SIGNIFICANT_MARGIN})"

    print(f"  {_describe('Signal C (combined average)', c, a)}")
    print(f"  {_describe('Signal D (agreement-filtered)', d, a)}")
    print(f"  Signal B (peer premium alone): correlation={_fmt(b['correlation'])}, hit_rate={_fmt(b['hit_rate'])} — "
          f"{'has some signal' if (b['correlation'] or 0) > 0.10 else 'no meaningful signal on its own'}")
    print(f"  Signal D coverage: {result['n_agree']} of {result['n_predictions']} observations "
          f"({100 * result['n_agree'] / result['n_predictions']:.0f}%) had peer agreement — "
          f"a filter this restrictive trades away most of the data even if its correlation looks better")

    return {"result": result, "signal_results": signal_results}


if __name__ == "__main__":
    run_full_analysis()
