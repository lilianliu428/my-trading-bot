"""
Point-in-time data fetching for the comps backtest.

Because yfinance only exposes ~5 years of annual statements, computing
trailing_growth as a strict 3-year CAGR (features.py's rule for LIVE
valuations) would leave only 1-2 usable historical snapshots per ticker.
For the backtest specifically, trailing_growth instead uses whatever
window of years is actually available (down to 1 year), flagged with the
window used — a deliberately looser rule than live feature computation,
justified by the backtest's lower-fidelity, sanity-check purpose (see
LAYER_2_COMPS_SPEC.md "Testing Requirements" #3).

Two known approximations, both flagged in output:
    - "As of" date = fiscal-year-end + REPORTING_LAG_DAYS, approximating
      when the 10-K became public (avoids look-ahead bias). yfinance
      doesn't expose actual filing dates.
    - Shares outstanding: yfinance has no historical share count, so every
      snapshot for a ticker uses its CURRENT share count. This affects only
      the market-cap side of the point-in-time EV multiple, not the
      per-share fair value ratio in the same way it would a real backtest.
"""

import datetime

import yfinance as yf

from valuation.comps.features import CompanyFeatures, _get_field

REPORTING_LAG_DAYS = 90


def _column_date(df, i):
    label = df.columns[i]
    return label.date() if hasattr(label, "date") else label


def get_point_in_time_snapshots(ticker):
    """
    Build point-in-time feature snapshots from yfinance annual statements,
    one per available fiscal year, oldest first.

    Returns (snapshots, data_flags). Each snapshot is a dict:
        as_of_date, fiscal_year_end, features (CompanyFeatures),
        total_debt, cash, shares_outstanding
    """
    data_flags = []
    yf_ticker = yf.Ticker(ticker)

    try:
        currency = yf_ticker.info.get("financialCurrency", "USD")
    except Exception:
        currency = "USD"
    if currency != "USD":
        return [], [f"Non-USD reporting currency ({currency}) — skipped"]

    income_stmt = yf_ticker.income_stmt
    balance_sheet = yf_ticker.balance_sheet
    cash_flow = yf_ticker.cashflow

    if income_stmt is None or income_stmt.empty or balance_sheet is None or balance_sheet.empty:
        return [], ["Missing financial statements"]

    n_years = income_stmt.shape[1]
    if n_years < 2:
        return [], [f"Only {n_years} year(s) of income statement history — no usable snapshots"]

    shares_outstanding = yf_ticker.info.get("sharesOutstanding")
    if shares_outstanding:
        data_flags.append("All snapshots use CURRENT shares outstanding (yfinance has no historical share count)")

    tax_rate = yf_ticker.info.get("effectiveTaxRate") or 0.21

    snapshots = []
    for i in range(n_years):
        col_income = income_stmt.iloc[:, i]
        revenue = _get_field(col_income, ["Total Revenue", "Revenue", "Operating Revenue"])
        operating_income = _get_field(col_income, ["Operating Income", "EBIT"])
        if revenue is None or revenue <= 0:
            continue

        trailing_growth = None
        growth_window_years = None
        for back in (3, 2, 1):
            if i + back < n_years:
                revenue_back = _get_field(income_stmt.iloc[:, i + back], ["Total Revenue", "Revenue", "Operating Revenue"])
                if revenue_back is not None and revenue_back > 0:
                    trailing_growth = (revenue / revenue_back) ** (1 / back) - 1
                    growth_window_years = back
                    break

        operating_margin = operating_income / revenue if operating_income is not None else None

        roic = None
        debt_snapshot = 0.0
        cash_snapshot = 0.0
        if i < balance_sheet.shape[1]:
            col_balance = balance_sheet.iloc[:, i]
            equity = _get_field(col_balance, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"])
            debt_snapshot = _get_field(col_balance, ["Total Debt"]) or 0.0
            cash_snapshot = _get_field(col_balance, ["Cash And Cash Equivalents", "Cash"]) or 0.0
            if equity is not None and operating_income is not None and operating_income > 0:
                invested_capital = equity + debt_snapshot - cash_snapshot
                if invested_capital > 0:
                    roic = (operating_income * (1 - tax_rate)) / invested_capital

        import math
        log_revenue = math.log(revenue)

        ebitda = None
        if operating_income is not None and cash_flow is not None and not cash_flow.empty and i < cash_flow.shape[1]:
            da = _get_field(cash_flow.iloc[:, i], ["Depreciation And Amortization", "Depreciation"])
            if da is not None and (operating_income + da) > 0:
                ebitda = operating_income + da

        # --- extra features semi_forest.py's validated model needs, beyond
        # the original 4 this function was built for. Same formulas as
        # feature_selection.py's compute_extended_features_for_ticker(),
        # applied to this column instead of column 0.
        gross_profit = _get_field(col_income, ["Gross Profit"])
        gross_margin = gross_profit / revenue if gross_profit is not None else None

        rnd = _get_field(col_income, ["Research And Development"])
        r_and_d_intensity = rnd / revenue if rnd is not None else None

        ebitda_margin = ebitda / revenue if ebitda is not None else None

        # growth_stability: yoy pairs at or before column i only (never a
        # more-recent column), matching trailing_growth's look-backward-only
        # discipline so this stays point-in-time / avoids look-ahead bias.
        growth_stability = None
        yoy_growth = []
        for j in range(i, n_years - 1):
            r_new = _get_field(income_stmt.iloc[:, j], ["Total Revenue", "Revenue", "Operating Revenue"])
            r_old = _get_field(income_stmt.iloc[:, j + 1], ["Total Revenue", "Revenue", "Operating Revenue"])
            if r_new is not None and r_old is not None and r_old > 0:
                yoy_growth.append(r_new / r_old - 1)
        if len(yoy_growth) >= 2:
            import numpy as np
            mean_g, std_g = float(np.mean(yoy_growth)), float(np.std(yoy_growth))
            if mean_g != 0:
                growth_stability = 1 - std_g / mean_g

        snap_flags = []
        if growth_window_years is not None and growth_window_years != 3:
            snap_flags.append(f"trailing_growth uses a {growth_window_years}yr window, not the standard 3yr (limited history at this point in time)")

        features = CompanyFeatures(
            ticker=ticker,
            trailing_growth=trailing_growth,
            operating_margin=operating_margin,
            roic=roic,
            log_revenue=log_revenue,
            revenue_ttm=revenue,
            ebitda_ttm=ebitda,
            data_flags=snap_flags,
            growth_stability=growth_stability,
            gross_margin=gross_margin,
            ebitda_margin=ebitda_margin,
            r_and_d_intensity=r_and_d_intensity,
        )

        fiscal_year_end = _column_date(income_stmt, i)
        as_of_date = fiscal_year_end + datetime.timedelta(days=REPORTING_LAG_DAYS)

        snapshots.append({
            "as_of_date": as_of_date,
            "fiscal_year_end": fiscal_year_end,
            "features": features,
            "total_debt": debt_snapshot,
            "cash": cash_snapshot,
            "shares_outstanding": shares_outstanding,
        })

    snapshots.sort(key=lambda s: s["as_of_date"])
    return snapshots, data_flags


def fetch_price_series(ticker, start_date, end_date):
    """
    Fetch daily closes for [start_date, end_date] ONCE, cached as a list of
    (date, close) tuples for repeated nearest-date lookups — avoids one
    network call per rebalancing date.
    """
    hist = yf.Ticker(ticker).history(
        start=start_date.isoformat(), end=end_date.isoformat(), auto_adjust=True
    )
    if hist is None or hist.empty:
        return []
    return [
        (ts.date() if hasattr(ts, "date") else ts, float(close))
        for ts, close in hist["Close"].items()
    ]


def nearest_price(price_series, date, max_gap_days=10):
    """Nearest available close to `date` within max_gap_days, else None."""
    if not price_series:
        return None
    best_date, best_price = min(price_series, key=lambda p: abs((p[0] - date).days))
    if abs((best_date - date).days) > max_gap_days:
        return None
    return best_price
