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
import math

import yfinance as yf

from valuation.comps.features import CompanyFeatures, _get_field

REPORTING_LAG_DAYS = 90

# --- Constants for the point-in-time Category A feature reconstructions
# below. Mirror valuation/tech/shared/{ex_goodwill,rd_capitalization,
# sbc_dilution}.py and valuation/inputs/growth.py::compute_fundamental_growth,
# but reimplemented against column i (the as-of-that-snapshot year) instead
# of column 0 (today) — those Layer 1/2 modules are live-only (hardcoded
# .iloc[:, 0]) and are not modified per the alpha-layer spec's constraint.
RD_USEFUL_LIFE = 5
SBC_LOOKBACK_YEARS = 3
SBC_MAX_ANNUAL_RATE = 0.05
REINVESTMENT_MAX_YEARS = 5


def _column_date(df, i):
    label = df.columns[i]
    return label.date() if hasattr(label, "date") else label


def _goodwill_at(balance_col):
    """Goodwill for one balance-sheet column, with the same fallback
    ex_goodwill.py uses (combined intangibles line when goodwill isn't
    separately reported)."""
    goodwill = _get_field(balance_col, ["Goodwill"])
    if goodwill is None:
        goodwill = _get_field(balance_col, ["Goodwill And Other Intangible Assets"])
    return goodwill or 0.0


def _compute_ex_goodwill_at(operating_income, tax_rate, equity, debt, cash, goodwill):
    """Point-in-time ex-goodwill ROIC + goodwill ratio for one column.
    Returns (roic_ex_goodwill, goodwill_ratio), either None if undefined."""
    if operating_income is None or operating_income <= 0 or equity is None:
        return None, None
    invested_capital = equity + debt - cash
    if invested_capital <= 0:
        return None, None
    goodwill_ratio = goodwill / invested_capital
    invested_capital_ex_goodwill = invested_capital - goodwill
    if invested_capital_ex_goodwill <= 0:
        return None, goodwill_ratio
    nopat = operating_income * (1 - tax_rate)
    return nopat / invested_capital_ex_goodwill, goodwill_ratio


def _compute_rd_capitalized_margin_at(income_stmt, i, n_years, tax_rate, revenue, ebit_i,
                                       useful_life=RD_USEFUL_LIFE):
    """
    Point-in-time R&D-capitalized NOPAT / revenue for snapshot column i.
    Same Damodaran capitalization formula as rd_capitalization.py, but the
    R&D history window is [i, i+useful_life) — years at-or-before this
    snapshot's fiscal year — instead of [0, useful_life).
    """
    if "Research And Development" not in income_stmt.index or ebit_i is None or not revenue:
        return None
    rd_row = income_stmt.loc["Research And Development"]
    window = [k for k in range(useful_life) if i + k < n_years]
    if not window:
        return None
    # oldest-to-newest, matching rd_capitalization.py's convention
    rd_history = []
    for k in reversed(window):
        val = rd_row.iloc[i + k]
        if val is not None and not (isinstance(val, float) and math.isnan(val)) and float(val) > 0:
            rd_history.append(float(val))
    if not rd_history:
        return None

    rd_asset = 0.0
    for idx, rd in enumerate(rd_history):
        years_ago = (len(rd_history) - 1) - idx
        remaining_life_fraction = max(0, useful_life - years_ago - 1) / useful_life
        rd_asset += rd * remaining_life_fraction

    rd_amortization = 0.0
    for idx, rd in enumerate(rd_history):
        years_ago = (len(rd_history) - 1) - idx
        if years_ago < useful_life:
            rd_amortization += rd / useful_life

    current_rd = rd_history[-1]
    adjusted_ebit = ebit_i + current_rd - rd_amortization
    adjusted_nopat = adjusted_ebit * (1 - tax_rate)
    return adjusted_nopat / revenue


def _compute_sbc_dilution_rate_at(income_stmt, i, n_years, lookback=SBC_LOOKBACK_YEARS,
                                   max_rate=SBC_MAX_ANNUAL_RATE):
    """
    Point-in-time share-count CAGR for snapshot column i, capped at ±max_rate.
    Same formula as sbc_dilution.py's historical_dilution_rate, but measured
    over [i, i+lookback] (this snapshot's year back to `lookback` years
    earlier) instead of [0, lookback] (today back to `lookback` years ago).
    """
    share_row = None
    for field in ["Diluted Average Shares", "Basic Average Shares"]:
        if field in income_stmt.index:
            share_row = income_stmt.loc[field]
            break
    if share_row is None:
        return None
    actual_lookback = min(lookback, n_years - 1 - i)
    if actual_lookback < 1:
        return None
    newest = share_row.iloc[i]
    oldest = share_row.iloc[i + actual_lookback]
    if newest is None or oldest is None:
        return None
    if isinstance(newest, float) and math.isnan(newest):
        return None
    if isinstance(oldest, float) and math.isnan(oldest):
        return None
    newest, oldest = float(newest), float(oldest)
    if newest <= 0 or oldest <= 0:
        return None
    raw_rate = (newest / oldest) ** (1.0 / actual_lookback) - 1.0
    return max(-max_rate, min(max_rate, raw_rate))


def _compute_normalized_reinvestment_at(income_stmt, balance_sheet, cash_flow, i, tax_rate,
                                         max_years=REINVESTMENT_MAX_YEARS):
    """
    Point-in-time normalized reinvestment rate for snapshot column i: same
    max(median, weighted_3y) formula as growth.py::compute_fundamental_growth
    (ΔWC excluded — reinvestment = capex - depreciation only), but the
    per-year history window is [i, i+max_years) instead of [0, max_years).
    """
    n_income = income_stmt.shape[1]
    n_cash = cash_flow.shape[1] if cash_flow is not None else 0
    n_balance = balance_sheet.shape[1] if balance_sheet is not None else 0
    max_available = min(n_income, n_cash, n_balance) - i
    if max_available < 1:
        return None
    n_years_to_use = min(max_years, max_available)

    reinvestment_rates = []
    for year_offset in range(n_years_to_use):
        col_idx = i + year_offset
        try:
            income_y = income_stmt.iloc[:, col_idx]
            cash_y = cash_flow.iloc[:, col_idx]
        except Exception:
            continue
        ebit_y = _get_field(income_y, ["EBIT", "Operating Income"])
        if ebit_y is None or ebit_y <= 0:
            continue
        nopat_y = ebit_y * (1 - tax_rate)

        capex_y = _get_field(cash_y, ["Capital Expenditure"]) or 0.0
        depreciation_y = _get_field(cash_y, ["Depreciation And Amortization"]) or 0.0
        reinvestment_y = max(0.0, abs(capex_y) - depreciation_y)
        reinvestment_rates.append(reinvestment_y / nopat_y)

    if not reinvestment_rates:
        return None
    if len(reinvestment_rates) < 3:
        return sum(reinvestment_rates) / len(reinvestment_rates)

    sorted_rates = sorted(reinvestment_rates)
    n = len(sorted_rates)
    median_rate = (sorted_rates[n // 2] if n % 2 == 1
                   else (sorted_rates[n // 2 - 1] + sorted_rates[n // 2]) / 2)
    weighted_avg = 0.5 * reinvestment_rates[0] + 0.3 * reinvestment_rates[1] + 0.2 * reinvestment_rates[2]
    return max(median_rate, weighted_avg)


def get_point_in_time_snapshots(ticker):
    """
    Build point-in-time feature snapshots from yfinance annual statements,
    one per available fiscal year, oldest first.

    Returns (snapshots, data_flags). Each snapshot is a dict:
        as_of_date, fiscal_year_end, features (CompanyFeatures),
        total_debt, cash, shares_outstanding,
        goodwill, roic_ex_goodwill, goodwill_ratio, rd_capitalized_margin,
        sbc_dilution_rate, normalized_reinvestment (Category A alpha-layer
        features; None where not computable for that ticker/year)
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

        # --- Category A features for the alpha research layer (see
        # ALPHA_LAYER_SPEC.md). Reconstructed point-in-time from the same
        # income_stmt/balance_sheet/cash_flow already fetched above for
        # this ticker — no new fetching. None where the underlying data or
        # preconditions aren't available (e.g. no R&D line, insufficient
        # trailing years); consumers should treat None as "excluded", not 0.
        goodwill = None
        roic_ex_goodwill = None
        goodwill_ratio = None
        if i < balance_sheet.shape[1]:
            goodwill = _goodwill_at(balance_sheet.iloc[:, i])
            roic_ex_goodwill, goodwill_ratio = _compute_ex_goodwill_at(
                operating_income, tax_rate, equity, debt_snapshot, cash_snapshot, goodwill,
            )
        rd_capitalized_margin = _compute_rd_capitalized_margin_at(
            income_stmt, i, n_years, tax_rate, revenue, operating_income,
        )
        sbc_dilution_rate = _compute_sbc_dilution_rate_at(income_stmt, i, n_years)
        normalized_reinvestment = _compute_normalized_reinvestment_at(
            income_stmt, balance_sheet, cash_flow, i, tax_rate,
        )

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
            # Category A (alpha-layer) additions — see comment above.
            "goodwill": goodwill,
            "roic_ex_goodwill": roic_ex_goodwill,
            "goodwill_ratio": goodwill_ratio,
            "rd_capitalized_margin": rd_capitalized_margin,
            "sbc_dilution_rate": sbc_dilution_rate,
            "normalized_reinvestment": normalized_reinvestment,
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
