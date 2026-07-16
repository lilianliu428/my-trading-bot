"""
Comps multiples computation — EV/Sales and EV/EBITDA, the regression targets.

Also owns winsorize(), used both here (on raw observed multiples) and by
regression.py (on each feature independently) — see LAYER_2_COMPS_SPEC.md Q4.
"""

import math

import numpy as np
import yfinance as yf


def _get_field(row, field_names):
    for name in field_names:
        if name in row.index:
            val = row[name]
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return float(val)
    return None


def compute_ev_components(ticker):
    """
    Fetch enterprise-value building blocks from yfinance, using the same
    fields as valuation/inputs/wacc.py's get_capital_structure() for
    consistency with Layer 1 (same shares_outstanding, same current_price).

    Returns dict with: market_cap, total_debt, cash, ev, shares_outstanding,
    current_price, data_flags. Any missing component is None and ev is None.
    """
    data_flags = []
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info

    market_cap = info.get("marketCap")
    total_debt = info.get("totalDebt")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares_outstanding = info.get("sharesOutstanding")

    cash = None
    balance_sheet = yf_ticker.balance_sheet
    if balance_sheet is not None and not balance_sheet.empty:
        cash = _get_field(balance_sheet.iloc[:, 0], ["Cash And Cash Equivalents", "Cash"])
    if cash is None:
        data_flags.append("No cash balance found — treating as 0 for EV")
        cash = 0.0

    if total_debt is None:
        data_flags.append("No total debt found — treating as 0 for EV")
        total_debt = 0.0

    ev = None
    if market_cap is not None:
        ev = market_cap + total_debt - cash
    else:
        data_flags.append("No market cap — EV unavailable")

    return {
        "market_cap": market_cap,
        "total_debt": total_debt,
        "cash": cash,
        "ev": ev,
        "shares_outstanding": shares_outstanding,
        "current_price": current_price,
        "data_flags": data_flags,
    }


def compute_multiples_for_ticker(ticker):
    """
    Compute observed EV/Sales and EV/EBITDA from yfinance.
    Returns {"ev_sales": float | None, "ev_ebitda": float | None, "data_flags": list}
    """
    data_flags = []
    ev_components = compute_ev_components(ticker)
    data_flags.extend(ev_components["data_flags"])
    ev = ev_components["ev"]

    if ev is None:
        data_flags.append(f"EV unavailable for {ticker} — no multiples computable")
        return {"ev_sales": None, "ev_ebitda": None, "data_flags": data_flags}

    yf_ticker = yf.Ticker(ticker)
    income_stmt = yf_ticker.income_stmt
    cash_flow = yf_ticker.cashflow

    if income_stmt is None or income_stmt.empty:
        data_flags.append("No income statement — no multiples computable")
        return {"ev_sales": None, "ev_ebitda": None, "data_flags": data_flags}

    latest_income = income_stmt.iloc[:, 0]
    revenue_ttm = _get_field(latest_income, ["Total Revenue", "Revenue", "Operating Revenue"])
    operating_income_ttm = _get_field(latest_income, ["Operating Income", "EBIT"])

    ev_sales = None
    if revenue_ttm is not None and revenue_ttm > 0:
        ev_sales = ev / revenue_ttm
    else:
        data_flags.append("No usable TTM revenue — ev_sales unavailable")

    ev_ebitda = None
    if operating_income_ttm is not None and cash_flow is not None and not cash_flow.empty:
        da_ttm = _get_field(cash_flow.iloc[:, 0], ["Depreciation And Amortization", "Depreciation"])
        if da_ttm is not None:
            ebitda_ttm = operating_income_ttm + da_ttm
            if ebitda_ttm > 0:
                ev_ebitda = ev / ebitda_ttm
            else:
                data_flags.append(f"EBITDA non-positive ({ebitda_ttm:,.0f}) — ev_ebitda unavailable")
        else:
            data_flags.append("No D&A data — ev_ebitda unavailable")
    elif operating_income_ttm is None:
        data_flags.append("No operating income — ev_ebitda unavailable")

    return {"ev_sales": ev_sales, "ev_ebitda": ev_ebitda, "data_flags": data_flags}


def winsorize(values, lower=0.01, upper=0.99):
    """
    Cap values at percentile bounds. Prevents extreme outliers from warping
    the regression. Apply independently per feature and per multiple.
    """
    values = np.asarray(values, dtype=float)
    lo = np.percentile(values, lower * 100)
    hi = np.percentile(values, upper * 100)
    return np.clip(values, lo, hi)


if __name__ == "__main__":
    for t in ["AMD", "MSFT", "NVDA"]:
        m = compute_multiples_for_ticker(t)
        print(f"\n{t}: ev_sales={m['ev_sales']}  ev_ebitda={m['ev_ebitda']}")
        if m["data_flags"]:
            print(f"  flags: {m['data_flags']}")
