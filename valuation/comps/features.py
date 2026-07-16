"""
Comps feature computation — the regression right-hand-side variables.

Four features drive both the EV/Sales and EV/EBITDA regressions:
    trailing_growth   — 3-year revenue CAGR
    operating_margin  — TTM operating income / TTM revenue
    roic              — reused from the Layer 1 growth module for consistency
    log_revenue       — natural log of TTM revenue

"TTM" here follows the same convention as the rest of the valuation engine
(valuation/dcf.py, valuation/inputs/growth.py): the latest annual filing
column from yfinance, not a quarterly-rolled trailing-twelve-months figure.
"""

import math
from dataclasses import dataclass, field

import yfinance as yf

from valuation.inputs.growth import compute_fundamental_growth


@dataclass
class CompanyFeatures:
    ticker: str
    trailing_growth: float | None       # 3-year revenue CAGR (decimal)
    operating_margin: float | None      # TTM operating margin (decimal)
    roic: float | None                  # TTM ROIC (decimal)
    log_revenue: float | None           # log(TTM revenue in USD)
    revenue_ttm: float | None           # raw TTM revenue (for downstream calc)
    ebitda_ttm: float | None            # TTM EBITDA (None if negative/unavailable)
    data_flags: list = field(default_factory=list)

    # --- Extended candidate features (feature-selection diagnostic only —
    # compute_features_for_ticker() below leaves these as None; they're
    # populated by feature_selection.py's compute_extended_features_for_ticker().
    # Appended at the end with defaults so every existing call site (keyword
    # or positional) that only knows about the original 7 fields is unaffected. ---
    trailing_growth_5y: float | None = None    # revenue CAGR over the longest available window up to 5y
    growth_stability: float | None = None      # 1 - std(yoy growth)/mean(yoy growth) over available years
    gross_margin: float | None = None          # gross profit / revenue (TTM)
    ebitda_margin: float | None = None         # ebitda_ttm / revenue_ttm
    r_and_d_intensity: float | None = None     # R&D expense / revenue (TTM)
    fcf_conversion: float | None = None        # FCF / net income, capped [0, 3]

    def is_complete(self):
        """
        True if all four regression features are usable — non-None AND
        finite. NaN/inf can slip in from upstream data quirks (e.g.
        yfinance returning NaN instead of None for a missing field); a
        plain None-check wouldn't catch those, and statsmodels raises on
        a NaN/inf design matrix downstream in regression.py.
        """
        values = (self.trailing_growth, self.operating_margin, self.roic, self.log_revenue)
        return all(
            v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
            for v in values
        )


def _get_field(row, field_names):
    """Try each candidate field name in order; return first usable float."""
    for name in field_names:
        if name in row.index:
            val = row[name]
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return float(val)
    return None


def compute_features_for_ticker(ticker):
    """
    Fetch financial data from yfinance and compute the 4 regression features.

    Returns a CompanyFeatures with None values for any feature that could
    not be computed (data_flags explains why); never raises.
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

    n_income_years = income_stmt.shape[1]
    latest = income_stmt.iloc[:, 0]

    revenue_ttm = _get_field(latest, ["Total Revenue", "Revenue", "Operating Revenue"])
    operating_income_ttm = _get_field(latest, ["Operating Income", "EBIT"])

    if revenue_ttm is None or revenue_ttm <= 0:
        data_flags.append("No usable TTM revenue")
        return CompanyFeatures(ticker, None, None, None, None, None, None, data_flags)

    log_revenue = math.log(revenue_ttm)

    # --- operating_margin ---
    if operating_income_ttm is None:
        data_flags.append("No usable TTM operating income — operating_margin unavailable")
        operating_margin = None
    else:
        operating_margin = operating_income_ttm / revenue_ttm

    # --- trailing_growth: 3-year revenue CAGR (needs current + 3-years-ago) ---
    if n_income_years >= 4:
        revenue_3y_ago = _get_field(income_stmt.iloc[:, 3], ["Total Revenue", "Revenue", "Operating Revenue"])
        if revenue_3y_ago is not None and revenue_3y_ago > 0:
            trailing_growth = (revenue_ttm / revenue_3y_ago) ** (1 / 3) - 1
        else:
            data_flags.append("3-years-ago revenue unusable — trailing_growth unavailable")
            trailing_growth = None
    else:
        data_flags.append(f"Only {n_income_years} years of income statement history (<4) — trailing_growth unavailable")
        trailing_growth = None

    # --- roic: reuse Layer 1 growth module for consistency ---
    try:
        fund_growth = compute_fundamental_growth(ticker)
        roic = fund_growth.get("roic")
        data_flags.extend(fund_growth.get("data_flags", []))
        if roic is None:
            data_flags.append("compute_fundamental_growth returned no ROIC")
        elif isinstance(roic, float) and (math.isnan(roic) or math.isinf(roic)):
            # Known yfinance quirk: effectiveTaxRate can come back as NaN
            # rather than None, and growth.py's `info.get(...) or 0.21`
            # fallback doesn't catch it (NaN is truthy in Python), so NaN
            # silently propagates through tax_rate -> ebit_after_tax -> roic.
            # Reject it here rather than let it reach the regression, where
            # statsmodels would raise on the resulting NaN/inf design matrix.
            data_flags.append(f"compute_fundamental_growth returned non-finite ROIC ({roic}) — treating as unavailable")
            roic = None
    except Exception as e:
        data_flags.append(f"compute_fundamental_growth failed: {type(e).__name__}: {e}")
        roic = None

    # --- ebitda_ttm = operating_income_ttm + D&A_ttm ---
    ebitda_ttm = None
    if operating_income_ttm is not None and cash_flow is not None and not cash_flow.empty:
        da_ttm = _get_field(cash_flow.iloc[:, 0], ["Depreciation And Amortization", "Depreciation"])
        if da_ttm is not None:
            candidate = operating_income_ttm + da_ttm
            if candidate > 0:
                ebitda_ttm = candidate
            else:
                data_flags.append(f"EBITDA non-positive ({candidate:,.0f}) — ev_ebitda unavailable")
        else:
            data_flags.append("No D&A data — EBITDA unavailable")
    elif operating_income_ttm is not None:
        data_flags.append("No cash flow statement — EBITDA unavailable")

    return CompanyFeatures(
        ticker=ticker,
        trailing_growth=trailing_growth,
        operating_margin=operating_margin,
        roic=roic,
        log_revenue=log_revenue,
        revenue_ttm=revenue_ttm,
        ebitda_ttm=ebitda_ttm,
        data_flags=data_flags,
    )


if __name__ == "__main__":
    for t in ["AMD", "MSFT", "NVDA"]:
        f = compute_features_for_ticker(t)
        print(f"\n{t}: complete={f.is_complete()}")
        print(f"  trailing_growth={f.trailing_growth}")
        print(f"  operating_margin={f.operating_margin}")
        print(f"  roic={f.roic}")
        print(f"  log_revenue={f.log_revenue}  revenue_ttm={f.revenue_ttm}")
        print(f"  ebitda_ttm={f.ebitda_ttm}")
        if f.data_flags:
            print(f"  flags: {f.data_flags}")
