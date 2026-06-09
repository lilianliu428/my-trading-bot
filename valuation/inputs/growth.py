"""
Growth modeling for DCF — three-stage model with ROIC fade.

Architecture:
    Stage 1 (Years 1 to N1):  High growth — current rates persist
    Stage 2 (Years N1+1 to N2): Transition — g and ROIC fade linearly
    Stage 3 (Year N2+1+):     Stable — g = rf, ROIC = industry average

Key decisions backed by Damodaran (Investment Valuation, Ch. 11-12):
    - Terminal growth ≤ risk-free rate (long-run economic growth proxy)
    - Reinvestment rate in stable growth = g / ROIC (formula, not guess)
    - ROIC fades toward industry average, not WACC (moats are real)
    - High-growth period length proportional to excess returns

Modules:
    compute_fundamental_growth(ticker)   — g = reinvestment × ROIC from financials
    compute_historical_growth(ticker)    — log-linear regression on FCF (revenue fallback)
    get_consensus_growth(ticker)         — analyst estimates from yfinance
    build_growth_profile(ticker)         — three-stage orchestration
"""

import math
import numpy as np
import yfinance as yf

from valuation.data_sources.fred import get_risk_free_rate


# === Constants ===

# US federal corporate tax rate, fallback if yfinance returns nothing
DEFAULT_TAX_RATE = 0.21

# Transition period length (Damodaran-compatible: 5 years when transitions are used)
TRANSITION_YEARS = 5

# High-growth period bounds (years)
MIN_HIGH_GROWTH_YEARS = 3
MAX_HIGH_GROWTH_YEARS = 15

# Coefficient mapping excess returns to high-growth period length
# Formula: high_growth_years = min(15, max(3, (ROIC - WACC) * 50))
# Calibration: a company with 10pp excess return gets 5 years; 20pp gets 10 years
EXCESS_RETURN_TO_YEARS_COEFF = 50

# Industry-average ROIC by bucket (Option A — hardcoded heuristics)
# TODO Phase 1.4: compute these from our database of 555 tickers, store, refresh quarterly
INDUSTRY_AVERAGE_ROIC = {
    "mature_tech": 0.17,         # MSFT, AAPL, GOOGL — moats persist
    "saas_growth": 0.15,         # CRM, NOW — strong but younger moats
    "semiconductor": 0.12,        # NVDA, AMD — cyclical
    "consumer_defensive": 0.13,   # KO, PG — brand moats
    "consumer_cyclical": 0.10,    # AMZN, TSLA — less stable
    "financial_bank": 0.08,       # JPM, BAC — commodity-ish
    "financial_other": 0.10,      # V, MA, BLK — varies
    "healthcare_pharma": 0.15,    # patents during life
    "healthcare_other": 0.12,
    "industrial": 0.10,           # GE, CAT — competitive
    "energy": 0.08,               # XOM, CVX — commodity
    "materials": 0.08,
    "utility": 0.07,              # regulated to be close to WACC
    "reit": 0.08,
    "insurance": 0.09,
    "communication": 0.11,
    "default": 0.10,              # fallback if bucket unknown
}


# === Public functions (called from build_growth_profile and outside this module) ===

def compute_fundamental_growth(ticker):
    """
    Compute sustainable growth rate using fundamentals.

    Formula:
        g = reinvestment_rate × ROIC

    Where:
        reinvestment_rate = (Capex - Depreciation + ΔWorking Capital) / EBIT(1-t)
        ROIC = EBIT(1-t) / Invested Capital
        Invested Capital = Total Equity + Total Debt - Cash

    Returns:
        dict with: ebit_after_tax, reinvestment, reinvestment_rate,
                   invested_capital, roic, fundamental_growth, tax_rate, data_flags
    """
    print(f"  Computing fundamental growth for {ticker}...")
    yf_ticker = yf.Ticker(ticker)
    income_stmt = yf_ticker.income_stmt
    balance_sheet = yf_ticker.balance_sheet
    cash_flow = yf_ticker.cashflow

    if any(df is None or df.empty for df in [income_stmt, balance_sheet, cash_flow]):
        raise ValueError(f"Missing financial statement data for {ticker}")

    latest_income = income_stmt.iloc[:, 0]
    latest_balance = balance_sheet.iloc[:, 0]
    latest_cashflow = cash_flow.iloc[:, 0]
    prev_balance = balance_sheet.iloc[:, 1] if income_stmt.shape[1] >= 2 else None

    data_flags = []

    # EBIT
    ebit = None
    for field in ["EBIT", "Operating Income"]:
        if field in latest_income.index:
            ebit = float(latest_income[field])
            break
    if ebit is None:
        raise ValueError(f"No EBIT field found for {ticker}")

    if ebit <= 0:
        data_flags.append("EBIT is negative or zero — fundamental growth undefined")
        return _null_fundamental_result(data_flags)

    # Tax rate
    info = yf_ticker.info
    tax_rate = info.get("effectiveTaxRate") or DEFAULT_TAX_RATE
    ebit_after_tax = ebit * (1 - tax_rate)

    # Capex and depreciation
    capex = None
    for field in ["Capital Expenditure", "Capital Expenditures"]:
        if field in latest_cashflow.index:
            capex = abs(float(latest_cashflow[field]))
            break
    if capex is None:
        raise ValueError(f"No capex field found for {ticker}")

    depreciation = None
    for field in ["Depreciation Amortization Depletion", "Depreciation And Amortization", "Depreciation"]:
        if field in latest_cashflow.index:
            depreciation = float(latest_cashflow[field])
            break
    if depreciation is None:
        raise ValueError(f"No depreciation field found for {ticker}")

    # Change in working capital
    def working_capital(balance):
        ca = balance.get("Current Assets")
        cl = balance.get("Current Liabilities")
        if ca is None or cl is None:
            return None
        return float(ca) - float(cl)

    wc_current = working_capital(latest_balance)
    wc_prev = working_capital(prev_balance) if prev_balance is not None else None

    if wc_current is None or wc_prev is None:
        change_in_wc = 0.0
        data_flags.append("Working capital data missing — assumed ΔWC = 0")
    else:
        change_in_wc = wc_current - wc_prev

    reinvestment = capex - depreciation + change_in_wc

    # Reinvestment rate — clamp to 0 if negative
    if reinvestment < 0:
        reinvestment_rate = 0
        data_flags.append("Negative reinvestment — clamped to 0")
    else:
        reinvestment_rate = reinvestment / ebit_after_tax

    # Invested capital
    total_equity = None
    for field in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
        if field in latest_balance.index:
            total_equity = float(latest_balance[field])
            break
    if total_equity is None:
        raise ValueError(f"No equity field found for {ticker}")

    total_debt = float(latest_balance.get("Total Debt", 0))
    cash = float(latest_balance.get("Cash And Cash Equivalents", 0))
    invested_capital = total_equity + total_debt - cash

    if invested_capital <= 0:
        data_flags.append("Invested capital is negative or zero — ROIC undefined")
        return {
            "ebit_after_tax": ebit_after_tax,
            "reinvestment": reinvestment,
            "reinvestment_rate": reinvestment_rate,
            "invested_capital": invested_capital,
            "roic": None,
            "fundamental_growth": None,
            "tax_rate": tax_rate,
            "data_flags": data_flags,
        }

    roic = ebit_after_tax / invested_capital
    fundamental_growth = reinvestment_rate * roic

    return {
        "ebit_after_tax": ebit_after_tax,
        "reinvestment": reinvestment,
        "reinvestment_rate": reinvestment_rate,
        "invested_capital": invested_capital,
        "roic": roic,
        "fundamental_growth": fundamental_growth,
        "tax_rate": tax_rate,
        "data_flags": data_flags,
    }


def compute_historical_growth(ticker, years=5):
    """
    Estimate historical growth using log-linear regression.

    Why log-linear: uses all data points, robust to single-year outliers.
    More accurate than geometric mean (which uses only first/last values)
    or arithmetic mean (which is biased by spikes).

    Fallback chain:
        1. Log-linear regression on FCF (if all positive)
        2. Log-linear regression on revenue (if FCF has any negatives)
        3. None (if neither has enough data)

    Returns:
        dict with: series_used, method_used, growth_rate, r_squared, n_years, data_flags
    """
    print(f"  Computing historical growth for {ticker}...")
    yf_ticker = yf.Ticker(ticker)
    cash_flow = yf_ticker.cashflow

    data_flags = []

    if cash_flow is None or cash_flow.empty:
        data_flags.append("No cash flow data")
        return _try_revenue_growth(ticker, years, data_flags)

    # Extract FCF series, chronological order
    fcf_values = []
    for col in cash_flow.columns:
        col_data = cash_flow[col]
        fcf = None
        for field in ["Free Cash Flow", "FreeCashFlow"]:
            if field in col_data.index:
                fcf = float(col_data[field])
                break
        if fcf is None:
            ocf = col_data.get("Operating Cash Flow") or col_data.get("Cash Flow From Continuing Operating Activities")
            capex_val = col_data.get("Capital Expenditure")
            if ocf is not None and capex_val is not None:
                fcf = float(ocf) + float(capex_val)
        if fcf is None or math.isnan(fcf):
            continue
        fcf_values.append(fcf)

    fcf_values = fcf_values[::-1]  # chronological order
    if len(fcf_values) > years + 1:
        fcf_values = fcf_values[-(years + 1):]

    # Check usability
    if any(v <= 0 for v in fcf_values):
        data_flags.append("FCF series has non-positive values — falling back to revenue")
        return _try_revenue_growth(ticker, years, data_flags)

    if len(fcf_values) < 3:
        data_flags.append("Not enough FCF data for regression — falling back to revenue")
        return _try_revenue_growth(ticker, years, data_flags)

    growth_rate, r_squared = _log_linear_regression(fcf_values)

    return {
        "series_used": fcf_values,
        "method_used": "log_linear_fcf",
        "growth_rate": growth_rate,
        "r_squared": r_squared,
        "n_years": len(fcf_values),
        "data_flags": data_flags,
    }


def get_consensus_growth(ticker):
    """
    Near-term growth from analyst consensus (yfinance).

    Tries: long-term earnings growth → next-year earnings growth → revenue growth.

    Returns:
        dict with: consensus_growth, source, n_analysts, data_flags
    """
    print(f"  Fetching consensus growth for {ticker}...")
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info

    data_flags = []
    n_analysts = info.get("numberOfAnalystOpinions")

    consensus = info.get("earningsQuarterlyGrowth")
    source = "earningsQuarterlyGrowth"

    if consensus is None or consensus <= 0:
        consensus = info.get("earningsGrowth")
        source = "earningsGrowth (next year)"

    if consensus is None or consensus <= 0:
        consensus = info.get("revenueGrowth")
        source = "revenueGrowth (TTM)"
        data_flags.append("Using TTM revenue growth — no forward consensus")

    if consensus is None:
        data_flags.append("No consensus growth available")
        return {
            "consensus_growth": None,
            "source": None,
            "n_analysts": n_analysts,
            "data_flags": data_flags,
        }

    # Sanity guards
    if consensus > 0.50:
        data_flags.append(f"Consensus {consensus*100:.0f}% implausible — capped at 30%")
        consensus = 0.30
    if consensus < -0.20:
        data_flags.append(f"Consensus {consensus*100:.0f}% deeply negative — capped at -10%")
        consensus = -0.10

    return {
        "consensus_growth": consensus,
        "source": source,
        "n_analysts": n_analysts,
        "data_flags": data_flags,
    }


def build_growth_profile(ticker, wacc, bucket="default"):
    """
    Build three-stage growth profile per Damodaran's framework.

    Stage 1 (Years 1 to N1): High growth at current fundamental rate
    Stage 2 (Years N1+1 to N2): Transition — linear fade of g and ROIC
    Stage 3 (Year N2+1+): Stable — g = rf, ROIC = industry average

    Reinvestment rate in stable growth = g_stable / roic_stable (Damodaran formula).

    Args:
        ticker: e.g. "MSFT"
        wacc: weighted average cost of capital (decimal), used for excess returns
        bucket: business model bucket from our tagging system (mature_tech, etc.)

    Returns:
        dict with:
            yearly_growth (list): year-by-year growth for high-growth + transition
            yearly_roic (list): year-by-year ROIC corresponding to above
            yearly_reinvestment (list): year-by-year reinvestment rate
            terminal_growth (float)
            terminal_roic (float)
            terminal_reinvestment_rate (float)
            high_growth_years (int)
            transition_years (int)
            fundamental_growth, consensus_growth, historical_growth (floats)
            current_roic, industry_roic (floats)
            excess_return (float): current ROIC - WACC
            r_squared_historical (float)
            data_flags (list)
    """
    print(f"\n  === Building growth profile for {ticker} (bucket: {bucket}) ===")

    # Pull all signals
    fund_info = compute_fundamental_growth(ticker)
    hist_info = compute_historical_growth(ticker)
    cons_info = get_consensus_growth(ticker)
    rf = get_risk_free_rate()

    data_flags = []
    data_flags.extend(fund_info["data_flags"])
    data_flags.extend(hist_info["data_flags"])
    data_flags.extend(cons_info["data_flags"])

    fund_g = fund_info["fundamental_growth"]
    hist_g = hist_info["growth_rate"]
    cons_g = cons_info["consensus_growth"]
    current_roic = fund_info["roic"]

    # Initial growth = blend of fundamental, consensus, historical (60/20/20)
    initial_g = _blend_growth(fund_g, cons_g, hist_g, data_flags)

    # Industry-average ROIC for the bucket
    industry_roic = INDUSTRY_AVERAGE_ROIC.get(bucket, INDUSTRY_AVERAGE_ROIC["default"])

    # Terminal values
    terminal_g = rf  # Damodaran: terminal g ≤ risk-free rate
    terminal_roic = industry_roic
    terminal_reinvestment_rate = terminal_g / terminal_roic  # The Damodaran formula

    # High-growth period length from excess returns
    if current_roic is None:
        high_growth_years = MIN_HIGH_GROWTH_YEARS
        data_flags.append("No ROIC — using minimum high-growth period")
        excess_return = None
    else:
        excess_return = current_roic - wacc
        high_growth_years = int(max(
            MIN_HIGH_GROWTH_YEARS,
            min(MAX_HIGH_GROWTH_YEARS, excess_return * EXCESS_RETURN_TO_YEARS_COEFF)
        ))

    # Build the year-by-year arrays
    yearly_growth = []
    yearly_roic = []
    yearly_reinvestment = []

    # Stage 1: High growth, constant rates
    stage1_roic = current_roic if current_roic is not None else industry_roic
    stage1_reinv = (initial_g / stage1_roic) if stage1_roic > 0 else 0

    for year in range(1, high_growth_years + 1):
        yearly_growth.append(initial_g)
        yearly_roic.append(stage1_roic)
        yearly_reinvestment.append(stage1_reinv)

    # Stage 2: Transition — linear fade across all three variables
    for i in range(1, TRANSITION_YEARS + 1):
        progress = i / TRANSITION_YEARS  # 0 → 1 across transition
        g = initial_g + progress * (terminal_g - initial_g)
        roic = stage1_roic + progress * (terminal_roic - stage1_roic)
        reinvestment = g / roic if roic > 0 else 0
        yearly_growth.append(g)
        yearly_roic.append(roic)
        yearly_reinvestment.append(reinvestment)

    return {
        "yearly_growth": yearly_growth,
        "yearly_roic": yearly_roic,
        "yearly_reinvestment": yearly_reinvestment,
        "terminal_growth": terminal_g,
        "terminal_roic": terminal_roic,
        "terminal_reinvestment_rate": terminal_reinvestment_rate,
        "high_growth_years": high_growth_years,
        "transition_years": TRANSITION_YEARS,
        "fundamental_growth": fund_g,
        "consensus_growth": cons_g,
        "historical_growth": hist_g,
        "current_roic": current_roic,
        "industry_roic": industry_roic,
        "excess_return": excess_return,
        "r_squared_historical": hist_info.get("r_squared"),
        "n_analysts": cons_info.get("n_analysts"),
        "data_flags": data_flags,
    }


# === Private helpers ===

def _null_fundamental_result(data_flags):
    """Return-shaped null for failed fundamental computation."""
    return {
        "ebit_after_tax": None,
        "reinvestment": None,
        "reinvestment_rate": None,
        "invested_capital": None,
        "roic": None,
        "fundamental_growth": None,
        "tax_rate": None,
        "data_flags": data_flags,
    }


def _log_linear_regression(values):
    """
    Fit ln(value) = α + β·t. Returns (annual_growth, r_squared).
    Assumes values are all positive.
    """
    n = len(values)
    t = np.arange(n)
    log_values = np.log(values)

    beta, alpha = np.polyfit(t, log_values, deg=1)
    growth_rate = float(np.exp(beta) - 1)

    predicted = alpha + beta * t
    ss_residual = np.sum((log_values - predicted) ** 2)
    ss_total = np.sum((log_values - np.mean(log_values)) ** 2)
    r_squared = float(1 - ss_residual / ss_total) if ss_total > 0 else 0.0

    return growth_rate, r_squared


def _try_revenue_growth(ticker, years, data_flags):
    """Fallback: log-linear regression on revenue when FCF unusable."""
    yf_ticker = yf.Ticker(ticker)
    income_stmt = yf_ticker.income_stmt

    if income_stmt is None or income_stmt.empty:
        data_flags.append("No income statement for revenue fallback")
        return _null_historical_result(data_flags)

    revenue_values = []
    for col in income_stmt.columns:
        col_data = income_stmt[col]
        rev = None
        for field in ["Total Revenue", "Revenue", "Operating Revenue"]:
            if field in col_data.index:
                rev = float(col_data[field])
                break
        if rev is None or math.isnan(rev) or rev <= 0:
            continue
        revenue_values.append(rev)

    revenue_values = revenue_values[::-1]
    if len(revenue_values) > years + 1:
        revenue_values = revenue_values[-(years + 1):]

    if len(revenue_values) < 3:
        data_flags.append("Not enough revenue data for regression")
        return _null_historical_result(data_flags)

    growth_rate, r_squared = _log_linear_regression(revenue_values)

    return {
        "series_used": revenue_values,
        "method_used": "log_linear_revenue",
        "growth_rate": growth_rate,
        "r_squared": r_squared,
        "n_years": len(revenue_values),
        "data_flags": data_flags,
    }


def _null_historical_result(data_flags):
    """Return-shaped null for failed historical computation."""
    return {
        "series_used": [],
        "method_used": None,
        "growth_rate": None,
        "r_squared": None,
        "n_years": 0,
        "data_flags": data_flags,
    }


def _blend_growth(fund_g, cons_g, hist_g, data_flags):
    """
    Damodaran-style triangulation: 60% fundamental, 20% consensus, 20% historical.
    Renormalizes weights when signals are missing.
    """
    weights = {"fundamental": 0.60, "consensus": 0.20, "historical": 0.20}
    available = {}
    if fund_g is not None:
        available["fundamental"] = fund_g
    if cons_g is not None:
        available["consensus"] = cons_g
    if hist_g is not None:
        available["historical"] = hist_g

    if not available:
        data_flags.append("No growth signals — using risk-free rate as initial growth")
        return get_risk_free_rate()

    total_weight = sum(weights[k] for k in available)
    blended = sum((weights[k] / total_weight) * available[k] for k in available)

    used = ", ".join(f"{k}={available[k]*100:.1f}%" for k in available)
    data_flags.append(f"Initial growth blended from: {used} → {blended*100:.1f}%")
    return blended


if __name__ == "__main__":
    from valuation.inputs.wacc import compute_wacc

    wacc_info = compute_wacc("MSFT")
    wacc = wacc_info["wacc"]

    profile = build_growth_profile("MSFT", wacc, bucket="mature_tech")

    print(f"\n=== MSFT Growth Profile (Three-Stage) ===\n")
    print(f"Inputs:")
    print(f"  Current ROIC:        {profile['current_roic']*100:.1f}%")
    print(f"  Industry ROIC:       {profile['industry_roic']*100:.1f}%")
    print(f"  WACC:                {wacc*100:.2f}%")
    print(f"  Excess return:       {profile['excess_return']*100:+.2f}pp")
    print(f"  Risk-free rate:      {get_risk_free_rate()*100:.2f}%")

    print(f"\nGrowth signals:")
    if profile['fundamental_growth'] is not None:
        print(f"  Fundamental:         {profile['fundamental_growth']*100:.1f}%")
    if profile['consensus_growth'] is not None:
        print(f"  Consensus:           {profile['consensus_growth']*100:.1f}%")
    if profile['historical_growth'] is not None:
        r2 = profile['r_squared_historical']
        print(f"  Historical:          {profile['historical_growth']*100:.1f}%  (R² = {r2:.2f})")
    if profile['n_analysts']:
        print(f"  # analysts:          {profile['n_analysts']}")

    print(f"\nStructure:")
    print(f"  High-growth years:   {profile['high_growth_years']}")
    print(f"  Transition years:    {profile['transition_years']}")
    print(f"  Total modeled years: {len(profile['yearly_growth'])}")

    print(f"\nYear-by-year:")
    print(f"  {'Year':>5}  {'Growth':>7}  {'ROIC':>7}  {'Reinv':>7}")
    for i, (g, r, rv) in enumerate(zip(
        profile['yearly_growth'],
        profile['yearly_roic'],
        profile['yearly_reinvestment']
    ), start=1):
        stage = "HG" if i <= profile['high_growth_years'] else "TR"
        print(f"  {i:>3} {stage}  {g*100:>6.1f}%  {r*100:>6.1f}%  {rv*100:>6.1f}%")

    print(f"\n  Terminal:")
    print(f"     g  =  {profile['terminal_growth']*100:.2f}% (risk-free rate)")
    print(f"   ROIC =  {profile['terminal_roic']*100:.1f}% (industry avg)")
    print(f"  Reinv =  {profile['terminal_reinvestment_rate']*100:.1f}% (= g/ROIC)")

    if profile['data_flags']:
        print(f"\nFlags:")
        for f in profile['data_flags']:
            print(f"  - {f}")