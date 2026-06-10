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
# Coefficient mapping excess returns to high-growth period length
EXCESS_RETURN_TO_YEARS_COEFF = 50

# Industry-average ROIC by bucket (calibrated against Damodaran's published data)
# TODO Phase 1.4: compute these empirically from our 555-ticker database
INDUSTRY_AVERAGE_ROIC = {
    "mature_tech": 0.17,
    "saas_growth": 0.15,
    "semiconductor": 0.12,
    "consumer_defensive": 0.13,
    "consumer_cyclical": 0.10,
    "financial_bank": 0.08,
    "financial_other": 0.10,
    "healthcare_pharma": 0.15,
    "healthcare_other": 0.12,
    "industrial": 0.10,
    "energy": 0.08,
    "materials": 0.08,
    "utility": 0.07,
    "reit": 0.08,
    "insurance": 0.09,
    "communication": 0.11,
    "default": 0.10,
}

# Per-bucket cap on initial growth rate
# Calibrated to plausible sustained growth for each industry
# TODO Phase 1.4: replace with empirical 75th percentile from historical data
BUCKET_INITIAL_GROWTH_CAP = {
    "saas_growth": 0.35,
    "semiconductor": 0.30,
    "healthcare_pharma": 0.25,
    "mature_tech": 0.20,
    "consumer_cyclical": 0.20,
    "communication": 0.18,
    "industrial": 0.15,
    "healthcare_other": 0.15,
    "financial_other": 0.15,
    "materials": 0.12,
    "energy": 0.12,
    "consumer_defensive": 0.10,
    "financial_bank": 0.10,
    "insurance": 0.10,
    "reit": 0.08,
    "utility": 0.06,
    "default": 0.20,
}

# Per-bucket multiplier on high-growth period length
# Cyclical industries get shorter periods (booms end)
# Moated industries get longer periods (defenses persist)
BUCKET_HIGH_GROWTH_MULTIPLIER = {
    "saas_growth": 1.2,
    "mature_tech": 1.0,
    "consumer_defensive": 1.0,
    "healthcare_pharma": 0.8,
    "communication": 0.9,
    "consumer_cyclical": 0.8,
    "industrial": 0.7,
    "healthcare_other": 0.8,
    "financial_other": 0.8,
    "financial_bank": 0.7,
    "insurance": 0.8,
    "reit": 0.7,
    "semiconductor": 0.5,
    "energy": 0.5,
    "materials": 0.5,
    "utility": 0.6,
    "default": 1.0,
}

# Blend weights (Damodaran-style triangulation, refined toward fundamental anchor)
BLEND_WEIGHTS = {"fundamental": 0.70, "consensus": 0.20, "historical": 0.10}


# === Public functions (called from build_growth_profile and outside this module) ===

def compute_roic_history(ticker, years=5):
    """
    Compute ROIC for each of the last N years.
    Used for mean-reversion adjustments and maturity detection.

    Returns:
        dict with:
            roic_series (list): ROIC values, oldest to newest
            years_available (int): how many we got
            mean_roic (float): average across the series
            current_roic (float): most recent (last in series)
            stability (float): coefficient of variation (std/mean) — low = stable
            data_flags (list)
    """
    yf_ticker = yf.Ticker(ticker)
    income_stmt = yf_ticker.income_stmt
    balance_sheet = yf_ticker.balance_sheet

    data_flags = []

    if income_stmt is None or income_stmt.empty or balance_sheet is None or balance_sheet.empty:
        data_flags.append("Missing financial statements for ROIC history")
        return _null_roic_history(data_flags)

    # yfinance returns most recent year first. We iterate columns.
    n_cols = min(income_stmt.shape[1], balance_sheet.shape[1])
    if n_cols < 2:
        data_flags.append("Not enough years of financial data")
        return _null_roic_history(data_flags)

    info = yf_ticker.info
    tax_rate = info.get("effectiveTaxRate") or DEFAULT_TAX_RATE

    roic_series = []

    for i in range(min(n_cols, years)):
        income_col = income_stmt.iloc[:, i]
        balance_col = balance_sheet.iloc[:, i]

        # EBIT for this year
        ebit = None
        for field in ["EBIT", "Operating Income"]:
            if field in income_col.index:
                val = income_col[field]
                if val is not None and not math.isnan(float(val)):
                    ebit = float(val)
                    break
        if ebit is None or ebit <= 0:
            continue  # skip years with missing or negative EBIT

        # Invested capital for this year
        total_equity = None
        for field in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
            if field in balance_col.index:
                val = balance_col[field]
                if val is not None and not math.isnan(float(val)):
                    total_equity = float(val)
                    break
        if total_equity is None or total_equity <= 0:
            continue

        total_debt_val = balance_col.get("Total Debt", 0)
        cash_val = balance_col.get("Cash And Cash Equivalents", 0)
        total_debt = float(total_debt_val) if total_debt_val is not None and not math.isnan(float(total_debt_val)) else 0
        cash = float(cash_val) if cash_val is not None and not math.isnan(float(cash_val)) else 0

        invested_capital = total_equity + total_debt - cash
        if invested_capital <= 0:
            continue

        roic_for_year = (ebit * (1 - tax_rate)) / invested_capital
        roic_series.append(roic_for_year)

    if not roic_series:
        data_flags.append("No usable ROIC history computed")
        return _null_roic_history(data_flags)

    # yfinance gives newest first; reverse to chronological (oldest first)
    roic_series = roic_series[::-1]

    mean_roic = sum(roic_series) / len(roic_series)
    current_roic = roic_series[-1]

    # Coefficient of variation: how stable is ROIC?
    # Low CV (< 0.15) = stable mature company
    # High CV (> 0.4) = volatile, possibly in transition
    if len(roic_series) > 1 and mean_roic > 0:
        variance = sum((r - mean_roic) ** 2 for r in roic_series) / len(roic_series)
        std_dev = variance ** 0.5
        stability = std_dev / mean_roic
    else:
        stability = None

    return {
        "roic_series": roic_series,
        "years_available": len(roic_series),
        "mean_roic": mean_roic,
        "current_roic": current_roic,
        "stability": stability,
        "data_flags": data_flags,
    }


def _null_roic_history(data_flags):
    """Return-shaped null for failed ROIC history computation."""
    return {
        "roic_series": [],
        "years_available": 0,
        "mean_roic": None,
        "current_roic": None,
        "stability": None,
        "data_flags": data_flags,
    }

def compute_margin_history(ticker, years=5):
    """
    Compute gross margin for each of the last N years.
    Used for trajectory detection (declining margins = growth headwind).

    Returns:
        dict with:
            margin_series (list): gross margins, oldest to newest
            current_margin (float): most recent
            trend (float): linear slope of margin over time (per year change)
            data_flags (list)
    """
    yf_ticker = yf.Ticker(ticker)
    income_stmt = yf_ticker.income_stmt

    data_flags = []

    if income_stmt is None or income_stmt.empty:
        data_flags.append("Missing income statement for margin history")
        return _null_margin_history(data_flags)

    margin_series = []
    n_cols = min(income_stmt.shape[1], years)

    for i in range(n_cols):
        col = income_stmt.iloc[:, i]

        revenue = None
        for field in ["Total Revenue", "Revenue", "Operating Revenue"]:
            if field in col.index:
                val = col[field]
                if val is not None and not math.isnan(float(val)):
                    revenue = float(val)
                    break
        if revenue is None or revenue <= 0:
            continue

        cogs = None
        for field in ["Cost Of Revenue", "Cost of Revenue", "Reconciled Cost Of Revenue"]:
            if field in col.index:
                val = col[field]
                if val is not None and not math.isnan(float(val)):
                    cogs = float(val)
                    break
        if cogs is None:
            continue

        gross_margin = (revenue - cogs) / revenue
        margin_series.append(gross_margin)

    if len(margin_series) < 2:
        data_flags.append("Not enough margin data for trend")
        return _null_margin_history(data_flags)

    # Reverse to chronological order (oldest first)
    margin_series = margin_series[::-1]
    current_margin = margin_series[-1]

    # Linear trend: slope of margin over time
    # Positive = expanding, negative = compressing
    n = len(margin_series)
    t = np.arange(n)
    slope, _intercept = np.polyfit(t, margin_series, deg=1)
    trend = float(slope)  # change in margin per year

    return {
        "margin_series": margin_series,
        "current_margin": current_margin,
        "trend": trend,
        "data_flags": data_flags,
    }


def _null_margin_history(data_flags):
    return {
        "margin_series": [],
        "current_margin": None,
        "trend": None,
        "data_flags": data_flags,
    }

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

    # EBIT — banks and insurance companies don't have it in the standard form
    ebit = None
    for field in ["EBIT", "Operating Income"]:
        if field in latest_income.index:
            ebit = float(latest_income[field])
            break
    if ebit is None:
        data_flags.append("No EBIT found (likely financial company) — fundamental growth unavailable")
        return _null_fundamental_result(data_flags)

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

    # Initial growth = blend of fundamental, consensus, historical (70/20/10)
    initial_g_raw = _blend_growth(fund_g, cons_g, hist_g, data_flags)

    # Apply per-bucket initial growth cap
    bucket_cap = BUCKET_INITIAL_GROWTH_CAP.get(bucket, BUCKET_INITIAL_GROWTH_CAP["default"])
    if initial_g_raw > bucket_cap:
        data_flags.append(
            f"Initial growth {initial_g_raw * 100:.1f}% capped at bucket max {bucket_cap * 100:.0f}%"
        )
        initial_g = bucket_cap
    else:
        initial_g = initial_g_raw

    # Industry-average ROIC for the bucket
    industry_roic = INDUSTRY_AVERAGE_ROIC.get(bucket, INDUSTRY_AVERAGE_ROIC["default"])

    # Terminal values
    terminal_g = rf  # Damodaran: terminal g ≤ risk-free rate
    terminal_roic = industry_roic
    terminal_reinvestment_rate = terminal_g / terminal_roic

    # High-growth period length from excess returns × bucket multiplier × boom adjustment
    if current_roic is None:
        high_growth_years = MIN_HIGH_GROWTH_YEARS
        data_flags.append("No ROIC — using minimum high-growth period")
        excess_return = None
    else:
        excess_return = current_roic - wacc
        bucket_multiplier = BUCKET_HIGH_GROWTH_MULTIPLIER.get(
            bucket, BUCKET_HIGH_GROWTH_MULTIPLIER["default"]
        )
        boom_multiplier = _detect_boom(fund_g, hist_g, data_flags)

        raw_years = excess_return * EXCESS_RETURN_TO_YEARS_COEFF * bucket_multiplier * boom_multiplier
        high_growth_years = int(max(
            MIN_HIGH_GROWTH_YEARS,
            min(MAX_HIGH_GROWTH_YEARS, raw_years)
        ))

        if bucket_multiplier != 1.0:
            data_flags.append(
                f"Bucket multiplier {bucket_multiplier} applied (industry: {bucket})"
            )

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
    Triangulation blend: 70% fundamental, 20% consensus, 10% historical.
    Renormalizes weights when signals are missing.
    """
    weights = BLEND_WEIGHTS
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

def _detect_boom(fund_g, hist_g, data_flags):
    """
    Detect when recent growth wildly exceeds sustainable rate.
    Returns multiplier to shorten high-growth period.

    Ratio thresholds chosen so that:
        - 4x sustainable rate = extreme boom (mostly unsustainable)
        - 2.5x = strong boom
        - 1.5x = moderate boom
        - below 1.5x = no adjustment
    """
    if hist_g is None or fund_g is None or fund_g <= 0:
        return 1.0

    ratio = hist_g / fund_g

    if ratio > 4.0:
        data_flags.append(f"Extreme boom: recent={hist_g*100:.0f}% vs sustainable={fund_g*100:.0f}% → hg period cut 70%")
        return 0.3
    elif ratio > 2.5:
        data_flags.append(f"Strong boom: recent={hist_g*100:.0f}% vs sustainable={fund_g*100:.0f}% → hg period cut 50%")
        return 0.5
    elif ratio > 1.5:
        data_flags.append(f"Moderate boom: recent={hist_g*100:.0f}% vs sustainable={fund_g*100:.0f}% → hg period cut 30%")
        return 0.7
    else:
        return 1.0



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