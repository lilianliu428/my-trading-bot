"""
Growth modeling for DCF.

Replaces hardcoded growth rates with defensible estimates derived from:
    1. Fundamental growth (g = reinvestment_rate × ROIC) — sustainable rate
    2. Historical growth (last 5 years FCF) — sanity check
    3. Consensus growth (analyst estimates) — near-term anchor
    4. Growth fade — combines into year-by-year profile

This module is the conceptual heart of Pillar 1.3.
"""

import yfinance as yf
import math


# Terminal growth cap — no company can grow faster than the economy forever
# US nominal GDP growth (real GDP + inflation) typically 3-5%
# We use 4% as a defensible ceiling
GDP_TERMINAL_CAP = 0.04

# US federal corporate tax rate, used as fallback if yfinance doesn't return one
DEFAULT_TAX_RATE = 0.21


def compute_fundamental_growth(ticker):
    """
    Compute sustainable growth rate using fundamental analysis.

    Formula:
        g = reinvestment_rate × ROIC

    Where:
        reinvestment_rate = (Capex - Depreciation + ΔWorking Capital) / EBIT(1-t)
        ROIC = EBIT(1-t) / Invested Capital
        Invested Capital = Total Equity + Total Debt - Cash

    Returns:
        dict with:
            ebit_after_tax (float): EBIT × (1 - tax_rate)
            reinvestment (float): dollars reinvested (numerator of reinvestment rate)
            reinvestment_rate (float): as decimal, e.g. 0.40 = 40%
            invested_capital (float): denominator of ROIC
            roic (float): as decimal, e.g. 0.30 = 30%
            fundamental_growth (float): reinvestment_rate × roic, the answer
            tax_rate (float): used in computation
            data_flags (list): any warnings about edge cases (negative values, etc.)

    Notes:
        - We use latest annual financial statements (not TTM)
        - If reinvestment is negative (rare), we set rate to 0 (can't have negative
          sustainable growth from reinvestment)
        - If EBIT is negative, fundamental growth is undefined (we return None)
    """
    print(f"  Computing fundamental growth for {ticker}...")
    yf_ticker = yf.Ticker(ticker)
    income_stmt = yf_ticker.income_stmt
    balance_sheet = yf_ticker.balance_sheet
    cash_flow = yf_ticker.cashflow

    if any(df is None or df.empty for df in [income_stmt, balance_sheet, cash_flow]):
        raise ValueError(f"Missing financial statement data for {ticker}")

    # Get most recent annual values (first column in yfinance)
    latest_income = income_stmt.iloc[:, 0]
    latest_balance = balance_sheet.iloc[:, 0]
    latest_cashflow = cash_flow.iloc[:, 0]

    # Also get previous year for working capital change
    if income_stmt.shape[1] >= 2:
        prev_balance = balance_sheet.iloc[:, 1]
    else:
        prev_balance = None

    data_flags = []

    # === Step 1: Get EBIT ===
    # Try multiple field names yfinance uses
    ebit = None
    for field in ["EBIT", "Operating Income"]:
        if field in latest_income.index:
            ebit = float(latest_income[field])
            break
    if ebit is None:
        raise ValueError(f"No EBIT field found for {ticker}")

    if ebit <= 0:
        data_flags.append("EBIT is negative or zero — fundamental growth undefined")
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

    # === Step 2: Tax rate ===
    info = yf_ticker.info
    tax_rate = info.get("effectiveTaxRate") or DEFAULT_TAX_RATE

    # === Step 3: After-tax EBIT (NOPAT — Net Operating Profit After Tax) ===
    ebit_after_tax = ebit*(1-tax_rate)

    # === Step 4: Capex and Depreciation ===
    # yfinance field names:
    #   "Capital Expenditure" (usually NEGATIVE in yfinance — money outflow)
    #   "Depreciation Amortization Depletion" or "Depreciation And Amortization"
    capex = None
    for field in ["Capital Expenditure", "Capital Expenditures"]:
        if field in latest_cashflow.index:
            capex = abs(float(latest_cashflow[field]))  # take absolute value
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

    # === Step 5: Change in working capital ===
    # Working capital = Current Assets - Current Liabilities
    # We need: this year's WC minus last year's WC
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

    # === Step 6: Reinvestment ===
    # Formula: reinvestment = capex - depreciation + ΔWC
    reinvestment = capex-depreciation+change_in_wc

    # === Step 7: Reinvestment rate ===
    # reinvestment_rate = reinvestment / ebit_after_tax
    # Handle the edge case where reinvestment is negative (clamp to 0)
    if reinvestment < 0:
        reinvestment_rate = 0
        data_flags.append("Negative reinvestment — clamped to 0")
    else:
        reinvestment_rate = reinvestment / ebit_after_tax

    # === Step 8: Invested capital ===
    # Invested Capital = Total Equity + Total Debt - Cash
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
            "reinvestment": None,
            "reinvestment_rate": None,
            "invested_capital": invested_capital,
            "roic": None,
            "fundamental_growth": None,
            "tax_rate": tax_rate,
            "data_flags": data_flags,
        }

    # === Step 9: ROIC ===
    # ROIC = ebit_after_tax / invested_capital
    roic = ebit_after_tax/invested_capital

    # === Step 10: Fundamental growth ===
    # g = reinvestment_rate × roic
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
    Compute average annual FCF growth over the last N years.

    Sanity check vs fundamental growth. If they're wildly different,
    something interesting is happening (margin expansion, one-time events,
    business model change, etc.).

    Method: pull the last N+1 years of FCF, compute year-over-year growth
    rates, return the median (more robust than mean since it ignores outliers).

    Returns:
        dict with:
            fcf_series (list): the actual FCF values used
            yoy_growth_rates (list): year-over-year growth rates
            median_growth (float): the answer
            n_years (int): how many growth rates we computed
            data_flags (list): warnings
    """
    print(f"  Computing historical growth for {ticker}...")
    yf_ticker = yf.Ticker(ticker)
    cash_flow = yf_ticker.cashflow

    if cash_flow is None or cash_flow.empty:
        raise ValueError(f"No cash flow data for {ticker}")

    data_flags = []

    # yfinance returns most recent year as first column. Iterate columns
    # left-to-right to get most recent first; reverse to get oldest first.

    fcf_values = []
    for col in cash_flow.columns:
        col_data = cash_flow[col]
        fcf = None
        for field in ["Free Cash Flow", "FreeCashFlow"]:
            if field in col_data.index:
                fcf = float(col_data[field])
                break
        # Some versions don't have FCF directly — derive from operating CF - capex
        if fcf is None:
            ocf = col_data.get("Operating Cash Flow") or col_data.get("Cash Flow From Continuing Operating Activities")
            capex_val = col_data.get("Capital Expenditure")
            if ocf is not None and capex_val is not None:
                fcf = float(ocf) + float(capex_val)

        # Filter out NaN and None
        if fcf is None or math.isnan(fcf):
            continue
        fcf_values.append(fcf)

    # Reverse so oldest is first, newest is last
    fcf_values = fcf_values[::-1]

    # Limit to last N+1 years (for N growth rates)
    if len(fcf_values) > years + 1:
        fcf_values = fcf_values[-(years + 1):]

    if len(fcf_values) < 2:
        data_flags.append("Not enough FCF history to compute growth")
        return {
            "fcf_series": fcf_values,
            "yoy_growth_rates": [],
            "median_growth": None,
            "n_years": 0,
            "data_flags": data_flags,
        }

    # Compute year-over-year growth rates
    yoy_growth_rates = []
    for i in range(1, len(fcf_values)):
        prev_fcf = fcf_values[i - 1]
        curr_fcf = fcf_values[i]
        # Skip if prior year was zero/negative (can't compute % growth meaningfully)
        if prev_fcf <= 0:
            data_flags.append(f"Skipping year {i}: prior FCF was non-positive")
            continue
        growth = (curr_fcf - prev_fcf) / prev_fcf
        yoy_growth_rates.append(growth)

    if not yoy_growth_rates:
        return {
            "fcf_series": fcf_values,
            "yoy_growth_rates": [],
            "median_growth": None,
            "n_years": 0,
            "data_flags": data_flags,
        }

    # Median is more robust than mean — ignores outliers
    sorted_rates = sorted(yoy_growth_rates)
    n = len(sorted_rates)
    if n % 2 == 1:
        median_growth = sorted_rates[n // 2]
    else:
        median_growth = (sorted_rates[n // 2 - 1] + sorted_rates[n // 2]) / 2

    return {
        "fcf_series": fcf_values,
        "yoy_growth_rates": yoy_growth_rates,
        "median_growth": median_growth,
        "n_years": len(yoy_growth_rates),
        "data_flags": data_flags,
    }


def get_consensus_growth(ticker):
    """
    Pull near-term growth expectations from analyst consensus via yfinance.

    Tries multiple sources in order of preference:
        1. 5-year forward earnings growth estimate (if available)
        2. Forward revenue growth (next fiscal year)
        3. Latest TTM revenue growth (fallback)

    Returns:
        dict with:
            consensus_growth (float): the estimate as decimal
            source (str): which estimate was used
            data_flags (list): warnings
    """
    print(f"  Fetching consensus growth for {ticker}...")
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info

    data_flags = []

    # 5-year forward earnings growth (from analyst consensus)
    earnings_growth_5y = info.get("earningsGrowth")  # next-year EPS growth
    # Note: yfinance's "earningsGrowth" is usually next-year, not 5-year.
    # True 5-year estimates are in info.get("earningsQuarterlyGrowth") or
    # accessible via .analyst_price_target — varies by version.

    # Try the long-term growth field first
    consensus = info.get("earningsQuarterlyGrowth")
    source = "earningsQuarterlyGrowth (next quarter consensus)"

    # Fall back to next-year earnings growth
    if consensus is None or consensus <= 0:
        consensus = info.get("earningsGrowth")
        source = "earningsGrowth (next year consensus)"

    # Fall back to revenue growth as proxy
    if consensus is None or consensus <= 0:
        consensus = info.get("revenueGrowth")
        source = "revenueGrowth (TTM, fallback)"
        data_flags.append("Using TTM revenue growth as proxy — no forward estimate available")

    if consensus is None:
        data_flags.append("No consensus growth available")
        return {
            "consensus_growth": None,
            "source": None,
            "data_flags": data_flags,
        }

    # Sanity-check: cap absurd values
    if consensus > 0.50:
        data_flags.append(f"Consensus growth {consensus*100:.0f}% seems implausible — capping at 30%")
        consensus = 0.30
    if consensus < -0.20:
        data_flags.append(f"Consensus growth {consensus*100:.0f}% deeply negative — capping at -10%")
        consensus = -0.10

    return {
        "consensus_growth": consensus,
        "source": source,
        "data_flags": data_flags,
    }


def build_growth_profile(ticker, num_years=10):
    """
    Combine all growth signals into a year-by-year DCF growth schedule.

    Strategy:
        Year 1-2: consensus growth (what analysts/market expect near-term)
        Year 3 to num_years: linear fade from consensus to fundamental
        Terminal: min(fundamental, GDP cap) — capped because no company
                  can grow faster than the economy forever

    Returns:
        dict with:
            yearly_growth (list of floats, length num_years): year-by-year rates
            terminal_growth (float): used in Gordon Growth terminal value
            fundamental_growth (float): the sustainable rate
            consensus_growth (float): the near-term anchor
            historical_growth (float): the sanity check
            roic (float)
            excess_return (float): roic - wacc proxy; positive = value creator
            data_flags (list)
    """
    print(f"\n  === Building growth profile for {ticker} ===")

    # Pull all three growth signals
    fund_info = compute_fundamental_growth(ticker)
    hist_info = compute_historical_growth(ticker)
    cons_info = get_consensus_growth(ticker)

    fund_g = fund_info["fundamental_growth"]
    hist_g = hist_info["median_growth"]
    cons_g = cons_info["consensus_growth"]

    data_flags = []
    data_flags.extend(fund_info["data_flags"])
    data_flags.extend(hist_info["data_flags"])
    data_flags.extend(cons_info["data_flags"])

    # === Decide on near-term growth (years 1-2) ===
    # Damodaran's triangulation: weighted blend of three signals
    #   60% fundamental (most defensible — derived from operating economics)
    #   20% consensus    (market expectation, but analysts skew optimistic)
    #   20% historical   (recent reality)
    # If a signal is missing, we redistribute its weight proportionally.
    weights = {"fundamental": 0.60, "consensus": 0.20, "historical": 0.20}
    available = {}
    if fund_g is not None:
        available["fundamental"] = fund_g
    if cons_g is not None:
        available["consensus"] = cons_g
    if hist_g is not None:
        available["historical"] = hist_g

    if not available:
        data_flags.append("No growth signals available — defaulting to GDP cap")
        near_term_g = GDP_TERMINAL_CAP
    else:
        # Renormalize weights over only the signals we actually have
        total_available_weight = sum(weights[k] for k in available)
        near_term_g = sum(
            (weights[k] / total_available_weight) * available[k]
            for k in available
        )
        used_signals = ", ".join(
            f"{k}={available[k] * 100:.1f}%" for k in available
        )
        data_flags.append(f"Blended near-term from: {used_signals}")

    # Decide on long-term fundamental rate
    if fund_g is not None:
        long_term_g = fund_g
    elif hist_g is not None:
        long_term_g = hist_g
        data_flags.append("No fundamental; using historical as long-term")
    else:
        # Last resort — use 4% terminal cap
        long_term_g = GDP_TERMINAL_CAP
        data_flags.append("No reliable growth estimate; defaulting to GDP cap")

    # Build the year-by-year profile
    yearly_growth = []
    fade_start_year = 2  # years 1 and 2 are near-term; year 3 starts the fade
    fade_end_year = num_years  # last year reaches long_term_g

    for year in range(1, num_years + 1):
        if year <= fade_start_year:
            # Near-term: use consensus
            g = near_term_g
        else:
            # Linear fade from near_term_g to long_term_g
            # year 3 → mostly near-term
            # year num_years → mostly long-term
            fade_progress = (year - fade_start_year) / (fade_end_year - fade_start_year)
            g = near_term_g + fade_progress * (long_term_g - near_term_g)
        yearly_growth.append(g)

    # Terminal growth: cap at GDP
    if long_term_g is None:
        terminal_growth = GDP_TERMINAL_CAP
    else:
        terminal_growth = min(long_term_g, GDP_TERMINAL_CAP)

    return {
        "yearly_growth": yearly_growth,
        "terminal_growth": terminal_growth,
        "fundamental_growth": fund_g,
        "consensus_growth": cons_g,
        "historical_growth": hist_g,
        "roic": fund_info.get("roic"),
        "reinvestment_rate": fund_info.get("reinvestment_rate"),
        "data_flags": data_flags,
    }


if __name__ == "__main__":
    profile = build_growth_profile("MSFT")

    print(f"\n=== MSFT Growth Profile ===")
    print(f"\nGrowth signals:")
    if profile["fundamental_growth"] is not None:
        print(f"  Fundamental (g = reinvest × ROIC): {profile['fundamental_growth']*100:.1f}%")
        print(f"    ROIC:               {profile['roic']*100:.1f}%")
        print(f"    Reinvestment rate:  {profile['reinvestment_rate']*100:.1f}%")
    if profile["historical_growth"] is not None:
        print(f"  Historical (5yr FCF median):       {profile['historical_growth']*100:.1f}%")
    if profile["consensus_growth"] is not None:
        print(f"  Consensus (analyst near-term):     {profile['consensus_growth']*100:.1f}%")

    print(f"\nYear-by-year growth profile:")
    for i, g in enumerate(profile["yearly_growth"], start=1):
        print(f"  Year {i:2d}:  {g*100:.1f}%")
    print(f"  Terminal: {profile['terminal_growth']*100:.1f}%")

    if profile["data_flags"]:
        print(f"\nFlags:")
        for f in profile["data_flags"]:
            print(f"  - {f}")