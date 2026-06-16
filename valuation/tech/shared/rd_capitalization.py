"""
R&D capitalization for tech companies.

R&D is treated as an expense by standard accounting, but it actually produces
value over multiple years (CUDA was R&D in 2007, still earning today). This
module recasts R&D as a multi-year investment.

Formula (Damodaran):
    1. Treat each year's R&D as a 5-year asset
    2. Asset value = sum of (R&D_year_i × remaining_useful_life_i)
    3. This year's amortization = sum of (R&D_year_i / 5)
    4. Adjusted EBIT = Reported EBIT + R&D - amortization
    5. Adjusted invested capital = Reported IC + R&D asset
"""

import math
import yfinance as yf


# Standard convention for tech R&D useful life.
# Damodaran uses 5 for tech, 10 for pharma. We're tech-focused.
RD_USEFUL_LIFE = 5

# Statutory US corporate tax rate. Used when we adjust EBIT to NOPAT.
# Damodaran's recommendation: use statutory rate for valuation, not effective,
# since effective rates fluctuate year-to-year on temporary items.
DEFAULT_TAX_RATE = 0.21


def capitalize_rd(ticker, useful_life=RD_USEFUL_LIFE):
    """
    Capitalize R&D spending as a multi-year asset.

    Pulls R&D history from yfinance income statement and computes:
    - The R&D "asset" sitting on a paper balance sheet
    - This year's R&D amortization (the spread-out cost recognition)
    - Adjusted operating income (add back R&D, subtract amortization)
    - Adjusted invested capital (original IC + R&D asset)

    Args:
        ticker (str): stock ticker
        useful_life (int): years over which R&D is amortized (default 5)

    Returns:
        dict with:
            rd_history (list): R&D spending, oldest to newest
            rd_asset (float): capitalized R&D on paper balance sheet
            rd_amortization (float): this year's amortization charge
            reported_ebit (float): unadjusted EBIT
            adjusted_ebit (float): EBIT after R&D capitalization
            reported_nopat (float): unadjusted NOPAT
            adjusted_nopat (float): NOPAT after R&D capitalization
            tax_rate (float): tax rate used
            data_flags (list): warnings or notes
    """
    yf_ticker = yf.Ticker(ticker)
    income_stmt = yf_ticker.income_stmt

    data_flags = []

    # Sanity check the data
    if income_stmt is None or income_stmt.empty:
        return _null_result("No income statement data", data_flags)

    if "Research And Development" not in income_stmt.index:
        return _null_result(
            f"{ticker} has no R&D line — capitalization not applicable",
            data_flags,
        )

    # Pull R&D for as many years as yfinance gives us, oldest to newest.
    # yfinance returns columns in newest-first order.
    rd_row = income_stmt.loc["Research And Development"]
    rd_history = []
    for col in income_stmt.columns:
        val = rd_row[col]
        if val is not None and not math.isnan(float(val)) and float(val) > 0:
            rd_history.append(float(val))

    # Reverse to oldest-first ordering, so the most recent year is at the end.
    rd_history = rd_history[::-1]

    if not rd_history:
        return _null_result(f"{ticker} R&D history is empty", data_flags)

    years_available = len(rd_history)
    if years_available < useful_life:
        data_flags.append(
            f"Only {years_available} years of R&D available, less than {useful_life}-year useful life"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Build the capitalized R&D asset
    # ─────────────────────────────────────────────────────────────────────
    # Each historical year's R&D contributes a portion of its original value
    # based on how much useful life remains.
    #
    # If useful_life=5 and we have 4 years of history (oldest to newest):
    #   Y1 (4 years old): 1/5 remaining
    #   Y2 (3 years old): 2/5 remaining
    #   Y3 (2 years old): 3/5 remaining
    #   Y4 (just spent):  4/5 remaining  <- newest year
    #
    # We use 4/5 (not 5/5) for the newest year on the assumption that R&D
    # was spent on average mid-year, so one year of amortization has already
    # been recognized by the time we look at it.
    rd_asset = 0.0
    for i, rd in enumerate(rd_history):
        # How many years ago was this R&D spent?
        # The last item (most recent) has i = len(history) - 1
        years_ago = (len(rd_history) - 1) - i
        # Remaining useful life as a fraction of original
        remaining_life_fraction = max(0, useful_life - years_ago - 1) / useful_life
        rd_asset += rd * remaining_life_fraction

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: This year's R&D amortization
    # ─────────────────────────────────────────────────────────────────────
    # Each capitalized year contributes 1/useful_life of itself to this year's
    # amortization charge. If a year's R&D is older than useful_life, it's
    # fully amortized and contributes nothing further.
    rd_amortization = 0.0
    for i, rd in enumerate(rd_history):
        years_ago = (len(rd_history) - 1) - i
        if years_ago < useful_life:
            rd_amortization += rd / useful_life

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Adjust operating income
    # ─────────────────────────────────────────────────────────────────────
    # Find EBIT (or Operating Income as fallback)
    reported_ebit = None
    latest_income = income_stmt.iloc[:, 0]
    for field in ["EBIT", "Operating Income"]:
        if field in latest_income.index:
            val = latest_income[field]
            if val is not None and not math.isnan(float(val)):
                reported_ebit = float(val)
                break

    if reported_ebit is None:
        return _null_result(f"{ticker}: no EBIT field — likely financial company", data_flags)

    # Current year's R&D (most recent)
    current_rd = rd_history[-1]

    # Adjusted EBIT: add back this year's R&D (it's investment), subtract amortization
    adjusted_ebit = reported_ebit + current_rd - rd_amortization

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Convert to NOPAT using tax rate
    # ─────────────────────────────────────────────────────────────────────
    # Try effective tax rate from yfinance, fall back to statutory
    info = yf_ticker.info
    tax_rate = info.get("effectiveTaxRate") or DEFAULT_TAX_RATE

    reported_nopat = reported_ebit * (1 - tax_rate)
    adjusted_nopat = adjusted_ebit * (1 - tax_rate)

    return {
        "ticker": ticker,
        "rd_history": rd_history,
        "years_available": years_available,
        "rd_asset": rd_asset,
        "rd_amortization": rd_amortization,
        "current_rd": current_rd,
        "reported_ebit": reported_ebit,
        "adjusted_ebit": adjusted_ebit,
        "reported_nopat": reported_nopat,
        "adjusted_nopat": adjusted_nopat,
        "tax_rate": tax_rate,
        "useful_life": useful_life,
        "data_flags": data_flags,
    }


def _null_result(reason, data_flags):
    """Return-shape for failed R&D capitalization."""
    data_flags.append(reason)
    return {
        "ticker": None,
        "rd_history": [],
        "years_available": 0,
        "rd_asset": None,
        "rd_amortization": None,
        "current_rd": None,
        "reported_ebit": None,
        "adjusted_ebit": None,
        "reported_nopat": None,
        "adjusted_nopat": None,
        "tax_rate": None,
        "useful_life": None,
        "data_flags": data_flags,
    }


# ════════════════════════════════════════════════════════════════════════
# Test harness
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    TICKERS = ["MSFT", "NVDA", "META", "AMD", "AAPL"]

    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        result = capitalize_rd(ticker)

        if result["adjusted_nopat"] is None:
            print(f"  Failed: {result['data_flags']}")
            continue

        print(f"\n  R&D history ({result['years_available']} years, oldest to newest):")
        for i, rd in enumerate(result['rd_history']):
            years_ago = (len(result['rd_history']) - 1) - i
            print(f"    {years_ago} years ago: ${rd/1e9:.1f}B")

        print(f"\n  Capitalized R&D asset:    ${result['rd_asset']/1e9:.1f}B")
        print(f"  This year's amortization: ${result['rd_amortization']/1e9:.1f}B")
        print(f"  Current year R&D:         ${result['current_rd']/1e9:.1f}B")

        print(f"\n  EBIT comparison:")
        print(f"    Reported:               ${result['reported_ebit']/1e9:.1f}B")
        print(f"    Adjusted:               ${result['adjusted_ebit']/1e9:.1f}B")
        ebit_change_pct = (result['adjusted_ebit'] / result['reported_ebit'] - 1) * 100
        print(f"    Change:                 {ebit_change_pct:+.1f}%")

        print(f"\n  NOPAT comparison:")
        print(f"    Reported:               ${result['reported_nopat']/1e9:.1f}B")
        print(f"    Adjusted:               ${result['adjusted_nopat']/1e9:.1f}B")

        if result['data_flags']:
            print(f"\n  Flags:")
            for f in result['data_flags']:
                print(f"    - {f}")