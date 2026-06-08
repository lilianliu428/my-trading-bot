"""
Compute synthetic cost of debt using Damodaran's interest coverage method.

Steps:
    1. Compute interest coverage ratio = EBIT / Interest Expense
    2. Look up synthetic credit rating from the coverage ratio
    3. Map rating to credit spread over risk-free rate
    4. Cost of debt (pre-tax) = risk-free rate + credit spread
    5. Cost of debt (after-tax) = pre-tax × (1 - tax rate)

Source: Damodaran's published "Ratings, Interest Coverage Ratios and Default
Spread" table. Updated manually.

Reference: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html
Last updated: 2026-06-08
"""

import yfinance as yf

# Damodaran's synthetic rating table.
# Each entry: (min_coverage, max_coverage, rating, spread_over_risk_free).
# Use the FIRST row your coverage falls into, scanning from highest to lowest.
SYNTHETIC_RATING_TABLE = [
    (12.5, float("inf"), "AAA",  0.0059),
    (9.5,  12.5,         "AA",   0.0081),
    (7.5,  9.5,          "A+",   0.0107),
    (6.0,  7.5,          "A",    0.0120),
    (4.5,  6.0,          "A-",   0.0141),
    (4.0,  4.5,          "BBB+", 0.0181),
    (3.0,  4.0,          "BBB",  0.0207),
    (2.5,  3.0,          "BB+",  0.0340),
    (2.0,  2.5,          "BB",   0.0400),
    (1.5,  2.0,          "B+",   0.0550),
    (1.0,  1.5,          "B",    0.0675),
    (0.5,  1.0,          "CCC",  0.0850),
    (-float("inf"), 0.5, "D",    0.1200),
]


def lookup_synthetic_rating(coverage_ratio):
    """
    Map an interest coverage ratio to (rating, spread).

    Returns:
        (rating: str, spread: float)
    """
    for min_cov, max_cov, rating, spread in SYNTHETIC_RATING_TABLE:
        if min_cov <= coverage_ratio < max_cov:
            return rating, spread
    # Should never get here, but just in case
    return "D", 0.1200


def compute_cost_of_debt(ticker, risk_free_rate):
    """
    Compute synthetic pre-tax and after-tax cost of debt for a ticker.

    Args:
        ticker: e.g. "MSFT"
        risk_free_rate: e.g. 0.0447

    Returns:
        dict with:
            ebit (float)
            interest_expense (float)
            interest_coverage (float)
            synthetic_rating (str)
            credit_spread (float)
            pre_tax_cost_of_debt (float)
            tax_rate (float)
            after_tax_cost_of_debt (float)
    """
    print(f"  Fetching financials for {ticker}...")
    yf_ticker = yf.Ticker(ticker)

    # Pull annual income statement
    income_stmt = yf_ticker.income_stmt  # most recent column = most recent fiscal year

    if income_stmt is None or income_stmt.empty:
        raise ValueError(f"No income statement data for {ticker}")

    # The most recent year is the first column
    latest = income_stmt.iloc[:, 0]

    # EBIT (Operating Income)
    if "EBIT" in latest.index:
        ebit = float(latest["EBIT"])
    elif "Operating Income" in latest.index:
        ebit = float(latest["Operating Income"])
    else:
        raise ValueError(f"No EBIT/Operating Income found for {ticker}")

    # Interest Expense (always positive number in yfinance)
    if "Interest Expense" in latest.index:
        interest_expense = float(latest["Interest Expense"])
    else:
        raise ValueError(f"No Interest Expense found for {ticker}")

    # Sanity check
    if interest_expense <= 0:
        # Company has no debt or no interest expense; treat as effectively AAA
        interest_coverage = float("inf")
    else:
        interest_coverage = ebit / interest_expense

    # Get synthetic rating and spread
    rating, spread = lookup_synthetic_rating(interest_coverage)

    # Cost of debt
    pre_tax_cost_of_debt = risk_free_rate + spread

    # Tax rate — try to pull from yfinance, fall back to 21% (US corporate rate)
    info = yf_ticker.info
    tax_rate = info.get("effectiveTaxRate") or 0.21

    after_tax_cost_of_debt = pre_tax_cost_of_debt * (1 - tax_rate)

    return {
        "ebit": ebit,
        "interest_expense": interest_expense,
        "interest_coverage": interest_coverage,
        "synthetic_rating": rating,
        "credit_spread": spread,
        "pre_tax_cost_of_debt": pre_tax_cost_of_debt,
        "tax_rate": tax_rate,
        "after_tax_cost_of_debt": after_tax_cost_of_debt,
    }


if __name__ == "__main__":
    from valuation.data_sources.fred import get_risk_free_rate

    rf = get_risk_free_rate()
    result = compute_cost_of_debt("MSFT", rf)

    print(f"\nMSFT Cost of Debt Analysis")
    print(f"  EBIT:                  ${result['ebit']/1e9:.1f}B")
    print(f"  Interest expense:      ${result['interest_expense']/1e9:.2f}B")
    print(f"  Interest coverage:     {result['interest_coverage']:.1f}x")
    print(f"  Synthetic rating:      {result['synthetic_rating']}")
    print(f"  Credit spread:         {result['credit_spread']*100:.2f}%")
    print(f"  Pre-tax cost of debt:  {result['pre_tax_cost_of_debt']*100:.2f}%")
    print(f"  Tax rate:              {result['tax_rate']*100:.1f}%")
    print(f"  After-tax cost of debt: {result['after_tax_cost_of_debt']*100:.2f}%")