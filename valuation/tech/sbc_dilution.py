"""
Stock-based compensation dilution modeling.

Tech companies pay employees in newly-issued stock. Over time, this dilutes
existing shareholders. For high-SBC companies (PLTR, SNOW, DDOG), share count
can grow 3-5% per year, shrinking each share's claim on the business.

Some companies offset SBC with buybacks (AAPL, MSFT). For shareholders, what
matters is the NET share count change. This module measures that and projects
it forward over the DCF horizon.

Strategy:
    1. Pull share count history from yfinance
    2. Compute compound annual growth rate over last 3 years
    3. Cap at ±5% to prevent unrealistic projections
    4. Project forward for 10 years
"""

import math
import yfinance as yf


# How many years of history to use for the dilution rate calculation
LOOKBACK_YEARS = 3

# Cap on annual rate of change. Real companies rarely sustain >5% net dilution
# or buyback rates for a full decade. Capping avoids absurd projections.
MAX_ANNUAL_RATE = 0.05

# How many years to project forward (matches DCF horizon)
PROJECTION_YEARS = 10


def project_share_count(ticker, current_shares=None, years=PROJECTION_YEARS):
    """
    Project share count growth based on historical net change.

    Computes the compound annual growth rate of share count over the last 3
    years (capturing both SBC dilution and buybacks), then projects forward.

    Args:
        ticker (str): stock ticker
        current_shares (float, optional): override current shares. If None,
            pulled from yfinance info.
        years (int): how many years forward to project

    Returns:
        dict with:
            current_shares (float)
            historical_dilution_rate (float): annual % change, signed
                positive = dilution, negative = buyback
            projected_shares (list): year-by-year projected share counts
            terminal_shares (float): share count at end of projection
            data_flags (list)
    """
    yf_ticker = yf.Ticker(ticker)
    data_flags = []

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Get current share count
    # ─────────────────────────────────────────────────────────────────────
    if current_shares is None:
        info = yf_ticker.info
        current_shares = info.get("sharesOutstanding")
        if current_shares is None or current_shares <= 0:
            return _null_result("No current share count available", data_flags)
        current_shares = float(current_shares)

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Get historical share count from balance sheet
    # ─────────────────────────────────────────────────────────────────────
    # yfinance exposes diluted average shares on the income statement
    income_stmt = yf_ticker.income_stmt

    if income_stmt is None or income_stmt.empty:
        return _null_result("No income statement for share history", data_flags)

    # Try a few field names — yfinance varies by company
    share_history = None
    for field in [
        "Diluted Average Shares",
        "Basic Average Shares",
        "Diluted NI Availto Com Stockholders",  # fallback
    ]:
        if field in income_stmt.index:
            share_history = income_stmt.loc[field]
            break

    if share_history is None:
        return _null_result(
            "No share count history available in income statement",
            data_flags,
        )

    # Extract valid numeric values (oldest to newest after reversal)
    historical_shares = []
    for col in income_stmt.columns:
        val = share_history[col]
        if val is not None and not math.isnan(float(val)) and float(val) > 0:
            historical_shares.append(float(val))

    if len(historical_shares) < 2:
        return _null_result("Not enough share history (need 2+ years)", data_flags)

    # yfinance returns newest-first; reverse to chronological
    historical_shares = historical_shares[::-1]

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Compute compound annual growth rate
    # ─────────────────────────────────────────────────────────────────────
    # Use up to LOOKBACK_YEARS of history
    n_years_available = len(historical_shares)
    lookback = min(LOOKBACK_YEARS, n_years_available - 1)

    if lookback < 1:
        return _null_result("Cannot compute rate from less than 2 data points", data_flags)

    oldest = historical_shares[-(lookback + 1)]  # 'lookback' years ago
    newest = historical_shares[-1]  # most recent year

    if oldest <= 0:
        return _null_result("Historical share count zero or negative", data_flags)

    # CAGR formula: (end / start) ^ (1/years) - 1
    raw_rate = (newest / oldest) ** (1.0 / lookback) - 1.0

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Cap at ±5% to prevent extreme projections
    # ─────────────────────────────────────────────────────────────────────
    if raw_rate > MAX_ANNUAL_RATE:
        capped_rate = MAX_ANNUAL_RATE
        data_flags.append(
            f"Dilution rate {raw_rate*100:.2f}% capped at {MAX_ANNUAL_RATE*100:.1f}% — "
            f"extreme rates rarely sustained over a decade"
        )
    elif raw_rate < -MAX_ANNUAL_RATE:
        capped_rate = -MAX_ANNUAL_RATE
        data_flags.append(
            f"Buyback rate {raw_rate*100:.2f}% capped at -{MAX_ANNUAL_RATE*100:.1f}% — "
            f"extreme rates rarely sustained over a decade"
        )
    else:
        capped_rate = raw_rate

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Project share count forward
    # ─────────────────────────────────────────────────────────────────────
    projected_shares = []
    shares_running = current_shares
    for year in range(1, years + 1):
        shares_running = shares_running * (1 + capped_rate)
        projected_shares.append(shares_running)

    terminal_shares = projected_shares[-1]
    total_change = (terminal_shares / current_shares - 1) * 100

    if abs(total_change) > 5:  # only flag meaningful changes
        direction = "growth" if total_change > 0 else "shrinkage"
        data_flags.append(
            f"Projected {abs(total_change):.0f}% share count {direction} over {years} years"
        )

    return {
        "ticker": ticker,
        "current_shares": current_shares,
        "historical_dilution_rate": capped_rate,
        "raw_dilution_rate": raw_rate,
        "lookback_years": lookback,
        "projected_shares": projected_shares,
        "terminal_shares": terminal_shares,
        "data_flags": data_flags,
    }


def _null_result(reason, data_flags):
    """Return-shape for failed computation."""
    data_flags.append(reason)
    return {
        "ticker": None,
        "current_shares": None,
        "historical_dilution_rate": None,
        "raw_dilution_rate": None,
        "lookback_years": None,
        "projected_shares": [],
        "terminal_shares": None,
        "data_flags": data_flags,
    }


# ════════════════════════════════════════════════════════════════════════
# Test harness
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Mix of: aggressive buyback (AAPL), neutral (MSFT), moderate dilution
    # (NVDA, GOOGL), heavy dilution (PLTR, DDOG, SNOW)
    TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "PLTR", "DDOG", "SNOW", "CRWD"]

    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        result = project_share_count(ticker)

        if result["current_shares"] is None:
            print(f"  Failed: {result['data_flags']}")
            continue

        print(f"\n  Current shares:               {result['current_shares']/1e9:.3f}B")
        print(f"  Raw dilution rate (3yr):      {result['raw_dilution_rate']*100:+.2f}% per year")
        print(f"  Capped dilution rate:         {result['historical_dilution_rate']*100:+.2f}% per year")
        print(f"  Lookback years used:          {result['lookback_years']}")

        print(f"\n  Projection (10 years):")
        for i in [0, 4, 9]:  # show year 1, 5, 10
            yr = i + 1
            shares = result['projected_shares'][i]
            pct_change = (shares / result['current_shares'] - 1) * 100
            print(f"    Year {yr:>2}:                     {shares/1e9:.3f}B ({pct_change:+.1f}%)")

        if result['data_flags']:
            print(f"\n  Flags:")
            for f in result['data_flags']:
                print(f"    - {f}")