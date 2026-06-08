"""
FRED API client for pulling macro data.

FRED = Federal Reserve Economic Data, maintained by St. Louis Fed.
Free, well-maintained, includes Treasury yields, inflation, GDP, etc.

Setup:
    1. Get an API key from https://fredaccount.stlouisfed.org/apikey
    2. Add to .env: FRED_API_KEY=your_key_here
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

_FRED_CLIENT = None


def _get_client():
    """Lazy-initialize the FRED client (don't create until first use)."""
    global _FRED_CLIENT
    if _FRED_CLIENT is None:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise ValueError(
                "FRED_API_KEY not found in environment. "
                "Add it to .env: FRED_API_KEY=your_key_here"
            )
        _FRED_CLIENT = Fred(api_key=api_key)
    return _FRED_CLIENT


def get_risk_free_rate():
    """
    Get the current US 10-year Treasury yield as a decimal.

    Returns:
        float: e.g. 0.0445 for 4.45%

    FRED publishes this daily. We grab the most recent non-null value
    (sometimes today's reading isn't posted yet, so we look back ~10 days
    to be safe).

    Series ID: DGS10 (10-Year Treasury Constant Maturity Rate)
    Note: FRED returns the rate as a percentage (e.g., 4.45), so we divide
    by 100 to convert to a decimal (0.0445) for use in formulas.
    """
    client = _get_client()
    # Get last 10 days of data; sometimes the latest day is missing
    end = datetime.now()
    start = end - timedelta(days=10)
    series = client.get_series("DGS10", observation_start=start, observation_end=end)
    # Drop NaN values and take the most recent
    series = series.dropna()
    if series.empty:
        raise ValueError("No recent 10-year Treasury data from FRED.")
    latest_value = series.iloc[-1]  # most recent observation
    latest_date = series.index[-1]
    print(f"  [FRED] 10Y Treasury: {latest_value:.2f}% (as of {latest_date.date()})")
    return latest_value / 100  # convert percentage to decimal


if __name__ == "__main__":
    # Smoke test
    rate = get_risk_free_rate()
    print(f"Risk-free rate: {rate:.4f} ({rate*100:.2f}%)")