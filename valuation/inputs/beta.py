"""
Compute beta from historical daily returns.

β = Cov(stock_returns, market_returns) / Var(market_returns)

We use SPY as the market proxy (S&P 500 ETF, widely accepted standard).
Default window: 5 years of daily returns (Damodaran's preference).
"""

import sqlite3
import numpy as np

DB_PATH = "/home/ubuntu/my-trading-bot/data.db"  # server path


def fetch_adj_close_series(ticker, db_path=DB_PATH):
    """
    Fetch adjusted close prices for a ticker, ordered by date ascending.

    Returns:
        list of tuples: [(date_str, adj_close), ...]
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT date, adj_close FROM price_history WHERE ticker = ? ORDER BY date ASC",
        (ticker,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def compute_daily_returns(prices):
    """
    Convert a list of prices into daily percentage returns.

    Formula: return_t = (price_t - price_{t-1}) / price_{t-1}

    Args:
        prices: list of floats (adjusted close prices, ordered by date)

    Returns:
        list of floats (length = len(prices) - 1; first day has no prior day to compare)

    Example:
        prices = [100, 102, 101, 105]
        returns = [0.02, -0.0098, 0.0396]
    """
    # TODO: implement
    returns = []
    for i in range(1, len(prices)):
        return_t = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(return_t)
    return returns


def compute_beta(ticker, market="SPY", db_path=DB_PATH):
    """
    Compute raw and adjusted beta for a ticker vs market proxy.

    Steps:
        1. Fetch adj_close series for both ticker and market
        2. Align dates (only use days where BOTH have data)
        3. Compute daily returns for each
        4. Compute Cov(ticker_returns, market_returns)
        5. Compute Var(market_returns)
        6. raw_beta = Cov / Var
        7. adjusted_beta = 0.67 × raw_beta + 0.33 × 1.0

    Returns:
        dict with:
            raw_beta (float)
            adjusted_beta (float)
            n_observations (int): days used in calculation
            start_date, end_date (str)
    """
    # Step 1: Fetch both series
    ticker_data = fetch_adj_close_series(ticker, db_path)
    market_data = fetch_adj_close_series(market, db_path)

    if not ticker_data or not market_data:
        raise ValueError(f"Missing price data for {ticker} or {market}")

    # Step 2: Align dates — convert to dicts for fast lookup, then find common dates
    ticker_dict = dict(ticker_data)  # {date_str: price}
    market_dict = dict(market_data)
    common_dates = sorted(set(ticker_dict.keys()) & set(market_dict.keys()))

    if len(common_dates) < 30:
        raise ValueError(f"Not enough overlapping days: {len(common_dates)}")

    ticker_prices = [ticker_dict[d] for d in common_dates]
    market_prices = [market_dict[d] for d in common_dates]

    # Step 3: Compute returns
    ticker_returns = compute_daily_returns(ticker_prices)
    market_returns = compute_daily_returns(market_prices)

    # Step 4-5: Compute covariance and variance
    # Hint: use numpy's np.cov() — returns a 2x2 matrix:
    #   [[Var(x), Cov(x,y)],
    #    [Cov(y,x), Var(y)]]
    # You want cov_matrix[0][1] for covariance and cov_matrix[1][1] for market variance
    cov_matrix = np.cov(ticker_returns, market_returns)
    covariance = cov_matrix[0][1]
    market_variance = cov_matrix[1][1]

    # Step 6: Raw beta
    raw_beta = covariance / market_variance

    # Step 7: Adjusted beta (Blume adjustment)
    adjusted_beta = 0.67 * raw_beta + 0.33 * 1.0

    return {
        "raw_beta": raw_beta,
        "adjusted_beta": adjusted_beta,
        "n_observations": len(common_dates) - 1,  # -1 because returns has one less day
        "start_date": common_dates[0],
        "end_date": common_dates[-1],
    }


if __name__ == "__main__":
    result = compute_beta("MSFT")
    print(f"MSFT Beta Analysis")
    print(f"  Period:         {result['start_date']} to {result['end_date']}")
    print(f"  Observations:   {result['n_observations']} trading days")
    print(f"  Raw beta:       {result['raw_beta']:.3f}")
    print(f"  Adjusted beta:  {result['adjusted_beta']:.3f}")