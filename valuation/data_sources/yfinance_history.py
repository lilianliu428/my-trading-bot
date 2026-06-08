"""
Pull historical daily prices from yfinance and store in price_history table.

This is a one-time backfill operation per ticker, but can be re-run to update
with newer data. Existing rows (ticker + date) are replaced via INSERT OR REPLACE.
"""

import sqlite3
from datetime import datetime, timedelta
import yfinance as yf

DB_PATH = "/home/ubuntu/my-trading-bot/data.db"  # adjust if running locally


def backfill_price_history(ticker, years=5, db_path=DB_PATH):
    """
    Pull `years` of daily price data from yfinance for the given ticker
    and insert into the price_history table.

    Args:
        ticker: e.g. "MSFT"
        years: how many years of history to pull
        db_path: path to SQLite database

    Returns:
        int: number of rows inserted/updated
    """
    print(f"  Fetching {years}yr of daily prices for {ticker}...")

    end = datetime.now()
    start = end - timedelta(days=365 * years + 30)  # extra buffer

    # yfinance call
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,  # we want both close AND adj_close
    )

    if df.empty:
        print(f"  ⚠ No data returned for {ticker}")
        return 0

    # yfinance returns multi-index columns sometimes (yfinance version-dependent)
    # Flatten if needed
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    # Insert into DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    rows_inserted = 0
    for date, row in df.iterrows():
        cursor.execute(
            """
            INSERT OR REPLACE INTO price_history
            (ticker, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                date.strftime("%Y-%m-%d"),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                float(row["Adj Close"]),
                int(row["Volume"]),
            ),
        )
        rows_inserted += 1

    conn.commit()
    conn.close()

    print(f"  ✓ Inserted/updated {rows_inserted} rows for {ticker}")
    return rows_inserted


if __name__ == "__main__":
    # Backfill MSFT and SPY for beta computation
    backfill_price_history("MSFT", years=5)
    backfill_price_history("SPY", years=5)