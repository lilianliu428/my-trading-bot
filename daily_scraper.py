"""
Daily scraper - runs once per day after market close.
Fetches prices for all tickers and writes to database.
"""
import sqlite3
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import ALPACA_API_KEY, ALPACA_SECRET
from database import DB_PATH

# Initialize Alpaca client once
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET)


def fetch_bars_for_tickers(tickers, days=350):
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start,
        end=end
    )

    print(f"Fetching {len(tickers)} tickers from Alpaca...")

    # retry up to 3 times
    for attempt in range(3):
        try:
            bars = client.get_stock_bars(request)
            return bars.df
        except Exception as e:
            print(f"⚠️  Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                import time
                time.sleep(5)

    print(f"❌ Failed to fetch batch after 3 attempts")
    return None

def calculate_indicators(df_for_one_ticker):
    """
    Given a dataframe of one ticker's price history,
    return the latest snapshot with all indicators calculated.
    """
    closes = df_for_one_ticker['close']

    # RSI calculation (14-day)
    delta = closes.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Moving averages
    ma_20 = closes.rolling(window=20).mean()
    ma_50 = closes.rolling(window=50).mean()
    ma_200 = closes.rolling(window=200).mean()

    latest = {
        'price': float(closes.iloc[-1]),
        'rsi': float(rsi.iloc[-1]) if not rsi.iloc[-1] != rsi.iloc[-1] else None,  # NaN check
        'ma_20': float(ma_20.iloc[-1]) if not ma_20.iloc[-1] != ma_20.iloc[-1] else None,
        'ma_50': float(ma_50.iloc[-1]) if not ma_50.iloc[-1] != ma_50.iloc[-1] else None,
        'ma_200': float(ma_200.iloc[-1]) if not ma_200.iloc[-1] != ma_200.iloc[-1] else None,
        'volume': int(df_for_one_ticker['volume'].iloc[-1]),
        'date': df_for_one_ticker.index[-1].strftime('%Y-%m-%d'),
    }
    return latest


def save_snapshots(snapshots):
    """Write snapshots to database."""
    conn = sqlite3.connect(DB_PATH)
    for ticker, data in snapshots.items():
        conn.execute("""
            INSERT OR REPLACE INTO snapshots 
            (ticker, date, price, rsi, ma_20, ma_50, ma_200, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, data['date'], data['price'], data['rsi'],
              data['ma_20'], data['ma_50'], data['ma_200'], data['volume']))
    conn.commit()
    conn.close()
    print(f"✅ Saved {len(snapshots)} snapshots to database")


def run_daily_scrape(tickers):
    """Main entry point - run the full daily scrape."""
    # Fetch all data in one batch (Alpaca allows multi-ticker requests)
    df = fetch_bars_for_tickers(tickers)
    if df is None:
        return {}  # skip this batch
    # Process per ticker
    snapshots = {}
    for ticker in tickers:
        try:
            ticker_data = df.loc[ticker]
            if len(ticker_data) < 14:  # need at least 14 days for RSI
                print(f"⚠️  Skipping {ticker}: not enough data")
                continue
            snapshots[ticker] = calculate_indicators(ticker_data)
        except KeyError:
            print(f"⚠️  No data for {ticker}")
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")

    # Save to DB
    save_snapshots(snapshots)
    return snapshots


if __name__ == "__main__":
    from scanner import get_all_tickers

    all_tickers = get_all_tickers()
    print(f"Loaded {len(all_tickers)} unique tickers")

    # Alpaca recommends batches of 100 max per request
    BATCH_SIZE = 50
    all_snapshots = {}
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i + BATCH_SIZE]
        print(f"\n--- Batch {i // BATCH_SIZE + 1}/{(len(all_tickers) - 1) // BATCH_SIZE + 1} ---")
        snapshots = run_daily_scrape(batch)
        all_snapshots.update(snapshots)

    print(f"\n🎉 Total: {len(all_snapshots)} snapshots saved to database")