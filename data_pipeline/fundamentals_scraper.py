"""
Fundamentals scraper - runs WEEKLY, not daily.
Fundamentals barely change so we cache aggressively.
Slow: 3 second gap between tickers, ~30 min total.
"""
import time
import sqlite3
import yfinance as yf
from datetime import datetime
from data_pipeline.database import DB_PATH


def fetch_fundamentals(ticker):
    """Get fundamentals for one ticker from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            'sector': info.get('sector'),
            'pe': info.get('trailingPE'),
            'op_margin': info.get('operatingMargins'),
            'roe': info.get('returnOnEquity'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'debt_equity': info.get('debtToEquity'),
            'free_cash_flow_positive': 1 if info.get('freeCashflow', 0) and info.get('freeCashflow', 0) > 0 else 0,
            'institutional_ownership': info.get('heldPercentInstitutions'),
            'insider_ownership': info.get('heldPercentInsiders'),
        }
    except Exception as e:
        print(f"❌ {ticker}: {e}")
        return None


def save_fundamentals(ticker, data):
    """Write fundamentals to database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO fundamentals
        (ticker, updated_at, sector, pe, op_margin, roe, revenue_growth, 
         earnings_growth, debt_equity, free_cash_flow_positive,
         institutional_ownership, insider_ownership, fund_score, max_score, core_passed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker, datetime.now().isoformat(),
        data.get('sector'), data.get('pe'), data.get('op_margin'), data.get('roe'),
        data.get('revenue_growth'), data.get('earnings_growth'), data.get('debt_equity'),
        data.get('free_cash_flow_positive'),
        data.get('institutional_ownership'), data.get('insider_ownership'),
        None, None, None  # scores computed later
    ))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    from ticker_universe import get_all_tickers

    tickers = get_all_tickers()
    print(f"Scraping fundamentals for {len(tickers)} tickers...")
    print(f"⏱️  Expected duration: {len(tickers) * 3 / 60:.0f} minutes\n")

    success = 0
    for i, ticker in enumerate(tickers):
        data = fetch_fundamentals(ticker)
        if data:
            save_fundamentals(ticker, data)
            success += 1
            if i % 20 == 0:
                print(f"[{i}/{len(tickers)}] {ticker}: sector={data['sector']}, PE={data['pe']}")
        time.sleep(3)  # be polite to Yahoo

    print(f"\n🎉 Done. Saved {success}/{len(tickers)} fundamentals to database.")