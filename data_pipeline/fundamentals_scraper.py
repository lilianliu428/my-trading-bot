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

        fcf = info.get('freeCashflow')
        return {
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'op_margin': info.get('operatingMargins'),
            'profit_margin': info.get('profitMargins'),
            'gross_margin': info.get('grossMargins'),
            'roe': info.get('returnOnEquity'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'quarterly_earnings_growth': info.get('earningsQuarterlyGrowth'),
            'debt_equity': info.get('debtToEquity'),
            'free_cash_flow_positive': 1 if fcf and fcf > 0 else 0,
            'free_cash_flow': fcf,
            'total_revenue': info.get('totalRevenue'),
            'book_value': info.get('bookValue'),
            'price_to_book': info.get('priceToBook'),
            'total_cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'current_ratio': info.get('currentRatio'),
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
        (ticker, updated_at, sector, industry, pe, forward_pe, peg_ratio,
         op_margin, profit_margin, gross_margin, roe,
         revenue_growth, earnings_growth, quarterly_earnings_growth,
         debt_equity, free_cash_flow_positive, free_cash_flow, total_revenue,
         book_value, price_to_book, total_cash, total_debt, current_ratio,
         institutional_ownership, insider_ownership,
         business_model, fund_score, max_score, core_passed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker, datetime.now().isoformat(),
        data.get('sector'), data.get('industry'), data.get('pe'),
        data.get('forward_pe'), data.get('peg_ratio'),
        data.get('op_margin'), data.get('profit_margin'), data.get('gross_margin'),
        data.get('roe'),
        data.get('revenue_growth'), data.get('earnings_growth'),
        data.get('quarterly_earnings_growth'),
        data.get('debt_equity'), data.get('free_cash_flow_positive'),
        data.get('free_cash_flow'), data.get('total_revenue'),
        data.get('book_value'), data.get('price_to_book'),
        data.get('total_cash'), data.get('total_debt'), data.get('current_ratio'),
        data.get('institutional_ownership'), data.get('insider_ownership'),
        None, None, None, None  # business_model + scores filled in later
    ))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    from data_pipeline.ticker_universe import get_all_tickers
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
