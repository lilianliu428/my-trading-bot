import yfinance as yf
import requests
import time
from fundamentals import check_fundamentals
from config import (DROP_THRESHOLD, RISE_THRESHOLD,
                    RSI_OVERSOLD, RSI_OVERBOUGHT, FUNDAMENTAL_MIN_SCORE)
from concurrent.futures import ThreadPoolExecutor, as_completed

class ta:
    @staticmethod
    def rsi(close, length=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=length).mean()
        avg_loss = loss.rolling(window=length).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

def get_all_tickers():
    import pandas as pd
    all_tickers = set()

    # S&P 500
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df = pd.read_csv(url)
        sp500 = df["Symbol"].tolist()
        all_tickers.update(sp500)
        print(f"Loaded {len(sp500)} S&P 500 tickers")
    except Exception as e:
        print(f"S&P 500 failed: {e}")

    # NASDAQ 100 only — not all 3000
    nasdaq100 = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "GOOG",
        "AVGO", "COST", "NFLX", "TMUS", "AMD", "PEP", "LIN", "CSCO", "ADBE",
        "TXN", "QCOM", "INTU", "AMGN", "AMAT", "ISRG", "MU", "BKNG", "LRCX",
        "REGN", "KLAC", "ADI", "PANW", "MDLZ", "INTC", "VRTX", "ADP", "GILD",
        "SBUX", "MELI", "SNPS", "CDNS", "ASML", "MAR", "CTAS", "CEG", "PYPL",
        "FTNT", "ORLY", "NXPI", "MRVL", "CRWD", "ABNB", "PCAR", "WDAY", "DDOG",
        "CHTR", "CPRT", "MNST", "TTWO", "TEAM", "FAST", "ODFL", "VRSK", "BIIB",
        "IDXX", "FANG", "ZS", "ANSS", "ILMN", "DXCM", "WBD", "SIRI", "DLTR",
        "ALGN", "ENPH", "JD", "LCID", "MTCH", "OKTA", "PAYX", "PDD", "ROST",
        "SGEN", "SPLK", "SWKS", "VRSN", "WBA", "XCEL", "XEL", "ZM", "COIN",
        "PLTR", "RBLX", "RIVN", "HOOD", "DDOG", "NET", "SNOW", "PATH", "U",
        "AFRM", "BILL", "COUP", "GTLB", "HUBS", "MDB", "PCTY", "SMAR", "TOST"
    ]
    all_tickers.update(nasdaq100)
    print(f"Added NASDAQ 100")

    # Dow Jones 30
    dow30 = [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "DOW", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD",
        "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WMT"
    ]
    all_tickers.update(dow30)
    print(f"Added Dow Jones 30")

    tickers = list(all_tickers)
    print(f"Total unique tickers: {len(tickers)}")
    return tickers

def analyze_stock(ticker):
    import time

    # try up to 3 times with backoff if rate limited
    hist = None
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")  # need 200+ days for MAs
            if not hist.empty and len(hist) >= 15:
                break
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            if "rate" in str(e).lower() or "too many" in str(e).lower():
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))  # 3s, 6s
                    continue
            print(f"Error analyzing {ticker}: {e}")
            return None

    if hist is None or hist.empty or len(hist) < 15:
        return None

    try:

        hist["RSI"] = ta.rsi(hist["Close"], length=14)
        latest_rsi = hist["RSI"].iloc[-1]
        current_price = hist["Close"].iloc[-1]

        high_30d = hist["Close"].tail(30).max()
        low_30d = hist["Close"].tail(30).min()
        drop_pct = ((current_price - high_30d) / high_30d) * 100
        rise_pct = ((current_price - low_30d) / low_30d) * 100

        # fetch enough history for moving averages
        hist_full = stock.history(period="1y")
        fund_score, max_score, core_passed, fund_reasons = check_fundamentals(ticker, hist_full)
        strong = core_passed == 5 or fund_score >= FUNDAMENTAL_MIN_SCORE

        signal = None
        emoji = ""

        if drop_pct < DROP_THRESHOLD and latest_rsi < RSI_OVERSOLD:
            signal = "BUY CANDIDATE" if strong else "POTENTIAL TRAP"
            emoji = "🔍" if strong else "⚠️"

        elif rise_pct > RISE_THRESHOLD and latest_rsi > RSI_OVERBOUGHT:
            signal = "MOMENTUM CONFIRMED" if strong else "TAKE PROFIT / AVOID"
            emoji = "🚀" if strong else "📈"

        if not signal:
            return None

        fund_summary = "\n".join(fund_reasons)
        return (
            f"{emoji} {signal}: {ticker}\n"
            f"Price: ${current_price:.2f}\n"
            f"RSI: {latest_rsi:.1f}\n"
            f"Drop: {drop_pct:.1f}% | Rise: {rise_pct:.1f}%\n"
            f"Fundamentals: {fund_score:.1f}/{max_score} (core {core_passed}/5)\n"
            f"{fund_summary}\n"
        )

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None


def scan_tickers_parallel(tickers, max_workers=20):
    results = []
    completed = 0
    total = len(tickers)

    # split into batches to avoid rate limiting
    batch_size = 50
    batches = [tickers[i:i + batch_size] for i in range(0, total, batch_size)]

    for batch in batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(analyze_stock, ticker): ticker
                for ticker in batch
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1
                if completed % 50 == 0:
                    print(f"Progress: {completed}/{total} stocks scanned")
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error with {ticker}: {e}")

        # pause between batches so Yahoo Finance doesn't block us
        print(f"Batch done, pausing...")
        time.sleep(1.5)

    return results