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

        # S&P MidCap 400
    # S&P MidCap 400
    try:
        midcap_url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        response = requests.get(midcap_url, headers=headers)
        midcap_df = pd.read_html(response.text)[0]
        midcap = midcap_df['Symbol'].tolist()
        all_tickers.update(midcap)
        print(f"Added {len(midcap)} S&P MidCap 400 tickers")
    except Exception as e:
        print(f"Error loading MidCap 400: {e}")

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

    # Major sector ETFs (for market regime + benchmarking)
    etfs = [
        "SPY",  # S&P 500
        "QQQ",  # NASDAQ 100
        "IWM",  # Russell 2000
        "DIA",  # Dow Jones
        "XLF",  # Financials
        "XLE",  # Energy
        "XLK",  # Technology
        "XLV",  # Healthcare
        "XLY",  # Consumer Discretionary
        "XLP",  # Consumer Staples
        "XLI",  # Industrials
        "XLU",  # Utilities
        "XLB",  # Materials
        "XLRE",  # Real Estate
        "XLC",  # Communication Services
        "VTI",  # Total US Market
        "VOO",  # S&P 500 alt
        "VXUS",  # International
    ]
    all_tickers.update(etfs)
    print(f"Added {len(etfs)} major ETFs")

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

        fund_score, max_score, core_passed, fund_reasons = check_fundamentals(ticker, hist)
        strong = core_passed == 5 or fund_score >= FUNDAMENTAL_MIN_SCORE

        signal = None
        emoji = ""

        # main signals (act-now)
        if drop_pct < DROP_THRESHOLD and latest_rsi < RSI_OVERSOLD:
            signal = "BUY CANDIDATE" if strong else "POTENTIAL TRAP"
            emoji = "🔍" if strong else "⚠️"

        elif rise_pct > RISE_THRESHOLD and latest_rsi > RSI_OVERBOUGHT:
            signal = "MOMENTUM CONFIRMED" if strong else "TAKE PROFIT / AVOID"
            emoji = "🚀" if strong else "📈"

        # warm zone signals (watch-only)
        elif strong and drop_pct < -3 and latest_rsi < 50:
            signal = "APPROACHING BUY"
            emoji = "👀"

        elif strong and abs(drop_pct) < 3 and 40 < latest_rsi < 60:
            signal = "STRONG WATCH"
            emoji = "📋"

        if not signal:
            return None

        # categorize for filtering
        if signal in ("BUY CANDIDATE", "POTENTIAL TRAP", "MOMENTUM CONFIRMED", "TAKE PROFIT / AVOID"):
            category = "main"
        elif signal == "APPROACHING BUY":
            category = "approaching_buy"
        elif signal == "STRONG WATCH":
            category = "strong_watch"
        else:
            category = "other"

        fund_summary = "\n".join(fund_reasons)
        message = (
            f"{emoji} {signal}: {ticker}\n"
            f"Price: ${current_price:.2f}\n"
            f"RSI: {latest_rsi:.1f}\n"
            f"Drop: {drop_pct:.1f}% | Rise: {rise_pct:.1f}%\n"
            f"Fundamentals: {fund_score:.1f}/{max_score} (core {core_passed}/5)\n"
            f"{fund_summary}\n"
        )

        return {"message": message, "category": category, "signal": signal, "ticker": ticker}

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None


def scan_tickers_parallel(tickers, max_workers=8, categories=None):
    """
    Scan tickers in parallel.
    categories: list of categories to keep. None = only main signals (default).
    Examples: ['main'], ['strong_watch'], ['main', 'approaching_buy']
    """
    if categories is None:
        categories = ['main']

    results = []
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_stock, ticker): ticker for ticker in batch}
            for future in as_completed(futures):
                result = future.result()
                if result and result['category'] in categories:
                    results.append(result)

        if i + batch_size < len(tickers):
            time.sleep(3)

    return results