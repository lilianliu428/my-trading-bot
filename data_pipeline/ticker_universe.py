import yfinance as yf
import requests
import time
from scoring.scorer import check_fundamentals
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

def analyze_stock_from_db(tickers, categories=None):
    """
    Scan tickers using Alpaca for prices + DB for fundamentals.
    Much faster than yfinance — no rate limits.
    categories: list of categories to keep. None = only main signals.
    """
    import sqlite3
    from datetime import datetime, timedelta
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from config import ALPACA_API_KEY, ALPACA_API_SECRET

    if categories is None:
        categories = ['main']

    DB_PATH = "/home/ubuntu/my-trading-bot/data.db"
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    # Load all fundamentals from DB into memory (one query, fast)
    conn = sqlite3.connect(DB_PATH)
    fund_rows = conn.execute("""
        SELECT ticker, sector, pe, op_margin, roe, revenue_growth, earnings_growth,
               debt_equity, free_cash_flow_positive, institutional_ownership, insider_ownership
        FROM fundamentals
    """).fetchall()

    bmark_rows = conn.execute("""
        SELECT sector, pe_median, op_margin_median, roe_median FROM sector_benchmarks
    """).fetchall()
    conn.close()

    fund_map = {row[0]: row[1:] for row in fund_rows}
    bmark_map = {row[0]: row[1:] for row in bmark_rows}

    def score_from_db(ticker, current_price=None, ma_20=None, ma_50=None, ma_200=None):
        row = fund_map.get(ticker)
        if not row:
            return 0.0, 19.5, 0, ["⚠️ Not in DB"]
        sector, pe, op_margin, roe, rev_growth, earn_growth, debt_eq, fcf, inst_own, insider_own = row
        bmark = bmark_map.get(sector) if sector else None
        pe_median = bmark[0] if bmark else None
        om_median = bmark[1] if bmark else None
        roe_median = bmark[2] if bmark else None

        score = 0.0; max_score = 0.0; core_passed = 0; reasons = []

        max_score += 2
        if rev_growth is not None:
            if rev_growth > 0.05: score += 2; core_passed += 1; reasons.append(f"✅ Revenue {rev_growth*100:.1f}%")
            else: reasons.append(f"❌ Revenue weak {rev_growth*100:.1f}%")

        max_score += 2
        if fcf is not None:
            if fcf: score += 2; core_passed += 1; reasons.append("✅ FCF positive")
            else: reasons.append("❌ FCF negative")

        max_score += 2
        if pe and pe > 0:
            if pe_median:
                if pe <= pe_median * 1.2: score += 2; core_passed += 1; reasons.append(f"✅ P/E {pe:.1f} vs median {pe_median:.1f}")
                else: reasons.append(f"❌ P/E {pe:.1f} above median {pe_median:.1f}")
            else:
                if pe < 40: score += 2; core_passed += 1; reasons.append(f"✅ P/E {pe:.1f}")
                else: reasons.append(f"❌ P/E high {pe:.1f}")

        max_score += 2
        if earn_growth is not None:
            if earn_growth > 0: score += 2; core_passed += 1; reasons.append(f"✅ Earnings {earn_growth*100:.1f}%")
            else: reasons.append(f"❌ Earnings shrinking {earn_growth*100:.1f}%")

        max_score += 2
        if debt_eq is not None:
            if debt_eq < 150: score += 2; core_passed += 1; reasons.append(f"✅ D/E {debt_eq:.0f}")
            else: reasons.append(f"❌ D/E high {debt_eq:.0f}")

        max_score += 1.5
        if op_margin is not None:
            if om_median:
                if op_margin >= om_median: score += 1.5; reasons.append(f"✅ Op margin {op_margin*100:.1f}%")
                else: reasons.append(f"❌ Op margin {op_margin*100:.1f}%")
            else:
                if op_margin > 0.15: score += 1.5; reasons.append(f"✅ Op margin {op_margin*100:.1f}%")
                else: reasons.append(f"❌ Op margin {op_margin*100:.1f}%")

        max_score += 1.5
        if roe is not None:
            if roe_median:
                if roe >= roe_median: score += 1.5; reasons.append(f"✅ ROE {roe*100:.1f}%")
                else: reasons.append(f"❌ ROE {roe*100:.1f}%")
            else:
                if roe > 0.15: score += 1.5; reasons.append(f"✅ ROE {roe*100:.1f}%")
                else: reasons.append(f"❌ ROE {roe*100:.1f}%")

        max_score += 1
        if inst_own is not None:
            if inst_own > 0.5: score += 1; reasons.append(f"✅ Inst own {inst_own*100:.1f}%")
            else: reasons.append(f"❌ Inst own {inst_own*100:.1f}%")

        max_score += 1
        if insider_own is not None:
            if insider_own > 0.01: score += 1; reasons.append(f"✅ Insider {insider_own*100:.1f}%")
            else: reasons.append(f"❌ Insider {insider_own*100:.1f}%")

        # --- TREND CONTEXT (informational only, doesn't affect score) ---
        if current_price is not None and ma_200 is not None:
            dist = ((current_price - ma_200) / ma_200) * 100
            if current_price > ma_200:
                reasons.append(f"📈 Above 200MA (+{dist:.1f}%)")
            else:
                reasons.append(f"📉 Below 200MA ({dist:.1f}%)")

        if ma_50 is not None and ma_200 is not None:
            if ma_50 > ma_200:
                reasons.append(f"📈 Golden cross (uptrend)")
            else:
                reasons.append(f"📉 Death cross (downtrend)")

        return score, max_score, core_passed, reasons

    # Fetch prices from Alpaca in batches of 100
    results = []
    end = datetime.now() - timedelta(minutes=15)
    start = end - timedelta(days=60)

    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=start, end=end)
            bars = client.get_stock_bars(req)
            df_all = bars.df
            if df_all.empty:
                continue

            for ticker in batch:
                try:
                    if ticker not in df_all.index.get_level_values(0):
                        continue
                    df = df_all.loc[ticker]
                    if len(df) < 15:
                        continue

                    closes = df["close"]
                    current_price = float(closes.iloc[-1])
                    high_30d = float(closes.tail(30).max())
                    low_30d = float(closes.tail(30).min())
                    drop_pct = ((current_price - high_30d) / high_30d) * 100
                    rise_pct = ((current_price - low_30d) / low_30d) * 100

                    delta = closes.diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

                    # Fetch MA data from snapshots table
                    ma_20 = ma_50 = ma_200 = None
                    try:
                        conn2 = sqlite3.connect(DB_PATH)
                        ma_row = conn2.execute(
                            "SELECT ma_20, ma_50, ma_200 FROM snapshots WHERE ticker=? ORDER BY date DESC LIMIT 1",
                            (ticker,)
                        ).fetchone()
                        conn2.close()
                        if ma_row:
                            ma_20, ma_50, ma_200 = ma_row
                    except:
                        pass

                    fund_score, max_score, core_passed, fund_reasons = score_from_db(ticker, current_price, ma_20, ma_50, ma_200)
                    strong = core_passed == 5 or fund_score >= FUNDAMENTAL_MIN_SCORE

                    signal = None; emoji = ""
                    if drop_pct < DROP_THRESHOLD and rsi < RSI_OVERSOLD:
                        signal = "BUY CANDIDATE" if strong else "POTENTIAL TRAP"
                        emoji = "🔍" if strong else "⚠️"
                    elif rise_pct > RISE_THRESHOLD and rsi > RSI_OVERBOUGHT:
                        signal = "MOMENTUM CONFIRMED" if strong else "TAKE PROFIT / AVOID"
                        emoji = "🚀" if strong else "📈"
                    elif strong and drop_pct < -3 and rsi < 50:
                        signal = "APPROACHING BUY"; emoji = "👀"
                    elif strong and abs(drop_pct) < 3 and 40 < rsi < 60:
                        signal = "STRONG WATCH"; emoji = "📋"

                    if not signal:
                        continue

                    if signal in ("BUY CANDIDATE", "POTENTIAL TRAP", "MOMENTUM CONFIRMED", "TAKE PROFIT / AVOID"):
                        category = "main"
                    elif signal == "APPROACHING BUY":
                        category = "approaching_buy"
                    else:
                        category = "strong_watch"

                    if category not in categories:
                        continue

                    fund_summary = "\n".join(fund_reasons)
                    message = (
                        f"{emoji} {signal}: {ticker}\n"
                        f"Price: ${current_price:.2f}\n"
                        f"RSI: {rsi:.1f}\n"
                        f"Drop: {drop_pct:.1f}% | Rise: {rise_pct:.1f}%\n"
                        f"Fundamentals: {fund_score:.1f}/{max_score} (core {core_passed}/5)\n"
                        f"{fund_summary}\n"
                    )
                    results.append({"message": message, "category": category, "signal": signal, "ticker": ticker})

                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    continue

        except Exception as e:
            print(f"Alpaca batch error: {e}")
            continue

    return results
