import pandas as pd
from telegram import Update
from telegram.ext import ContextTypes
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import (ALPACA_API_KEY, ALPACA_API_SECRET,
                    DROP_THRESHOLD, RISE_THRESHOLD,
                    RSI_OVERSOLD, RSI_OVERBOUGHT, FUNDAMENTAL_MIN_SCORE)

import sqlite3
import os

DB_PATH = "/home/ubuntu/my-trading-bot/data.db"


def get_fundamentals_from_db(ticker, current_price=None, ma_20=None, ma_50=None, ma_200=None):
    """Fetch pre-computed fundamentals from DB. Same return shape as check_fundamentals()."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT fund_score, max_score, core_passed, pe, op_margin, roe, "
            "revenue_growth, debt_equity, free_cash_flow_positive FROM fundamentals WHERE ticker = ?",
            (ticker,)
        ).fetchone()
        conn.close()

        if row is None:
            return 0.0, 19.5, 0, ["⚠️ Not in database yet (run fundamentals scraper)"]

        fund_score, max_score, core_passed, pe, op_margin, roe, rev_growth, debt_eq, fcf = row
        reasons = []
        if pe and pe > 0:  reasons.append(f"P/E: {pe:.1f}")
        if op_margin:      reasons.append(f"Op Margin: {op_margin*100:.1f}%")
        if roe:            reasons.append(f"ROE: {roe*100:.1f}%")
        if rev_growth:     reasons.append(f"Rev Growth: {rev_growth*100:.1f}%")
        if debt_eq:        reasons.append(f"D/E: {debt_eq:.2f}")
        reasons.append("✅ FCF Positive" if fcf else "❌ FCF Negative")
        return fund_score or 0.0, max_score or 19.5, core_passed or 0, reasons

    except Exception as e:
        return 0.0, 19.5, 0, [f"⚠️ DB error: {e}"]


import sqlite3
import os

DB_PATH = "/home/ubuntu/my-trading-bot/data.db"


def get_fundamentals_from_db(ticker, current_price=None, ma_20=None, ma_50=None, ma_200=None):
    """
    Replicates check_fundamentals() logic using stored DB values.
    Returns (fund_score, max_score, core_passed, reasons) — same shape as before.
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        row = conn.execute("""
            SELECT sector, pe, op_margin, roe, revenue_growth, earnings_growth,
                   debt_equity, free_cash_flow_positive,
                   institutional_ownership, insider_ownership
            FROM fundamentals WHERE ticker = ?
        """, (ticker,)).fetchone()

        if row is None:
            conn.close()
            return 0.0, 19.5, 0, ["⚠️ Not in database yet (run fundamentals scraper)"]

        sector, pe, op_margin, roe, rev_growth, earn_growth, debt_eq, fcf, inst_own, insider_own = row

        # Load sector benchmarks if available
        bmark = conn.execute("""
            SELECT pe_median, op_margin_median, roe_median
            FROM sector_benchmarks WHERE sector = ?
        """, (sector,)).fetchone() if sector else None
        conn.close()

        pe_median = bmark[0] if bmark else None
        om_median = bmark[1] if bmark else None
        roe_median = bmark[2] if bmark else None

        score = 0.0
        max_score = 0.0
        core_passed = 0
        reasons = []

        # --- CORE CHECKS (weight 2 each, max 10) ---

        # 1. Revenue growth
        max_score += 2
        if rev_growth is not None:
            if rev_growth > 0.05:
                score += 2; core_passed += 1
                reasons.append(f"✅ Revenue growing {rev_growth*100:.1f}%")
            else:
                reasons.append(f"❌ Revenue growth weak {rev_growth*100:.1f}%")

        # 2. Free cash flow
        max_score += 2
        if fcf is not None:
            if fcf:
                score += 2; core_passed += 1
                reasons.append("✅ Positive free cash flow")
            else:
                reasons.append("❌ Negative free cash flow")

        # 3. P/E ratio
        max_score += 2
        if pe and pe > 0:
            if pe_median is not None:
                if pe <= pe_median * 1.2:
                    score += 2; core_passed += 1
                    reasons.append(f"✅ P/E {pe:.1f} reasonable vs {sector} median {pe_median:.1f}")
                else:
                    reasons.append(f"❌ P/E {pe:.1f} above {sector} median {pe_median:.1f}")
            else:
                if pe < 40:
                    score += 2; core_passed += 1
                    reasons.append(f"✅ P/E reasonable at {pe:.1f}")
                else:
                    reasons.append(f"❌ P/E too high at {pe:.1f}")

        # 4. Earnings growth
        max_score += 2
        if earn_growth is not None:
            if earn_growth > 0:
                score += 2; core_passed += 1
                reasons.append(f"✅ Earnings growing {earn_growth*100:.1f}%")
            else:
                reasons.append(f"❌ Earnings shrinking {earn_growth*100:.1f}%")

        # 5. Debt/equity
        max_score += 2
        if debt_eq is not None:
            if debt_eq < 150:
                score += 2; core_passed += 1
                reasons.append(f"✅ Debt/equity manageable at {debt_eq:.0f}")
            else:
                reasons.append(f"❌ High debt/equity at {debt_eq:.0f}")

        # --- ADDON CHECKS ---

        # Operating margin (weight 1.5)
        max_score += 1.5
        if op_margin is not None:
            if om_median is not None:
                if op_margin >= om_median:
                    score += 1.5
                    reasons.append(f"✅ Op margin {op_margin*100:.1f}% beats {sector} median {om_median*100:.1f}%")
                else:
                    reasons.append(f"❌ Op margin {op_margin*100:.1f}% below {sector} median {om_median*100:.1f}%")
            else:
                if op_margin > 0.15:
                    score += 1.5
                    reasons.append(f"✅ Strong op margin {op_margin*100:.1f}%")
                else:
                    reasons.append(f"❌ Weak op margin {op_margin*100:.1f}%")

        # ROE (weight 1.5)
        max_score += 1.5
        if roe is not None:
            if roe_median is not None:
                if roe >= roe_median:
                    score += 1.5
                    reasons.append(f"✅ ROE {roe*100:.1f}% beats {sector} median {roe_median*100:.1f}%")
                else:
                    reasons.append(f"❌ ROE {roe*100:.1f}% below {sector} median {roe_median*100:.1f}%")
            else:
                if roe > 0.15:
                    score += 1.5
                    reasons.append(f"✅ Strong ROE {roe*100:.1f}%")
                else:
                    reasons.append(f"❌ Weak ROE {roe*100:.1f}%")

        # Institutional ownership (weight 1)
        max_score += 1
        if inst_own is not None:
            if inst_own > 0.5:
                score += 1
                reasons.append(f"✅ High institutional ownership {inst_own*100:.1f}%")
            else:
                reasons.append(f"❌ Low institutional ownership {inst_own*100:.1f}%")

        # Insider ownership (weight 1)
        max_score += 1
        if insider_own is not None:
            if insider_own > 0.01:
                score += 1
                reasons.append(f"✅ Insiders own {insider_own*100:.1f}%")
            else:
                reasons.append(f"❌ Low insider ownership {insider_own*100:.1f}%")

        # Note: technical checks (MA200, golden cross) skipped here —
        # those use live price data already computed above in search_stock()

        # Fetch MA from snapshots if not provided
        if ma_200 is None or ma_50 is None:
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

        # Trend context (informational, doesn't affect score)
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

    except Exception as e:
        return 0.0, 19.5, 0, [f"⚠️ DB error: {e}"]


# Initialize Alpaca client once
alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)


def calculate_indicators_from_bars(df):
    """Given a dataframe of price bars, calculate RSI and MAs."""
    closes = df['close']

    # RSI
    delta = closes.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))

    return {
        'price': float(closes.iloc[-1]),
        'rsi': float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None,
        'ma_20': float(closes.tail(20).mean()) if len(closes) >= 20 else None,
        'ma_50': float(closes.tail(50).mean()) if len(closes) >= 50 else None,
        'ma_200': float(closes.tail(200).mean()) if len(closes) >= 200 else None,
        'high_30d': float(closes.tail(30).max()),
        'low_30d': float(closes.tail(30).min()),
    }


def fetch_live_data(ticker):
    """Fetch fresh data from Alpaca for a single ticker."""
    end = datetime.now() - timedelta(minutes=15)  # 15-min delayed (free tier)
    start = end - timedelta(days=350)

    request = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Day,
        start=start,
        end=end
    )

    try:
        bars = alpaca_client.get_stock_bars(request)
        df = bars.df
        if df.empty:
            return None
        # df is multi-indexed by (symbol, timestamp), get just this ticker
        return df.loc[ticker]
    except Exception as e:
        print(f"Alpaca error for {ticker}: {e}")
        return None


async def search_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /search TICKER  e.g. /search MSFT")
        return

    ticker = context.args[0].upper()
    await update.message.reply_text(f"🔍 Analyzing {ticker} (live)...")

    # Fetch live data from Alpaca
    df = fetch_live_data(ticker)
    if df is None or df.empty:
        await update.message.reply_text(f"❌ No data found for {ticker}")
        return

    # Calculate indicators
    import pandas as pd
    indicators = calculate_indicators_from_bars(df)

    current_price = indicators['price']
    latest_rsi = indicators['rsi']
    high_30d = indicators['high_30d']
    low_30d = indicators['low_30d']
    drop_pct = ((current_price - high_30d) / high_30d) * 100
    rise_pct = ((current_price - low_30d) / low_30d) * 100

    # Get fundamentals from database
    fund_score, max_score, core_passed, fund_reasons = get_fundamentals_from_db(ticker, current_price=current_price)
    strong = core_passed == 5 or fund_score >= FUNDAMENTAL_MIN_SCORE

    # Determine signal
    signal = None
    emoji = ""

    if drop_pct < DROP_THRESHOLD and latest_rsi < RSI_OVERSOLD:
        signal = "BUY CANDIDATE" if strong else "POTENTIAL TRAP"
        emoji = "🔍" if strong else "⚠️"
    elif rise_pct > RISE_THRESHOLD and latest_rsi > RSI_OVERBOUGHT:
        signal = "MOMENTUM CONFIRMED" if strong else "TAKE PROFIT / AVOID"
        emoji = "🚀" if strong else "📈"
    elif strong and drop_pct < -3 and latest_rsi < 50:
        signal = "APPROACHING BUY"
        emoji = "👀"
    elif strong and abs(drop_pct) < 3 and 40 < latest_rsi < 60:
        signal = "STRONG WATCH"
        emoji = "📋"
    else:
        signal = "NO SIGNAL"
        emoji = "📊"

    fund_summary = "\n".join(fund_reasons)

    msg = (
        f"{emoji} {signal}: {ticker} (LIVE)\n"
        f"Price: ${current_price:.2f}\n"
        f"RSI: {latest_rsi:.1f}\n"
        f"Drop: {drop_pct:.1f}% | Rise: {rise_pct:.1f}%\n"
        f"Fundamentals: {fund_score:.1f}/{max_score} (core {core_passed}/5)\n"
        f"{fund_summary}\n"
    )

    await update.message.reply_text(msg)
