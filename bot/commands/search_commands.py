import pandas as pd
import sqlite3
from telegram import Update
from telegram.ext import ContextTypes
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import (ALPACA_API_KEY, ALPACA_API_SECRET,
                    DROP_THRESHOLD, RISE_THRESHOLD,
                    RSI_OVERSOLD, RSI_OVERBOUGHT, FUNDAMENTAL_MIN_SCORE)
from scoring.dispatcher import score_ticker

DB_PATH = "/home/ubuntu/my-trading-bot/data.db"

# Initialize Alpaca client once
alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)


def calculate_indicators_from_bars(df):
    """Given a dataframe of price bars, calculate RSI and MAs."""
    closes = df['close']

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
    end = datetime.now() - timedelta(minutes=15)
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
        return df.loc[ticker]
    except Exception as e:
        print(f"Alpaca error for {ticker}: {e}")
        return None


def get_ma_from_db(ticker):
    """Fetch latest MA values from snapshots table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT ma_20, ma_50, ma_200 FROM snapshots WHERE ticker=? ORDER BY date DESC LIMIT 1",
            (ticker,)
        ).fetchone()
        conn.close()
        return row if row else (None, None, None)
    except:
        return (None, None, None)


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

    # Calculate technicals
    indicators = calculate_indicators_from_bars(df)
    current_price = indicators['price']
    latest_rsi = indicators['rsi']
    high_30d = indicators['high_30d']
    low_30d = indicators['low_30d']
    drop_pct = ((current_price - high_30d) / high_30d) * 100
    rise_pct = ((current_price - low_30d) / low_30d) * 100

    # Get business-model-aware fundamental score from dispatcher
    fund_score, max_score, all_cores_passed, fund_reasons, bucket = score_ticker(ticker)
    strong = all_cores_passed or fund_score >= FUNDAMENTAL_MIN_SCORE

    # Append MA trend context
    ma_20, ma_50, ma_200 = get_ma_from_db(ticker)
    if ma_200 is not None:
        dist = ((current_price - ma_200) / ma_200) * 100
        if current_price > ma_200:
            fund_reasons.append(f"📈 Above 200MA (+{dist:.1f}%)")
        else:
            fund_reasons.append(f"📉 Below 200MA ({dist:.1f}%)")
    if ma_50 is not None and ma_200 is not None:
        if ma_50 > ma_200:
            fund_reasons.append(f"📈 Golden cross (uptrend)")
        else:
            fund_reasons.append(f"📉 Death cross (downtrend)")

    # Determine signal
    signal = None; emoji = ""
    if drop_pct < DROP_THRESHOLD and latest_rsi < RSI_OVERSOLD:
        signal = "BUY CANDIDATE" if strong else "POTENTIAL TRAP"
        emoji = "🔍" if strong else "⚠️"
    elif rise_pct > RISE_THRESHOLD and latest_rsi > RSI_OVERBOUGHT:
        signal = "MOMENTUM CONFIRMED" if strong else "TAKE PROFIT / AVOID"
        emoji = "🚀" if strong else "📈"
    elif strong and drop_pct < -3 and latest_rsi < 50:
        signal = "APPROACHING BUY"; emoji = "👀"
    elif strong and abs(drop_pct) < 3 and 40 < latest_rsi < 60:
        signal = "STRONG WATCH"; emoji = "📋"
    else:
        signal = "NO SIGNAL"; emoji = "📊"

    fund_summary = "\n".join(fund_reasons)
    msg = (
        f"{emoji} {signal}: {ticker} (LIVE)\n"
        f"Bucket: {bucket}\n"
        f"Price: ${current_price:.2f}\n"
        f"RSI: {latest_rsi:.1f}\n"
        f"Drop: {drop_pct:.1f}% | Rise: {rise_pct:.1f}%\n"
        f"Fundamentals: {fund_score:.1f}/{max_score}\n"
        f"{fund_summary}\n"
    )

    await update.message.reply_text(msg)
