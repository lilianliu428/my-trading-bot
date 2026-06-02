from telegram import Update
from telegram.ext import ContextTypes
from scanner import analyze_stock
from fundamentals import check_fundamentals
import yfinance as yf
import pandas_ta as ta

async def search_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /search TICKER  e.g. /search QQQM")
        return
    ticker = context.args[0].upper()
    await update.message.reply_text(f"🔍 Analyzing {ticker}...")

    result = analyze_stock(ticker)
    if result:
        await update.message.reply_text(result)
        return

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty:
            await update.message.reply_text(f"❌ No data found for {ticker}")
            return

        hist["RSI"] = ta.rsi(hist["Close"], length=14)
        latest_rsi = hist["RSI"].iloc[-1]
        current_price = hist["Close"].iloc[-1]
        high_30d = hist["Close"].tail(30).max()
        low_30d = hist["Close"].tail(30).min()
        drop_pct = ((current_price - high_30d) / high_30d) * 100
        rise_pct = ((current_price - low_30d) / low_30d) * 100

        fund_score, max_score, core_passed, fund_reasons = check_fundamentals(ticker)

        msg = (
            f"📊 NO SIGNAL: {ticker}\n"
            f"Price: ${current_price:.2f}\n"
            f"RSI: {latest_rsi:.1f}\n"
            f"Drop: {drop_pct:.1f}% | Rise: {rise_pct:.1f}%\n"
            f"Fundamentals: {fund_score:.1f}/{max_score} (core {core_passed}/5)\n"
            + "\n".join(fund_reasons)
            + "\n\nDoesn't meet buy/sell/momentum thresholds right now."
        )
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")