from telegram import Update
from telegram.ext import ContextTypes
import yfinance as yf

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ticker = update.message.text.upper().strip()
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info['last_price']
        await update.message.reply_text(f"{ticker}: ${price:.2f}")
    except:
        await update.message.reply_text(f"Couldn't find ticker: {ticker}")