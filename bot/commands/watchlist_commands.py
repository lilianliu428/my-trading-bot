from telegram import Update
from telegram.ext import ContextTypes
from watchlist import load_watchlist, save_watchlist

async def add_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /add TICKER  e.g. /add TSLA")
        return
    ticker = context.args[0].upper()
    watchlist = load_watchlist()
    if ticker in watchlist:
        await update.message.reply_text(f"{ticker} is already in your watchlist.")
        return
    watchlist.append(ticker)
    save_watchlist(watchlist)
    await update.message.reply_text(f"✅ Added {ticker} to watchlist.")

async def remove_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /remove TICKER  e.g. /remove TSLA")
        return
    ticker = context.args[0].upper()
    watchlist = load_watchlist()
    if ticker not in watchlist:
        await update.message.reply_text(f"{ticker} is not in your watchlist.")
        return
    watchlist.remove(ticker)
    save_watchlist(watchlist)
    await update.message.reply_text(f"🗑️ Removed {ticker} from watchlist.")

async def list_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    watchlist = load_watchlist()
    if not watchlist:
        await update.message.reply_text("Your watchlist is empty.")
        return
    msg = "📋 Your watchlist:\n\n" + "\n".join(f"• {t}" for t in watchlist)
    await update.message.reply_text(msg)