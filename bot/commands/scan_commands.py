from telegram import Update, Bot
from telegram.ext import ContextTypes
from watchlist import load_watchlist
from data_pipeline.ticker_universe import analyze_stock_from_db, get_all_tickers
from config import TOKEN, YOUR_CHAT_ID


async def scan_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    watchlist = load_watchlist()
    if not watchlist:
        await update.message.reply_text("Your watchlist is empty. Add stocks with /add TICKER")
        return

    await update.message.reply_text(f"🔍 Scanning {len(watchlist)} watchlist stocks...")
    results = analyze_stock_from_db(watchlist, categories=['main', 'approaching_buy', 'strong_watch'])

    if results:
        for r in results:
            await update.message.reply_text(r['message'])
    else:
        await update.message.reply_text("No signals from your watchlist right now.")


async def screen_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tickers = get_all_tickers()
    await update.message.reply_text(f"🔍 Screening {len(tickers)} stocks... (2-3 min)")

    results = analyze_stock_from_db(tickers, categories=['main'])

    if results:
        messages = [r['message'] for r in results]
        chunks = [messages[i:i+5] for i in range(0, len(messages), 5)]
        for chunk in chunks:
            await update.message.reply_text("📊 Screen Results:\n\n" + "\n".join(chunk))
    else:
        await update.message.reply_text("📊 Screen complete — no stocks triggered your criteria today.")


async def scan_watchlist_and_send():
    bot = Bot(token=TOKEN)
    watchlist = load_watchlist()
    print("Running scheduled scan...")
    results = analyze_stock_from_db(watchlist, categories=['main'])
    if results:
        message = "📊 Scan Results:\n\n" + "\n".join([r['message'] for r in results])
    else:
        message = "📊 Scan complete — no stocks triggered your criteria today."
    await bot.send_message(chat_id=YOUR_CHAT_ID, text=message)
