from telegram import Update, Bot
from telegram.ext import ContextTypes
from watchlist import load_watchlist
from scanner import scan_tickers_parallel, get_all_tickers
from config import TOKEN, YOUR_CHAT_ID


async def scan_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    watchlist = load_watchlist()
    if not watchlist:
        await update.message.reply_text("Your watchlist is empty. Add stocks with /add TICKER")
        return

    await update.message.reply_text(f"🔍 Scanning {len(watchlist)} stocks...")

    results = scan_tickers_parallel(watchlist, max_workers=15)

    if results:
        messages = [r['message'] for r in results]
        for msg in messages:
            await update.message.reply_text(msg)
    else:
        await update.message.reply_text("No signals from your watchlist right now.")


async def screen_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tickers = get_all_tickers()

    from fundamentals.sector_benchmarks import build_sector_benchmarks, _sector_cache
    if not _sector_cache:
        await update.message.reply_text(
            "🔍 Building sector benchmarks (first run only)... about 1-2 minutes."
        )
        build_sector_benchmarks(tickers)

    await update.message.reply_text("🔍 Screening now... about 2-3 minutes.")

    # default to main signals only (no warm zones in screen)
    results = scan_tickers_parallel(tickers, max_workers=15)

    if results:
        messages = [r['message'] for r in results]
        chunks = [messages[i:i + 5] for i in range(0, len(messages), 5)]
        for chunk in chunks:
            await update.message.reply_text(
                "📊 Screen Results:\n\n" + "\n".join(chunk)
            )
    else:
        await update.message.reply_text(
            "📊 Screen complete — no stocks triggered your criteria today."
        )

async def scan_watchlist_and_send():
    bot = Bot(token=TOKEN)
    watchlist = load_watchlist()
    print("Running morning scan...")
    alerts = scan_tickers_parallel(watchlist, max_workers=10)
    if alerts:
        message = "📊 Morning Scan Results:\n\n" + "\n".join(alerts)
    else:
        message = "📊 Morning scan complete — no stocks triggered your criteria today."
    await bot.send_message(chat_id=YOUR_CHAT_ID, text=message)