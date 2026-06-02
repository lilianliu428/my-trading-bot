from telegram import Update, Bot
from telegram.ext import ContextTypes
from watchlist import load_watchlist
from scanner import scan_tickers_parallel, get_all_tickers
from config import TOKEN, YOUR_CHAT_ID

async def scan_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Running scan now...")
    await scan_watchlist_and_send()


async def screen_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 Building sector benchmarks then screening... about 3-4 minutes total."
    )
    tickers = get_all_tickers()

    # build sector benchmarks first
    from fundamentals.sector_benchmarks import build_sector_benchmarks
    build_sector_benchmarks(tickers)

    # now scan with sector-aware logic
    alerts = scan_tickers_parallel(tickers, max_workers=20)
    if alerts:
        chunks = [alerts[i:i + 5] for i in range(0, len(alerts), 5)]
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