from telegram import Update
from telegram.ext import ContextTypes
from scanner import scan_tickers_parallel
from handlers.scan_commands import get_all_tickers


async def category_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Usage: /category <type>\n\n"
            "Available categories:\n"
            "• buy_candidate — strong fundamentals + significant drop + oversold (act now)\n"
            "• approaching_buy — strong fundamentals + moderate drop (getting close)\n"
            "• strong_watch — strong fundamentals, neutral price (waiting for opportunity)\n"
            "• take_profit — strong fundamentals + overbought (consider locking in gains)\n"
            "• traps — weak fundamentals with extreme price action (avoid these)\n\n"
            "Example: /category strong_watch"
        )
        return

    cat_arg = context.args[0].lower()

    category_map = {
        'buy_candidate': ('main', 'BUY CANDIDATE'),
        'take_profit': ('main', 'MOMENTUM CONFIRMED'),
        'traps': ('main', ['POTENTIAL TRAP', 'TAKE PROFIT / AVOID']),
        'strong_watch': ('strong_watch', None),
        'approaching_buy': ('approaching_buy', None),
    }

    if cat_arg not in category_map:
        await update.message.reply_text(
            f"Unknown category '{cat_arg}'. Run /category for help."
        )
        return

    target_category, target_signal = category_map[cat_arg]

    await update.message.reply_text(f"🔍 Finding all {cat_arg} stocks... 3-4 min.")

    tickers = get_all_tickers()
    from fundamentals.sector_benchmarks import build_sector_benchmarks, _sector_cache
    if not _sector_cache:
        build_sector_benchmarks(tickers)

    results = scan_tickers_parallel(tickers, max_workers=15, categories=[target_category])

    # filter by specific signal name if needed
    if target_signal:
        if isinstance(target_signal, list):
            results = [r for r in results if r['signal'] in target_signal]
        else:
            results = [r for r in results if r['signal'] == target_signal]

    if not results:
        await update.message.reply_text(f"No stocks currently in {cat_arg}.")
        return

    messages = [r['message'] for r in results]
    chunks = [messages[i:i + 5] for i in range(0, len(messages), 5)]
    for i, chunk in enumerate(chunks):
        header = f"📋 {cat_arg.upper()} ({len(messages)} found):\n\n" if i == 0 else ""
        await update.message.reply_text(header + "\n".join(chunk))