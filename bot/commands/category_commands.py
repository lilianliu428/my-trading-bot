from telegram import Update
from telegram.ext import ContextTypes
from data_pipeline.database import get_all_latest_snapshots
from data_pipeline.ticker_universe import analyze_stock_from_db, get_all_tickers


async def category_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Usage: /category <type>\n\n"
            "Types:\n"
            "  buy_candidate — strong fundamentals + big drop + RSI oversold\n"
            "  traps — weak fundamentals but technical buy setup\n"
            "  take_profit — momentum confirmed (strong)\n"
            "  strong_watch — strong fundamentals, neutral price/RSI\n"
            "  approaching_buy — strong fundamentals + small drop + RSI < 50"
        )
        return

    cat_arg = context.args[0].lower()
    category_map = {
        'buy_candidate':   ('main', ['BUY CANDIDATE']),
        'take_profit':     ('main', ['MOMENTUM CONFIRMED']),
        'traps':           ('main', ['POTENTIAL TRAP', 'TAKE PROFIT / AVOID']),
        'strong_watch':    ('strong_watch', None),
        'approaching_buy': ('approaching_buy', None),
    }

    if cat_arg not in category_map:
        await update.message.reply_text(f"Unknown category '{cat_arg}'. Try: buy_candidate, traps, take_profit, strong_watch, approaching_buy")
        return

    target_category, target_signals = category_map[cat_arg]

    await update.message.reply_text(f"🔍 Scanning all stocks for {cat_arg}... (1-2 min)")

    tickers = get_all_tickers()
    categories_to_scan = [target_category]
    results = analyze_stock_from_db(tickers, categories=categories_to_scan)

    # filter to specific signals if needed
    if target_signals:
        results = [r for r in results if r['signal'] in target_signals]

    if not results:
        await update.message.reply_text(f"No stocks in {cat_arg} right now.")
        return

    messages = [r['message'] for r in results]
    chunks = [messages[i:i+5] for i in range(0, len(messages), 5)]
    for i, chunk in enumerate(chunks):
        header = f"📋 {cat_arg.upper()} ({len(messages)} found):\n\n" if i == 0 else ""
        await update.message.reply_text(header + "\n".join(chunk))
