import re
from telegram import Update
from telegram.ext import ContextTypes
from data_pipeline.database import get_all_latest_snapshots
from data_pipeline.ticker_universe import analyze_stock_from_db, get_all_tickers


def _extract_ticker_from_message(msg: str) -> str | None:
    """Pull the ticker symbol out of a result message header like '🔍 BUY CANDIDATE: NVDA'."""
    match = re.search(r":\s*([A-Z]{1,5})\b", msg)
    return match.group(1) if match else None


def _inject_pattern_line(msg: str) -> str:
    """Insert a pattern-match line right after the 'Bucket:' line in a result message."""
    ticker = _extract_ticker_from_message(msg)
    if not ticker:
        return msg

    try:
        from scoring.anchor_matcher import match, interpret_matches
        matches = match(ticker, top_k=3)
        if not matches or matches[0][2] < 0.60:
            return msg  # weak match, omit entirely

        interp = interpret_matches(matches)
        verdict = interp.get("verdict")
        top = matches[0]

        if verdict in ("loser_warning", "recovery_signal"):
            pattern_line = interp["message"]
        elif verdict == "winner_pattern":
            pattern_line = f"Pattern: 🟢 Looks like {top[0]}-{top[4]} ({top[2]:.0%}, winner)"
        elif verdict == "mixed":
            pattern_line = f"Pattern: 🟡 Looks like {top[0]}-{top[4]} ({top[2]:.0%}, mixed)"
        else:
            return msg  # weak verdict, omit entirely
    except Exception as e:
        print(f"Pattern matching failed for {ticker}: {e}")
        return msg

    # Insert after the 'Bucket:' line if present, otherwise after the header line
    lines = msg.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Bucket:"):
            lines.insert(i + 1, pattern_line)
            return "\n".join(lines)
    # No Bucket: line found, insert after first line
    lines.insert(1, pattern_line)
    return "\n".join(lines)


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

    # Inject pattern-match lines into each result message
    messages = [_inject_pattern_line(r['message']) for r in results]
    chunks = [messages[i:i+5] for i in range(0, len(messages), 5)]
    for i, chunk in enumerate(chunks):
        header = f"📋 {cat_arg.upper()} ({len(messages)} found):\n\n" if i == 0 else ""
        await update.message.reply_text(header + "\n".join(chunk))