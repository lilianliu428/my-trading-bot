from telegram import Update
from telegram.ext import ContextTypes


async def category_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        # ... existing help message ...
        return

    from cache import get_by_category, get_cache_age

    cat_arg = context.args[0].lower()
    category_map = {
        'buy_candidate': ('main', 'BUY CANDIDATE'),
        'take_profit': ('main', 'MOMENTUM CONFIRMED'),
        'traps': ('main', ['POTENTIAL TRAP', 'TAKE PROFIT / AVOID']),
        'strong_watch': ('strong_watch', None),
        'approaching_buy': ('approaching_buy', None),
    }

    if cat_arg not in category_map:
        await update.message.reply_text(f"Unknown category '{cat_arg}'.")
        return

    target_category, target_signal = category_map[cat_arg]

    # check cache age
    age = get_cache_age()
    if age is None:
        await update.message.reply_text(
            "⏳ Cache is being built for the first time. Try again in a few minutes."
        )
        return

    results = get_by_category(target_category, target_signal)

    age_min = age // 60

    if not results:
        await update.message.reply_text(
            f"No stocks currently in {cat_arg}. (Cache: {age_min} min old)"
        )
        return

    messages = [r['message'] for r in results]
    chunks = [messages[i:i + 5] for i in range(0, len(messages), 5)]
    for i, chunk in enumerate(chunks):
        header = f"📋 {cat_arg.upper()} ({len(messages)} found, cache {age_min} min old):\n\n" if i == 0 else ""
        await update.message.reply_text(header + "\n".join(chunk))
