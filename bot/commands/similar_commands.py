"""
/similar TICKER — Telegram command to show historical pattern matches.

Usage: /similar NVDA
       /similar AAPL --cross
"""

from telegram import Update
from telegram.ext import ContextTypes

from scoring.anchor_matcher import (
    match,
    interpret_matches,
    format_matches_for_telegram,
)


async def similar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /similar TICKER [--cross]

    Shows top 3 historical company-phase matches for a given ticker.
    Use --cross flag to enable within-industry cross-bucket matching.
    """
    if not context.args:
        await update.message.reply_text(
            "Usage: `/similar TICKER` (e.g. `/similar NVDA`)\n"
            "Add `--cross` to enable cross-bucket matching within the same industry.",
            parse_mode="Markdown",
        )
        return

    ticker = context.args[0].upper()
    cross_industry = "--cross" in context.args

    await update.message.reply_text(f"🔎 Analyzing {ticker}...")

    matches = match(ticker, top_k=3, cross_industry=cross_industry)
    if not matches:
        await update.message.reply_text(
            f"❌ Couldn't find matches for {ticker}. "
            f"Either no fundamentals in DB or no anchors in its bucket. "
            f"Try `--cross` for cross-industry matching."
        )
        return

    interpretation = interpret_matches(matches)
    message = format_matches_for_telegram(ticker, matches, interpretation)

    if cross_industry:
        message += "\n\n_(cross-industry matching enabled)_"

    await update.message.reply_text(message, parse_mode="Markdown")