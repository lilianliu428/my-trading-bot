from telegram.ext import Application, MessageHandler, CommandHandler, filters
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from bot.commands import (add_stock, remove_stock, list_stocks,
                         scan_now, screen_all, handle_message,
                         scan_watchlist_and_send, search_stock,
                         category_command)
from config import TOKEN
from telegram import BotCommand


async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("search", "Search a single stock (live)"),
        BotCommand("screen", "Scan all S&P 500 for signals"),
        BotCommand("category", "View stocks by signal category"),
        BotCommand("scan", "Scan your watchlist"),
        BotCommand("add", "Add ticker to watchlist"),
        BotCommand("remove", "Remove ticker from watchlist"),
        BotCommand("list", "Show your watchlist"),
    ])
    # initial cache build in background (don't block startup)
#    from cache import refresh_cache
 #   import threading
  #  threading.Thread(target=refresh_cache, daemon=True).start()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(scan_watchlist_and_send, trigger="cron", hour=9, minute=0)

    # refresh cache every hour
#    scheduler.add_job(refresh_cache, trigger="cron", minute=0)

    scheduler.start()
    print("Scheduler started — scan runs daily at 9:00 AM, cache refreshes hourly")

app = Application.builder().token(TOKEN).post_init(post_init).build()

app.add_handler(CommandHandler("add", add_stock))
app.add_handler(CommandHandler("remove", remove_stock))
app.add_handler(CommandHandler("list", list_stocks))
app.add_handler(CommandHandler("scan", scan_now))
app.add_handler(CommandHandler("screen", screen_all))
app.add_handler(CommandHandler("search", search_stock))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
app.add_handler(CommandHandler("category", category_command))

print("Bot is running...")
print("Commands: /add | /remove | /list | /scan | /screen | /search |/category")
app.run_polling()
