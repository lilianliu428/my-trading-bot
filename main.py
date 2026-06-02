from telegram.ext import Application, MessageHandler, CommandHandler, filters
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from handlers import (add_stock, remove_stock, list_stocks,
                      scan_now, screen_all, handle_message,
                      scan_watchlist_and_send, search_stock,
                      category_command)
from config import TOKEN

async def post_init(application):
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scan_watchlist_and_send, "cron", hour=9, minute=0)
    scheduler.start()
    print("Scheduler started — scan runs daily at 9:00 AM")

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
print("Commands: /add | /remove | /list | /scan | /screen | /search")
app.run_polling()