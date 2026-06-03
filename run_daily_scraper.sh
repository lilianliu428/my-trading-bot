#!/bin/bash
cd /home/ubuntu/my-trading-bot
source .venv/bin/activate
python -m data_pipeline.daily_scraper >> /home/ubuntu/my-trading-bot/cron_daily.log 2>&1
echo "--- Daily scraper finished at $(date) ---" >> /home/ubuntu/my-trading-bot/cron_daily.log
