#!/bin/bash
cd /home/ubuntu/my-trading-bot
source .venv/bin/activate
python -m data_pipeline.fundamentals_scraper >> /home/ubuntu/my-trading-bot/cron_fundamentals.log 2>&1
echo "--- Fundamentals scraper finished at $(date) ---" >> /home/ubuntu/my-trading-bot/cron_fundamentals.log
