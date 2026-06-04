#!/bin/bash
# Daily database backup - keeps 7 days on rotation
BACKUP_DIR="/home/ubuntu/db_backups"
DAY_OF_WEEK=$(date +%u)  # 1=Mon, 7=Sun
BACKUP_FILE="$BACKUP_DIR/data.db.day${DAY_OF_WEEK}"

cp /home/ubuntu/my-trading-bot/data.db "$BACKUP_FILE"
echo "$(date) - Backed up to $BACKUP_FILE ($(du -h $BACKUP_FILE | cut -f1))" >> /home/ubuntu/my-trading-bot/cron_backup.log
