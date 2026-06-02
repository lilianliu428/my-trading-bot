import json
import os
from config import WATCHLIST_FILE

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    return ["AAPL", "MSFT", "GOOGL"]

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlist, f)