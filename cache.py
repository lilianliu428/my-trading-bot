import threading
import time
from data_pipeline.ticker_universe import scan_tickers_parallel, get_all_tickers
from scoring.sectors.benchmarks import build_sector_benchmarks, _sector_cache

_results_cache = []
_last_updated = 0
_lock = threading.Lock()
CACHE_DURATION = 3600  # refresh every hour


def get_cached_results():
    """Return cached scan results. Returns empty list if no cache yet."""
    with _lock:
        return list(_results_cache)


def get_cache_age():
    """How old is the cache in seconds. Returns None if empty."""
    if _last_updated == 0:
        return None
    return int(time.time() - _last_updated)


def refresh_cache():
    """Run full scan, cache all categories. Blocking call."""
    global _results_cache, _last_updated

    print("🔄 Refreshing cache...")
    tickers = get_all_tickers()

    if not _sector_cache:
        build_sector_benchmarks(tickers)

    # scan everything including warm zones
    results = scan_tickers_parallel(
        tickers,
        max_workers=15,
        categories=['main', 'strong_watch', 'approaching_buy']
    )

    with _lock:
        _results_cache = results
        _last_updated = time.time()

    print(f"✅ Cache refreshed: {len(results)} signals")


def get_by_category(target_category, target_signal=None):
    """Filter cached results by category and optional signal name."""
    results = get_cached_results()

    if target_category:
        results = [r for r in results if r['category'] == target_category]

    if target_signal:
        if isinstance(target_signal, list):
            results = [r for r in results if r['signal'] in target_signal]
        else:
            results = [r for r in results if r['signal'] == target_signal]

    return results