import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import time

# cache so we don't recalculate every scan
_sector_cache = {}
_cache_timestamp = 0
CACHE_DURATION = 86400  # refresh once per day


def fetch_stock_metrics(ticker):
    """Get sector + metrics for one stock."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "sector": info.get("sector"),
            "pe": info.get("trailingPE"),
            "op_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
        }
    except:
        return None


def build_sector_benchmarks(tickers):
    """Calculate median metrics per sector across all given tickers."""
    global _sector_cache, _cache_timestamp

    # use cache if still fresh
    if _sector_cache and time.time() - _cache_timestamp < CACHE_DURATION:
        return _sector_cache

    print("Building sector benchmarks (one-time per day)...")
    sector_data = {}

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_stock_metrics, t) for t in tickers]
        for future in as_completed(futures):
            data = future.result()
            if not data or not data["sector"]:
                continue
            sector = data["sector"]
            if sector not in sector_data:
                sector_data[sector] = {"pe": [], "op_margin": [], "roe": []}
            try:
                pe = float(data["pe"]) if data["pe"] else None
                if pe and 0 < pe < 500:
                    sector_data[sector]["pe"].append(pe)
            except (TypeError, ValueError):
                pass

            try:
                om = float(data["op_margin"]) if data["op_margin"] is not None else None
                if om is not None:
                    sector_data[sector]["op_margin"].append(om)
            except (TypeError, ValueError):
                pass

            try:
                roe = float(data["roe"]) if data["roe"] is not None else None
                if roe is not None:
                    sector_data[sector]["roe"].append(roe)
            except (TypeError, ValueError):
                pass

    benchmarks = {}
    for sector, metrics in sector_data.items():
        benchmarks[sector] = {
            "pe_median": statistics.median(metrics["pe"]) if metrics["pe"] else None,
            "op_margin_median": statistics.median(metrics["op_margin"]) if metrics["op_margin"] else None,
            "roe_median": statistics.median(metrics["roe"]) if metrics["roe"] else None,
        }
        print(f"  {sector}: P/E median {benchmarks[sector]['pe_median']:.1f}, "
              f"OpMargin {benchmarks[sector]['op_margin_median'] * 100:.1f}%, "
              f"ROE {benchmarks[sector]['roe_median'] * 100:.1f}%")

    _sector_cache = benchmarks
    _cache_timestamp = time.time()
    return benchmarks


def get_sector_benchmark(sector, metric):
    """Helper to look up a specific benchmark."""
    if sector in _sector_cache:
        return _sector_cache[sector].get(metric)
    return None