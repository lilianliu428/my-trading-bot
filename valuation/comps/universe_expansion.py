"""
One-off diagnostic tool: source candidate tickers for the semiconductors and
mature_tech comps universes from Yahoo's sector/industry taxonomy, classify
each via the project's existing classify() function, and validate data
availability via features.py/multiples.py.

Source preference order (task spec):
    1. yfinance's Industry/Sector API (yf.Industry(key).top_companies) —
       tried first; this is a real, currently-working yfinance 1.4.1
       feature, not a screener scrape.
    2. Yahoo Finance screener page scrape via requests + BeautifulSoup —
       only attempted if (1) returns nothing.
    3. Static hardcoded fallback list — only used if both (1) and (2) fail.

Read-only discovery tool — does NOT mutate universe.py. Results are
reviewed and applied to COMPS_UNIVERSE by hand. Run directly:
    python3 -m valuation.comps.universe_expansion
"""

import time

import yfinance as yf

from scoring.sectors.classifier import classify
from valuation.comps.universe import COMPS_UNIVERSE
from valuation.comps.features import compute_features_for_ticker
from valuation.comps.multiples import compute_multiples_for_ticker

MIN_MARKET_CAP = 500e6
MIN_REVENUE_TTM = 100e6
SLEEP_BETWEEN_CALLS = 0.15  # basic rate-limit courtesy between per-ticker yfinance calls

# Yahoo industry taxonomy keys feeding each bucket's candidate pool — chosen
# to match scoring/sectors/classifier.py's INDUSTRY_RULES mapping, and the
# task's SIC/GICS guidance (semis: SIC 3674 / GICS Semiconductors &
# Semiconductor Equipment; mature tech: SIC 7370-7379/3571 / GICS Software,
# IT Services, Communications Equipment, Computer Hardware).
SOURCE_INDUSTRIES = {
    "semiconductors": ["semiconductors", "semiconductor-equipment-materials"],
    "mature_tech": [
        "software-infrastructure", "software-application",
        "information-technology-services", "communication-equipment",
        "computer-hardware",
    ],
}

# Static fallback, used only if both the yfinance Industry API and the Yahoo
# scrape fail. Real, well-known US-listed tickers (as of this writing).
STATIC_FALLBACK = {
    "semiconductors": [
        "NVDA", "AMD", "AVGO", "INTC", "QCOM", "TXN", "MU", "ADI", "NXPI", "MCHP", "ON",
        "MRVL", "SWKS", "QRVO", "MPWR", "LSCC", "CRUS", "SLAB", "DIOD", "POWI", "VSH",
        "RMBS", "SYNA", "AMAT", "LRCX", "KLAC", "TER", "ENTG", "ASML", "TSM", "UMC",
        "ARM", "COHR", "IPGP", "MTSI", "OLED", "WOLF", "ALGM", "SITM", "GFS", "CEVA",
    ],
    "mature_tech": [
        "MSFT", "AAPL", "GOOGL", "META", "AMZN", "ORCL", "CRM", "ADBE", "IBM", "CSCO",
        "NOW", "INTU", "WDAY", "TEAM", "DDOG", "SNOW", "PANW", "NFLX", "DIS", "TMUS",
        "VZ", "T", "HPQ", "DELL", "ANSS", "CDNS", "SNPS", "FTNT", "PAYX", "ADSK",
        "SSNC", "JKHY", "CTSH", "ACN", "GEN", "AKAM", "JNPR", "FFIV", "CIEN", "MSI",
    ],
}


def fetch_candidates_via_yfinance_industry(bucket):
    """Source 1: yf.Industry(key).top_companies for every source industry key."""
    tickers = set()
    for key in SOURCE_INDUSTRIES[bucket]:
        try:
            top = yf.Industry(key).top_companies
            if top is not None and not top.empty:
                tickers.update(top.index.tolist())
        except Exception as e:
            print(f"  [WARN] yf.Industry('{key}') failed: {type(e).__name__}: {e}")
    return sorted(tickers)


def fetch_candidates_via_scrape(bucket):
    """Source 2: requests + BeautifulSoup scrape of Yahoo's legacy screener pages."""
    import requests
    from bs4 import BeautifulSoup

    url = (
        "https://finance.yahoo.com/screener/predefined/sector?category=technology&industry=semiconductors"
        if bucket == "semiconductors"
        else "https://finance.yahoo.com/screener/predefined/sector?category=technology"
    )
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        tickers = sorted({a.get_text(strip=True) for a in soup.select("a[data-test='quoteLink']")})
        return [t for t in tickers if t]
    except Exception as e:
        print(f"  [WARN] Yahoo scrape fallback failed: {type(e).__name__}: {e}")
        return []


def source_candidates(bucket):
    """Try each source in preference order; return (tickers, source_used)."""
    candidates = fetch_candidates_via_yfinance_industry(bucket)
    if candidates:
        return candidates, "yfinance Industry.top_companies"

    print("  yfinance Industry API returned nothing — trying Yahoo scrape fallback...")
    candidates = fetch_candidates_via_scrape(bucket)
    if candidates:
        return candidates, "Yahoo Finance screener scrape"

    print("  Scrape fallback also failed — using static hardcoded list...")
    return list(STATIC_FALLBACK[bucket]), "static hardcoded fallback"


def classify_and_prefilter(ticker, bucket, data_flags):
    """
    Cheap first-pass filter (one .info fetch): confirm classify() agrees
    this belongs in `bucket`, and check market cap / revenue thresholds.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        data_flags.append(f"{ticker}: info fetch failed ({type(e).__name__}: {e})")
        return False

    market_cap = info.get("marketCap")
    total_revenue = info.get("totalRevenue")
    row = {
        "ticker": ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "revenue_growth": info.get("revenueGrowth"),
        "profit_margin": info.get("profitMargins"),
        "total_revenue": total_revenue,
    }
    derived_bucket = classify(row)

    if derived_bucket != bucket:
        data_flags.append(f"{ticker}: classify() -> '{derived_bucket}', not '{bucket}' — dropped")
        return False
    if market_cap is None or market_cap < MIN_MARKET_CAP:
        data_flags.append(f"{ticker}: market cap {market_cap} below ${MIN_MARKET_CAP / 1e6:.0f}M minimum")
        return False
    if total_revenue is None or total_revenue < MIN_REVENUE_TTM:
        data_flags.append(f"{ticker}: TTM revenue {total_revenue} below ${MIN_REVENUE_TTM / 1e6:.0f}M minimum")
        return False
    return True


def validate_features(ticker, data_flags):
    """
    Expensive second-pass filter: does features.py/multiples.py actually
    produce a complete, usable observation for this ticker? (This is the
    same completeness bar Phase 1/2 of the comps engine itself requires —
    is_complete() encodes the "3+ years of continuous history" criterion
    via trailing_growth's 4-column requirement.)
    """
    feats = compute_features_for_ticker(ticker)
    if not feats.is_complete():
        missing = [f for f in ("trailing_growth", "operating_margin", "roic", "log_revenue") if getattr(feats, f) is None]
        data_flags.append(f"{ticker}: incomplete feature vector (missing: {missing})")
        return False

    mults = compute_multiples_for_ticker(ticker)
    if mults["ev_sales"] is None:
        data_flags.append(f"{ticker}: no ev_sales multiple computable ({mults['data_flags']})")
        return False
    return True


def expand_bucket(bucket):
    print(f"\n{'=' * 80}\nEXPANDING: {bucket}\n{'=' * 80}")
    existing = set(COMPS_UNIVERSE.get(bucket, []))

    candidates, source = source_candidates(bucket)
    print(f"  Source used: {source}")
    print(f"  Raw candidates found: {len(candidates)}")

    new_candidates = sorted(set(candidates) - existing)
    print(f"  Already in universe: {len(candidates) - len(new_candidates)}  |  New candidates to evaluate: {len(new_candidates)}")

    data_flags = []
    prefilter_survivors = []
    for i, t in enumerate(new_candidates):
        if classify_and_prefilter(t, bucket, data_flags):
            prefilter_survivors.append(t)
        time.sleep(SLEEP_BETWEEN_CALLS)
    print(f"  Passed bucket/market-cap/revenue prefilter: {len(prefilter_survivors)} / {len(new_candidates)}")

    final_survivors = []
    for t in prefilter_survivors:
        if validate_features(t, data_flags):
            final_survivors.append(t)
        time.sleep(SLEEP_BETWEEN_CALLS)
    print(f"  Passed feature-completeness validation: {len(final_survivors)} / {len(prefilter_survivors)}")

    final_universe = sorted(existing | set(final_survivors))
    print(f"\n  Final universe size: {len(final_universe)} (existing {len(existing)} + {len(final_survivors)} newly validated)")

    return {
        "bucket": bucket,
        "source": source,
        "raw_candidates_found": len(candidates),
        "new_candidates_evaluated": len(new_candidates),
        "prefilter_survivors": prefilter_survivors,
        "final_survivors": final_survivors,
        "final_universe": final_universe,
        "data_flags": data_flags,
    }


def run_universe_expansion(buckets=("semiconductors", "mature_tech")):
    results = {}
    for bucket in buckets:
        results[bucket] = expand_bucket(bucket)

    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    for bucket, r in results.items():
        status = "PASS (>= 40)" if len(r["final_universe"]) >= 40 else "BELOW 40 — needs more sourcing"
        print(f"\n{bucket}: {len(r['final_universe'])} tickers total  [{status}]")
        print(f"  source: {r['source']}")
        print(f"  new tickers validated and added: {r['final_survivors']}")

    print("\nFailure reasons:")
    for bucket, r in results.items():
        print(f"\n  [{bucket}] ({len(r['data_flags'])} total)")
        for f in r["data_flags"]:
            print(f"    - {f}")

    return results


if __name__ == "__main__":
    run_universe_expansion()
