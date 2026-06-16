"""
test_cycle_classifier.py

Sanity-check the semi cycle classifier on a known set of tickers.
No DCF math yet - just confirming the classifier produces sensible buckets.
"""

from valuation.tech.semiconductors.cycle_classifier import classify_semi_cycle


# Known semis - we should classify all of these correctly via the hardcoded map
KNOWN_SEMIS = [
    "NVDA",   # expect ai_infrastructure
    "AMD",    # expect ai_infrastructure
    "AVGO",   # expect ai_infrastructure
    "ASML",   # expect ai_infrastructure
    "INTC",   # expect ai_infrastructure
    "MU",     # expect memory
    "WDC",    # expect memory
    "TXN",    # expect diversified_analog
    "ADI",    # expect diversified_analog
    "MCHP",   # expect diversified_analog
]

# Unknown ticker(s) to exercise the fallback path
UNKNOWN_SEMIS = [
    "ALAB",      # newer AI networking - should hit yfinance fallback
    "FAKE123",   # nonsense ticker - should hit final default
]


def main():
    print("=" * 80)
    print("Semi Cycle Classifier - Sanity Check")
    print("=" * 80)

    print("\n--- KNOWN SEMIS (hardcoded path) ---\n")
    for ticker in KNOWN_SEMIS:
        r = classify_semi_cycle(ticker)
        print(f"{ticker:6s} → {r['cycle']:22s}  runway={r['runway_years']:2d}yr  "
              f"terminal_g={r['terminal_growth']*100:.1f}%  [{r['source']}]")
        print(f"        {r['rationale']}\n")

    print("\n--- UNKNOWN SEMIS (fallback path) ---\n")
    for ticker in UNKNOWN_SEMIS:
        r = classify_semi_cycle(ticker)
        print(f"{ticker:6s} → {r['cycle']:22s}  runway={r['runway_years']:2d}yr  "
              f"terminal_g={r['terminal_growth']*100:.1f}%  [{r['source']}]")
        print(f"        {r['rationale']}\n")


if __name__ == "__main__":
    main()