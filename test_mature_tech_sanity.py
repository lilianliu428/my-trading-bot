"""
test_mature_tech_sanity.py

Run the tech DCF on mature_tech and communication tickers to see if the
catastrophic underpricing pattern we saw on non-AI-leader semis is
semi-specific or generic. If MSFT/GOOGL/AAPL/META come out reasonable,
the semi bucket has a specific issue. If they also come out at -70%+,
the runway formula has a general problem across all tech.
"""

from valuation.tech.mature.mature_dcf import compute_tech_intrinsic_value


TICKERS = [
    ("MSFT",  "mature_tech"),
    ("GOOGL", "mature_tech"),
    ("AAPL",  "mature_tech"),
    ("META",  "communication"),
    ("ORCL",  "mature_tech"),  # mature but acquisitive, useful comparison
]


def main():
    results = []

    for ticker, bucket in TICKERS:
        print(f"\n{'─' * 70}")
        print(f"  {ticker}  ({bucket})")
        print(f"{'─' * 70}")

        try:
            r = compute_tech_intrinsic_value(ticker, bucket=bucket)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append((ticker, bucket, None, None, None, "exception"))
            continue

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            results.append((ticker, bucket, None, None, None, "error"))
            continue

        fv = r["per_share_value"]
        mp = r["current_price"]
        ud = r["upside_downside"]

        print(f"  Fair value:           ${fv:.2f}")
        print(f"  Market price:         ${mp:.2f}")
        if ud is not None:
            print(f"  Upside/(downside):    {ud * 100:+.1f}%")
        print(f"  WACC:                 {r['wacc'] * 100:.2f}%")
        print(f"  High-growth years:    {r['growth_profile']['high_growth_years']}")
        print(f"  Initial growth:       {r['growth_profile']['yearly_growth'][0] * 100:.2f}%")
        print(f"  Terminal growth:      {r['growth_profile']['yearly_growth'][-1] * 100:.2f}%")
        print(f"  Stage 1 ROIC:         {r['growth_profile']['yearly_roic'][0] * 100:.2f}%")
        print(f"  Terminal ROIC:        {r['growth_profile']['yearly_roic'][-1] * 100:.2f}%")
        print(f"  Heavy acquirer:       {r['goodwill_analysis']['is_heavy_acquirer']}")
        print(f"  Terminal % of FV:     {r['terminal_pct_of_value'] * 100:.0f}%")

        results.append((ticker, bucket, fv, mp, ud, "ok"))

    # Summary
    print(f"\n\n{'=' * 75}")
    print("  SUMMARY")
    print(f"{'=' * 75}")
    print(f"  {'Ticker':<8} {'Bucket':<15} {'Fair Value':>12} {'Market':>10} {'Upside':>10}")
    print(f"  {'-' * 65}")
    for ticker, bucket, fv, mp, ud, status in results:
        if fv is None:
            print(f"  {ticker:<8} {bucket:<15} {'n/a':>12} {'n/a':>10} {'n/a':>10}  [{status}]")
        else:
            ud_str = f"{ud * 100:+.1f}%" if ud is not None else "n/a"
            print(f"  {ticker:<8} {bucket:<15} ${fv:>10.2f} ${mp:>8.2f} {ud_str:>10}")


if __name__ == "__main__":
    main()