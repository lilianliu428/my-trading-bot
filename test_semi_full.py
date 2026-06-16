"""
test_semi_full.py

Run the full tech DCF on the semiconductor universe and print fair values
side by side with market prices. The goal is to see whether the model
produces defensible numbers across the spectrum: heavy acquirers (AVGO, AMD),
organic compounders (NVDA), turnarounds (INTC), memory (MU), and analog (TXN, ADI).

We do not tune the model to make any specific ticker "look right." We surface
what the model says, then judge it as a whole.
"""

from valuation.tech.mature.mature_dcf import compute_tech_intrinsic_value


# Representative semi list spanning the three cycles
SEMI_TICKERS = [
    # AI infrastructure
    "NVDA",   # organic, AI leader
    "AMD",    # heavy acquirer (Xilinx), AI challenger
    "AVGO",   # heavy acquirer (VMware++), AI-custom-silicon
    "ASML",   # lithography monopoly
    "INTC",   # turnaround, AI ambitions
    # Memory
    "MU",     # DRAM/NAND
    # Diversified analog
    "TXN",    # broad analog
    "ADI",    # broad analog
]


def main():
    results = []

    for ticker in SEMI_TICKERS:
        print(f"\n{'─' * 70}")
        print(f"  {ticker}")
        print(f"{'─' * 70}")

        try:
            r = compute_tech_intrinsic_value(ticker, bucket="semiconductors")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append((ticker, None, None, None, "exception"))
            continue

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            results.append((ticker, None, None, None, "error"))
            continue

        fv = r["per_share_value"]
        mp = r["current_price"]
        ud = r["upside_downside"]
        status = "ok"

        print(f"  Fair value:           ${fv:.2f}")
        print(f"  Market price:         ${mp:.2f}")
        if ud is not None:
            print(f"  Upside/(downside):    {ud * 100:+.1f}%")
        print(f"  WACC:                 {r['wacc'] * 100:.2f}%")
        print(f"  High-growth years:    {r['growth_profile']['high_growth_years']}")
        print(f"  Initial growth:       {r['growth_profile']['yearly_growth'][0] * 100:.2f}%")
        print(f"  Terminal growth:      {r['growth_profile']['yearly_growth'][-1] * 100:.2f}%")
        print(f"  Terminal ROIC:        {r['growth_profile']['yearly_roic'][-1] * 100:.2f}%")
        print(f"  Heavy acquirer:       {r['goodwill_analysis']['is_heavy_acquirer']}")
        print(f"  Terminal % of FV:     {r['terminal_pct_of_value'] * 100:.0f}%")

        results.append((ticker, fv, mp, ud, status))

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Ticker':<8} {'Fair Value':>12} {'Market':>10} {'Upside':>10}  {'Status':<10}")
    print(f"  {'-' * 60}")
    for ticker, fv, mp, ud, status in results:
        if fv is None:
            print(f"  {ticker:<8} {'n/a':>12} {'n/a':>10} {'n/a':>10}  {status}")
        else:
            ud_str = f"{ud * 100:+.1f}%" if ud is not None else "n/a"
            print(f"  {ticker:<8} ${fv:>10.2f} ${mp:>8.2f} {ud_str:>10}  {status}")


if __name__ == "__main__":
    main()