"""
test_heavy_acquirer_routing.py

Verify that heavy-acquirer routing fires for AVGO/AMD and stays silent for NVDA/MSFT.
Prints the data_flags so we can see what the model decided.
"""

from valuation.inputs.wacc import compute_wacc
from valuation.inputs.growth import build_growth_profile

TICKERS = [
    ("AVGO", "semiconductors",  "expected: heavy acquirer (multiple large M&A)"),
    ("AMD",  "semiconductors",  "expected: heavy acquirer (Xilinx)"),
    ("NVDA", "semiconductors",  "expected: NOT heavy acquirer (organic)"),
    ("MSFT", "mature_tech",     "expected: NOT heavy acquirer (light goodwill)"),
]


def main():
    for ticker, bucket, expectation in TICKERS:
        print("\n" + "=" * 70)
        print(f"  {ticker}  [{bucket}]  -  {expectation}")
        print("=" * 70)

        try:
            wacc_result = compute_wacc(ticker)
            wacc = wacc_result["wacc"]
            profile = build_growth_profile(ticker, wacc, bucket=bucket)

            heavy_flags = [f for f in profile["data_flags"] if "Heavy acquirer" in f]
            if heavy_flags:
                print("  ✓ Heavy-acquirer routing TRIGGERED:")
                for f in heavy_flags:
                    print(f"    → {f}")
            else:
                print("  ✓ Heavy-acquirer routing NOT triggered (normal path used)")

            # Print key derived numbers for sanity-check
            print(f"\n  Stage 1 ROIC:      {profile['yearly_roic'][0] * 100:.2f}%")
            print(f"  Terminal ROIC:     {profile['yearly_roic'][-1] * 100:.2f}%")
            print(f"  High-growth years: {profile['high_growth_years']}")
            print(f"  Initial growth:    {profile['yearly_growth'][0] * 100:.2f}%")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()