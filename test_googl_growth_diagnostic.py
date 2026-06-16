"""
test_googl_growth_diagnostic.py

GOOGL came out at +195% upside, which is the loudest anomaly in either
sanity-check run. Trace exactly what each growth signal returned so we can
see whether the +195% is honest model output (GOOGL really IS undervalued
by the market according to first principles) or whether one of the signals
is misfiring and dragging the blend up.

Comparison ticker: MSFT (came out at +28%, which feels more reasonable for
a similar large-cap AI-exposed mature tech name). The contrast should tell
us what's structurally different.
"""

from valuation.inputs.growth import (
    compute_fundamental_growth,
    compute_historical_growth,
    get_consensus_growth,
    build_growth_profile,
)
from valuation.inputs.wacc import compute_wacc


def trace(ticker, bucket):
    print("=" * 75)
    print(f"  GROWTH SIGNAL TRACE: {ticker}  ({bucket})")
    print("=" * 75)

    print("\n--- Fundamental growth ---")
    fund = compute_fundamental_growth(ticker)
    print(f"  EBIT after-tax:           ${fund['ebit_after_tax']/1e9:.2f}B")
    print(f"  Reinvestment:             ${fund['reinvestment']/1e9:.2f}B")
    print(f"  Reinvestment rate:        {fund['reinvestment_rate']*100:.2f}%")
    print(f"  Invested capital:         ${fund['invested_capital']/1e9:.2f}B")
    print(f"  Goodwill:                 ${fund['goodwill']/1e9:.2f}B")
    gw_ratio = fund.get('goodwill_ratio')
    if gw_ratio is not None:
        print(f"  Goodwill ratio:           {gw_ratio*100:.1f}%")
    print(f"  ROIC (reported):          {fund['roic']*100:.2f}%")
    if fund.get('roic_ex_goodwill') is not None:
        print(f"  ROIC (ex-goodwill):       {fund['roic_ex_goodwill']*100:.2f}%")
    print(f"  Fundamental growth:       {fund['fundamental_growth']*100:.2f}%")

    print("\n--- Historical growth ---")
    hist = compute_historical_growth(ticker)
    if hist['growth_rate'] is not None:
        print(f"  Series used:              {hist['series_used']}")
        print(f"  Method:                   {hist['method_used']}")
        print(f"  Growth rate:              {hist['growth_rate']*100:.2f}%")
        print(f"  R²:                       {hist['r_squared']:.3f}")
        print(f"  N years:                  {hist['n_years']}")
    else:
        print(f"  Growth rate:              n/a")

    print("\n--- Consensus growth ---")
    cons = get_consensus_growth(ticker)
    if cons['consensus_growth'] is not None:
        print(f"  Consensus growth:         {cons['consensus_growth']*100:.2f}%")
    if cons['data_flags']:
        for flag in cons['data_flags']:
            print(f"  Flag:                     {flag}")

    print("\n--- Final blended profile ---")
    wacc_result = compute_wacc(ticker)
    profile = build_growth_profile(ticker, wacc_result['wacc'], bucket=bucket)
    print(f"  Initial growth (Year 1):  {profile['yearly_growth'][0]*100:.2f}%")
    print(f"  Terminal growth:          {profile['yearly_growth'][-1]*100:.2f}%")
    print(f"  Stage 1 ROIC:             {profile['yearly_roic'][0]*100:.2f}%")
    print(f"  Terminal ROIC:            {profile['yearly_roic'][-1]*100:.2f}%")
    print(f"  High-growth years:        {profile['high_growth_years']}")
    print(f"  WACC:                     {wacc_result['wacc']*100:.2f}%")
    print(f"\n  Relevant data flags:")
    for flag in profile['data_flags']:
        if any(k in flag.lower() for k in ['blend', 'cap', 'boom', 'maturity', 'margin', 'regime', 'heavy']):
            print(f"    - {flag}")
    print()


if __name__ == "__main__":
    trace("GOOGL", "mature_tech")
    print("\n")
    trace("MSFT", "mature_tech")