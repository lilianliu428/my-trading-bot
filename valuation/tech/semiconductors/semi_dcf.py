"""
semi_dcf.py

Semiconductor-specific DCF orchestrator.

This is the home for all semi-specific valuation logic. Today it's a thin
wrapper around the shared tech DCF, because most of the semi-specific behavior
lives upstream in `valuation/inputs/growth.py` (heavy-acquirer routing,
Path A runway cap from cycle classifier, cycle-specific terminal growth).

As we add more semi-specific logic, it consolidates here:
- [planned] Capex-led revenue projection with 18-month lag
- [planned] Customer concentration adjustment to WACC
- [planned] Fab useful-life extension (production assets generate revenue 12-15
  years; standard depreciation assumes 7-10 — extend the productive life
  assumption for terminal value)
- [planned] Cycle-position sense check (where in cycle are we?)

For now, this file:
1. Validates that the ticker is actually a known semi
2. Surfaces the cycle classification result alongside the DCF result
3. Routes through compute_tech_intrinsic_value (the shared tech engine)
"""

from valuation.tech.mature.mature_dcf import compute_tech_intrinsic_value
from valuation.tech.semiconductors.cycle_classifier import classify_semi_cycle


def compute_semi_intrinsic_value(ticker):
    """
    Semi-specific DCF entry point.

    Returns the same dict shape as compute_tech_intrinsic_value, with an
    added 'semi_cycle' field containing the cycle classifier's output.

    Parameters
    ----------
    ticker : str
        Stock ticker (case-insensitive)

    Returns
    -------
    dict
        Same shape as compute_tech_intrinsic_value, plus:
        - semi_cycle: dict with cycle, runway_years, terminal_growth,
          rationale, source from classify_semi_cycle
    """
    ticker = ticker.upper()

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Classify the semi by cycle.
    # The classifier itself never raises (has a defensible default for
    # unknown tickers) — but we surface its result so the bot can show
    # the user why it valued this ticker the way it did.
    # ─────────────────────────────────────────────────────────────────────
    semi_cycle = classify_semi_cycle(ticker)

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Run the shared tech DCF with bucket="semiconductors".
    # The bucket parameter triggers downstream semi-specific behavior in
    # valuation/inputs/growth.py:
    #   - Path A runway cap (cycle classifier's max overrides excess-return
    #     formula when the formula overshoots)
    #   - Cycle-specific terminal_growth (memory 2.0%, AI-infra 3.5%,
    #     analog 2.8%) instead of risk-free rate
    # Heavy-acquirer routing also fires here if goodwill > 40% of invested
    # capital — that's bucket-agnostic, triggered by data not by name.
    # ─────────────────────────────────────────────────────────────────────
    result = compute_tech_intrinsic_value(ticker, bucket="semiconductors")

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Attach the semi cycle info to the result for downstream use
    # (bot display, debugging, model comparison).
    # ─────────────────────────────────────────────────────────────────────
    if isinstance(result, dict):
        result["semi_cycle"] = semi_cycle

    return result


# ════════════════════════════════════════════════════════════════════════
# Test harness — runs the full semi list
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    SEMI_TICKERS = [
        "NVDA",   # AI-infrastructure, organic
        "AMD",    # AI-infrastructure, heavy acquirer (Xilinx)
        "AVGO",   # AI-infrastructure, heavy acquirer (VMware++)
        "INTC",   # AI-infrastructure, turnaround
        "MU",     # Memory
        "TXN",    # Diversified analog
        "ADI",    # Diversified analog, heavy acquirer (Maxim)
    ]

    results = []

    for ticker in SEMI_TICKERS:
        print(f"\n{'─' * 70}")
        print(f"  {ticker}")
        print(f"{'─' * 70}")

        try:
            r = compute_semi_intrinsic_value(ticker)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append((ticker, None, None, None, "exception"))
            continue

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            results.append((ticker, None, None, None, "error"))
            continue

        cycle = r["semi_cycle"]
        fv = r["per_share_value"]
        mp = r["current_price"]
        ud = r["upside_downside"]

        print(f"  Cycle:                {cycle['cycle']} (source: {cycle['source']})")
        print(f"  Cycle max runway:     {cycle['runway_years']} yr")
        print(f"  Cycle terminal g:     {cycle['terminal_growth']*100:.1f}%")
        print(f"")
        print(f"  Fair value:           ${fv:.2f}")
        print(f"  Market price:         ${mp:.2f}")
        if ud is not None:
            print(f"  Upside/(downside):    {ud * 100:+.1f}%")
        print(f"  WACC:                 {r['wacc'] * 100:.2f}%")
        print(f"  High-growth years:    {r['growth_profile']['high_growth_years']}")
        print(f"  Initial growth:       {r['growth_profile']['yearly_growth'][0] * 100:.2f}%")
        print(f"  Heavy acquirer:       {r['goodwill_analysis']['is_heavy_acquirer']}")

        results.append((ticker, cycle['cycle'], fv, mp, ud, "ok"))

    # Summary
    print(f"\n\n{'=' * 75}")
    print("  SEMI DCF SUMMARY")
    print(f"{'=' * 75}")
    print(f"  {'Ticker':<8} {'Cycle':<22} {'Fair Value':>12} {'Market':>10} {'Upside':>10}")
    print(f"  {'-' * 70}")
    for row in results:
        ticker = row[0]
        if len(row) == 5 or row[2] is None:
            status = row[-1]
            print(f"  {ticker:<8} {'':<22} {'n/a':>12} {'n/a':>10} {'n/a':>10}  [{status}]")
        else:
            _, cycle, fv, mp, ud, _ = row
            ud_str = f"{ud * 100:+.1f}%" if ud is not None else "n/a"
            print(f"  {ticker:<8} {cycle:<22} ${fv:>10.2f} ${mp:>8.2f} {ud_str:>10}")