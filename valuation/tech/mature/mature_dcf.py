"""
Tech-specific DCF orchestrator.

Combines three tech-specific adjustments on top of the generic DCF engine:
1. R&D capitalization → adjusted NOPAT and invested capital
2. Ex-goodwill ROIC → used for fundamental growth calculation
3. SBC dilution projection → projected share count for per-share value

Used for tickers in TECH_BUCKETS. For other buckets, use the generic
compute_intrinsic_value_for_ticker() in valuation/dcf.py instead.
"""

import yfinance as yf

from valuation.tech import TECH_BUCKETS
from valuation.tech.shared.rd_capitalization import capitalize_rd
from valuation.tech.shared.ex_goodwill import compute_ex_goodwill_roic
from valuation.tech.shared.sbc_dilution import project_share_count

# Reinvestment cap — same as in generic growth.py
MAX_REINV = 0.80


def compute_tech_intrinsic_value(ticker, bucket=None):
    """
    Tech-specific DCF entry point.

    Returns a dict with same shape as generic DCF result, plus tech-specific
    fields documenting which adjustments fired and by how much.
    """
    if bucket not in TECH_BUCKETS:
        return {
            "error": f"{ticker} in bucket '{bucket}' is not a tech ticker. Use generic DCF.",
            "data_flags": [f"Bucket {bucket} not in TECH_BUCKETS"],
        }

    # Lazy imports to avoid circular dependencies with generic modules
    from valuation.inputs.wacc import compute_wacc
    from valuation.inputs.growth import build_growth_profile

    data_flags = [f"Using tech DCF for {ticker} (bucket: {bucket})"]

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Compute the three tech adjustments
    # ─────────────────────────────────────────────────────────────────────
    rd = capitalize_rd(ticker)
    if rd["adjusted_nopat"] is None:
        return {
            "error": f"R&D capitalization failed for {ticker}",
            "data_flags": data_flags + rd["data_flags"],
        }
    data_flags.extend(rd["data_flags"])

    goodwill = compute_ex_goodwill_roic(ticker)
    if goodwill["roic_ex_goodwill"] is None:
        return {
            "error": f"Ex-goodwill ROIC computation failed for {ticker}",
            "data_flags": data_flags + goodwill["data_flags"],
        }
    data_flags.extend(goodwill["data_flags"])

    sbc = project_share_count(ticker)
    if sbc["terminal_shares"] is None:
        # SBC failure is not fatal — fall back to current shares
        data_flags.append(f"SBC projection failed; using current shares for per-share value")
        data_flags.extend(sbc["data_flags"])
        terminal_shares = None
    else:
        terminal_shares = sbc["terminal_shares"]
        data_flags.extend(sbc["data_flags"])

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Compute WACC (same as generic)
    # ─────────────────────────────────────────────────────────────────────
    wacc_result = compute_wacc(ticker)
    wacc = wacc_result["wacc"]

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Build growth profile, then override with tech ROIC values
    # ─────────────────────────────────────────────────────────────────────
    # Use the generic growth profile to get growth rates, transition logic,
    # boom detection, etc. Then we'll override the ROIC values for stage 1
    # and re-derive fundamental growth using ex-goodwill ROIC.
    growth = build_growth_profile(ticker, wacc, bucket=bucket)
    data_flags.extend(growth.get("data_flags", []))

    # Override fundamental growth using ex-goodwill ROIC
    # fundamental_growth = reinvestment_rate × ROIC
    # We use the same reinvestment rate the generic model computed, but with
    # ex-goodwill ROIC. This captures the AMD-style insight that organic
    # growth is funded at operational economics, not goodwill-diluted ones.
    reinvestment_rate = None
    if growth.get("fundamental_growth") is not None and growth.get("current_roic"):
        # Back-solve: implied reinvestment_rate from generic fundamental
        if growth["current_roic"] > 0:
            reinvestment_rate = growth["fundamental_growth"] / growth["current_roic"]

    if reinvestment_rate is not None and goodwill["roic_ex_goodwill"] > 0:
        tech_fundamental_growth = reinvestment_rate * goodwill["roic_ex_goodwill"]
        data_flags.append(
            f"Tech-adjusted fundamental growth: {tech_fundamental_growth * 100:.1f}% "
            f"(was {growth['fundamental_growth'] * 100:.1f}% with reported ROIC)"
        )
    else:
        tech_fundamental_growth = growth.get("fundamental_growth")

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Project cash flows using ADJUSTED NOPAT as starting point
    # ─────────────────────────────────────────────────────────────────────
    # This is the key tech adjustment: start from R&D-capitalized NOPAT,
    # not reported NOPAT. NOPAT is higher for R&D-heavy companies.
    starting_nopat = rd["adjusted_nopat"]

    yearly_nopat = []
    yearly_fcff = []
    yearly_pv_fcff = []

    current_nopat = starting_nopat
    for t, (g, reinv) in enumerate(
            zip(growth["yearly_growth"], growth["yearly_reinvestment"]),
            start=1,
    ):
        current_nopat = current_nopat * (1 + g)
        # Apply same reinvestment cap as generic model
        effective_reinv = min(reinv, MAX_REINV)
        fcff = current_nopat * (1 - effective_reinv)
        # Mid-year discounting
        pv_fcff = fcff / ((1 + wacc) ** (t - 0.5))

        yearly_nopat.append(current_nopat)
        yearly_fcff.append(fcff)
        yearly_pv_fcff.append(pv_fcff)

    pv_explicit = sum(yearly_pv_fcff)

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Terminal value (same as generic, Gordon Growth)
    # ─────────────────────────────────────────────────────────────────────
    n_years = len(growth["yearly_growth"])
    terminal_nopat = current_nopat * (1 + growth["terminal_growth"])
    terminal_reinvestment = min(growth["terminal_reinvestment_rate"], MAX_REINV)
    terminal_fcff = terminal_nopat * (1 - terminal_reinvestment)

    if wacc <= growth["terminal_growth"]:
        return {
            "error": f"WACC ({wacc * 100:.2f}%) <= terminal growth ({growth['terminal_growth'] * 100:.2f}%) — terminal value undefined",
            "data_flags": data_flags,
        }

    terminal_value = terminal_fcff / (wacc - growth["terminal_growth"])
    pv_terminal = terminal_value / ((1 + wacc) ** (n_years - 0.5))

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Sum to firm value, then equity value
    # ─────────────────────────────────────────────────────────────────────
    firm_value = pv_explicit + pv_terminal
    equity_value = firm_value - goodwill["total_debt"] + goodwill["cash"]

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Per-share value using PROJECTED shares
    # ─────────────────────────────────────────────────────────────────────
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info
    current_shares = info.get("sharesOutstanding")
    current_price = info.get("currentPrice")

    if current_shares is None or current_shares <= 0:
        return {
            "error": f"No shares outstanding for {ticker}",
            "data_flags": data_flags,
        }

    # Use terminal projected shares if available
    if terminal_shares is not None:
        shares_for_per_share = terminal_shares
        per_share_value = equity_value / terminal_shares
        # Compute what it would have been with current shares (for comparison)
        per_share_value_current_shares = equity_value / current_shares
        data_flags.append(
            f"Per-share value uses projected terminal shares ({terminal_shares / 1e9:.2f}B), "
            f"not current ({current_shares / 1e9:.2f}B). "
            f"Difference: {(per_share_value / per_share_value_current_shares - 1) * 100:+.1f}%"
        )
    else:
        shares_for_per_share = current_shares
        per_share_value = equity_value / current_shares
        per_share_value_current_shares = per_share_value

    upside_downside = (per_share_value / current_price - 1) if current_price else None

    return {
        "ticker": ticker,
        "bucket": bucket,
        "model_version": "tech_v1",
        # Core valuation
        "per_share_value": per_share_value,
        "current_price": current_price,
        "upside_downside": upside_downside,
        "firm_value": firm_value,
        "equity_value": equity_value,
        "pv_explicit": pv_explicit,
        "pv_terminal": pv_terminal,
        "terminal_pct_of_value": pv_terminal / firm_value if firm_value else None,
        # Cost of capital
        "wacc": wacc,
        # Cash flows
        "starting_nopat": starting_nopat,
        "yearly_nopat": yearly_nopat,
        "yearly_fcff": yearly_fcff,
        "yearly_pv_fcff": yearly_pv_fcff,
        "terminal_value": terminal_value,
        # Growth structure
        "growth_profile": growth,
        # Tech-specific outputs
        "rd_adjustment": rd,
        "goodwill_analysis": goodwill,
        "sbc_projection": sbc,
        "shares_used_for_per_share": shares_for_per_share,
        "current_shares": current_shares,
        "tech_fundamental_growth": tech_fundamental_growth,
        # Comparison to no-SBC-adjustment case
        "per_share_value_at_current_shares": per_share_value_current_shares,
        # Status
        "data_flags": data_flags,
    }


# ════════════════════════════════════════════════════════════════════════
# Test harness
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Mix of: non-acquirer (NVDA), moderate (MSFT, GOOGL), heavy SBC (PLTR),
    # heavy acquirer (AVGO, AMD)
    TEST_CASES = [
        ("MSFT", "mature_tech"),
        ("NVDA", "semiconductors"),
        ("META", "communication"),
        ("AAPL", "mature_tech"),
        ("AMD", "semiconductors"),
        ("AVGO", "semiconductors"),
        ("PLTR", "saas_growth"),
    ]

    for ticker, bucket in TEST_CASES:
        print(f"\n{'=' * 70}")
        print(f"  {ticker} ({bucket})")
        print(f"{'=' * 70}")

        result = compute_tech_intrinsic_value(ticker, bucket=bucket)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            for f in result.get("data_flags", []):
                print(f"    - {f}")
            continue

        print(f"\n  Per-share value:        ${result['per_share_value']:.2f}")
        print(f"  Current price:          ${result['current_price']:.2f}")
        ud = result['upside_downside']
        if ud is not None:
            print(f"  Upside/(downside):      {ud * 100:+.1f}%")

        print(f"\n  Tech adjustments:")
        rd = result['rd_adjustment']
        print(f"    R&D asset:            ${rd['rd_asset'] / 1e9:.1f}B")
        print(f"    NOPAT reported:       ${rd['reported_nopat'] / 1e9:.1f}B")
        print(f"    NOPAT adjusted:       ${rd['adjusted_nopat'] / 1e9:.1f}B")

        gw = result['goodwill_analysis']
        print(f"    ROIC reported:        {gw['roic_reported'] * 100:.1f}%")
        print(f"    ROIC ex-goodwill:     {gw['roic_ex_goodwill'] * 100:.1f}%")
        print(f"    Heavy acquirer:       {gw['is_heavy_acquirer']}")

        sbc = result['sbc_projection']
        print(f"    Dilution rate:        {sbc['historical_dilution_rate'] * 100:+.2f}%/yr")
        print(f"    Current shares:       {result['current_shares'] / 1e9:.2f}B")
        print(f"    Projected shares:     {result['shares_used_for_per_share'] / 1e9:.2f}B")

        print(f"\n  Valuation breakdown:")
        print(f"    PV explicit:          ${result['pv_explicit'] / 1e9:.0f}B")
        print(f"    PV terminal:          ${result['pv_terminal'] / 1e9:.0f}B")
        print(f"    Terminal % of FV:     {result['terminal_pct_of_value'] * 100:.0f}%")
        print(f"    WACC:                 {result['wacc'] * 100:.2f}%")
        print(f"    Firm value:           ${result['firm_value'] / 1e9:.0f}B")
        print(f"    Equity value:         ${result['equity_value'] / 1e9:.0f}B")