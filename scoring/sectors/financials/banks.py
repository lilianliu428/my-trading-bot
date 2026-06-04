"""
Banks scoring profile.

For traditional and investment banks.
Examples: JPM, BAC, WFC, C, GS, MS, USB, TFC, PNC, BK.

What's DIFFERENT about banks:
  - Debt/equity is MEANINGLESS (banks ARE leverage by design)
  - Operating margin / FCF use a different P&L structure
  - P/B matters way more than for other industries (NAV is real)
  - ROE is THE single best "well-run bank" check
  - Growth bar is LOW (banks grow ~3-7%; 0% growth isn't a red flag yet)

What we MEASURE:
  - ROE > 10% (well-managed bank)
  - P/B reasonable (< 1.8)
  - P/E reasonable (< 15)
  - Profit margin > 20% (banks should be very profitable)
  - Revenue not declining
"""


def score(row):
    """
    Score a bank.
    `row` is a dict from the fundamentals table.
    Returns (score, max_score, all_cores_passed, reasons)
    """
    score = 0.0
    max_score = 0.0
    reasons = []
    cores_passed = []

    rev_growth = row.get("revenue_growth")
    pe = row.get("pe")
    pb = row.get("price_to_book")
    roe = row.get("roe")
    profit_margin = row.get("profit_margin")

    # ─── CORE 1: ROE > 10% ─── weight 3
    max_score += 3
    if roe is not None:
        passed = roe > 0.10
        if passed:
            score += 3
            if roe > 0.15:
                reasons.append(f"✅ ROE {roe*100:.1f}% (excellent, well-run)")
            else:
                reasons.append(f"✅ ROE {roe*100:.1f}% (>10%)")
        else:
            reasons.append(f"❌ ROE {roe*100:.1f}% (need >10%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ ROE unknown")
        cores_passed.append(False)

    # ─── CORE 2: P/B reasonable ─── weight 3
    max_score += 3
    if pb is not None and pb > 0:
        passed = pb < 1.8
        if passed:
            score += 3
            if pb < 1.0:
                reasons.append(f"✅ P/B {pb:.2f} (trading below book — value zone)")
            elif pb < 1.3:
                reasons.append(f"✅ P/B {pb:.2f} (cheap)")
            else:
                reasons.append(f"✅ P/B {pb:.2f} (reasonable)")
        else:
            reasons.append(f"❌ P/B {pb:.2f} (>=1.8, expensive)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ P/B unknown")
        cores_passed.append(False)

    # ─── CORE 3: P/E reasonable ─── weight 3
    max_score += 3
    if pe is not None and pe > 0:
        passed = pe < 15
        if passed:
            score += 3
            reasons.append(f"✅ P/E {pe:.1f} (<15)")
        else:
            reasons.append(f"❌ P/E {pe:.1f} (>=15, rich for a bank)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ P/E unknown")
        cores_passed.append(False)

    # ─── CORE 4: Profit margin > 20% ─── weight 3
    max_score += 3
    if profit_margin is not None:
        passed = profit_margin > 0.20
        if passed:
            score += 3
            reasons.append(f"✅ Profit margin {profit_margin*100:.1f}% (>20%)")
        else:
            reasons.append(f"❌ Profit margin {profit_margin*100:.1f}% (need >20%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Profit margin unknown")
        cores_passed.append(False)

    # ─── CORE 5: Revenue not declining ─── weight 3
    max_score += 3
    if rev_growth is not None:
        passed = rev_growth >= 0.00
        if passed:
            score += 3
            if rev_growth > 0.05:
                reasons.append(f"✅ Revenue growing {rev_growth*100:.1f}% (healthy for a bank)")
            else:
                reasons.append(f"✅ Revenue {rev_growth*100:.1f}% (flat is OK for banks)")
        else:
            reasons.append(f"❌ Revenue declining {rev_growth*100:.1f}%")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Revenue growth unknown")
        cores_passed.append(False)

    all_cores_passed = all(cores_passed)
    return score, max_score, all_cores_passed, reasons
