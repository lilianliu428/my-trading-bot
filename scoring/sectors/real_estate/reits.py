"""
REIT (Real Estate Investment Trust) scoring profile.

For property-owning trusts that distribute most income as dividends.
Examples: O, PLD, AMT, EQIX, SPG, CCI, WELL, PSA, EQR, VTR.

What's DIFFERENT about REITs:
  - P/E is MISLEADING. Real estate depreciation makes net income look low
    but FFO (Funds From Operations) is closer to true cash flow.
  - We don't have FFO directly, so we use FCF + profit margin as proxies.
  - REITs use a LOT of leverage by design — debt/equity not useful here.
  - Growth is slow; 3-5% revenue growth is healthy.
  - Dividend safety matters most (current ratio is a rough proxy).

What we MEASURE:
  - Revenue growth > 3% (modest bar)
  - FCF positive (real cash flow)
  - Profit margin > 10% (REITs should make real money)
  - Current ratio > 0.5 (REITs run thin liquidity by design, just need adequacy)
  - Reasonable institutional ownership (REITs usually attract institutions)
"""


def score(row):
    score = 0.0
    max_score = 0.0
    reasons = []
    cores_passed = []

    rev_growth = row.get("revenue_growth")
    profit_margin = row.get("profit_margin")
    fcf_positive = row.get("free_cash_flow_positive")
    current_ratio = row.get("current_ratio")
    inst_own = row.get("institutional_ownership")

    # ─── CORE 1: Revenue growth > 3% ─── weight 3
    max_score += 3
    if rev_growth is not None:
        passed = rev_growth > 0.03
        if passed:
            score += 3
            reasons.append(f"✅ Revenue growing {rev_growth*100:.1f}% (healthy for REIT)")
        else:
            reasons.append(f"❌ Revenue only {rev_growth*100:.1f}% (need >3%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Revenue growth unknown")
        cores_passed.append(False)

    # ─── CORE 2: FCF positive ─── weight 3
    max_score += 3
    if fcf_positive is not None:
        passed = bool(fcf_positive)
        if passed:
            score += 3
            reasons.append("✅ FCF positive (real cash)")
        else:
            reasons.append("❌ FCF negative")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ FCF unknown")
        cores_passed.append(False)

    # ─── CORE 3: Profit margin > 10% ─── weight 3
    max_score += 3
    if profit_margin is not None:
        passed = profit_margin > 0.10
        if passed:
            score += 3
            reasons.append(f"✅ Profit margin {profit_margin*100:.1f}% (>10%)")
        else:
            reasons.append(f"❌ Profit margin {profit_margin*100:.1f}% (need >10%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Profit margin unknown")
        cores_passed.append(False)

    # ─── CORE 4: Current ratio > 0.5 ─── weight 3
    # REITs run very thin liquidity by design (they distribute almost all cash)
    max_score += 3
    if current_ratio is not None:
        passed = current_ratio > 0.5
        if passed:
            score += 3
            reasons.append(f"✅ Current ratio {current_ratio:.2f} (adequate)")
        else:
            reasons.append(f"❌ Current ratio {current_ratio:.2f} (tight even for REIT)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Current ratio unknown")
        cores_passed.append(False)

    # ─── CORE 5: Institutional ownership > 50% ─── weight 3
    max_score += 3
    if inst_own is not None:
        passed = inst_own > 0.50
        if passed:
            score += 3
            reasons.append(f"✅ Inst ownership {inst_own*100:.1f}% (institutionally trusted)")
        else:
            reasons.append(f"❌ Inst ownership only {inst_own*100:.1f}% (need >50%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Institutional ownership unknown")
        cores_passed.append(False)

    # Informational note
    reasons.append("ℹ️ Note: REIT P/E is misleading; FFO is the truer metric (not in our data yet)")

    all_cores_passed = all(cores_passed)
    return score, max_score, all_cores_passed, reasons
