"""
Biotech scoring profile.

For drug discovery & early-stage life sciences.
Examples: MRNA, REGN, VRTX, BIIB, GILD, INCY, BMRN, SGEN, NVAX, BNTX.

What's DIFFERENT about biotech:
  - P/E and earnings growth are often NULL or wildly volatile.
    Many biotechs lose money for years before a drug pays off.
  - Cash runway is THE survival metric: total_cash / annual_burn.
    We use total_cash >> total_debt as a runway proxy.
  - Revenue can be lumpy (depends on drug approvals, royalties).
  - Pipeline value, FDA approvals, Phase 3 trials are ideally tracked too
    — but we don't have that data yet (Stage 6).

What we MEASURE:
  - Cash >> Debt (runway proxy — strong if cash > 1.5x debt)
  - Current ratio > 1.5 (short-term solvency)
  - Revenue exists (any revenue at all)
  - Gross margin > 60% (drugs are high-margin once approved)
  - Institutional ownership > 40% (some smart money in)

We INTENTIONALLY don't require:
  - P/E (often null pre-profitability)
  - Earnings growth (often null or extreme)
  - FCF positive (many biotechs burn cash investing in R&D)
  - Debt/equity (irrelevant for early-stage)
"""


def score(row):
    score = 0.0
    max_score = 0.0
    reasons = []
    cores_passed = []

    total_cash = row.get("total_cash")
    total_debt = row.get("total_debt")
    current_ratio = row.get("current_ratio")
    total_revenue = row.get("total_revenue")
    gross_margin = row.get("gross_margin")
    inst_own = row.get("institutional_ownership")

    # ─── CORE 1: Cash > 1.5x debt (runway proxy) ─── weight 3
    max_score += 3
    if total_cash is not None and total_debt is not None:
        if total_debt == 0:
            passed = total_cash > 0
            if passed:
                score += 3
                reasons.append(f"✅ Cash ${total_cash/1e9:.1f}B, zero debt (excellent)")
            else:
                reasons.append("❌ No cash, no debt — distressed?")
        else:
            ratio = total_cash / total_debt
            passed = ratio > 1.5
            if passed:
                score += 3
                reasons.append(f"✅ Cash/debt ratio {ratio:.1f}x (strong runway)")
            else:
                reasons.append(f"❌ Cash/debt only {ratio:.1f}x (need >1.5x for biotech)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Cash or debt unknown")
        cores_passed.append(False)

    # ─── CORE 2: Current ratio > 1.5 ─── weight 3
    max_score += 3
    if current_ratio is not None:
        passed = current_ratio > 1.5
        if passed:
            score += 3
            reasons.append(f"✅ Current ratio {current_ratio:.2f} (>1.5)")
        else:
            reasons.append(f"❌ Current ratio {current_ratio:.2f} (tight)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Current ratio unknown")
        cores_passed.append(False)

    # ─── CORE 3: Has revenue ─── weight 3
    max_score += 3
    if total_revenue is not None:
        passed = total_revenue > 0
        if passed:
            score += 3
            reasons.append(f"✅ Revenue ${total_revenue/1e9:.1f}B (commercial stage)")
        else:
            reasons.append("❌ No revenue (pre-commercial)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Pre-revenue or unknown")
        cores_passed.append(False)

    # ─── CORE 4: Gross margin > 60% ─── weight 3
    max_score += 3
    if gross_margin is not None:
        passed = gross_margin > 0.60
        if passed:
            score += 3
            reasons.append(f"✅ Gross margin {gross_margin*100:.1f}% (>60%, drug economics)")
        else:
            reasons.append(f"❌ Gross margin {gross_margin*100:.1f}% (need >60%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Gross margin unknown")
        cores_passed.append(False)

    # ─── CORE 5: Institutional ownership > 40% ─── weight 3
    max_score += 3
    if inst_own is not None:
        passed = inst_own > 0.40
        if passed:
            score += 3
            reasons.append(f"✅ Inst ownership {inst_own*100:.1f}% (smart money in)")
        else:
            reasons.append(f"❌ Inst ownership {inst_own*100:.1f}% (need >40%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Institutional ownership unknown")
        cores_passed.append(False)

    # Informational note
    reasons.append("ℹ️ Biotech: pipeline & FDA catalysts matter most (not yet in our data)")

    all_cores_passed = all(cores_passed)
    return score, max_score, all_cores_passed, reasons
