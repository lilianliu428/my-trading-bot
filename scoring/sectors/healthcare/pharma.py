"""
Pharma scoring profile.

For mature drug manufacturers with revenue and earnings.
Examples: JNJ, PFE, LLY, MRK, BMY, ABBV, NVS, AZN.

What's DIFFERENT about pharma vs biotech:
  - These are profitable, established companies with multiple approved drugs.
  - P/E matters and is usually 15-30 range.
  - High gross margins (60-80%) typical.
  - Cash machines — should be FCF positive with strong margins.
  - Patent cliffs are a real risk (informational note included).

What we MEASURE:
  - Revenue growth >= 0% (slow growth OK; declines are red flags)
  - P/E reasonable (< 35)
  - FCF positive
  - Profit margin > 15%
  - Gross margin > 60% (drug economics)
"""


def score(row):
    score = 0.0
    max_score = 0.0
    reasons = []
    cores_passed = []

    rev_growth = row.get("revenue_growth")
    pe = row.get("pe")
    fcf_positive = row.get("free_cash_flow_positive")
    profit_margin = row.get("profit_margin")
    gross_margin = row.get("gross_margin")

    # ─── CORE 1: Revenue not declining ─── weight 3
    max_score += 3
    if rev_growth is not None:
        passed = rev_growth >= 0
        if passed:
            score += 3
            if rev_growth > 0.05:
                reasons.append(f"✅ Revenue growing {rev_growth*100:.1f}% (healthy)")
            else:
                reasons.append(f"✅ Revenue {rev_growth*100:.1f}% (slow but stable)")
        else:
            reasons.append(f"❌ Revenue declining {rev_growth*100:.1f}% (patent cliff?)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Revenue growth unknown")
        cores_passed.append(False)

    # ─── CORE 2: P/E reasonable ─── weight 3
    max_score += 3
    if pe is not None and pe > 0:
        passed = pe < 35
        if passed:
            score += 3
            reasons.append(f"✅ P/E {pe:.1f} (<35)")
        else:
            reasons.append(f"❌ P/E {pe:.1f} (>=35, rich)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ P/E unknown")
        cores_passed.append(False)

    # ─── CORE 3: FCF positive ─── weight 3
    max_score += 3
    if fcf_positive is not None:
        passed = bool(fcf_positive)
        if passed:
            score += 3
            reasons.append("✅ FCF positive")
        else:
            reasons.append("❌ FCF negative")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ FCF unknown")
        cores_passed.append(False)

    # ─── CORE 4: Profit margin > 15% ─── weight 3
    max_score += 3
    if profit_margin is not None:
        passed = profit_margin > 0.15
        if passed:
            score += 3
            reasons.append(f"✅ Profit margin {profit_margin*100:.1f}% (>15%)")
        else:
            reasons.append(f"❌ Profit margin {profit_margin*100:.1f}% (need >15%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Profit margin unknown")
        cores_passed.append(False)

    # ─── CORE 5: Gross margin > 60% ─── weight 3
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

    # Informational note
    reasons.append("ℹ️ Pharma: watch patent cliffs and pipeline replenishment")

    all_cores_passed = all(cores_passed)
    return score, max_score, all_cores_passed, reasons
