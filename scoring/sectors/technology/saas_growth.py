"""
SaaS / Growth-stage tech scoring profile.

For unprofitable or barely-profitable software companies with high growth.
Examples: CRWD, ZS, NET, DDOG, SNOW, MDB, OKTA, GTLB, PLTR.

What we IGNORE (intentionally):
  - P/E ratio (often None or absurd for SaaS)
  - Earnings growth (often None)
  - Debt/equity (SaaS rarely has meaningful debt)

What we MEASURE:
  - Revenue growth (the #1 SaaS metric)
  - Gross margin (SaaS should be 65%+)
  - Rule of 40 (growth + FCF margin >= 40%)
  - FCF positive (cash generation, not just bookings)
  - Cash runway (current ratio as proxy)
"""


def _safe_pct(x):
    """yfinance often returns 0.23 for 23%. Just multiply by 100 in display."""
    return x * 100 if x is not None else None


def score(row):
    """
    Score a SaaS growth company.
    `row` is a dict from the fundamentals table.
    Returns (score, max_score, all_cores_passed, reasons)
    """
    score = 0.0
    max_score = 0.0
    reasons = []
    cores_passed = []  # track each core's pass/fail

    rev_growth = row.get("revenue_growth")
    gross_margin = row.get("gross_margin")
    op_margin = row.get("op_margin")
    fcf = row.get("free_cash_flow")
    total_revenue = row.get("total_revenue")
    current_ratio = row.get("current_ratio")
    fcf_positive = row.get("free_cash_flow_positive")

    # ─── CORE 1: Revenue growth > 20% ─── weight 3
    max_score += 3
    if rev_growth is not None:
        passed = rev_growth > 0.20
        if passed:
            score += 3
            reasons.append(f"✅ Revenue growing {rev_growth*100:.1f}% (>20%)")
        else:
            reasons.append(f"❌ Revenue only {rev_growth*100:.1f}% (need >20%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Revenue growth unknown")
        cores_passed.append(False)

    # ─── CORE 2: Gross margin > 65% ─── weight 3
    max_score += 3
    if gross_margin is not None:
        passed = gross_margin > 0.65
        if passed:
            score += 3
            reasons.append(f"✅ Gross margin {gross_margin*100:.1f}% (>65%)")
        else:
            reasons.append(f"❌ Gross margin {gross_margin*100:.1f}% (need >65%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Gross margin unknown")
        cores_passed.append(False)

    # ─── CORE 3: Rule of 40 ─── weight 3
    # Rule of 40 = revenue growth + FCF margin >= 40
    # FCF margin = FCF / Revenue. Fallback: operating margin if FCF/revenue missing.
    max_score += 3
    if rev_growth is not None and total_revenue and fcf is not None:
        fcf_margin = fcf / total_revenue
        rule40 = (rev_growth + fcf_margin) * 100
        passed = rule40 >= 40
        if passed:
            score += 3
            reasons.append(f"✅ Rule of 40: {rule40:.0f} (growth+FCF margin)")
        else:
            reasons.append(f"❌ Rule of 40: {rule40:.0f} (need >=40)")
        cores_passed.append(passed)
    elif rev_growth is not None and op_margin is not None:
        # fallback when FCF/revenue is missing
        rule40 = (rev_growth + op_margin) * 100
        passed = rule40 >= 40
        if passed:
            score += 3
            reasons.append(f"✅ Rule of 40 (op margin proxy): {rule40:.0f}")
        else:
            reasons.append(f"❌ Rule of 40 (op margin proxy): {rule40:.0f}")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Rule of 40 — missing data")
        cores_passed.append(False)

    # ─── CORE 4: Positive FCF ─── weight 3
    max_score += 3
    if fcf_positive is not None:
        passed = bool(fcf_positive)
        if passed:
            score += 3
            reasons.append("✅ FCF positive")
        else:
            reasons.append("❌ FCF negative (burning cash)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ FCF unknown")
        cores_passed.append(False)

    # ─── CORE 5: Cash runway (current ratio > 1.5) ─── weight 3
    max_score += 3
    if current_ratio is not None:
        passed = current_ratio > 1.5
        if passed:
            score += 3
            reasons.append(f"✅ Current ratio {current_ratio:.2f} (good runway)")
        else:
            reasons.append(f"❌ Current ratio {current_ratio:.2f} (tight)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Current ratio unknown")
        cores_passed.append(False)

    all_cores_passed = all(cores_passed)
    return score, max_score, all_cores_passed, reasons
