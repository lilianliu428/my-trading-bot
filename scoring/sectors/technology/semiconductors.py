"""
Semiconductor scoring profile.

For chip designers, foundries, and equipment makers.
Examples: NVDA, AMD, AVGO, TSM, INTC, TXN, AMAT, LRCX, KLAC, ASML, ADI, MU, MRVL, NXPI.

What's DIFFERENT about semis:
  - Cyclical industry — boom/bust tied to chip demand
  - Gross margin is the cycle indicator (40-60% is normal; 65%+ suggests peak)
  - Capex is huge — FCF AFTER capex matters more than reported earnings
  - P/E can mislead near cycle extremes
  - We add a SOXX position context to flag where in the cycle we likely are

What we MEASURE:
  - Revenue growth > 10% (semis grow faster than mature tech at peak, slower at trough)
  - Gross margin > 40% (industry baseline; bonus context if much higher)
  - ROE > 12%
  - FCF positive
  - Debt manageable
"""

import sqlite3
from data_pipeline.database import DB_PATH


def _get_soxx_context():
    """
    Look at SOXX position vs its own history to infer chip cycle phase.
    Returns (label, explanation) — informational only, doesn't affect score.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        # Get SOXX's latest price and 200MA
        row = conn.execute("""
            SELECT price, ma_200 FROM snapshots
            WHERE ticker='SOXX' ORDER BY date DESC LIMIT 1
        """).fetchone()
        conn.close()

        if not row or not row[1]:
            return None, None

        price, ma_200 = row
        dist_pct = ((price - ma_200) / ma_200) * 100

        if dist_pct > 20:
            return ("🔥 SOXX HIGH",
                    f"SOXX is {dist_pct:+.0f}% above 200MA — likely late/peak cycle. "
                    f"Chip companies' great numbers may not last; be cautious.")
        elif dist_pct > 5:
            return ("📈 SOXX elevated",
                    f"SOXX is {dist_pct:+.0f}% above 200MA — mid/late cycle. Strong demand but watch for slowdown.")
        elif dist_pct > -10:
            return ("⚖️ SOXX neutral",
                    f"SOXX is {dist_pct:+.0f}% vs 200MA — mid-cycle, no extreme.")
        elif dist_pct > -20:
            return ("📉 SOXX weak",
                    f"SOXX is {dist_pct:+.0f}% below 200MA — possible early cycle / accumulation zone.")
        else:
            return ("💎 SOXX LOW",
                    f"SOXX is {dist_pct:+.0f}% below 200MA — likely trough. Quality semis may be on sale.")
    except Exception as e:
        return None, None


def score(row):
    """
    Score a semiconductor company.
    `row` is a dict from the fundamentals table.
    Returns (score, max_score, all_cores_passed, reasons)
    """
    score = 0.0
    max_score = 0.0
    reasons = []
    cores_passed = []

    rev_growth = row.get("revenue_growth")
    gross_margin = row.get("gross_margin")
    roe = row.get("roe")
    op_margin = row.get("op_margin")
    fcf_positive = row.get("free_cash_flow_positive")
    debt_eq = row.get("debt_equity")

    # ─── CORE 1: Revenue growth > 10% ─── weight 3
    max_score += 3
    if rev_growth is not None:
        passed = rev_growth > 0.10
        if passed:
            score += 3
            reasons.append(f"✅ Revenue growing {rev_growth*100:.1f}% (>10%)")
        else:
            reasons.append(f"❌ Revenue only {rev_growth*100:.1f}% (need >10%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Revenue growth unknown")
        cores_passed.append(False)

    # ─── CORE 2: Gross margin > 40% ─── weight 3
    max_score += 3
    if gross_margin is not None:
        passed = gross_margin > 0.40
        if passed:
            score += 3
            if gross_margin > 0.65:
                reasons.append(f"✅ Gross margin {gross_margin*100:.1f}% (strong; but watch — may be cycle peak)")
            else:
                reasons.append(f"✅ Gross margin {gross_margin*100:.1f}% (>40%)")
        else:
            reasons.append(f"❌ Gross margin {gross_margin*100:.1f}% (need >40%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Gross margin unknown")
        cores_passed.append(False)

    # ─── CORE 3: ROE > 12% ─── weight 3
    max_score += 3
    if roe is not None:
        passed = roe > 0.12
        if passed:
            score += 3
            reasons.append(f"✅ ROE {roe*100:.1f}% (>12%)")
        else:
            reasons.append(f"❌ ROE {roe*100:.1f}% (need >12%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ ROE unknown")
        cores_passed.append(False)

    # ─── CORE 4: FCF positive + decent op margin ─── weight 3
    max_score += 3
    if fcf_positive is not None and op_margin is not None:
        passed = bool(fcf_positive) and op_margin > 0.15
        if passed:
            score += 3
            reasons.append(f"✅ Real earnings: FCF+, op margin {op_margin*100:.1f}%")
        else:
            details = []
            if not fcf_positive: details.append("FCF negative")
            if op_margin <= 0.15: details.append(f"op margin only {op_margin*100:.1f}%")
            reasons.append(f"❌ Weak earnings ({', '.join(details)})")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Earnings data unknown")
        cores_passed.append(False)

    # ─── CORE 5: Debt manageable ─── weight 3
    max_score += 3
    if debt_eq is not None:
        passed = debt_eq < 150
        if passed:
            score += 3
            reasons.append(f"✅ D/E {debt_eq:.0f} (<150)")
        else:
            reasons.append(f"❌ D/E {debt_eq:.0f} (>=150, high for cyclicals)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Debt/equity unknown")
        cores_passed.append(False)

    # ─── CYCLE CONTEXT (informational, doesn't affect score) ───
    cycle_label, cycle_msg = _get_soxx_context()
    if cycle_label:
        reasons.append(f"{cycle_label}: {cycle_msg}")

    all_cores_passed = all(cores_passed)
    return score, max_score, all_cores_passed, reasons
