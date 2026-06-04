"""
Default scoring profile.

The catch-all for tickers that don't fit a specialized bucket:
  - Mid-cap industrials with weird metrics
  - Companies with sparse data
  - ETFs
  - Sectors we haven't written yet (energy, materials, utilities, etc)

This is roughly equivalent to the original cross-sector scorer,
but cleaner and uses the new DB structure.

We measure standard fundamentals:
  - Revenue growth > 5%
  - FCF positive
  - P/E reasonable (sector benchmark or absolute)
  - Earnings growing
  - Debt/equity manageable
"""

import sqlite3
from data_pipeline.database import DB_PATH


def _get_pe_median(sector):
    """Pull sector P/E median from benchmarks table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        b = conn.execute(
            "SELECT pe_median FROM sector_benchmarks WHERE sector=?", (sector,)
        ).fetchone()
        conn.close()
        return b[0] if b else None
    except:
        return None


def score(row):
    """
    Generic fundamental scoring.
    `row` is a dict from the fundamentals table.
    Returns (score, max_score, all_cores_passed, reasons)
    """
    score = 0.0
    max_score = 0.0
    reasons = []
    cores_passed = []

    rev_growth = row.get("revenue_growth")
    fcf_positive = row.get("free_cash_flow_positive")
    pe = row.get("pe")
    earn_growth = row.get("earnings_growth")
    debt_eq = row.get("debt_equity")
    sector = row.get("sector")
    pe_median = _get_pe_median(sector) if sector else None

    # ─── CORE 1: Revenue growth > 5% ─── weight 3
    max_score += 3
    if rev_growth is not None:
        passed = rev_growth > 0.05
        if passed:
            score += 3
            reasons.append(f"✅ Revenue growing {rev_growth*100:.1f}%")
        else:
            reasons.append(f"❌ Revenue weak {rev_growth*100:.1f}%")
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
            reasons.append("✅ FCF positive")
        else:
            reasons.append("❌ FCF negative")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ FCF unknown")
        cores_passed.append(False)

    # ─── CORE 3: P/E reasonable ─── weight 3
    max_score += 3
    if pe is not None and pe > 0:
        if pe_median:
            passed = pe <= pe_median * 1.2
            if passed:
                score += 3
                reasons.append(f"✅ P/E {pe:.1f} vs {sector} median {pe_median:.1f}")
            else:
                reasons.append(f"❌ P/E {pe:.1f} above {sector} median {pe_median:.1f}")
        else:
            passed = pe < 30
            if passed:
                score += 3
                reasons.append(f"✅ P/E {pe:.1f} (<30)")
            else:
                reasons.append(f"❌ P/E {pe:.1f} (>=30)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ P/E unknown")
        cores_passed.append(False)

    # ─── CORE 4: Earnings growing ─── weight 3
    max_score += 3
    if earn_growth is not None:
        passed = earn_growth > 0
        if passed:
            score += 3
            reasons.append(f"✅ Earnings growing {earn_growth*100:.1f}%")
        else:
            reasons.append(f"❌ Earnings shrinking {earn_growth*100:.1f}%")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Earnings growth unknown")
        cores_passed.append(False)

    # ─── CORE 5: Debt manageable ─── weight 3
    max_score += 3
    if debt_eq is not None:
        passed = debt_eq < 150
        if passed:
            score += 3
            reasons.append(f"✅ D/E {debt_eq:.0f} (<150)")
        else:
            reasons.append(f"❌ D/E {debt_eq:.0f} (high)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Debt/equity unknown")
        cores_passed.append(False)

    all_cores_passed = all(cores_passed)
    return score, max_score, all_cores_passed, reasons
