"""
Mature tech scoring profile.

For profitable, established tech companies past hyper-growth phase.
Examples: AAPL, MSFT, ORCL, ADBE, CSCO, IBM, INTC, TXN, CRM (debatable).

What we MEASURE differently from SaaS:
  - Lower growth bar (5% not 20%) — these are cash cows
  - Profitability/quality bar HIGHER (ROE, margins, FCF must be real)
  - P/E matters again (they have actual earnings)
  - Debt manageable (some leverage OK for buybacks)
"""


def score(row):
    """
    Score a mature tech company.
    `row` is a dict from the fundamentals table.
    Returns (score, max_score, all_cores_passed, reasons)
    """
    score = 0.0
    max_score = 0.0
    reasons = []
    cores_passed = []

    rev_growth = row.get("revenue_growth")
    pe = row.get("pe")
    roe = row.get("roe")
    op_margin = row.get("op_margin")
    fcf_positive = row.get("free_cash_flow_positive")
    debt_eq = row.get("debt_equity")
    sector = row.get("sector")

    # Use sector benchmark if available
    pe_median = None
    try:
        import sqlite3
        from data_pipeline.database import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        b = conn.execute("SELECT pe_median FROM sector_benchmarks WHERE sector=?", (sector,)).fetchone()
        conn.close()
        if b: pe_median = b[0]
    except:
        pass

    # ─── CORE 1: Revenue growth > 5% ─── weight 3
    max_score += 3
    if rev_growth is not None:
        passed = rev_growth > 0.05
        if passed:
            score += 3
            reasons.append(f"✅ Revenue growing {rev_growth*100:.1f}% (>5%)")
        else:
            reasons.append(f"❌ Revenue only {rev_growth*100:.1f}% (need >5%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Revenue growth unknown")
        cores_passed.append(False)

    # ─── CORE 2: P/E reasonable ─── weight 3
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
            passed = pe < 35  # absolute cap if no benchmark
            if passed:
                score += 3
                reasons.append(f"✅ P/E {pe:.1f} (<35)")
            else:
                reasons.append(f"❌ P/E {pe:.1f} (>=35)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ P/E unknown")
        cores_passed.append(False)

    # ─── CORE 3: ROE > 15% ─── weight 3
    max_score += 3
    if roe is not None:
        passed = roe > 0.15
        if passed:
            score += 3
            reasons.append(f"✅ ROE {roe*100:.1f}% (>15%)")
        else:
            reasons.append(f"❌ ROE {roe*100:.1f}% (need >15%)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ ROE unknown")
        cores_passed.append(False)

    # ─── CORE 4: Real cash machine (FCF+ AND op margin > 20%) ─── weight 3
    max_score += 3
    if fcf_positive is not None and op_margin is not None:
        passed = bool(fcf_positive) and op_margin > 0.20
        if passed:
            score += 3
            reasons.append(f"✅ Cash machine: FCF+, op margin {op_margin*100:.1f}%")
        else:
            details = []
            if not fcf_positive: details.append("FCF negative")
            if op_margin <= 0.20: details.append(f"op margin only {op_margin*100:.1f}%")
            reasons.append(f"❌ Not a cash machine ({', '.join(details)})")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Cash/margin data unknown")
        cores_passed.append(False)

    # ─── CORE 5: Debt manageable ─── weight 3
    max_score += 3
    if debt_eq is not None:
        passed = debt_eq < 200
        if passed:
            score += 3
            reasons.append(f"✅ D/E {debt_eq:.0f} (<200, manageable)")
        else:
            reasons.append(f"❌ D/E {debt_eq:.0f} (>=200, high)")
        cores_passed.append(passed)
    else:
        reasons.append("⚠️ Debt/equity unknown")
        cores_passed.append(False)

    all_cores_passed = all(cores_passed)
    return score, max_score, all_cores_passed, reasons
