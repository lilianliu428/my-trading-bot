from scoring.sectors.benchmarks import get_sector_benchmark

def check_revenue_growth(info):
    rev = info.get("revenueGrowth", None)
    if rev is None:
        return 0, None
    if rev > 0.05:
        return 1, f"✅ Revenue growing {rev*100:.1f}%"
    return 0, f"❌ Revenue growth weak {rev*100:.1f}%"

def check_free_cash_flow(info):
    fcf = info.get("freeCashflow", None)
    if fcf is None:
        return 0, None
    if fcf > 0:
        return 1, f"✅ Positive free cash flow"
    return 0, f"❌ Negative free cash flow"


def check_pe_ratio(info):
    pe = info.get("trailingPE", None)
    sector = info.get("sector", None)
    if pe is None or pe <= 0:
        return 0, None

    benchmark = get_sector_benchmark(sector, "pe_median") if sector else None

    if benchmark is not None:
        if pe <= benchmark * 1.2:  # within 20% of sector median or lower
            return 1, f"✅ P/E {pe:.1f} reasonable vs {sector} median {benchmark:.1f}"
        return 0, f"❌ P/E {pe:.1f} above {sector} median {benchmark:.1f}"

    if 0 < pe < 40:
        return 1, f"✅ P/E reasonable at {pe:.1f}"
    return 0, f"❌ P/E too high at {pe:.1f}"

def check_earnings_growth(info):
    eg = info.get("earningsGrowth", None)
    if eg is None:
        return 0, None
    if eg > 0:
        return 1, f"✅ Earnings growing {eg*100:.1f}%"
    return 0, f"❌ Earnings shrinking {eg*100:.1f}%"

def check_debt_equity(info):
    de = info.get("debtToEquity", None)
    if de is None:
        return 0, None
    if de < 150:
        return 1, f"✅ Debt/equity manageable at {de:.0f}"
    return 0, f"❌ High debt/equity at {de:.0f}"