import yfinance as yf
from scoring.metrics.financial_health import (
    check_revenue_growth, check_free_cash_flow, check_pe_ratio,
    check_earnings_growth, check_debt_equity
)
from scoring.metrics.profitability import check_operating_margin, check_roe
from scoring.metrics.ownership import check_institutional_ownership, check_insider_ownership
from scoring.technical.indicators import (
    check_above_200ma, check_golden_cross, check_distance_from_200ma
)

CORE_CHECKS = [
    (check_revenue_growth, 2),
    (check_free_cash_flow, 2),
    (check_pe_ratio, 2),
    (check_earnings_growth, 2),
    (check_debt_equity, 2),
]

ADDON_CHECKS = [
    (check_operating_margin, 1.5),
    (check_roe, 1.5),
    (check_institutional_ownership, 1),
    (check_insider_ownership, 1),
]

TECHNICAL_CHECKS = [
    (check_above_200ma, 1.5),
    (check_golden_cross, 1.5),
    (check_distance_from_200ma, 1),
]


def check_fundamentals(ticker, hist=None):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if hist is None:
            hist = stock.history(period="1y")  # need 200+ days

        score = 0
        max_score = 0
        core_passed = 0
        reasons = []

        for check, weight in CORE_CHECKS:
            points, reason = check(info)
            score += points * weight
            max_score += weight
            core_passed += points
            if reason:
                reasons.append(reason)

        for check, weight in ADDON_CHECKS:
            points, reason = check(info)
            score += points * weight
            max_score += weight
            if reason:
                reasons.append(reason)

        for check, weight in TECHNICAL_CHECKS:
            points, reason = check(info, hist)
            score += points * weight
            max_score += weight
            if reason:
                reasons.append(reason)

        return score, max_score, core_passed, reasons

    except Exception as e:
        return 0, 0, 0, ["⚠️ Could not fetch fundamentals"]