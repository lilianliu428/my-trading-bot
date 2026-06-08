"""
Macro inputs for DCF: risk-free rate and equity risk premium.

These are economy-wide numbers — they apply to every stock we value,
not company-specific.
"""

from valuation.data_sources.fred import get_risk_free_rate


# Damodaran publishes the implied equity risk premium monthly at
# https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html
#
# Updated manually for now. We'll add automatic scraping in a later phase.
# Last updated: 2026-06-05 (placeholder — check Damodaran's site for current)
IMPLIED_ERP = 0.0460   # 4.60%
ERP_LAST_UPDATED = "2026-06-05"


def get_macro_inputs():
    """
    Get current macro inputs: risk-free rate (live) and ERP (hardcoded).

    Returns:
        dict with:
            risk_free_rate (float, decimal): from FRED
            equity_risk_premium (float, decimal): from Damodaran (manual)
            erp_last_updated (str): when ERP was last refreshed manually
    """
    rf = get_risk_free_rate()
    return {
        "risk_free_rate": rf,
        "equity_risk_premium": IMPLIED_ERP,
        "erp_last_updated": ERP_LAST_UPDATED,
    }


if __name__ == "__main__":
    macro = get_macro_inputs()
    print(f"Risk-free rate:       {macro['risk_free_rate']*100:.2f}%")
    print(f"Equity risk premium:  {macro['equity_risk_premium']*100:.2f}%")
    print(f"ERP last updated:     {macro['erp_last_updated']}")
    print()
    print(f"Implied 'market return' = rf + ERP = "
          f"{(macro['risk_free_rate'] + macro['equity_risk_premium'])*100:.2f}%")