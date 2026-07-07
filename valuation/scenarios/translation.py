"""
valuation/scenarios/translation.py

Translates business-level scenario drivers (market share, target year, target
margin) into DCF inputs (initial growth, runway, terminal growth).

This is where the scenario's narrative meets the math. The translation is
deliberately mechanical — given assumptions about where the company will be
in N years, what growth path gets us there?

Key insight: if we know
    current_revenue (from data)
    target_revenue = TAM(target_year) × target_share (from scenario)
then the implied CAGR is solvable algebraically.

From there:
    initial_growth = min(implied_cagr, bucket_cap)
    high_growth_years = years_until_target
    terminal_growth = cycle's terminal growth assumption
"""

import math
from typing import Optional


def derive_dcf_inputs(
    current_revenue: float,
    current_year: int,
    target_year: int,
    target_revenue: float,
    target_margin: float,
    terminal_growth: float,
    initial_growth_cap: float = 0.40,
) -> dict:
    """
    Convert scenario business drivers into DCF inputs.

    Args:
        current_revenue: TTM revenue (USD)
        current_year: current calendar year
        target_year: scenario's target year
        target_revenue: implied revenue at target_year (TAM × share)
        target_margin: operating margin at target_year (decimal)
        terminal_growth: long-run growth rate after high-growth period (decimal)
        initial_growth_cap: hard cap on initial growth — even bull scenarios
                            shouldn't project >40% Year 1

    Returns:
        dict with derived_initial_growth, derived_high_growth_years,
        derived_terminal_growth, implied_cagr, data_flags
    """
    data_flags = []

    years_until_target = target_year - current_year

    if years_until_target <= 0:
        data_flags.append(f"Target year {target_year} not in future — using 5-year horizon")
        years_until_target = 5

    if current_revenue <= 0:
        raise ValueError(f"Current revenue must be positive (got {current_revenue})")
    if target_revenue <= 0:
        raise ValueError(f"Target revenue must be positive (got {target_revenue})")

    # Compounded annual growth rate that gets us from current to target
    implied_cagr = (target_revenue / current_revenue) ** (1 / years_until_target) - 1

    # Cap the initial growth at the cap (even bull scenarios shouldn't project >40%)
    derived_initial_growth = min(implied_cagr, initial_growth_cap)

    if implied_cagr > initial_growth_cap:
        data_flags.append(
            f"Implied CAGR {implied_cagr*100:.1f}% capped at {initial_growth_cap*100:.0f}%. "
            f"Scenario target requires growth faster than bucket cap — runway extended to compensate."
        )
        # When growth is capped, the target won't be reached in years_until_target.
        # Extend runway so cumulative growth gets there.
        # If we cap growth at 40% and target requires 50%, we need more years.
        # capped_revenue_at_target_year = current_revenue * (1 + cap) ** target_year
        # We need capped_revenue >= target_revenue
        # So: years = ln(target/current) / ln(1+cap)
        extended_years = math.ceil(
            math.log(target_revenue / current_revenue) / math.log(1 + initial_growth_cap)
        )
        derived_high_growth_years = extended_years
    else:
        derived_high_growth_years = years_until_target

    # Hard floor and ceiling on runway — even capped, shouldn't be extreme
    derived_high_growth_years = max(3, min(derived_high_growth_years, 15))

    return {
        "derived_initial_growth": derived_initial_growth,
        "derived_high_growth_years": derived_high_growth_years,
        "derived_terminal_growth": terminal_growth,
        "implied_cagr": implied_cagr,
        "years_until_target": years_until_target,
        "data_flags": data_flags,
    }