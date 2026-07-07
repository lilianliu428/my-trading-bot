"""
valuation/scenarios/orchestrator.py

Runs scenario-weighted valuation for a ticker.

Flow:
1. Look up scenarios for the ticker from the industry config
2. For each scenario:
   a. Get current revenue from yfinance
   b. Compute target revenue = TAM(target_year) × target_share
   c. Translate to DCF inputs (initial_growth, runway, terminal_growth)
   d. Run the DCF with those inputs (using existing engine)
3. Combine results into probability-weighted ScenarioValuationOutput
"""

import yfinance as yf
from datetime import datetime

from valuation.scenarios.models import (
    ScenarioResult,
    ScenarioValuationOutput,
)
from valuation.scenarios.translation import derive_dcf_inputs
from valuation.scenarios.industries import ai_accelerators
from valuation.tech.mature.mature_dcf import compute_tech_intrinsic_value


# Map ticker → which industry's scenarios apply
TICKER_INDUSTRY = {
    "AMD":  "ai_accelerators",
    "NVDA": "ai_accelerators",
    "AVGO": "ai_accelerators",
}


def compute_scenario_valuation(ticker: str) -> ScenarioValuationOutput:
    """
    Run scenario-weighted DCF for a single ticker.

    Args:
        ticker: stock ticker (case-insensitive)

    Returns:
        ScenarioValuationOutput with per-scenario results and weighted aggregates
    """
    ticker = ticker.upper()
    data_flags = []

    if ticker not in TICKER_INDUSTRY:
        raise KeyError(
            f"Ticker {ticker} not configured for scenario valuation. "
            f"Supported: {list(TICKER_INDUSTRY.keys())}"
        )

    industry = TICKER_INDUSTRY[ticker]
    if industry == "ai_accelerators":
        scenarios = ai_accelerators.get_scenarios(ticker)
        get_tam = ai_accelerators.get_tam
    else:
        raise NotImplementedError(f"Industry {industry} not implemented yet")

    # ─── Get current revenue and price from yfinance ──────────────────
    yf_ticker = yf.Ticker(ticker)
    income_stmt = yf_ticker.income_stmt
    if income_stmt is None or income_stmt.empty:
        raise ValueError(f"No income statement data for {ticker}")

    # Get TTM revenue (latest fiscal year)
    revenue_row = None
    for field in ["Total Revenue", "Revenue", "Operating Revenue"]:
        if field in income_stmt.index:
            revenue_row = income_stmt.loc[field]
            break
    if revenue_row is None:
        raise ValueError(f"No revenue row found for {ticker}")
    current_revenue = float(revenue_row.iloc[0])

    # Current market price
    current_price = float(yf_ticker.info.get("currentPrice") or yf_ticker.info.get("regularMarketPrice"))

    current_year = datetime.now().year

    # ─── Run DCF for each scenario ────────────────────────────────────
    results = []
    for scenario in scenarios:
        # Derive target revenue from TAM and share
        target_year = scenario.drivers.target_year
        tam_at_target = get_tam(target_year)
        target_revenue = tam_at_target * scenario.drivers.target_share

        # Derive DCF inputs from business drivers
        try:
            # Use cycle's terminal growth (AI infrastructure = 3.5%)
            terminal_g = 0.035
            derivation = derive_dcf_inputs(
                current_revenue=current_revenue,
                current_year=current_year,
                target_year=target_year,
                target_revenue=target_revenue,
                target_margin=scenario.drivers.target_margin,
                terminal_growth=terminal_g,
            )
            data_flags.extend([f"[{scenario.name}] {f}" for f in derivation["data_flags"]])
        except Exception as e:
            data_flags.append(f"[{scenario.name}] Driver derivation failed: {e}")
            continue

        # Apply scenario overrides if present
        derived_initial_growth = (
            scenario.drivers.override_initial_growth
            if scenario.drivers.override_initial_growth is not None
            else derivation["derived_initial_growth"]
        )
        derived_high_growth_years = (
            scenario.drivers.override_high_growth_years
            if scenario.drivers.override_high_growth_years is not None
            else derivation["derived_high_growth_years"]
        )
        derived_terminal_growth = (
            scenario.drivers.override_terminal_growth
            if scenario.drivers.override_terminal_growth is not None
            else derivation["derived_terminal_growth"]
        )

        # ─── Run the existing DCF engine with scenario-derived inputs ────
        # NOTE: For v1 we use the standard tech DCF. The DCF engine doesn't
        # natively accept growth/runway overrides as parameters, so for now
        # the scenario inputs are *informational* — the actual valuation
        # uses the engine's standard inputs.
        #
        # TODO: parameterize compute_tech_intrinsic_value to accept overrides.
        # This is a separate refactor in session 2.
        try:
            dcf_result = compute_tech_intrinsic_value(
                ticker,
                bucket="semiconductors",
                initial_growth_override=derived_initial_growth,
                high_growth_years_override=derived_high_growth_years,
                terminal_growth_override=derived_terminal_growth,
            )
        except Exception as e:
            data_flags.append(f"[{scenario.name}] DCF engine failed: {e}")
            continue

        if "error" in dcf_result:
            data_flags.append(f"[{scenario.name}] DCF returned error: {dcf_result['error']}")
            continue

        # Build the scenario result
        scenario_result = ScenarioResult(
            scenario_name=scenario.name,
            probability=scenario.probability,
            narrative=scenario.narrative,
            derived_initial_growth=derived_initial_growth,
            derived_high_growth_years=derived_high_growth_years,
            derived_terminal_growth=derived_terminal_growth,
            fair_value_per_share=dcf_result["per_share_value"],
            upside_downside=dcf_result["upside_downside"],
            raw_dcf_result={
                # Just enough for debugging — full dict is huge
                "wacc": dcf_result["wacc"],
                "high_growth_years": dcf_result["growth_profile"]["high_growth_years"],
                "initial_growth": dcf_result["growth_profile"]["yearly_growth"][0],
            },
        )
        results.append(scenario_result)

    if not results:
        raise RuntimeError(f"All scenarios failed for {ticker} — see data_flags")

    # ─── Aggregate ────────────────────────────────────────────────────
    expected_value = sum(r.fair_value_per_share * r.probability for r in results)
    expected_upside = (expected_value - current_price) / current_price if current_price > 0 else None
    range_low = min(r.fair_value_per_share for r in results)
    range_high = max(r.fair_value_per_share for r in results)
    mode_scenario = max(results, key=lambda r: r.probability).scenario_name

    return ScenarioValuationOutput(
        ticker=ticker,
        industry=industry,
        current_price=current_price,
        scenarios=results,
        expected_value=expected_value,
        expected_upside=expected_upside,
        range_low=range_low,
        range_high=range_high,
        mode_scenario=mode_scenario,
        data_flags=data_flags,
    )


# ════════════════════════════════════════════════════════════════════════
# Test harness
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    output = compute_scenario_valuation("AMD")

    print(f"\n{'=' * 70}")
    print(f"  SCENARIO VALUATION: {output.ticker}")
    print(f"  Industry: {output.industry}")
    print(f"  Current price: ${output.current_price:.2f}")
    print(f"{'=' * 70}\n")

    for r in output.scenarios:
        print(f"  {r.scenario_name} (P={r.probability*100:.0f}%)")
        print(f"    {r.narrative}")
        print(f"    Implied initial growth: {r.derived_initial_growth*100:.1f}%")
        print(f"    High-growth years:      {r.derived_high_growth_years}")
        print(f"    Fair value:             ${r.fair_value_per_share:.2f}")
        if r.upside_downside is not None:
            print(f"    Upside:                 {r.upside_downside*100:+.1f}%")
        print()

    print(f"{'-' * 70}")
    print(f"  EXPECTED VALUE:       ${output.expected_value:.2f}")
    if output.expected_upside is not None:
        print(f"  Expected upside:      {output.expected_upside*100:+.1f}%")
    print(f"  Range:                ${output.range_low:.2f} → ${output.range_high:.2f}")
    print(f"  Mode scenario:        {output.mode_scenario}")
    print(f"  Current price:        ${output.current_price:.2f}")
