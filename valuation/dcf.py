"""
DCF Valuation Engine — Phase 1.1

Naive two-stage DCF: explicit forecast period at high growth, then terminal
value via Gordon Growth.

This is intentionally crude. Assumptions are hard-coded and will be improved
in later phases. The goal of Phase 1.1 is to get the pipeline working
end-to-end so we can see the whole picture before adding sophistication.
"""

from dataclasses import dataclass


@dataclass
class DCFInputs:
    """All the inputs needed for a DCF. Every assumption visible."""
    # Current state of the business
    current_fcf: float  # latest annual free cash flow, in dollars
    total_debt: float  # total debt from balance sheet
    total_cash: float  # cash + equivalents
    shares_outstanding: float  # shares outstanding (we'll need to add this)

    # Growth assumptions
    high_growth_rate: float  # e.g. 0.12 = 12% during explicit forecast
    high_growth_years: int  # e.g. 5
    terminal_growth_rate: float  # e.g. 0.03 = 3% forever after

    # Discount rate
    discount_rate: float  # e.g. 0.091 = 9.1% (cost of equity for now)


@dataclass
class DCFResult:
    """What the DCF outputs."""
    firm_value: float  # total present value of all cash flows
    equity_value: float  # after subtracting net debt
    per_share_value: float  # equity value / shares outstanding
    # For debugging — keep the intermediate numbers
    explicit_cash_flows: list  # cash flows during high-growth period
    pv_explicit: list  # present values of those
    terminal_value: float  # at end of high-growth period
    pv_terminal: float  # discounted back to today


def project_cash_flows(current_fcf, growth_rate, num_years):
    """
    Project free cash flows for the explicit forecast period.

    Returns a list of length num_years where:
        list[0] = year 1 FCF (current FCF grown by growth_rate, once)
        list[1] = year 2 FCF
        ...
        list[num_years - 1] = year num_years FCF

    Example: current_fcf=100, growth_rate=0.10, num_years=3
        Returns [110, 121, 133.1]
    """
    res_list = []
    for year in range(1, num_years + 1):  # note: starts at 1, ends at num_years
        fcf = current_fcf * (1 + growth_rate) ** year
        res_list.append(fcf)
    return res_list



def compute_terminal_value(final_year_fcf, terminal_growth_rate, discount_rate):
    """
    Compute terminal value at the end of the explicit forecast period
    using the Gordon Growth model.

    The terminal value represents all cash flows from year (n+1) to infinity,
    assuming they grow at a constant rate (terminal_growth_rate) forever.

    Formula:
        TV = (CF_next_year) / (discount_rate - terminal_growth_rate)
        where CF_next_year = final_year_fcf × (1 + terminal_growth_rate)

    Note: this is the value AT THE END OF year n, not discounted to today yet.
    You'll discount it back in the main function.

    Note: this formula requires discount_rate > terminal_growth_rate.
    If terminal_growth_rate >= discount_rate, the math gives infinite or
    negative values — economically meaningless.
    """
    CF_next_year = final_year_fcf * (1 + terminal_growth_rate)
    TV = (CF_next_year) / (discount_rate - terminal_growth_rate)
    return TV



def discount_to_present(future_value, discount_rate, years):
    """
    Discount a single future value back to today.

    Formula: PV = FV / (1 + r)^t

    Example: discount_to_present(110, 0.10, 1) = 100.0
    """
    PV = future_value / (1 + discount_rate) ** years
    return PV


def compute_intrinsic_value(inputs: DCFInputs) -> DCFResult:
    """
    The main function. Takes DCFInputs, returns DCFResult.

    Steps:
        1. Project cash flows for the high-growth period
        2. Compute terminal value at end of high-growth period
        3. Discount all cash flows + terminal value back to today
        4. Sum them = firm value
        5. Subtract net debt (debt - cash) to get equity value
        6. Divide by shares outstanding to get per-share value
    """
    # Step 1: Project the explicit cash flows
    cash_flows = project_cash_flows(
        current_fcf=inputs.current_fcf,
        growth_rate=inputs.high_growth_rate,
        num_years=inputs.high_growth_years,
    )

    # Step 2: Terminal value (at end of last forecast year)
    terminal_value = compute_terminal_value(
        final_year_fcf=cash_flows[-1],  # last element of the list
        terminal_growth_rate=inputs.terminal_growth_rate,
        discount_rate=inputs.discount_rate,
    )

    # Step 3a: Discount each explicit cash flow back to today
    pv_explicit = []
    for year, cf in enumerate(cash_flows, start=1):
        pv = discount_to_present(cf, inputs.discount_rate, year)
        pv_explicit.append(pv)

    # Step 3b: Discount the terminal value back to today
    pv_terminal = discount_to_present(
        terminal_value,
        inputs.discount_rate,
        inputs.high_growth_years,  # discount by full forecast length
    )
    # Step 4: Firm value = PV of all explicit cash flows + PV of terminal value
    firm_value = sum(pv_explicit) + pv_terminal

    # Step 5: Equity value = firm value - net debt
    net_debt = inputs.total_debt - inputs.total_cash
    equity_value = firm_value - net_debt

    # Step 6: Per-share value
    per_share_value = equity_value / inputs.shares_outstanding

    # Step 7: Package up everything (including debug info) and return
    return DCFResult(
        firm_value=firm_value,
        equity_value=equity_value,
        per_share_value=per_share_value,
        explicit_cash_flows=cash_flows,
        pv_explicit=pv_explicit,
        terminal_value=terminal_value,
        pv_terminal=pv_terminal,
    )

def compute_intrinsic_value_for_ticker(
    ticker,
    current_fcf,
    high_growth_rate=0.15,
    high_growth_years=10,
    terminal_growth_rate=0.04,
):
    """
    End-to-end DCF for a ticker using real WACC, real shares, real price.

    Pulls all market-derived inputs automatically. Only the growth
    assumptions are passed in — those still require judgment.

    Args:
        ticker: e.g. "MSFT"
        current_fcf: latest annual FCF in dollars (from our fundamentals DB)
        high_growth_rate, high_growth_years, terminal_growth_rate: growth assumptions

    Returns:
        dict with the DCFResult fields plus wacc_info for transparency
    """
    from valuation.inputs.wacc import compute_wacc

    # Pull WACC and capital-structure inputs
    wacc_info = compute_wacc(ticker)

    # Build DCF inputs from real data
    inputs = DCFInputs(
        current_fcf=current_fcf,
        total_debt=wacc_info["total_debt"],
        total_cash=0,  # TODO: pull total_cash separately for cleaner net debt
        shares_outstanding=wacc_info["shares_outstanding"],
        high_growth_rate=high_growth_rate,
        high_growth_years=high_growth_years,
        terminal_growth_rate=terminal_growth_rate,
        discount_rate=wacc_info["wacc"],
    )

    # Run the DCF
    result = compute_intrinsic_value(inputs)

    return {
        "result": result,
        "wacc_info": wacc_info,
        "current_price": wacc_info["current_price"],
        "upside_downside": (result.per_share_value - wacc_info["current_price"])
                          / wacc_info["current_price"],
    }

if __name__ == "__main__":
    # Phase 1.2: real WACC, real shares, real price
    bundle = compute_intrinsic_value_for_ticker(
        ticker="MSFT",
        current_fcf=37.01e9,
        high_growth_rate=0.15,
        high_growth_years=10,
        terminal_growth_rate=0.04,
    )

    r = bundle["result"]
    w = bundle["wacc_info"]

    print(f"\n=== MSFT DCF (Phase 1.2 — real WACC) ===")
    print(f"\nDiscount rate (WACC): {w['wacc']*100:.2f}%")
    print(f"  Beta:                  {w['beta']:.3f}")
    print(f"  Cost of equity:        {w['cost_of_equity']*100:.2f}%")
    print(f"  After-tax cost of debt: {w['after_tax_cost_of_debt']*100:.2f}%")
    print(f"  Equity weight:         {w['equity_weight']*100:.1f}%")
    print(f"  Debt weight:           {w['debt_weight']*100:.1f}%")

    print(f"\nValuation:")
    print(f"  Firm value:            ${r.firm_value / 1e9:.1f}B")
    print(f"  Equity value:          ${r.equity_value / 1e9:.1f}B")
    print(f"  Per-share value:       ${r.per_share_value:.2f}")
    print(f"  Current market price:  ${bundle['current_price']:.2f}")
    print(f"  Upside/(downside):     {bundle['upside_downside']*100:+.1f}%")
    print(f"  Terminal % of total:   {r.pv_terminal / r.firm_value * 100:.0f}%")