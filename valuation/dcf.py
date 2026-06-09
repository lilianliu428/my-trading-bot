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

def compute_intrinsic_value_for_ticker(ticker, bucket="default"):
    """
    End-to-end DCF using three-stage growth + ROIC fade model.

    Pulls all market-derived inputs:
        - WACC from real macro + capital structure
        - Growth profile (year-by-year g, ROIC, reinvestment)
        - Current EBIT and tax rate from financials

    Args:
        ticker: e.g. "MSFT"
        bucket: business model bucket (mature_tech, consumer_defensive, etc.)

    Returns:
        dict with the full DCF breakdown:
            firm_value, equity_value, per_share_value,
            current_price, upside_downside,
            yearly_cash_flows, pv_explicit, pv_terminal,
            terminal_value, wacc, growth_profile, data_flags
    """
    from valuation.inputs.wacc import compute_wacc
    from valuation.inputs.growth import build_growth_profile
    import yfinance as yf

    # Step 1: Get WACC, capital structure, and current price
    wacc_info = compute_wacc(ticker)
    wacc = wacc_info["wacc"]

    # Step 2: Get growth profile
    growth = build_growth_profile(ticker, wacc, bucket=bucket)

    # Step 3: Get current EBIT(1-t) — starting cash flow
    yf_ticker = yf.Ticker(ticker)
    income_stmt = yf_ticker.income_stmt
    latest = income_stmt.iloc[:, 0]

    ebit = None
    for field in ["EBIT", "Operating Income"]:
        if field in latest.index:
            ebit = float(latest[field])
            break

    if ebit is None or ebit <= 0:
        return {
            "error": "Could not compute DCF: EBIT unavailable or non-positive",
            "ticker": ticker,
            "wacc": wacc,
            "growth_profile": growth,
            "data_flags": ["No usable EBIT — DCF cannot be computed for financial companies in current model"],
        }

    tax_rate = yf_ticker.info.get("effectiveTaxRate") or 0.21
    starting_nopat = ebit * (1 - tax_rate)

    # Step 4: Project year-by-year cash flows
    # FCFF_t = NOPAT_t × (1 - reinvestment_rate_t)
    # NOPAT_t = NOPAT_{t-1} × (1 + growth_rate_t)

    yearly_nopat = []
    yearly_fcff = []
    yearly_pv_fcff = []

    current_nopat = starting_nopat
    for t, (g, reinv) in enumerate(zip(growth["yearly_growth"], growth["yearly_reinvestment"]), start=1):
        current_nopat = current_nopat * (1 + g)
        fcff = current_nopat * (1 - reinv)
        pv_fcff = fcff / ((1 + wacc) ** t)

        yearly_nopat.append(current_nopat)
        yearly_fcff.append(fcff)
        yearly_pv_fcff.append(pv_fcff)

    pv_explicit = sum(yearly_pv_fcff)

    # Step 5: Terminal value (Gordon Growth with proper reinvestment formula)
    # TV = NOPAT_{N+1} × (1 - reinvestment_rate_terminal) / (WACC - g_terminal)
    n_years = len(growth["yearly_growth"])
    terminal_nopat = current_nopat * (1 + growth["terminal_growth"])
    terminal_reinvestment = growth["terminal_reinvestment_rate"]
    terminal_fcff = terminal_nopat * (1 - terminal_reinvestment)
    terminal_value = terminal_fcff / (wacc - growth["terminal_growth"])
    pv_terminal = terminal_value / ((1 + wacc) ** n_years)

    # Step 6: Combine
    firm_value = pv_explicit + pv_terminal

    # Equity = firm value - net debt
    # Pull total_cash for proper net debt (was a placeholder before)
    balance_sheet = yf_ticker.balance_sheet
    latest_balance = balance_sheet.iloc[:, 0]
    total_cash = float(latest_balance.get("Cash And Cash Equivalents", 0))

    equity_value = firm_value - wacc_info["total_debt"] + total_cash
    per_share_value = equity_value / wacc_info["shares_outstanding"]

    current_price = wacc_info["current_price"]
    upside_downside = (per_share_value - current_price) / current_price

    return {
        "ticker": ticker,
        "bucket": bucket,
        "firm_value": firm_value,
        "equity_value": equity_value,
        "per_share_value": per_share_value,
        "current_price": current_price,
        "upside_downside": upside_downside,
        "pv_explicit": pv_explicit,
        "pv_terminal": pv_terminal,
        "terminal_value": terminal_value,
        "terminal_pct_of_value": pv_terminal / firm_value,
        "yearly_nopat": yearly_nopat,
        "yearly_fcff": yearly_fcff,
        "yearly_pv_fcff": yearly_pv_fcff,
        "wacc": wacc,
        "growth_profile": growth,
        "starting_nopat": starting_nopat,
        "data_flags": growth["data_flags"],
    }

if __name__ == "__main__":
    result = compute_intrinsic_value_for_ticker("MSFT", bucket="mature_tech")

    if "error" in result:
        print(f"\nERROR: {result['error']}")
    else:
        print(f"\n=== {result['ticker']} DCF (Three-Stage Model) ===")

        print(f"\nDiscount rate (WACC):  {result['wacc']*100:.2f}%")
        gp = result['growth_profile']
        print(f"\nGrowth profile:")
        print(f"  High-growth period:  {gp['high_growth_years']} years")
        print(f"  Transition period:   {gp['transition_years']} years")
        print(f"  Terminal growth:     {gp['terminal_growth']*100:.2f}%")
        print(f"  Terminal ROIC:       {gp['terminal_roic']*100:.1f}%")

        print(f"\nValuation:")
        print(f"  Starting NOPAT:      ${result['starting_nopat']/1e9:.1f}B")
        print(f"  PV explicit CFs:     ${result['pv_explicit']/1e9:.1f}B  ({(1-result['terminal_pct_of_value'])*100:.0f}%)")
        print(f"  PV terminal:         ${result['pv_terminal']/1e9:.1f}B  ({result['terminal_pct_of_value']*100:.0f}%)")
        print(f"  Firm value:          ${result['firm_value']/1e9:.1f}B")
        print(f"  Equity value:        ${result['equity_value']/1e9:.1f}B")
        print(f"  Per-share value:     ${result['per_share_value']:.2f}")
        print(f"  Current price:       ${result['current_price']:.2f}")
        print(f"  Upside/(downside):   {result['upside_downside']*100:+.1f}%")

        if result['data_flags']:
            print(f"\nFlags:")
            for f in result['data_flags']:
                print(f"  - {f}")