"""
Compute WACC (Weighted Average Cost of Capital) for a ticker.

WACC = (E/V) × r_e + (D/V) × r_d × (1-t)

Where:
    E = market value of equity (market cap)
    D = market value of debt (we use book value as proxy)
    V = E + D
    r_e = cost of equity (from CAPM)
    r_d = pre-tax cost of debt
    t   = tax rate

This is the discount rate used in firm-value DCF, applied to free cash
flow to the firm (FCFF).
"""

import yfinance as yf

from valuation.data_sources.fred import get_risk_free_rate
from valuation.inputs.macro import IMPLIED_ERP
from valuation.inputs.beta import compute_beta
from valuation.inputs.cost_of_debt import compute_cost_of_debt


def get_capital_structure(ticker):
    """
    Get market cap (equity) and total debt for a ticker from yfinance.

    Returns:
        dict with:
            market_cap (float): market value of equity
            total_debt (float): book value of debt (proxy for market value)
            equity_weight (float): E / (E + D)
            debt_weight (float): D / (E + D)
            current_price (float): latest price
            shares_outstanding (float): used downstream in DCF
    """
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info

    market_cap = info.get("marketCap")
    total_debt = info.get("totalDebt")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares_outstanding = info.get("sharesOutstanding")

    if not all([market_cap, total_debt is not None, current_price, shares_outstanding]):
        raise ValueError(
            f"Missing capital structure data for {ticker}. "
            f"Got market_cap={market_cap}, total_debt={total_debt}, "
            f"price={current_price}, shares={shares_outstanding}"
        )

    total_capital = market_cap + total_debt
    equity_weight = market_cap / total_capital
    debt_weight = total_debt / total_capital

    return {
        "market_cap": market_cap,
        "total_debt": total_debt,
        "total_capital": total_capital,
        "equity_weight": equity_weight,
        "debt_weight": debt_weight,
        "current_price": current_price,
        "shares_outstanding": shares_outstanding,
    }


def compute_wacc(ticker):
    """
    Compute WACC for a ticker by pulling all inputs and combining.

    Returns:
        dict with all the pieces (for transparency/debugging) plus the
        final wacc.
    """
    print(f"\n=== Computing WACC for {ticker} ===")

    # Step 1: Risk-free rate
    rf = get_risk_free_rate()

    # Step 2: Cost of equity via CAPM
    print(f"  Computing beta...")
    beta_info = compute_beta(ticker)
    beta = beta_info["adjusted_beta"]
    erp = IMPLIED_ERP
    cost_of_equity = rf + beta * erp

    # Step 3: Cost of debt
    cod_info = compute_cost_of_debt(ticker, rf)
    pre_tax_cost_of_debt = cod_info["pre_tax_cost_of_debt"]
    after_tax_cost_of_debt = cod_info["after_tax_cost_of_debt"]
    tax_rate = cod_info["tax_rate"]

    # Step 4: Capital structure weights
    print(f"  Getting capital structure...")
    cap_struct = get_capital_structure(ticker)

    # Step 5: Combine into WACC
    wacc = (
        cap_struct["equity_weight"] * cost_of_equity
        + cap_struct["debt_weight"] * after_tax_cost_of_debt
    )

    return {
        # Inputs
        "risk_free_rate": rf,
        "beta": beta,
        "equity_risk_premium": erp,
        "cost_of_equity": cost_of_equity,
        "pre_tax_cost_of_debt": pre_tax_cost_of_debt,
        "after_tax_cost_of_debt": after_tax_cost_of_debt,
        "tax_rate": tax_rate,
        # Capital structure
        "market_cap": cap_struct["market_cap"],
        "total_debt": cap_struct["total_debt"],
        "equity_weight": cap_struct["equity_weight"],
        "debt_weight": cap_struct["debt_weight"],
        "current_price": cap_struct["current_price"],
        "shares_outstanding": cap_struct["shares_outstanding"],
        # Output
        "wacc": wacc,
    }


if __name__ == "__main__":
    result = compute_wacc("MSFT")

    print(f"\n=== MSFT WACC Summary ===")
    print(f"\nCost of equity (CAPM):")
    print(f"  Risk-free rate:     {result['risk_free_rate']*100:.2f}%")
    print(f"  Beta:               {result['beta']:.3f}")
    print(f"  ERP:                {result['equity_risk_premium']*100:.2f}%")
    print(f"  → Cost of equity:   {result['cost_of_equity']*100:.2f}%")

    print(f"\nCost of debt:")
    print(f"  Pre-tax:            {result['pre_tax_cost_of_debt']*100:.2f}%")
    print(f"  Tax rate:           {result['tax_rate']*100:.1f}%")
    print(f"  → After-tax:        {result['after_tax_cost_of_debt']*100:.2f}%")

    print(f"\nCapital structure:")
    print(f"  Market cap:         ${result['market_cap']/1e9:.0f}B")
    print(f"  Total debt:         ${result['total_debt']/1e9:.0f}B")
    print(f"  Equity weight:      {result['equity_weight']*100:.1f}%")
    print(f"  Debt weight:        {result['debt_weight']*100:.1f}%")

    print(f"\n→ WACC: {result['wacc']*100:.2f}%")