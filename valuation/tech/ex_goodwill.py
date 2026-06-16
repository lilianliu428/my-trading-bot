"""
Ex-goodwill ROIC computation.

Standard ROIC includes goodwill in invested capital. For companies with
heavy acquisition history (AMD bought Xilinx, AVGO bought VMware, etc.),
this inflates the denominator and crushes reported ROIC, even when the
underlying business is operationally healthy.

Ex-goodwill ROIC subtracts goodwill from invested capital, isolating the
operational return on capital. We use ex-goodwill ROIC for forecasting
because the next dollar of reinvestment doesn't replicate past acquisition
premiums — it goes to organic operations.

Both ROICs are returned:
- roic_reported: includes goodwill (capital allocation quality)
- roic_ex_goodwill: excludes goodwill (operational economics)
- goodwill_ratio: how much of invested capital is goodwill (sanity check)

For forecasting, use ex-goodwill. For evaluating management's M&A track
record, look at the difference between the two.
"""

import math
import yfinance as yf


DEFAULT_TAX_RATE = 0.21


def compute_ex_goodwill_roic(ticker):
    """
    Compute ROIC both with and without goodwill in invested capital.

    Args:
        ticker (str): stock ticker

    Returns:
        dict with:
            ebit (float): operating income
            nopat (float): EBIT × (1 - tax_rate)
            tax_rate (float): tax rate used
            total_equity (float)
            total_debt (float)
            cash (float)
            goodwill (float): goodwill from balance sheet
            invested_capital_reported (float): equity + debt - cash
            invested_capital_ex_goodwill (float): above minus goodwill
            roic_reported (float): NOPAT / invested_capital_reported
            roic_ex_goodwill (float): NOPAT / invested_capital_ex_goodwill
            goodwill_ratio (float): goodwill / invested_capital_reported
            is_heavy_acquirer (bool): True if goodwill > 30% of invested capital
            data_flags (list): warnings or notes
    """
    yf_ticker = yf.Ticker(ticker)
    income_stmt = yf_ticker.income_stmt
    balance_sheet = yf_ticker.balance_sheet

    data_flags = []

    # Sanity check the data
    if income_stmt is None or income_stmt.empty:
        return _null_result("No income statement data", data_flags)
    if balance_sheet is None or balance_sheet.empty:
        return _null_result("No balance sheet data", data_flags)

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Get EBIT (operating income)
    # ─────────────────────────────────────────────────────────────────────
    latest_income = income_stmt.iloc[:, 0]
    ebit = None
    for field in ["EBIT", "Operating Income"]:
        if field in latest_income.index:
            val = latest_income[field]
            if val is not None and not math.isnan(float(val)):
                ebit = float(val)
                break

    if ebit is None:
        return _null_result(
            f"{ticker}: no EBIT — likely financial company, ROIC not applicable",
            data_flags,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Get tax rate, compute NOPAT
    # ─────────────────────────────────────────────────────────────────────
    info = yf_ticker.info
    tax_rate = info.get("effectiveTaxRate") or DEFAULT_TAX_RATE
    nopat = ebit * (1 - tax_rate)

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Get balance sheet components
    # ─────────────────────────────────────────────────────────────────────
    latest_balance = balance_sheet.iloc[:, 0]

    # Total equity (with fallback fields)
    total_equity = None
    for field in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
        if field in latest_balance.index:
            val = latest_balance[field]
            if val is not None and not math.isnan(float(val)):
                total_equity = float(val)
                break

    if total_equity is None or total_equity <= 0:
        return _null_result(f"{ticker}: no usable equity value", data_flags)

    # Total debt
    total_debt = _safe_float(latest_balance.get("Total Debt", 0))

    # Cash
    cash = _safe_float(latest_balance.get("Cash And Cash Equivalents", 0))

    # Goodwill (with fallback to combined intangibles field)
    goodwill = _safe_float(latest_balance.get("Goodwill", 0))
    if goodwill == 0:
        # Some companies report combined "Goodwill And Other Intangible Assets"
        goodwill = _safe_float(latest_balance.get("Goodwill And Other Intangible Assets", 0))
        if goodwill > 0:
            data_flags.append(
                "Goodwill not separately reported; used 'Goodwill And Other Intangible Assets'"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Compute both versions of invested capital
    # ─────────────────────────────────────────────────────────────────────
    invested_capital_reported = total_equity + total_debt - cash
    invested_capital_ex_goodwill = invested_capital_reported - goodwill

    if invested_capital_reported <= 0:
        return _null_result(f"{ticker}: invested capital is zero or negative", data_flags)

    if invested_capital_ex_goodwill <= 0:
        data_flags.append(
            f"Goodwill (${goodwill/1e9:.1f}B) exceeds operational capital — "
            f"company is mostly an acquisition vehicle"
        )
        # Don't divide by zero or negative; flag and use reported instead
        invested_capital_ex_goodwill = invested_capital_reported

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Compute both ROICs
    # ─────────────────────────────────────────────────────────────────────
    roic_reported = nopat / invested_capital_reported
    roic_ex_goodwill = nopat / invested_capital_ex_goodwill

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Sanity check ratios
    # ─────────────────────────────────────────────────────────────────────
    goodwill_ratio = goodwill / invested_capital_reported if invested_capital_reported > 0 else 0
    is_heavy_acquirer = goodwill_ratio > 0.30

    if is_heavy_acquirer:
        data_flags.append(
            f"Heavy acquirer: {goodwill_ratio*100:.0f}% of invested capital is goodwill. "
            f"Reported ROIC ({roic_reported*100:.1f}%) understates operational economics. "
            f"Use roic_ex_goodwill ({roic_ex_goodwill*100:.1f}%) for forecasting."
        )

    return {
        "ticker": ticker,
        "ebit": ebit,
        "nopat": nopat,
        "tax_rate": tax_rate,
        "total_equity": total_equity,
        "total_debt": total_debt,
        "cash": cash,
        "goodwill": goodwill,
        "invested_capital_reported": invested_capital_reported,
        "invested_capital_ex_goodwill": invested_capital_ex_goodwill,
        "roic_reported": roic_reported,
        "roic_ex_goodwill": roic_ex_goodwill,
        "goodwill_ratio": goodwill_ratio,
        "is_heavy_acquirer": is_heavy_acquirer,
        "data_flags": data_flags,
    }


def _safe_float(value):
    """Convert a balance sheet value to float, treating None/NaN as 0."""
    if value is None:
        return 0.0
    try:
        v = float(value)
        if math.isnan(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def _null_result(reason, data_flags):
    """Return-shape for failed computation."""
    data_flags.append(reason)
    return {
        "ticker": None,
        "ebit": None,
        "nopat": None,
        "tax_rate": None,
        "total_equity": None,
        "total_debt": None,
        "cash": None,
        "goodwill": None,
        "invested_capital_reported": None,
        "invested_capital_ex_goodwill": None,
        "roic_reported": None,
        "roic_ex_goodwill": None,
        "goodwill_ratio": None,
        "is_heavy_acquirer": False,
        "data_flags": data_flags,
    }


# ════════════════════════════════════════════════════════════════════════
# Test harness
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Mix of: non-acquirer (NVDA), moderate (MSFT, GOOGL), heavy (AVGO, AMD)
    TICKERS = ["NVDA", "MSFT", "GOOGL", "AVGO", "AMD"]

    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        result = compute_ex_goodwill_roic(ticker)

        if result["nopat"] is None:
            print(f"  Failed: {result['data_flags']}")
            continue

        print(f"\n  EBIT:                          ${result['ebit']/1e9:.1f}B")
        print(f"  NOPAT:                         ${result['nopat']/1e9:.1f}B")
        print(f"  Tax rate:                      {result['tax_rate']*100:.1f}%")

        print(f"\n  Invested capital (reported):   ${result['invested_capital_reported']/1e9:.1f}B")
        print(f"  Less: Goodwill:                ${result['goodwill']/1e9:.1f}B")
        print(f"  Invested capital (ex-goodwill):${result['invested_capital_ex_goodwill']/1e9:.1f}B")
        print(f"  Goodwill ratio:                {result['goodwill_ratio']*100:.1f}%")

        print(f"\n  ROIC (reported):               {result['roic_reported']*100:.1f}%")
        print(f"  ROIC (ex-goodwill):            {result['roic_ex_goodwill']*100:.1f}%")
        print(f"  Heavy acquirer:                {result['is_heavy_acquirer']}")

        if result['data_flags']:
            print(f"\n  Flags:")
            for f in result['data_flags']:
                print(f"    - {f}")