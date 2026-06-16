# Tech-specific DCF

Specialized DCF model for technology companies. Used instead of the generic
`valuation/dcf.py` for tickers in tech-adjacent buckets.

## Why a separate model?

Generic DCF treats all companies the same. But tech companies have unique
characteristics that the generic model handles badly:

1. **R&D is investment, not expense.** Tech companies spend billions on R&D
   that produces returns for years. Standard accounting treats it as a recurring
   expense, which understates earnings and overstates reinvestment efficiency.

2. **Stock-based comp dilutes ownership.** SaaS companies issue 3-5% of shares
   per year as employee comp. Over 10 years, share count grows 30-40%. Standard
   DCF divides by today's shares — wildly wrong for high-SBC companies.

3. **Serial acquirers carry massive goodwill.** AMD bought Xilinx for $49B, AVGO
   bought VMware for $69B. The acquisition premiums sit on the balance sheet as
   goodwill, crushing reported ROIC. Operationally, these are healthy businesses.

4. **Terminal value explodes for low-WACC names.** Gordon Growth perpetuity
   gives weird answers for safe, high-quality tech compounders (MSFT, GOOGL).
   Exit multiples are more honest.

## Files

- `__init__.py` — TECH_BUCKETS definition
- `rd_capitalization.py` — R&D capitalization logic
- `sbc_dilution.py` — Share-count projection from SBC
- `ex_goodwill.py` — Ex-goodwill ROIC everywhere
- `tech_dcf.py` — Entry point that orchestrates all adjustments

## When NOT to use this

- Non-tech buckets (use generic DCF or specialized model when built)
- Banks, REITs, insurance (need Residual Income or NAV, not DCF at all)

## See also

- `docs/MODEL_TECH.md` — full explanation of each adjustment
- `docs/MODEL_GENERIC.md` — what the generic DCF does and its limitations