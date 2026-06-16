"""
Tech-specific DCF model.

Adds tech-industry adjustments on top of the generic DCF engine:
- R&D capitalization (treat R&D as investment, not expense)
- SBC dilution modeling (project share count growth from stock-based comp)
- Ex-goodwill ROIC everywhere it matters (serial acquirers like AMD, AVGO)
- Exit multiple terminal value (avoid Gordon Growth explosion)

Used for tickers in TECH_BUCKETS:
- mature_tech
- saas_growth
- semiconductors
- communication
"""

TECH_BUCKETS = {
    "mature_tech",
    "saas_growth",
    "semiconductors",
    "communication",
}