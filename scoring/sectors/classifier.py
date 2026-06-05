"""
Classifies a ticker into a business-model bucket.

Returns one of:
  saas_growth, mature_tech, semiconductors,
  bank, insurance, financial_other,
  biotech, pharma, healthcare_other,
  reit, industrial, consumer_cyclical, consumer_defensive,
  energy, utility, materials, communication, default
"""

# ============================================================
# Hardcoded overrides — for edge cases rules can't handle well
# ============================================================
OVERRIDES = {
    # Tech edge cases
    "PLTR": "saas_growth",      # classified as software but has gov't contracts
    "SNOW": "saas_growth",
    "DDOG": "saas_growth",
    "NET": "saas_growth",
    "ZS":  "saas_growth",
    "CRWD": "saas_growth",
    "MDB": "saas_growth",
    "OKTA": "saas_growth",
    "PATH": "saas_growth",
    "U":   "saas_growth",
    "GTLB": "saas_growth",
    "TEAM": "saas_growth",
    "HUBS": "saas_growth",
    "COIN": "default",          # crypto exchange — unique

    # ---- Mature internet giants (classified as "Internet Content & Information")
    # but are huge ad-driven cash cows, not unprofitable SaaS ----
    "GOOGL": "mature_tech",
    "GOOG":  "mature_tech",
    "META":  "mature_tech",
    "AMZN":  "mature_tech",     # e-commerce + AWS, mature cash machine
    "NFLX":  "mature_tech",     # streaming is now mature
    "BKNG":  "mature_tech",     # Booking.com — mature internet
    "EBAY":  "mature_tech",
    "PYPL":  "mature_tech",
    "BIDU":  "mature_tech",
    "JD":    "mature_tech",     # Chinese e-commerce, mature
    "TSLA": "consumer_cyclical", # yfinance puts in Consumer Cyclical, keep that
    "RBLX": "saas_growth",      # gaming SaaS-ish
    "ABNB": "consumer_cyclical",
    
    # Healthcare edge cases — biotechs without revenue
    "MRNA": "biotech",
    "VRTX": "biotech",
    "REGN": "biotech",
    "BIIB": "biotech",
    "GILD": "biotech",
    "AMGN": "biotech",
    "INCY": "biotech",
    "SGEN": "biotech",
    
    # ETFs — skip business model scoring
    "SPY": "default", "QQQ": "default", "IWM": "default", "DIA": "default",
    "XLF": "default", "XLE": "default", "XLK": "default", "XLV": "default",
    "XLY": "default", "XLP": "default", "XLI": "default", "XLU": "default",
    "XLB": "default", "XLRE": "default", "XLC": "default",
    "VTI": "default", "VOO": "default", "VXUS": "default",
}


# ============================================================
# Industry → bucket mapping (yfinance industry field)
# ============================================================
INDUSTRY_RULES = {
    # ---- Technology subsectors ----
    "Software - Infrastructure": "saas_growth",  # checked dynamically — could be mature
    "Software - Application":    "saas_growth",
    "Information Technology Services": "mature_tech",
    "Communication Equipment":   "mature_tech",
    "Computer Hardware":         "mature_tech",
    "Consumer Electronics":      "mature_tech",
    "Electronic Components":     "mature_tech",
    "Scientific & Technical Instruments": "mature_tech",
    "Solar":                     "mature_tech",
    
    "Semiconductors":            "semiconductors",
    "Semiconductor Equipment & Materials": "semiconductors",
    
    # ---- Financials ----
    "Banks - Diversified":       "bank",
    "Banks - Regional":          "bank",
    "Banks":                     "bank",
    "Credit Services":           "financial_other",
    "Capital Markets":           "financial_other",
    "Financial Data & Stock Exchanges": "financial_other",
    "Asset Management":          "financial_other",
    "Insurance - Diversified":   "insurance",
    "Insurance - Life":          "insurance",
    "Insurance - Property & Casualty": "insurance",
    "Insurance - Reinsurance":   "insurance",
    "Insurance - Specialty":     "insurance",
    "Insurance Brokers":         "insurance",
    
    # ---- Healthcare ----
    "Biotechnology":             "biotech",
    "Drug Manufacturers - General": "pharma",
    "Drug Manufacturers - Specialty & Generic": "pharma",
    "Medical Devices":           "healthcare_other",
    "Medical Instruments & Supplies": "healthcare_other",
    "Medical Care Facilities":   "healthcare_other",
    "Healthcare Plans":          "healthcare_other",
    "Health Information Services": "healthcare_other",
    "Diagnostics & Research":    "healthcare_other",
    "Medical Distribution":      "healthcare_other",
    
    # ---- Real Estate ----
    "REIT - Diversified":        "reit",
    "REIT - Industrial":         "reit",
    "REIT - Office":             "reit",
    "REIT - Residential":        "reit",
    "REIT - Retail":             "reit",
    "REIT - Healthcare Facilities": "reit",
    "REIT - Specialty":          "reit",
    "REIT - Hotel & Motel":      "reit",
    "REIT - Mortgage":           "reit",
    "Real Estate Services":      "default",
    
    # ---- Communication ----
    "Internet Content & Information": "saas_growth",  # GOOGL/META — but big and mature, special case
    "Entertainment":             "communication",
    "Telecom Services":          "communication",
    "Broadcasting":              "communication",
    "Publishing":                "communication",
    "Advertising Agencies":      "communication",
    "Electronic Gaming & Multimedia": "communication",
}


def classify(row):
    """
    Given a fundamentals row dict with 'ticker', 'sector', 'industry',
    'revenue_growth', 'profit_margin', 'total_revenue' etc, return a bucket name.
    """
    ticker = row.get("ticker")
    sector = row.get("sector")
    industry = row.get("industry")
    rev_growth = row.get("revenue_growth")
    profit_margin = row.get("profit_margin")
    revenue = row.get("total_revenue")

    # 1. Hardcoded overrides win
    if ticker in OVERRIDES:
        return OVERRIDES[ticker]

    # 2. Industry-based rules
    bucket = INDUSTRY_RULES.get(industry)
    if bucket:
        # Refinement: "Software - Infrastructure/Application" is sometimes mature, not saas_growth.
        # Use revenue + growth + profitability to disambiguate.
        if bucket == "saas_growth":
            # Mega-cap profitable software = mature regardless of growth pace (MSFT)
            if (revenue is not None and revenue > 100e9
                    and profit_margin is not None and profit_margin > 0.20):
                return "mature_tech"
            # Solidly profitable software not growing explosively = mature
            # Captures ORCL, ADSK, CDNS, PAYX, PTC, FTNT — but NOT PLTR (high growth)
            if (profit_margin is not None and profit_margin > 0.18
                    and rev_growth is not None and rev_growth < 0.25):
                return "mature_tech"
        return bucket

    # 3. Sector-based fallback
    sector_fallback = {
        "Technology":           "mature_tech",
        "Financial Services":   "financial_other",
        "Healthcare":           "healthcare_other",
        "Real Estate":          "default",
        "Communication Services": "communication",
        "Industrials":          "industrial",
        "Consumer Cyclical":    "consumer_cyclical",
        "Consumer Defensive":   "consumer_defensive",
        "Energy":               "energy",
        "Utilities":            "utility",
        "Basic Materials":      "materials",
    }
    return sector_fallback.get(sector, "default")


def classify_all_in_db():
    """Update business_model column for all tickers in the DB."""
    import sqlite3
    from data_pipeline.database import DB_PATH
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT ticker, sector, industry, revenue_growth, profit_margin, total_revenue
        FROM fundamentals
    """).fetchall()
    
    updates = []
    for row in rows:
        bucket = classify(dict(row))
        updates.append((bucket, row["ticker"]))
    
    conn.executemany(
        "UPDATE fundamentals SET business_model = ? WHERE ticker = ?",
        updates
    )
    conn.commit()
    
    # Report distribution
    dist = conn.execute("""
        SELECT business_model, COUNT(*) FROM fundamentals
        GROUP BY business_model ORDER BY COUNT(*) DESC
    """).fetchall()
    conn.close()
    
    print("\nBusiness model distribution:")
    for bm, count in dist:
        print(f"  {bm or '(null)'}: {count}")


if __name__ == "__main__":
    classify_all_in_db()
