"""
Dispatcher — routes a ticker to the right bucket scorer.

Input: ticker (string)
Output: (score, max_score, all_cores_passed, reasons, bucket_name)
"""

import sqlite3
from data_pipeline.database import DB_PATH
from scoring.sectors.classifier import classify

# Import all bucket scorers
from scoring.sectors.technology.saas_growth import score as score_saas
from scoring.sectors.technology.mature_tech import score as score_mature_tech
from scoring.sectors.technology.semiconductors import score as score_semis
from scoring.sectors.financials.banks import score as score_banks
from scoring.sectors.default import score as score_default


# Map bucket name → scorer function
SCORERS = {
    "saas_growth":      score_saas,
    "mature_tech":      score_mature_tech,
    "semiconductors":   score_semis,
    "bank":             score_banks,
    # Buckets that exist in classifier but don't have dedicated scorers YET
    # all fall through to default for now:
    "insurance":        score_default,
    "financial_other":  score_default,
    "biotech":          score_default,
    "pharma":           score_default,
    "healthcare_other": score_default,
    "reit":             score_default,
    "industrial":       score_default,
    "consumer_cyclical": score_default,
    "consumer_defensive": score_default,
    "energy":           score_default,
    "utility":          score_default,
    "materials":        score_default,
    "communication":    score_default,
    "default":          score_default,
}


def score_ticker(ticker):
    """
    Look up a ticker's fundamentals, classify it, route to the right scorer.
    Returns (score, max_score, all_cores_passed, reasons, bucket_name).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM fundamentals WHERE ticker=?", (ticker,)).fetchone()
    conn.close()

    if row is None:
        return 0.0, 15.0, False, [f"⚠️ {ticker} not in database"], "unknown"

    row_dict = dict(row)
    bucket = classify(row_dict)
    scorer = SCORERS.get(bucket, score_default)

    score, max_score, all_cores, reasons = scorer(row_dict)
    return score, max_score, all_cores, reasons, bucket


def score_row(row_dict):
    """
    Same as score_ticker but takes a row dict directly (avoids extra DB hit).
    Useful when called inside analyze_stock_from_db which already loads all rows.
    """
    bucket = classify(row_dict)
    scorer = SCORERS.get(bucket, score_default)
    score, max_score, all_cores, reasons = scorer(row_dict)
    return score, max_score, all_cores, reasons, bucket
