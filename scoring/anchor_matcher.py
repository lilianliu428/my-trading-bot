"""
Anchor Matcher — finds the historical company-phases most similar to a current ticker.

Uses cosine similarity on fingerprint vectors built from fundamentals:
  - log-scale revenue (orders of magnitude vary widely)
  - decimal margins (gross, op, fcf, net)
  - decimal revenue growth

Bucket-aware by default. Optional within-industry cross-matching for
transformation cases (e.g. mature_tech → saas_growth pivots like IBM).

Interpretation logic is cautious on losses: a strong loser match outweighs
two weak winner matches. Quality over quantity.

Key API:
    match(ticker, top_k=3, cross_industry=False) -> list of match tuples
    interpret_matches(matches) -> verdict dict with message
"""

import math
import sqlite3
from typing import Optional

from scoring.historical_anchors import ANCHORS, all_anchor_points

DB_PATH = "/home/ubuntu/my-trading-bot/data.db"


# ---------------------------------------------------------------
# Industry groupings — for optional cross-bucket matching
# ---------------------------------------------------------------
INDUSTRY_GROUPS = {
    "tech": ["saas_growth", "mature_tech", "semiconductors"],
    "financial": ["bank", "reit"],
    "healthcare": ["biotech", "pharma"],
    "consumer": ["consumer_cyclical"],
    "industrial": ["industrial"],
}


def _bucket_to_industry(bucket: str) -> Optional[str]:
    """Return the parent industry for a bucket, or None if unrecognized."""
    for industry, buckets in INDUSTRY_GROUPS.items():
        if bucket in buckets:
            return industry
    return None


# ---------------------------------------------------------------
# Fingerprint construction — turn DB row into anchor-shaped dict
# ---------------------------------------------------------------
def build_fingerprint(ticker: str) -> Optional[dict]:
    """
    Pull the latest fundamentals for `ticker` from data.db and shape into
    a dict matching the anchor schema. Returns None if no data.

    Note: some fields may be None (e.g. banks lack meaningful gross_margin).
    cosine_similarity() handles missing fields by skipping them.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticker, total_revenue, quarterly_earnings_growth,
               gross_margin, op_margin, profit_margin,
               free_cash_flow, business_model
        FROM fundamentals
        WHERE ticker = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (ticker.upper(),),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    _, revenue, growth, gross_margin, op_margin, profit_margin, fcf, bucket = row

    # Compute fcf_margin from raw FCF / revenue
    fcf_margin = (fcf / revenue) if (fcf is not None and revenue) else None

    # profit_margin from yfinance is net margin (decimal)
    net_margin = profit_margin

    return {
        "ticker": ticker.upper(),
        "bucket": bucket,
        "revenue": float(revenue) if revenue else None,
        "revenue_growth": float(growth) if growth is not None else None,
        "gross_margin": float(gross_margin) if gross_margin is not None else None,
        "op_margin": float(op_margin) if op_margin is not None else None,
        "fcf_margin": float(fcf_margin) if fcf_margin is not None else None,
        "net_margin": float(net_margin) if net_margin is not None else None,
    }


# ---------------------------------------------------------------
# Cosine similarity — handles missing fields gracefully
# ---------------------------------------------------------------
# Fields used for comparison. Equal weights for now (per design decision).
COMPARE_FIELDS = [
    "revenue",          # log-scaled
    "revenue_growth",   # decimal
    "gross_margin",     # decimal
    "op_margin",        # decimal
    "fcf_margin",       # decimal
    "net_margin",       # decimal
]


def _normalize_field(field: str, value: float) -> float:
    """
    Normalize a field value to a comparable scale.
    Revenue gets log-scaled. Other fields stay as decimals.
    """
    if value is None:
        return None
    if field == "revenue":
        # log10 of revenue. Floor at $100M to avoid log(0) or log(negative)
        if value <= 0:
            return None
        return math.log10(max(value, 1e8))
    # Margins and growth are already 0..1 range decimals
    return value


# ---------------------------------------------------------------
# Similarity: scaled euclidean distance with exponential decay
# ---------------------------------------------------------------
# We use scaled euclidean rather than cosine because cosine only captures
# the "angle" of the fingerprint vector — direction, not magnitude. Two
# companies with similar margin ratios but very different absolute margins
# would score 0.99 in pure cosine. Euclidean captures the actual differences.
#
# Per-field scale factors normalize each field's contribution. Without these,
# log-revenue (range ~9-12) would dominate margin differences (range ~0-1).
FIELD_SCALES = {
    "revenue":         3.0,   # log10, typical range 9-12 (1B to 1T)
    "revenue_growth":  2.5,   # decimal, typical range -0.5 to +2.0
    "gross_margin":    1.0,   # decimal, typical range 0 to 0.95
    "op_margin":       1.15,  # decimal, typical range -0.5 to +0.65
    "fcf_margin":      1.0,   # decimal, typical range -0.5 to +0.5
    "net_margin":      1.0,   # decimal, typical range -0.5 to +0.5
}

# Decay rate for distance → similarity mapping. exp(-3d) gives:
#   d=0    → sim=1.00 (identical)
#   d=0.15 → sim=0.64 (close match — same bucket, similar phase)
#   d=0.25 → sim=0.47 (moderate — same industry, different phase)
#   d=0.50 → sim=0.22 (weak — different shape)
#   d=0.75 → sim=0.11 (essentially unrelated)
DECAY_RATE = 3.0


def cosine_similarity(fp_a: dict, fp_b: dict) -> float:
    """
    Compute similarity between two fingerprints using scaled euclidean
    distance with exponential decay.

    Returns float in [0, 1] where:
        1.0 = identical fingerprints
        ~0.6 = strong match (similar phase, similar magnitude)
        ~0.4 = moderate match (same bucket, somewhat different)
        <0.3 = weak/no meaningful match

    Skips fields that are None in either fingerprint. Needs at least 2
    overlapping non-null fields to return a meaningful score.

    Note: kept name `cosine_similarity` for API stability — the algorithm
    is actually scaled euclidean. Future cleanup could rename.
    """
    sum_sq = 0.0
    n = 0
    for field in COMPARE_FIELDS:
        val_a = _normalize_field(field, fp_a.get(field))
        val_b = _normalize_field(field, fp_b.get(field))
        if val_a is None or val_b is None:
            continue
        scale = FIELD_SCALES.get(field, 1.0)
        diff = (val_a - val_b) / scale
        sum_sq += diff * diff
        n += 1

    if n < 2:
        # Need at least 2 dimensions for meaningful similarity
        return 0.0

    # Average distance per dimension — keeps scale stable as field count varies
    distance = math.sqrt(sum_sq / n)

    # Exponential decay: distance → similarity
    return math.exp(-DECAY_RATE * distance)


# ---------------------------------------------------------------
# Main matcher API
# ---------------------------------------------------------------
def match(
    ticker: str,
    top_k: int = 3,
    cross_industry: bool = False,
) -> list:
    """
    Find the top-K most similar historical anchors for `ticker`.

    Args:
        ticker: stock symbol to match
        top_k: how many top matches to return
        cross_industry: if True, allow matching within the same parent industry
                        (e.g. saas_growth <-> mature_tech for tech transformations).
                        If False (default), only match against same-bucket anchors.

    Returns:
        List of tuples sorted by similarity descending:
            [(anchor_ticker, phase_label, similarity, outcome, year, notes), ...]
        Empty list if no fingerprint can be built or no anchors qualify.
    """
    fp = build_fingerprint(ticker)
    if fp is None:
        return []
    if fp["bucket"] is None:
        return []

    candidate_industry = _bucket_to_industry(fp["bucket"])

    scored = []
    for anchor_ticker, phase_label, anchor_data in all_anchor_points():
        anchor_bucket = anchor_data["bucket"]

        # Bucket / industry filtering
        if cross_industry:
            anchor_industry = _bucket_to_industry(anchor_bucket)
            if anchor_industry != candidate_industry:
                continue
        else:
            if anchor_bucket != fp["bucket"]:
                continue

        sim = cosine_similarity(fp, anchor_data)
        scored.append(
            (
                anchor_ticker,
                phase_label,
                sim,
                anchor_data["outcome"],
                anchor_data["year"],
                anchor_data.get("notes", ""),
            )
        )

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------
# Interpretation — apply cautious-on-losses logic
# ---------------------------------------------------------------
LOSER_WARNING_THRESHOLD = 0.50   # if top match is loser >= this sim, strong warning
SECONDARY_LOSER_THRESHOLD = 0.45  # if 2nd/3rd match is loser >= this sim, caution
STRONG_MATCH_THRESHOLD = 0.55    # similarity above which we consider a match strong


def interpret_matches(matches: list) -> dict:
    """
    Apply asymmetric cautious-on-losses logic to a list of matches.

    Returns dict with:
        verdict: "winner_pattern" | "loser_warning" | "recovery_signal" |
                 "mixed" | "weak_match"
        message: short explanation suitable for Telegram
        top_match: the highest-similarity match tuple
    """
    if not matches:
        return {
            "verdict": "weak_match",
            "message": "No comparable historical patterns found.",
            "top_match": None,
        }

    top = matches[0]
    top_ticker, top_phase, top_sim, top_outcome, top_year, top_notes = top

    # Strong loser match — strongest warning, regardless of other matches
    if top_outcome == "loser" and top_sim >= LOSER_WARNING_THRESHOLD:
        return {
            "verdict": "loser_warning",
            "message": (
                f"⚠️ Pattern warning: resembles {top_ticker} {top_year} "
                f"({top_phase}) at {top_sim:.0%} similarity — historical outcome was LOSER. "
                f"Consider waiting for clearer signals."
            ),
            "top_match": top,
        }

    # Recovery phase match — incumbent comeback signal
    if top_outcome == "winner" and "recovery" in top_phase.lower() and top_sim >= STRONG_MATCH_THRESHOLD:
        return {
            "verdict": "recovery_signal",
            "message": (
                f"🔄 Recovery pattern: resembles {top_ticker} {top_year} "
                f"({top_phase}) at {top_sim:.0%} similarity. "
                f"Incumbent-transformation winner pattern."
            ),
            "top_match": top,
        }

    # Strong winner — but check for loser in 2nd/3rd
    if top_outcome == "winner" and top_sim >= STRONG_MATCH_THRESHOLD:
        loser_caution = None
        for m in matches[1:]:
            _, m_phase, m_sim, m_outcome, m_year, _ = m
            if m_outcome == "loser" and m_sim >= SECONDARY_LOSER_THRESHOLD:
                loser_caution = (m[0], m_year, m_sim)
                break

        msg = (
            f"🟢 Winner pattern: resembles {top_ticker} {top_year} "
            f"({top_phase}) at {top_sim:.0%} similarity."
        )
        if loser_caution:
            msg += (
                f" ⚠️ Note: also resembles {loser_caution[0]} {loser_caution[1]} "
                f"({loser_caution[2]:.0%}, loser) — proceed with care."
            )
        return {
            "verdict": "winner_pattern",
            "message": msg,
            "top_match": top,
        }

    # Mixed outcome match or moderate similarity
    if top_outcome == "mixed":
        return {
            "verdict": "mixed",
            "message": (
                f"〰️ Mixed pattern: resembles {top_ticker} {top_year} "
                f"({top_phase}) at {top_sim:.0%} similarity — historical outcome was MIXED."
            ),
            "top_match": top,
        }

    # Weak match — not a strong signal either way
    return {
        "verdict": "weak_match",
        "message": (
            f"Closest match: {top_ticker} {top_year} ({top_phase}) "
            f"at {top_sim:.0%} similarity. No strong pattern signal."
        ),
        "top_match": top,
    }


# ---------------------------------------------------------------
# Formatting helper for Telegram output
# ---------------------------------------------------------------
def format_matches_for_telegram(ticker: str, matches: list, interpretation: dict) -> str:
    """
    Build a multi-line Telegram message showing top matches + interpretation.
    """
    if not matches:
        return f"🔎 {ticker} — no comparable historical patterns found."

    lines = [f"🔎 *{ticker.upper()}* — Pattern Match"]
    fp = build_fingerprint(ticker)
    if fp and fp.get("bucket"):
        lines.append(f"Bucket: `{fp['bucket']}`")
    lines.append("")
    lines.append("*Top matches:*")

    outcome_emoji = {
        "winner": "🟢",
        "loser": "🔴",
        "mixed": "🟡",
    }

    for i, m in enumerate(matches, start=1):
        anchor_ticker, phase, sim, outcome, year, _ = m
        emoji = outcome_emoji.get(outcome, "⚪")
        lines.append(
            f"{i}. {emoji} {anchor_ticker} {year} ({phase}) — {sim:.0%}"
        )

    lines.append("")
    lines.append(interpretation["message"])

    return "\n".join(lines)


# ---------------------------------------------------------------
# CLI sanity test
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys

    test_tickers = sys.argv[1:] if len(sys.argv) > 1 else ["NVDA", "AAPL", "INTC", "JPM"]

    for ticker in test_tickers:
        print(f"\n{'=' * 60}")
        print(f"Testing: {ticker}")
        print("=" * 60)

        fp = build_fingerprint(ticker)
        if fp is None:
            print(f"  No fingerprint built for {ticker} (not in DB?)")
            continue

        print(f"  Bucket: {fp['bucket']}")
        print(f"  Revenue: ${fp['revenue']/1e9:.1f}B" if fp.get('revenue') else "  Revenue: N/A")
        print(f"  Growth: {fp['revenue_growth']*100:+.1f}%" if fp.get('revenue_growth') is not None else "  Growth: N/A")
        print(f"  Op margin: {fp['op_margin']*100:+.1f}%" if fp.get('op_margin') is not None else "  Op margin: N/A")

        matches = match(ticker, top_k=3)
        if not matches:
            print(f"  No matches found.")
            continue

        print(f"\n  Top 3 matches:")
        for i, m in enumerate(matches, start=1):
            t, phase, sim, outcome, year, _ = m
            print(f"    {i}. {t} {year} ({phase}) — {sim:.0%} [{outcome}]")

        interp = interpret_matches(matches)
        print(f"\n  Verdict: {interp['verdict']}")
        print(f"  {interp['message']}")