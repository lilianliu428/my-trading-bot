"""
The 18 pre-declared alphas from ALPHA_LAYER_SPEC.md "Alpha construction".
This list is fixed — no alphas added after seeing results (that's the
selection-bias failure mode the spec calls out from the Layer 2
feature-selection sessions).

Each alpha function takes the assembled panel DataFrame (valuation/alpha/
panel.py::build_panel output) and returns a pd.Series aligned to the
DataFrame's index: one alpha value per (ticker, date) row. Positive means
long, per the spec's sign convention.
"""

import pandas as pd

from valuation.alpha.operators import rank, zscore, neutralize

DATE_COL = "date"
BUCKET_COL = "bucket"


def _center(df, values, date_col=DATE_COL):
    """
    Subtract the per-date mean so the alpha column sums to ~0 within each
    date (construction step 4 in the spec). This is a per-date additive
    shift: it does not change rank order, so Spearman IC and decile-based
    portfolio metrics are identical with or without it. It exists purely so
    an alpha's sign is meaningful (positive = above that date's average
    tilt), matching the spec's "positive means long" convention.
    """
    tmp = pd.DataFrame({date_col: df[date_col].values, "_v": values.values}, index=df.index)
    group_mean = tmp.groupby(date_col)["_v"].transform("mean")
    return values - group_mean


def _raw(df, col, negate=False):
    """Step 1+3+4 for a single-feature raw alpha: rank, optionally negate, center."""
    work = df[[DATE_COL, col]].copy()
    r = rank(work, col, DATE_COL)
    if negate:
        r = -r
    return _center(df, r)


def _neutralized(df, col, negate=False):
    """Step 1+2+3+4: neutralize by bucket, rank, optionally negate, center."""
    work = df[[DATE_COL, BUCKET_COL, col]].copy()
    work["_neut"] = neutralize(work, col, BUCKET_COL, DATE_COL)
    r = rank(work, "_neut", DATE_COL)
    if negate:
        r = -r
    return _center(df, r)


# ════════════════════════════════════════════════════════════════════════
# Single-feature alphas (raw)
# ════════════════════════════════════════════════════════════════════════

def A1(df):
    """rank(ex_goodwill_roic)"""
    return _raw(df, "ex_goodwill_roic")


def A2(df):
    """rank(rd_capitalized_margin)"""
    return _raw(df, "rd_capitalized_margin")


def A3(df):
    """rank(normalized_reinvestment)"""
    return _raw(df, "normalized_reinvestment")


def A4(df):
    """-rank(sbc_dilution_rate) — dilution is bad"""
    return _raw(df, "sbc_dilution_rate", negate=True)


def A5(df):
    """rank(dcf_upside)"""
    return _raw(df, "dcf_upside")


def A6(df):
    """rank(gross_margin) — baseline"""
    return _raw(df, "gross_margin")


def A7(df):
    """rank(trailing_growth) — baseline"""
    return _raw(df, "trailing_growth")


def A8(df):
    """rank(roic) — baseline"""
    return _raw(df, "roic")


def A9(df):
    """-rank(ev_ebitda) — baseline value"""
    return _raw(df, "ev_ebitda", negate=True)


# ════════════════════════════════════════════════════════════════════════
# Bucket-neutralized versions
# ════════════════════════════════════════════════════════════════════════

def A1n(df):
    """rank(neutralize(ex_goodwill_roic, bucket))"""
    return _neutralized(df, "ex_goodwill_roic")


def A2n(df):
    """rank(neutralize(rd_capitalized_margin, bucket))"""
    return _neutralized(df, "rd_capitalized_margin")


def A3n(df):
    """rank(neutralize(normalized_reinvestment, bucket))"""
    return _neutralized(df, "normalized_reinvestment")


def A5n(df):
    """rank(neutralize(dcf_upside, bucket))"""
    return _neutralized(df, "dcf_upside")


def A8n(df):
    """rank(neutralize(roic, bucket)) — baseline"""
    return _neutralized(df, "roic")


def A9n(df):
    """-rank(neutralize(ev_ebitda, bucket)) — baseline"""
    return _neutralized(df, "ev_ebitda", negate=True)


# ════════════════════════════════════════════════════════════════════════
# Composites
# ════════════════════════════════════════════════════════════════════════

def C1(df):
    """zscore(A1n) + zscore(A9n) — derived quality + value"""
    work = df[[DATE_COL]].copy()
    work["A1n"] = A1n(df)
    work["A9n"] = A9n(df)
    return _center(df, zscore(work, "A1n", DATE_COL) + zscore(work, "A9n", DATE_COL))


def C2(df):
    """zscore(A1n) + zscore(A2n) — the two most distinctive derived features"""
    work = df[[DATE_COL]].copy()
    work["A1n"] = A1n(df)
    work["A2n"] = A2n(df)
    return _center(df, zscore(work, "A1n", DATE_COL) + zscore(work, "A2n", DATE_COL))


def C3(df):
    """zscore(A8n) + zscore(A9n) — baseline quality + value (contrast with C1)"""
    work = df[[DATE_COL]].copy()
    work["A8n"] = A8n(df)
    work["A9n"] = A9n(df)
    return _center(df, zscore(work, "A8n", DATE_COL) + zscore(work, "A9n", DATE_COL))


ALPHAS = {
    "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5,
    "A6": A6, "A7": A7, "A8": A8, "A9": A9,
    "A1n": A1n, "A2n": A2n, "A3n": A3n, "A5n": A5n, "A8n": A8n, "A9n": A9n,
    "C1": C1, "C2": C2, "C3": C3,
}

assert len(ALPHAS) == 18, f"expected 18 pre-declared alphas, got {len(ALPHAS)}"

CATEGORY_A_ALPHA_IDS = {"A1", "A2", "A3", "A4", "A5", "A1n", "A2n", "A3n", "A5n", "C1", "C2"}
CATEGORY_B_ALPHA_IDS = {"A6", "A7", "A8", "A9", "A8n", "A9n", "C3"}
