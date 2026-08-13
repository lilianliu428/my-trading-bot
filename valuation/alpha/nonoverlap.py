"""
Non-overlapping-return evaluation, per methodology guidance from Prof.
Travis Johnson (McCombs).

The existing evaluation (evaluate.py + runner.py) correlates the signal at
date T against the 12-month-forward return from T. Consecutive monthly
dates share 11 of 12 months of that return window, so the ~30 rebalancing
dates are not independent observations and the reported t-stats overstate
significance.

The Hodrick insight this module implements: the overlap can be removed
without discarding the 12-month economic horizon, because

    cov(signal_t, sum of r over t+1..t+12)  ==  cov(sum of signal over
    t-12..t-1, r_t)

Averaging the signal over the trailing 12 months and correlating against
the (non-overlapping, by construction, since each window is exactly one
inter-rebalancing-date step) 1-month-forward return targets the same
underlying economics while eliminating the return-window overlap. Signal
persistence on its own is not the problem being fixed here — only
autocorrelation on the return side was; Newey-West standard errors are
included as a safeguard against any residual autocorrelation in the
resulting IC series (e.g. from signal persistence), not as a fix for a
different issue.

Fully additive: does not import-and-modify or duplicate operators.py,
alphas.py, or panel.py logic — every alpha is evaluated by calling the
SAME alphas.ALPHAS[alpha_id] functions the existing pipeline uses, either
on a feature-smoothed panel (variant a) or followed by smoothing its
output (variant b). The existing 12-month-forward evaluation path is
untouched and used here as-is for the "old" side of every comparison.
"""

import numpy as np
import pandas as pd

from valuation.alpha.panel import assemble_panel, FEATURE_COLUMNS
from valuation.alpha.alphas import ALPHAS
from valuation.alpha.evaluate import compute_ic_series, aggregate_ic_stats, newey_west_tstat
from valuation.alpha.runner import _classify_verdict

TRAILING_WINDOW = 12  # months
NW_LAGS = (2, 3)
NEW_FWD_COL = "fwd_return_1m"


def build_extended_panel(tickers, start_date, end_date, bucket_by_ticker,
                          snapshots_by_ticker, prices_by_ticker, spy_prices):
    """
    Thin wrapper around panel.assemble_panel (unmodified), requesting
    forward_windows=(1, 6, 12) so the panel carries fwd_return_1m
    alongside the existing fwd_return_6m/12m. Pass an `end_date` later
    than the original evaluation's where the cached price data supports
    it — a 1-month-forward target only needs one more month of price
    history per row, not twelve, so the usable panel can extend further
    at the tail than the original 12-month-forward evaluation could.

    Returns (panel_df, data_flags), same shape as assemble_panel. Rows for
    dates within the original evaluation's window are numerically
    identical to that run (same start_date, same per-row computation) —
    filter panel_df["date"] <= original_end_date to reproduce it exactly,
    which is how this module gets its "old" comparison baseline without a
    second fetch or a duplicated build.
    """
    return assemble_panel(
        tickers, start_date, end_date, bucket_by_ticker,
        snapshots_by_ticker, prices_by_ticker, spy_prices,
        forward_windows=(1, 6, 12),
    )


def _sorted_by_ticker_date(df):
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def smooth_feature_columns(panel_df, window=TRAILING_WINDOW):
    """
    Variant (a) prep: replace each of FEATURE_COLUMNS with its trailing
    `window`-month rolling average, per ticker. Trailing/right-aligned by
    construction (pandas .rolling default: window [i-window+1, i]), so
    this never looks ahead. min_periods=window (a full window required,
    no partial-window averages) for exactness against the Hodrick sum's
    exact `window`-period span.

    Sparse features (e.g. sbc_dilution_rate, ~23% coverage in this panel)
    will mostly go NaN under a strict min_periods=window requirement,
    since it needs `window` CONSECUTIVE non-null monthly rows — reported
    as reduced n_dates for the affected alpha rather than silently
    relaxed.
    """
    df = _sorted_by_ticker_date(panel_df).copy()
    for col in FEATURE_COLUMNS:
        df[col] = (
            df.groupby("ticker")[col]
              .rolling(window=window, min_periods=window)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df


def compute_variant_a_alpha(alpha_id, panel_df, window=TRAILING_WINDOW, alpha_col="_alpha"):
    """
    Variant (a): rank(trailing_avg(raw_feature)) — average the raw feature
    over the trailing window, THEN run the existing, unmodified alpha
    construction function on the smoothed panel. More faithful to the
    Hodrick equation's left-hand-side sum.

    Returns the ticker/date-sorted panel with a new `alpha_col` column —
    not a bare Series — so every downstream consumer reads date/ticker/
    fwd_return_* from the same frame the alpha was computed on, with no
    separate index-alignment step required.
    """
    smoothed = smooth_feature_columns(panel_df, window)
    smoothed[alpha_col] = ALPHAS[alpha_id](smoothed)
    return smoothed


def compute_variant_b_alpha(alpha_id, panel_df, window=TRAILING_WINDOW, alpha_col="_alpha"):
    """
    Variant (b): trailing_avg(rank(raw_feature)) — run the existing,
    unmodified alpha construction on the panel AS-IS (exactly today's
    standard alpha value), then take ITS trailing rolling average per
    ticker. More robust to outliers in the raw feature, since the
    rank/neutralize/center step already ran before any averaging.

    Returns the ticker/date-sorted panel with a new `alpha_col` column,
    same convention as compute_variant_a_alpha.
    """
    df = _sorted_by_ticker_date(panel_df).copy()
    df["_alpha_raw"] = ALPHAS[alpha_id](df)
    df[alpha_col] = (
        df.groupby("ticker")["_alpha_raw"]
          .rolling(window=window, min_periods=window)
          .mean()
          .reset_index(level=0, drop=True)
    )
    return df.drop(columns=["_alpha_raw"])


def evaluate_new_spec(augmented_df, alpha_col="_alpha", fwd_col=NEW_FWD_COL,
                       date_col="date", lags=NW_LAGS):
    """
    IC-based evaluation of a precomputed alpha column (from
    compute_variant_a_alpha / compute_variant_b_alpha) against the
    non-overlapping forward-return column. Reuses compute_ic_series /
    aggregate_ic_stats unmodified for the naive stats, then adds
    Newey-West t-stats (keyed "nw_tstat_{lag}") for each lag in `lags`,
    computed on the SAME per-date IC series in chronological order.
    """
    ic_by_date, skipped = compute_ic_series(augmented_df, alpha_col, fwd_col, date_col)
    result = aggregate_ic_stats(ic_by_date)

    dates_sorted = sorted(ic_by_date.keys())
    ic_chrono = [ic_by_date[d] for d in dates_sorted]
    for lag in lags:
        result[f"nw_tstat_{lag}"] = newey_west_tstat(ic_chrono, lag) if ic_chrono else None

    result["skipped_dates"] = skipped
    result["ic_by_date"] = ic_by_date
    return result


def _sign(x):
    if x is None:
        return None
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def run_comparison_table(old_panel_df, new_panel_df, alpha_ids=None,
                          window=TRAILING_WINDOW, lags=NW_LAGS):
    """
    Old vs. new(a) vs. new(b), all 18 pre-declared alphas.

    old_panel_df: evaluated exactly as the existing pipeline does (naive
    IC/t-stat against fwd_return_12m) — this path is not changed, it's
    the thing being critiqued, kept as the "old" comparison basis.
    new_panel_df: the extended panel (fwd_return_1m present) used for
    both new variants.

    Returns a DataFrame sorted by new_ic_ir_a (variant a is the one
    described as more faithful to the Hodrick equation, so it's the
    primary ranking column; both variants' full stats are included).
    """
    if alpha_ids is None:
        alpha_ids = list(ALPHAS.keys())

    rows = []
    for alpha_id in alpha_ids:
        old_ic_by_date, _ = compute_ic_series(old_panel_df, alpha_id, "fwd_return_12m", "date")
        old_stats = aggregate_ic_stats(old_ic_by_date)

        aug_a = compute_variant_a_alpha(alpha_id, new_panel_df, window)
        stats_a = evaluate_new_spec(aug_a, "_alpha", NEW_FWD_COL, "date", lags)

        aug_b = compute_variant_b_alpha(alpha_id, new_panel_df, window)
        stats_b = evaluate_new_spec(aug_b, "_alpha", NEW_FWD_COL, "date", lags)

        sign_old, sign_a, sign_b = _sign(old_stats["mean_ic"]), _sign(stats_a["mean_ic"]), _sign(stats_b["mean_ic"])

        rows.append({
            "alpha_id": alpha_id,
            "old_mean_ic": old_stats["mean_ic"], "old_tstat": old_stats["ic_tstat"],
            "old_n_dates": old_stats["n_dates"],
            "new_mean_ic_a": stats_a["mean_ic"], "new_ic_ir_a": stats_a["ic_ir"],
            "new_tstat_naive_a": stats_a["ic_tstat"],
            "new_tstat_nw2_a": stats_a.get("nw_tstat_2"), "new_tstat_nw3_a": stats_a.get("nw_tstat_3"),
            "n_dates_a": stats_a["n_dates"],
            "new_mean_ic_b": stats_b["mean_ic"], "new_ic_ir_b": stats_b["ic_ir"],
            "new_tstat_naive_b": stats_b["ic_tstat"],
            "new_tstat_nw2_b": stats_b.get("nw_tstat_2"), "new_tstat_nw3_b": stats_b.get("nw_tstat_3"),
            "n_dates_b": stats_b["n_dates"],
            "sign_flip_a": bool(sign_old is not None and sign_a is not None and sign_old != 0 and sign_a != 0 and sign_old != sign_a),
            "sign_flip_b": bool(sign_old is not None and sign_b is not None and sign_old != 0 and sign_b != 0 and sign_old != sign_b),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("new_ic_ir_a", ascending=False, na_position="last").reset_index(drop=True)


def run_holdout_new_spec(new_panel_df, alpha_ids=None, variant="a", window=TRAILING_WINDOW, top_n=3):
    """
    Chronological holdout under the new (trailing-signal, non-overlapping
    1m-return) specification — same structure as
    runner.holdout_analysis: split dates in half (no shuffling), rank by
    ic_ir on the early half only, evaluate the top `top_n` on the late
    half with no refitting.
    """
    if alpha_ids is None:
        alpha_ids = list(ALPHAS.keys())
    compute_variant = compute_variant_a_alpha if variant == "a" else compute_variant_b_alpha

    dates = sorted(new_panel_df["date"].unique())
    mid = len(dates) // 2
    early_dates, late_dates = set(dates[:mid]), set(dates[mid:])

    early_rows = []
    aug_by_alpha = {}
    for alpha_id in alpha_ids:
        aug = compute_variant(alpha_id, new_panel_df, window)
        aug_by_alpha[alpha_id] = aug
        early_stats = evaluate_new_spec(aug[aug["date"].isin(early_dates)], "_alpha", NEW_FWD_COL, "date")
        early_rows.append({
            "alpha_id": alpha_id, "early_mean_ic": early_stats["mean_ic"],
            "early_ic_ir": early_stats["ic_ir"], "early_n_dates": early_stats["n_dates"],
        })

    early_ranking = pd.DataFrame(early_rows).sort_values(
        "early_ic_ir", ascending=False, na_position="last").reset_index(drop=True)
    top_alphas = early_ranking.dropna(subset=["early_ic_ir"]).head(top_n)["alpha_id"].tolist()

    holdout_rows = []
    for alpha_id in top_alphas:
        aug = aug_by_alpha[alpha_id]
        late_stats = evaluate_new_spec(aug[aug["date"].isin(late_dates)], "_alpha", NEW_FWD_COL, "date")
        early_row = early_ranking.loc[early_ranking["alpha_id"] == alpha_id].iloc[0]

        sign_held = (
            np.sign(early_row["early_mean_ic"]) == np.sign(late_stats["mean_ic"])
            if early_row["early_mean_ic"] is not None and late_stats["mean_ic"] is not None
            else None
        )
        verdict = _classify_verdict(
            early_row["early_ic_ir"], late_stats["ic_ir"],
            early_row["early_mean_ic"], late_stats["mean_ic"],
        )
        holdout_rows.append({
            "alpha_id": alpha_id,
            "early_mean_ic": early_row["early_mean_ic"], "early_ic_ir": early_row["early_ic_ir"],
            "late_mean_ic": late_stats["mean_ic"], "late_ic_ir": late_stats["ic_ir"],
            "late_n_dates": late_stats["n_dates"],
            "sign_held": sign_held, "verdict": verdict,
        })

    return {
        "early_ranking": early_ranking,
        "holdout_table": pd.DataFrame(holdout_rows),
        "n_early_dates": len(early_dates), "n_late_dates": len(late_dates),
        "variant": variant,
    }
