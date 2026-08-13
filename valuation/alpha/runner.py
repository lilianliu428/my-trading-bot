"""
Orchestrates the alpha research layer: assemble the panel, compute all 18
pre-declared alphas, evaluate them (full-period main table + chronological
holdout), and produce the report tables. See ALPHA_LAYER_SPEC.md "Output".
"""

import datetime
import json

import numpy as np
import pandas as pd

from valuation.alpha.panel import build_panel, build_bucket_map, coverage_report, FORWARD_WINDOWS
from valuation.alpha.alphas import ALPHAS, CATEGORY_A_ALPHA_IDS, CATEGORY_B_ALPHA_IDS
from valuation.alpha.evaluate import evaluate_alpha
from valuation.comps.universe import COMPS_UNIVERSE

HORIZONS = (6, 12)
HOLDOUT_TOP_N = 3


def get_universe_and_buckets():
    """Union of the semiconductors + mature_tech universes (the two buckets
    this spec tests), each ticker mapped to its one primary bucket."""
    bucket_map = build_bucket_map()
    tickers = sorted(set(COMPS_UNIVERSE.get("semiconductors", [])) | set(COMPS_UNIVERSE.get("mature_tech", [])))
    return tickers, bucket_map


def assemble_panel_with_alphas(tickers, start_date, end_date, bucket_by_ticker):
    """Build the panel, then add all 18 alpha columns to it."""
    panel_df, data_flags = build_panel(tickers, start_date, end_date, bucket_by_ticker)
    for alpha_id, fn in ALPHAS.items():
        panel_df[alpha_id] = fn(panel_df)
    return panel_df, data_flags


def main_results_table(panel_df, horizons=HORIZONS):
    """All 18 alphas x both horizons, evaluated on the full period."""
    rows = []
    for alpha_id in ALPHAS:
        metrics = evaluate_alpha(panel_df, alpha_id, horizons=horizons)
        for w in horizons:
            m = metrics[w]
            rows.append({
                "alpha_id": alpha_id, "horizon": w,
                "mean_ic": m["mean_ic"], "std_ic": m["std_ic"], "ic_ir": m["ic_ir"],
                "ic_tstat": m["ic_tstat"], "pct_positive": m["pct_positive"],
                "sharpe": m["sharpe"], "decile_spread": m["decile_spread"],
                "hit_rate": m["hit_rate"], "hit_rate_extremes": m["hit_rate_extremes"],
                "turnover": metrics["turnover"], "n_dates": m["n_dates"],
                "n_skipped_dates": len(m["skipped_dates"]),
            })
    df = pd.DataFrame(rows)
    df = df.sort_values("ic_ir", ascending=False, na_position="last").reset_index(drop=True)
    return df


def _classify_verdict(early_ic_ir, late_ic_ir, early_mean_ic, late_mean_ic):
    if early_ic_ir is None or late_ic_ir is None or early_mean_ic is None or late_mean_ic is None:
        return "N/A (insufficient data)"
    if np.sign(early_mean_ic) != np.sign(late_mean_ic) and early_mean_ic != 0 and late_mean_ic != 0:
        return "Flips"
    if abs(late_ic_ir) >= 0.5 * abs(early_ic_ir):
        return "Holds"
    return "Degrades"


def holdout_analysis(panel_df, horizons=HORIZONS, top_n=HOLDOUT_TOP_N):
    """
    Chronological holdout: split rebalancing dates in half (no shuffling),
    rank alphas by ic_ir on the early half only, take the top `top_n`, and
    evaluate those (only) on the late half with no refitting.

    Returns dict {horizon: {"early_ranking": df, "holdout_table": df}}.
    """
    dates = sorted(panel_df["date"].unique())
    mid = len(dates) // 2
    early_dates, late_dates = set(dates[:mid]), set(dates[mid:])
    early_df = panel_df[panel_df["date"].isin(early_dates)]
    late_df = panel_df[panel_df["date"].isin(late_dates)]

    result = {}
    for w in horizons:
        early_rows = []
        for alpha_id in ALPHAS:
            m = evaluate_alpha(early_df, alpha_id, horizons=(w,))[w]
            early_rows.append({"alpha_id": alpha_id, "early_mean_ic": m["mean_ic"],
                                "early_ic_ir": m["ic_ir"], "early_n_dates": m["n_dates"]})
        early_ranking = pd.DataFrame(early_rows).sort_values(
            "early_ic_ir", ascending=False, na_position="last").reset_index(drop=True)

        top_alphas = early_ranking.dropna(subset=["early_ic_ir"]).head(top_n)["alpha_id"].tolist()

        holdout_rows = []
        for alpha_id in top_alphas:
            early_row = early_ranking[early_ranking["alpha_id"] == alpha_id].iloc[0]
            m_late = evaluate_alpha(late_df, alpha_id, horizons=(w,))[w]
            sign_held = (
                np.sign(early_row["early_mean_ic"]) == np.sign(m_late["mean_ic"])
                if early_row["early_mean_ic"] is not None and m_late["mean_ic"] is not None
                else None
            )
            verdict = _classify_verdict(early_row["early_ic_ir"], m_late["ic_ir"],
                                         early_row["early_mean_ic"], m_late["mean_ic"])
            holdout_rows.append({
                "alpha_id": alpha_id,
                "early_mean_ic": early_row["early_mean_ic"], "early_ic_ir": early_row["early_ic_ir"],
                "late_mean_ic": m_late["mean_ic"], "late_ic_ir": m_late["ic_ir"],
                "late_n_dates": m_late["n_dates"],
                "sign_held": sign_held, "verdict": verdict,
            })
        holdout_table = pd.DataFrame(holdout_rows)
        result[w] = {"early_ranking": early_ranking, "holdout_table": holdout_table,
                     "n_early_dates": len(early_dates), "n_late_dates": len(late_dates)}
    return result


def derived_vs_baseline_table(main_table):
    """A1/A1n vs A8/A8n (ex-goodwill vs reported ROIC), C1 vs C3."""
    pairs = [("A1", "A8"), ("A1n", "A8n"), ("C1", "C3")]
    rows = []
    for derived_id, baseline_id in pairs:
        for w in HORIZONS:
            d = main_table[(main_table.alpha_id == derived_id) & (main_table.horizon == w)]
            b = main_table[(main_table.alpha_id == baseline_id) & (main_table.horizon == w)]
            if d.empty or b.empty:
                continue
            d, b = d.iloc[0], b.iloc[0]
            rows.append({
                "derived": derived_id, "baseline": baseline_id, "horizon": w,
                "derived_ic_ir": d["ic_ir"], "baseline_ic_ir": b["ic_ir"],
                "derived_mean_ic": d["mean_ic"], "baseline_mean_ic": b["mean_ic"],
                "derived_tstat": d["ic_tstat"], "baseline_tstat": b["ic_tstat"],
                "derived_beats_baseline": (
                    d["ic_ir"] > b["ic_ir"] if pd.notna(d["ic_ir"]) and pd.notna(b["ic_ir"]) else None
                ),
            })
    return pd.DataFrame(rows)


def save_panel(panel_df, path):
    panel_df.to_pickle(path)


def load_panel(path):
    return pd.read_pickle(path)
