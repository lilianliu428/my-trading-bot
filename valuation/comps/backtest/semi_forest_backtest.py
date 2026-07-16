"""
Backtest: does semi_forest.py's validated RandomForest comps model (HIGH
confidence, hold-out R²=0.330 EV/Sales / 0.598 EV/EBITDA, per the
tree_regression.py diagnostic session) actually predict FORWARD RETURNS
for semiconductor tickers — not just cross-sectional multiples, which is
all the original validation tested?

Point-in-time cross-sectional backtest: at each monthly rebalancing date T,
reconstitute the universe from point-in-time snapshots (only tickers with
data as-of T — survivorship-safe), fit semi_forest's EXACT validated
RandomForest spec (8 features, max_depth 7 for EV/Sales / 5 for EV/EBITDA,
n_estimators=100 — reuses fit_semi_forest/predict_semi_forest UNCHANGED,
no reimplementation of the model) on that cross-section, predict each
ticker's fair value, and look forward 6m/12m for actual returns.

Reuses unchanged: get_point_in_time_snapshots (extended this session with
the 4 extra features semi_forest needs — see historical_data.py),
fetch_price_series, nearest_price; _monthly_dates (framework.py);
compute_all_metrics and every metric function (analysis.py); fit_semi_forest,
predict_semi_forest (models/semi_forest.py) — the model itself is untouched.

Read directly: python3 -m valuation.comps.backtest.semi_forest_backtest
"""

import datetime

from valuation.comps.backtest.historical_data import get_point_in_time_snapshots, fetch_price_series, nearest_price
from valuation.comps.backtest.framework import _monthly_dates
from valuation.comps.backtest import analysis
from valuation.comps.regression import MultipleObservation, MIN_OBSERVATIONS
from valuation.comps.models.semi_forest import fit_semi_forest, predict_semi_forest

THIN_UNIVERSE_THRESHOLD = 15  # cross-section below this ~= beyond the comfortable training distribution the 64-ticker universe validated on


def run_semi_forest_backtest(tickers, start_date, end_date, forward_windows=(6, 12)):
    data_flags = []
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    max_fwd_months = max(forward_windows)

    snapshots_by_ticker, prices_by_ticker = {}, {}
    for t in tickers:
        snaps, flags = get_point_in_time_snapshots(t)
        data_flags.extend(f"[{t}] {f}" for f in flags)
        snapshots_by_ticker[t] = snaps
        price_start = start - datetime.timedelta(days=14)
        price_end = end + datetime.timedelta(days=32 * max_fwd_months)
        prices_by_ticker[t] = fetch_price_series(t, price_start, price_end)

    all_as_of = [s["as_of_date"] for snaps in snapshots_by_ticker.values() for s in snaps]
    if not all_as_of:
        data_flags.append("No point-in-time snapshots available for any ticker — backtest cannot run")
        return {"master_rows": [], "data_flags": data_flags, "n_predictions": 0, "n_rebalancing_dates": 0}

    available_start = max(start, min(all_as_of))
    if available_start > start:
        data_flags.append(f"Requested start {start_date} predates available fundamentals — clipped to {available_start.isoformat()}")

    rebal_dates = [d for d in _monthly_dates(available_start, end) if d <= end]

    master_rows = []
    n_thin_dates, n_skipped_dates, cross_section_sizes = 0, 0, []

    for T in rebal_dates:
        cross_section = {}
        for t in tickers:
            usable = [s for s in snapshots_by_ticker[t] if s["as_of_date"] <= T]
            if usable:
                cross_section[t] = usable[-1]

        if len(cross_section) < MIN_OBSERVATIONS:
            n_skipped_dates += 1
            continue
        cross_section_sizes.append(len(cross_section))
        if len(cross_section) < THIN_UNIVERSE_THRESHOLD:
            n_thin_dates += 1

        ev_sales_obs, ev_ebitda_obs, prices_at_T = [], [], {}
        for t, snap in cross_section.items():
            price_T = nearest_price(prices_by_ticker[t], T)
            shares, revenue, ebitda = snap["shares_outstanding"], snap["features"].revenue_ttm, snap["features"].ebitda_ttm
            if price_T is None or not shares or not revenue or revenue <= 0:
                continue
            prices_at_T[t] = price_T
            ev = price_T * shares + (snap["total_debt"] or 0) - (snap["cash"] or 0)
            if ev / revenue > 0:
                ev_sales_obs.append(MultipleObservation(t, "ev_sales", ev / revenue, snap["features"]))
            if ebitda is not None and ebitda > 0 and ev / ebitda > 0:
                ev_ebitda_obs.append(MultipleObservation(t, "ev_ebitda", ev / ebitda, snap["features"]))

        if len(ev_sales_obs) < MIN_OBSERVATIONS:
            n_skipped_dates += 1
            continue

        # fit semi_forest's exact validated spec — reused unchanged, no live hyperparameter search
        ev_sales_model, _f1 = fit_semi_forest(ev_sales_obs, "ev_sales")
        ev_ebitda_model, _f2 = (fit_semi_forest(ev_ebitda_obs, "ev_ebitda") if len(ev_ebitda_obs) >= MIN_OBSERVATIONS else (None, []))

        for t, snap in cross_section.items():
            if t not in prices_at_T:
                continue
            price_T = prices_at_T[t]
            revenue, ebitda, shares = snap["features"].revenue_ttm, snap["features"].ebitda_ttm, snap["shares_outstanding"]
            debt, cash = snap["total_debt"] or 0, snap["cash"] or 0

            upside_ev_sales, upside_ev_ebitda = None, None

            pred_es, _missing, _flags = predict_semi_forest(ev_sales_model, snap["features"])
            if pred_es is not None and revenue and shares:
                fv = (pred_es * revenue - debt + cash) / shares
                upside_ev_sales = fv / price_T - 1

            if ev_ebitda_model is not None:
                pred_ee, _missing, _flags = predict_semi_forest(ev_ebitda_model, snap["features"])
                if pred_ee is not None and ebitda and ebitda > 0 and shares:
                    fv = (pred_ee * ebitda - debt + cash) / shares
                    upside_ev_ebitda = fv / price_T - 1

            if upside_ev_sales is None and upside_ev_ebitda is None:
                continue
            avail = [u for u in (upside_ev_sales, upside_ev_ebitda) if u is not None]
            upside_avg = sum(avail) / len(avail)

            row = {
                "ticker": t, "date": T,
                "upside_ev_sales": upside_ev_sales, "upside_ev_ebitda": upside_ev_ebitda, "upside_avg": upside_avg,
            }
            for w in forward_windows:
                future_date = T + datetime.timedelta(days=30 * w)
                price_future = nearest_price(prices_by_ticker[t], future_date)
                if price_future is not None:
                    row[f"actual_return_{w}m"] = price_future / price_T - 1
            master_rows.append(row)

    avg_cross_section = sum(cross_section_sizes) / len(cross_section_sizes) if cross_section_sizes else 0
    data_flags.append(
        f"{len(master_rows)} (ticker, date) predictions across {len(rebal_dates) - n_skipped_dates} of {len(rebal_dates)} "
        f"rebalancing dates ({n_skipped_dates} skipped for cross-section < {MIN_OBSERVATIONS}); "
        f"avg cross-section size={avg_cross_section:.1f}; {n_thin_dates} dates had a thin cross-section "
        f"(< {THIN_UNIVERSE_THRESHOLD} names — meaningfully smaller than the 64-ticker universe semi_forest was validated on, "
        f"predictions there are more of an extrapolation than a like-for-like reproduction of the validated model)"
    )

    return {
        "master_rows": master_rows, "data_flags": data_flags,
        "n_predictions": len(master_rows), "n_rebalancing_dates": len(rebal_dates),
        "n_thin_dates": n_thin_dates, "period_start": available_start.isoformat(), "period_end": end_date,
    }


def _rows_for_source(master_rows, source_key, forward_windows):
    rows = []
    for r in master_rows:
        if r.get(source_key) is None:
            continue
        row = {"ticker": r["ticker"], "date": r["date"], "predicted_upside": r[source_key]}
        for w in forward_windows:
            key = f"actual_return_{w}m"
            if key in r:
                row[key] = r[key]
        rows.append(row)
    return rows


def compute_metrics_by_source(master_rows, forward_windows=(6, 12)):
    """Returns {source: compute_all_metrics(...)} for ev_sales, ev_ebitda, and averaged upside."""
    return {
        source: analysis.compute_all_metrics(_rows_for_source(master_rows, source, forward_windows), list(forward_windows))
        for source in ("upside_ev_sales", "upside_ev_ebitda", "upside_avg")
    }


# ════════════════════════════════════════════════════════════════════════
# Sub-type analysis and reporting — reuses the verdict/success-criteria
# framework built for the mature_tech DCF backtest (generic: operates on
# a metrics dict, doesn't care whether upside came from a DCF or comps).
# ════════════════════════════════════════════════════════════════════════

from valuation.comps.backtest.dcf_approximation import print_metrics_report  # noqa: E402

# "AI infra (NVDA, AMD, some analog names)" in the task is ambiguous about
# which analog names — TXN/ADI/MRVL are already assigned to "stable" per
# the task's own stable-semis list, so AI infra is kept to the two
# unambiguous names rather than guessing at an overlap.
STABLE_SEMIS = ["AVGO", "TXN", "ADI", "MRVL", "MPWR"]
CYCLICAL_MEMORY = ["MU", "WDC", "STX"]
AI_INFRA = ["NVDA", "AMD"]
EQUIPMENT = ["AMAT", "LRCX", "KLAC"]

SUBTYPE_MAP = {}
for _t in STABLE_SEMIS:
    SUBTYPE_MAP[_t] = "stable"
for _t in CYCLICAL_MEMORY:
    SUBTYPE_MAP[_t] = "cyclical_memory"
for _t in AI_INFRA:
    SUBTYPE_MAP[_t] = "ai_infra"
for _t in EQUIPMENT:
    SUBTYPE_MAP[_t] = "equipment"


def run_full_analysis():
    from valuation.comps.universe import get_universe

    tickers = get_universe("semiconductors")
    print(f"Universe: {len(tickers)} semiconductor tickers")
    print(f"  Stable ({len(STABLE_SEMIS)}): {STABLE_SEMIS}")
    print(f"  Cyclical/memory ({len(CYCLICAL_MEMORY)}): {CYCLICAL_MEMORY}")
    print(f"  AI infra ({len(AI_INFRA)}): {AI_INFRA}")
    print(f"  Equipment ({len(EQUIPMENT)}): {EQUIPMENT}")

    result = run_semi_forest_backtest(tickers, "2022-01-01", "2025-12-31", forward_windows=[6, 12])
    print(f"\nPeriod: {result['period_start']} -> {result['period_end']}")
    print(f"Rebalancing dates: {result['n_rebalancing_dates']}   Total predictions: {result['n_predictions']}")
    print("\nData flags (last 5):")
    for f in result["data_flags"][-5:]:
        print(f"  {f}")

    master_rows = result["master_rows"]

    print(f"\n{'=' * 100}\nOVERALL — all semiconductors, by prediction source\n{'=' * 100}")
    overall_by_source = compute_metrics_by_source(master_rows)
    for source, metrics in overall_by_source.items():
        n = len(_rows_for_source(master_rows, source, [6, 12]))
        print_metrics_report(f"Source: {source}", metrics, n)

    print(f"\n{'=' * 100}\nSUB-TYPE ANALYSIS (averaged EV/Sales + EV/EBITDA upside)\n{'=' * 100}")
    subtype_results = {}
    for label, subtype_tickers in [("STABLE", STABLE_SEMIS), ("CYCLICAL/MEMORY", CYCLICAL_MEMORY),
                                    ("AI INFRA", AI_INFRA), ("EQUIPMENT", EQUIPMENT)]:
        sub_master = [r for r in master_rows if r["ticker"] in subtype_tickers]
        sub_rows = _rows_for_source(sub_master, "upside_avg", [6, 12])
        metrics = analysis.compute_all_metrics(sub_rows, [6, 12])
        print_metrics_report(f"{label} (n_predictions={len(sub_rows)})", metrics, len(sub_rows))
        subtype_results[label] = metrics

    return {"overall_by_source": overall_by_source, "subtype_results": subtype_results, "backtest_result": result}


if __name__ == "__main__":
    run_full_analysis()
