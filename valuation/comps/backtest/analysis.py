"""
Backtest metrics: correlation, decile spread, Sharpe, hit rate, R² of
predicted-vs-actual. See LAYER_2_COMPS_SPEC.md "Metrics" for definitions.

`rows` is a list of dicts, each: {ticker, date, predicted_upside,
actual_return_<W>m for each forward window W (only present if a forward
price was found)}.
"""

import numpy as np
from scipy import stats


def _pairs(rows, window):
    key = f"actual_return_{window}m"
    xs, ys = [], []
    for r in rows:
        if key in r and r.get("predicted_upside") is not None:
            xs.append(r["predicted_upside"])
            ys.append(r[key])
    return np.array(xs), np.array(ys)


def correlation(rows, window):
    """Pearson correlation of predicted_upside vs actual_return, and its p-value."""
    x, y = _pairs(rows, window)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None, None
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def hit_rate(rows, window):
    """Fraction of predictions where sign(predicted_upside) == sign(actual_return)."""
    x, y = _pairs(rows, window)
    if len(x) == 0:
        return None
    return float(np.mean(np.sign(x) == np.sign(y)))


def r_squared_predicted_vs_actual(rows, window):
    """Direct linear regression of actual return on predicted upside. Returns (r2, slope)."""
    x, y = _pairs(rows, window)
    if len(x) < 3 or np.std(x) == 0:
        return None, None
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else None
    return (float(r2) if r2 is not None else None), float(slope)


def _per_date_long_short_returns(rows, window):
    """
    For each rebalancing date, rank names by predicted_upside (known at
    that date — no look-ahead), form top/bottom decile equal-weight, and
    record the long-short spread of ACTUAL forward returns for that date.
    """
    key = f"actual_return_{window}m"
    by_date = {}
    for r in rows:
        if key in r and r.get("predicted_upside") is not None:
            by_date.setdefault(r["date"], []).append((r["predicted_upside"], r[key]))

    spreads = []
    for _date, pairs in by_date.items():
        if len(pairs) < 10:
            continue
        pairs = sorted(pairs, key=lambda p: p[0])
        decile_size = max(1, len(pairs) // 10)
        bottom = [p[1] for p in pairs[:decile_size]]
        top = [p[1] for p in pairs[-decile_size:]]
        spreads.append(float(np.mean(top) - np.mean(bottom)))
    return spreads


def decile_spread(rows, window):
    """Mean (across rebalancing dates) of top-decile minus bottom-decile actual return."""
    spreads = _per_date_long_short_returns(rows, window)
    if not spreads:
        return None
    return float(np.mean(spreads))


def sharpe(rows, window):
    """Annualized Sharpe of the long-top/short-bottom-decile monthly return series."""
    spreads = _per_date_long_short_returns(rows, window)
    if len(spreads) < 2:
        return None
    std = np.std(spreads)
    if std == 0:
        return None
    return float(np.mean(spreads) / std * np.sqrt(12))


def compute_all_metrics(rows, forward_windows):
    """Compute every metric for every forward window; keys match BacktestResult fields."""
    metrics = {}
    for w in forward_windows:
        corr, pval = correlation(rows, w)
        metrics[f"correlation_{w}m"] = corr
        metrics[f"correlation_{w}m_pvalue"] = pval
        metrics[f"decile_spread_{w}m"] = decile_spread(rows, w)
        metrics[f"sharpe_{w}m"] = sharpe(rows, w)
        metrics[f"hit_rate_{w}m"] = hit_rate(rows, w)
        r2, slope = r_squared_predicted_vs_actual(rows, w)
        metrics[f"r_squared_{w}m"] = r2
        metrics[f"slope_{w}m"] = slope
    return metrics
