"""
Evaluation for the alpha research layer: per-date Spearman rank IC,
aggregate IC statistics, long-short portfolio metrics, and turnover.
See ALPHA_LAYER_SPEC.md "Evaluation".
"""

import numpy as np
import pandas as pd
from scipy import stats

from valuation.alpha.operators import rank

MIN_TICKERS_FOR_IC = 15
MIN_TICKERS_FOR_DECILE = 20


def compute_ic_series(df, alpha_col, fwd_col, date_col="date"):
    """
    Per-date Spearman IC between alpha_col and fwd_col. Drops rows with
    missing alpha or forward return on that date. Dates with fewer than
    MIN_TICKERS_FOR_IC valid tickers are skipped (not silently dropped —
    returned in skipped_dates for the caller to report).

    Returns (ic_by_date: {date: ic}, skipped_dates: [(date, n_valid), ...]).
    """
    ic_by_date = {}
    skipped = []
    for date, g in df.groupby(date_col):
        sub = g[[alpha_col, fwd_col]].dropna()
        if len(sub) < MIN_TICKERS_FOR_IC:
            skipped.append((date, len(sub)))
            continue
        if sub[alpha_col].std() == 0 or sub[fwd_col].std() == 0:
            skipped.append((date, len(sub)))
            continue
        rho, _p = stats.spearmanr(sub[alpha_col], sub[fwd_col])
        if not np.isnan(rho):
            ic_by_date[date] = float(rho)
        else:
            skipped.append((date, len(sub)))
    return ic_by_date, skipped


def newey_west_se_of_mean(values, lags):
    """
    Newey-West (1987) HAC standard error of the sample mean of a time
    series, Bartlett kernel. `values` must already be in chronological
    order — this is purely a function of lag structure, so a shuffled
    input silently gives a meaningless answer.

    Used as a safeguard against autocorrelation in the IC series ITSELF
    (e.g. from a persistent underlying signal), which non-overlapping
    returns do not by themselves fix — that only removes the
    return-window overlap, not signal persistence. lags=0 reduces to the
    ordinary (non-HAC) standard error of the mean.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        return None
    demeaned = x - x.mean()
    gamma0 = float(np.sum(demeaned ** 2) / n)
    long_run_var = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        gamma_lag = float(np.sum(demeaned[lag:] * demeaned[:-lag]) / n)
        long_run_var += 2 * weight * gamma_lag
    if long_run_var <= 0:
        return None
    return float(np.sqrt(long_run_var / n))


def newey_west_tstat(values, lags):
    """t-stat on the sample mean of `values` using the Newey-West HAC standard error above."""
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return None
    se = newey_west_se_of_mean(x, lags)
    if not se:
        return None
    return float(x.mean() / se)


def aggregate_ic_stats(ic_by_date):
    """mean_ic, std_ic, ic_ir (primary ranking metric), ic_tstat, n_dates, pct_positive."""
    values = list(ic_by_date.values())
    n = len(values)
    if n == 0:
        return {"mean_ic": None, "std_ic": None, "ic_ir": None, "ic_tstat": None,
                "n_dates": 0, "pct_positive": None}
    arr = np.array(values)
    mean_ic = float(arr.mean())
    std_ic = float(arr.std(ddof=1)) if n > 1 else 0.0
    ic_ir = (mean_ic / std_ic) if std_ic > 0 else None
    ic_tstat = (mean_ic / (std_ic / np.sqrt(n))) if std_ic > 0 else None
    pct_positive = float((arr > 0).mean())
    return {"mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir, "ic_tstat": ic_tstat,
            "n_dates": n, "pct_positive": pct_positive}


def _long_short_spread_for_date(sub, alpha_col, fwd_col, min_decile=MIN_TICKERS_FOR_DECILE):
    """
    Top-bucket-minus-bottom-bucket equal-weight mean forward return for one
    date's cross-section. Decile if >= min_decile valid tickers, else
    tercile (flagged via the returned `used` label).
    """
    n = len(sub)
    if n < 10:
        return None, None
    sub_sorted = sub.sort_values(alpha_col)
    if n >= min_decile:
        bucket_size, used = max(1, n // 10), "decile"
    else:
        bucket_size, used = max(1, n // 3), "tercile"
    bottom = sub_sorted.iloc[:bucket_size][fwd_col].mean()
    top = sub_sorted.iloc[-bucket_size:][fwd_col].mean()
    return float(top - bottom), used


def compute_portfolio_metrics(df, alpha_col, fwd_col, horizon_months, date_col="date"):
    """
    Long top / short bottom decile (tercile if thin) per date; string the
    per-date long-short spreads into a series across dates.

    Annualization note: rebalancing is monthly but each spread is a return
    over `horizon_months` months, so consecutive observations overlap
    heavily and are not independent (see spec's "Honest reporting
    requirements" — flagged again in the final narrative, not silently
    corrected here). mean_return is compounded to a 12-month-equivalent
    rate; volatility is scaled by sqrt(12/horizon_months), treating one
    non-overlapping window per year as the base unit.

    Returns dict with mean_return, volatility, sharpe (all annualized),
    decile_spread (raw, un-annualized, per spec's literal definition),
    n_dates, thin_dates (dates that fell back to tercile).
    """
    spreads, thin_dates = [], []
    for date, g in df.groupby(date_col):
        sub = g[[alpha_col, fwd_col]].dropna()
        if len(sub) < 10:
            continue
        spread, used = _long_short_spread_for_date(sub, alpha_col, fwd_col)
        if spread is None:
            continue
        if used == "tercile":
            thin_dates.append(date)
        spreads.append(spread)

    if len(spreads) < 2:
        return {"mean_return": None, "volatility": None, "sharpe": None,
                "decile_spread": None, "n_dates": len(spreads), "thin_dates": thin_dates}

    arr = np.array(spreads)
    decile_spread = float(arr.mean())
    periods_per_year = 12 / horizon_months

    # periods_per_year is always a positive integer for our fixed 6/12-month
    # horizons (2 or 1), so raising (1 + mean_spread) to that power is a
    # well-defined real number for any mean_spread, including <= -1 (a
    # concentrated decile can realize a >100% swing) — no need to guard
    # against a negative base here the way a fractional exponent would.
    mean_spread = arr.mean()
    mean_return = float((1 + mean_spread) ** periods_per_year - 1)
    volatility = float(arr.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = (mean_return / volatility) if volatility > 0 else None

    return {"mean_return": mean_return, "volatility": volatility, "sharpe": sharpe,
            "decile_spread": decile_spread, "n_dates": len(spreads), "thin_dates": thin_dates}


def hit_rate(df, alpha_col, fwd_col, date_col="date"):
    """
    Per-observation directional accuracy: "when the alpha ranked a company
    above median, how often did that company actually beat the median
    return?" Complements pct_positive (a per-date, sign-of-IC metric) and
    decile_spread (a magnitude metric) with the intuitive accuracy number.

    For each date: split both alpha_col and fwd_col at their own median
    (median split, not a fixed threshold — scale-free, doesn't depend on
    how the alpha is centered). A ticker "hits" if which side of the alpha
    median it's on matches which side of the return median it's on.
    Dates below MIN_TICKERS_FOR_IC valid tickers, or with zero variance in
    either column, are skipped (same guards as compute_ic_series) and
    reported in skipped_dates rather than silently dropped.

    Returns dict: overall_hit_rate (pooled across all counted observations,
    not a per-date average), hit_rate_by_date (pd.Series), n_observations,
    skipped_dates.
    """
    hits_by_date = {}
    total_hits, total_obs = 0, 0
    skipped = []
    for date, g in df.groupby(date_col):
        sub = g[[alpha_col, fwd_col]].dropna()
        if len(sub) < MIN_TICKERS_FOR_IC:
            skipped.append((date, len(sub)))
            continue
        if sub[alpha_col].std() == 0 or sub[fwd_col].std() == 0:
            skipped.append((date, len(sub)))
            continue
        median_alpha = sub[alpha_col].median()
        median_return = sub[fwd_col].median()
        predicted_above = sub[alpha_col] > median_alpha
        actual_above = sub[fwd_col] > median_return
        hits = predicted_above == actual_above
        hits_by_date[date] = float(hits.mean())
        total_hits += int(hits.sum())
        total_obs += len(hits)

    overall_hit_rate = (total_hits / total_obs) if total_obs > 0 else None
    return {
        "overall_hit_rate": overall_hit_rate,
        "hit_rate_by_date": pd.Series(hits_by_date),
        "n_observations": total_obs,
        "skipped_dates": skipped,
    }


def hit_rate_extremes(df, alpha_col, fwd_col, date_col="date", min_decile=MIN_TICKERS_FOR_DECILE):
    """
    Same median-split hit-test as hit_rate, but only counted for tickers in
    the top or bottom bucket by alpha value that date — where conviction
    (and, if the alpha is well-behaved, accuracy) is highest, and where a
    long/short strategy would actually be trading. Median splits (for
    classifying hits) are still computed across the FULL cross-section that
    date, same as hit_rate; only which observations get scored is
    restricted to the extremes.

    Bucket sizing matches _long_short_spread_for_date's existing decile/
    tercile fallback: decile if >= min_decile valid tickers that date, else
    tercile (flagged in thin_dates, consistent with compute_portfolio_metrics).
    Dates with fewer than 10 valid tickers are skipped outright.

    Returns dict: overall_hit_rate_extremes, hit_rate_by_date_extremes
    (pd.Series), n_observations_extremes, skipped_dates, thin_dates.
    """
    hits_by_date = {}
    total_hits, total_obs = 0, 0
    skipped, thin_dates = [], []
    for date, g in df.groupby(date_col):
        sub = g[[alpha_col, fwd_col]].dropna()
        n = len(sub)
        if n < 10:
            skipped.append((date, n))
            continue
        if sub[alpha_col].std() == 0 or sub[fwd_col].std() == 0:
            skipped.append((date, n))
            continue

        median_alpha = sub[alpha_col].median()
        median_return = sub[fwd_col].median()

        sub_sorted = sub.sort_values(alpha_col)
        if n >= min_decile:
            bucket_size = max(1, n // 10)
        else:
            bucket_size = max(1, n // 3)
            thin_dates.append(date)
        extremes = pd.concat([sub_sorted.iloc[:bucket_size], sub_sorted.iloc[-bucket_size:]])

        predicted_above = extremes[alpha_col] > median_alpha
        actual_above = extremes[fwd_col] > median_return
        hits = predicted_above == actual_above
        hits_by_date[date] = float(hits.mean())
        total_hits += int(hits.sum())
        total_obs += len(hits)

    overall_hit_rate_extremes = (total_hits / total_obs) if total_obs > 0 else None
    return {
        "overall_hit_rate_extremes": overall_hit_rate_extremes,
        "hit_rate_by_date_extremes": pd.Series(hits_by_date),
        "n_observations_extremes": total_obs,
        "skipped_dates": skipped,
        "thin_dates": thin_dates,
    }


def compute_turnover(df, alpha_col, date_col="date", ticker_col="ticker"):
    """
    Mean of |alpha_rank_t - alpha_rank_{t-1}| across tickers common to
    consecutive dates, where alpha_rank is the per-date percentile rank of
    alpha_col — this puts every alpha (raw ranks and composite z-score
    sums alike) on the same (0, 1] scale so turnover is comparable across
    alphas regardless of the alpha's native units.
    """
    work = df[[date_col, ticker_col, alpha_col]].copy()
    work["_rank"] = rank(work, alpha_col, date_col)

    dates = sorted(work[date_col].unique())
    per_date_means = []
    for prev_d, cur_d in zip(dates[:-1], dates[1:]):
        prev = work.loc[work[date_col] == prev_d].set_index(ticker_col)["_rank"]
        cur = work.loc[work[date_col] == cur_d].set_index(ticker_col)["_rank"]
        common = prev.index.intersection(cur.index)
        if len(common) == 0:
            continue
        diff = (cur.loc[common] - prev.loc[common]).abs().dropna()
        if len(diff) > 0:
            per_date_means.append(float(diff.mean()))

    if not per_date_means:
        return None
    return float(np.mean(per_date_means))


def evaluate_alpha(df, alpha_col, horizons=(6, 12), date_col="date"):
    """
    Full evaluation of one alpha column across the given forward-return
    horizons: IC stats, portfolio metrics, turnover, hit rate. Returns
    {horizon: {..metrics.., "skipped_dates": [...]}} plus a shared
    "turnover" key (turnover doesn't depend on horizon).
    """
    result = {"turnover": compute_turnover(df, alpha_col, date_col)}
    for w in horizons:
        fwd_col = f"fwd_return_{w}m"
        ic_by_date, skipped = compute_ic_series(df, alpha_col, fwd_col, date_col)
        ic_stats = aggregate_ic_stats(ic_by_date)
        portfolio = compute_portfolio_metrics(df, alpha_col, fwd_col, w, date_col)
        hr = hit_rate(df, alpha_col, fwd_col, date_col)
        hre = hit_rate_extremes(df, alpha_col, fwd_col, date_col)
        result[w] = {
            **ic_stats, **portfolio,
            "hit_rate": hr["overall_hit_rate"],
            "hit_rate_n_observations": hr["n_observations"],
            "hit_rate_extremes": hre["overall_hit_rate_extremes"],
            "hit_rate_extremes_n_observations": hre["n_observations_extremes"],
            "skipped_dates": skipped,
        }
    return result
