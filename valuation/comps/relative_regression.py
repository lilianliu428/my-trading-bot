"""
Test whether reformulating the comps regression on SECTOR-RELATIVE
multiples (a ticker's multiple relative to its bucket median) produces
more robust, better-generalizing predictions than absolute multiples.

Hypothesis: absolute multiples embed macro/regime effects (rates, sector
rotation, sentiment) that fundamentals cannot explain. Sector-relative
multiples strip out that shared bucket-wide component, leaving only
within-bucket variation — which should be more explicable from company
fundamentals alone, and (the actual test here) should generalize better
out-of-sample.

    relative_multiple = (ticker_multiple - bucket_median) / bucket_median

Unlike the absolute-multiple regression (log-linear — see regression.py),
this target can be negative (a ticker trading below its bucket's median),
so it is fit with LINEAR regression directly — no log transform.

Reuses rather than duplicates: compute_extended_features_for_ticker,
ALL_CANDIDATE_FEATURES, filter_available_features, forward_selection
(feature_selection.py — generalized this session to accept pluggable
fit_fn/predict_fn); winsorize, MIN_OBSERVATIONS, _feature_vector_complete
(regression.py / multiples.py); split_universe, interpret
(holdout_validation.py — SAME seed=42, so the relative-vs-absolute
comparison uses identical train/holdout ticker sets).

Read-only diagnostic. Run directly:
python3 -m valuation.comps.relative_regression
"""

import math
from dataclasses import dataclass, field

import numpy as np
import statsmodels.api as sm

from valuation.comps.multiples import winsorize, compute_multiples_for_ticker
from valuation.comps.regression import MultipleObservation, MIN_OBSERVATIONS, _feature_vector_complete
from valuation.comps.feature_selection import (
    compute_extended_features_for_ticker, filter_available_features, forward_selection,
)
from valuation.comps.holdout_validation import split_universe, interpret, SPLIT_SEED
from valuation.comps.universe import COMPS_UNIVERSE


@dataclass
class RelativeRegressionResult:
    multiple_type: str
    n_observations: int
    r_squared: float | None
    coefficients: dict
    intercept: float | None
    residual_std: float | None
    data_flags: list = field(default_factory=list)
    _model_result: object = None
    feature_ranges: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)


def _finite(v):
    return v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def fit_relative_regression(observations, multiple_type, feature_names=None):
    """
    Linear (NOT log-linear) regression of relative_multiple on the given
    feature subset. relative_multiple can be negative — sits at 0 for a
    ticker exactly at its bucket's median — so no log transform. Otherwise
    mirrors regression.fit_regression()'s structure: winsorize target and
    each feature independently, OLS via statsmodels, same
    const-plus-feature_names design-matrix handling and the same
    residual_std/feature_ranges bookkeeping predict_relative_multiple()
    needs.

    feature_names is required (unlike fit_regression, there's no sensible
    "default" feature set for this exploratory model).
    """
    if not feature_names:
        raise ValueError("fit_relative_regression requires an explicit feature_names list")
    feature_names = list(feature_names)
    data_flags = []

    usable = [
        obs for obs in observations
        if obs.multiple_type == multiple_type
        and _feature_vector_complete(obs.features, feature_names)
        and _finite(obs.multiple_value)   # NOTE: no positivity filter — relative_multiple can be negative
    ]
    dropped = len(observations) - len(usable)
    if dropped:
        data_flags.append(f"Dropped {dropped} of {len(observations)} observations (missing features or non-finite relative multiple)")

    if len(usable) < MIN_OBSERVATIONS:
        data_flags.append(f"Only {len(usable)} usable observations (< {MIN_OBSERVATIONS} minimum) — regression not fit")
        return RelativeRegressionResult(
            multiple_type=multiple_type, n_observations=len(usable), r_squared=None,
            coefficients={}, intercept=None, residual_std=None, data_flags=data_flags,
            feature_names=feature_names,
        )

    X_raw = np.array([[getattr(obs.features, f) for f in feature_names] for obs in usable], dtype=float)
    y_raw = np.array([obs.multiple_value for obs in usable], dtype=float)

    y_wins = winsorize(y_raw)  # percentile-based, sign-agnostic — works fine on a negative-containing array
    X_wins = np.column_stack([winsorize(X_raw[:, i]) for i in range(X_raw.shape[1])])
    data_flags.append("Winsorized relative multiple and each feature at [1%, 99%]")

    # NO log transform here — the one substantive difference from fit_regression.
    X_design = sm.add_constant(X_wins, has_constant="add")
    model_result = sm.OLS(y_wins, X_design).fit()

    coefficients = {name: float(model_result.params[i + 1]) for i, name in enumerate(feature_names)}
    intercept = float(model_result.params[0])
    r_squared = float(model_result.rsquared)
    residual_std = float(np.sqrt(model_result.mse_resid))

    data_flags.append(f"Fit on {len(usable)} observations, R²={r_squared:.3f}")

    feature_ranges = {
        name: (float(np.min(X_wins[:, i])), float(np.max(X_wins[:, i])))
        for i, name in enumerate(feature_names)
    }

    return RelativeRegressionResult(
        multiple_type=multiple_type, n_observations=len(usable), r_squared=r_squared,
        coefficients=coefficients, intercept=intercept, residual_std=residual_std,
        data_flags=data_flags, _model_result=model_result, feature_ranges=feature_ranges,
        feature_names=feature_names,
    )


def predict_relative_multiple(regression, features):
    """
    Predict relative_multiple directly — no exp() reversal, this is a
    linear model. Returns (predicted, se) or (None, None) if the
    regression couldn't be fit or the target's features are incomplete.
    """
    if regression is None or regression._model_result is None:
        return None, None
    if not _feature_vector_complete(features, regression.feature_names):
        return None, None
    x = np.array([[1.0] + [getattr(features, f) for f in regression.feature_names]])
    pred = regression._model_result.get_prediction(x)
    predicted = float(pred.predicted_mean[0])
    se = float(pred.se_obs[0])
    return predicted, se


def compute_relative_observations(tickers_for_median, tickers_to_build, features_by_ticker, multiples_by_ticker, multiple_type):
    """
    Compute bucket_median from tickers_for_median's raw multiples (pass
    the TRAINING set here, never the full universe, to avoid leaking
    hold-out information into the median used to define the regression
    target), then build MultipleObservation objects (multiple_value =
    relative multiple) for tickers_to_build using that frozen median —
    including when tickers_to_build is the hold-out set, so the same
    training-derived median is what "actual relative_multiple" means for
    hold-out evaluation too.

    Returns (observations, bucket_median, data_flags). bucket_median is
    None (observations empty) if fewer than 2 valid raw multiples are
    available to compute a median from.
    """
    data_flags = []
    raw_values_for_median = [
        multiples_by_ticker[t][multiple_type]
        for t in tickers_for_median
        if multiples_by_ticker.get(t) and multiples_by_ticker[t].get(multiple_type) is not None
        and multiples_by_ticker[t][multiple_type] > 0
    ]

    if len(raw_values_for_median) < 2:
        data_flags.append(f"Fewer than 2 valid {multiple_type} observations to compute bucket median — cannot build relative multiples")
        return [], None, data_flags

    bucket_median = float(np.median(raw_values_for_median))
    if bucket_median == 0:
        data_flags.append(f"Bucket median {multiple_type} is exactly 0 — relative multiple undefined")
        return [], None, data_flags

    observations = []
    for t in tickers_to_build:
        f = features_by_ticker.get(t)
        m = multiples_by_ticker.get(t)
        if f is None or m is None or m.get(multiple_type) is None or m[multiple_type] <= 0:
            continue
        relative = (m[multiple_type] - bucket_median) / bucket_median
        observations.append(MultipleObservation(t, multiple_type, relative, f))

    return observations, bucket_median, data_flags


def evaluate_holdout_relative(regression, holdout_obs):
    """Fit-once-predict-many on genuinely unseen tickers. Returns (n_evaluated, holdout_r2 = corr(predicted, actual)**2)."""
    predicted_list, actual_list = [], []
    for obs in holdout_obs:
        predicted, _se = predict_relative_multiple(regression, obs.features)
        if predicted is not None:
            predicted_list.append(predicted)
            actual_list.append(obs.multiple_value)

    if len(predicted_list) < 2:
        return len(predicted_list), None

    preds = np.array(predicted_list)
    actuals = np.array(actual_list)
    if np.std(preds) == 0 or np.std(actuals) == 0:
        return len(predicted_list), None
    corr = np.corrcoef(preds, actuals)[0, 1]
    return len(predicted_list), float(corr ** 2)


def _check_anomaly(label, value):
    """Task constraint: report (don't silently ignore) NaN or |R²| > 1.0."""
    if value is not None and (math.isnan(value) if isinstance(value, float) else False):
        print(f"  [ANOMALY] {label} is NaN — investigating: this should be impossible for a squared correlation, indicates a bug")
        return True
    if value is not None and abs(value) > 1.0:
        print(f"  [ANOMALY] {label}={value} has |value|>1.0 — impossible for a squared correlation, indicates a bug")
        return True
    return False


def run_relative_holdout_validation_for_bucket(bucket):
    tickers = COMPS_UNIVERSE[bucket]
    train_tickers, holdout_tickers = split_universe(tickers)  # seed=42 — SAME split as the absolute-multiple run

    print(f"\n{'=' * 90}\nRELATIVE-MULTIPLE HOLD-OUT VALIDATION: {bucket}\n{'=' * 90}")
    print(f"  Universe: {len(tickers)} -> train={len(train_tickers)}, holdout={len(holdout_tickers)} (seed={SPLIT_SEED}, same split as absolute-multiple run)")

    print("  Computing extended features + multiples for full universe...")
    features_by_ticker = {t: compute_extended_features_for_ticker(t) for t in tickers}
    multiples_by_ticker = {t: compute_multiples_for_ticker(t) for t in tickers}

    train_features_list = [features_by_ticker[t] for t in train_tickers]
    kept_features, dropped_features = filter_available_features(train_features_list)
    print(f"  Candidate features kept (>20% missing filter, TRAINING set only): {kept_features}")
    if dropped_features:
        print(f"  Dropped: { {k: f'{v * 100:.0f}%' for k, v in dropped_features.items()} }")

    results = {}
    for multiple_type in ("ev_sales", "ev_ebitda"):
        train_obs, bucket_median, flags = compute_relative_observations(
            train_tickers, train_tickers, features_by_ticker, multiples_by_ticker, multiple_type,
        )
        for f in flags:
            print(f"  [{multiple_type}] {f}")

        if bucket_median is None:
            results[multiple_type] = {
                "selected_features": [], "training_cv_r2": None, "n_train": 0,
                "n_holdout_evaluated": 0, "holdout_r2": None,
                "verdict": "N/A (insufficient data)", "bucket_median": None,
            }
            continue

        holdout_obs, _same_median, _flags2 = compute_relative_observations(
            train_tickers, holdout_tickers, features_by_ticker, multiples_by_ticker, multiple_type,
        )

        fs_result = forward_selection(
            train_obs, multiple_type, kept_features,
            fit_fn=fit_relative_regression, predict_fn=predict_relative_multiple,
        )
        selected = fs_result["selected_features"]
        training_cv_r2 = fs_result["final_cv_r2"]

        if selected:
            final_reg = fit_relative_regression(train_obs, multiple_type, feature_names=selected)
            n_evaluated, holdout_r2 = evaluate_holdout_relative(final_reg, holdout_obs)
            n_train = final_reg.n_observations
        else:
            n_evaluated, holdout_r2, n_train = 0, None, 0

        verdict = interpret(training_cv_r2, holdout_r2)

        print(f"\n  --- {multiple_type} (bucket_median={bucket_median:.3f}, frozen from training set) ---")
        print(f"  Winning feature set (training-only forward selection): {selected}")
        print(f"  Training CV R²: {training_cv_r2}")
        print(f"  Fitted on {n_train} training obs; evaluated on {n_evaluated} of {len(holdout_tickers)} holdout tickers")
        print(f"  Hold-out R²: {holdout_r2}")
        print(f"  Verdict: {verdict}")
        _check_anomaly(f"{bucket}/{multiple_type} training_cv_r2", training_cv_r2)
        _check_anomaly(f"{bucket}/{multiple_type} holdout_r2", holdout_r2)

        results[multiple_type] = {
            "selected_features": selected, "training_cv_r2": training_cv_r2, "n_train": n_train,
            "n_holdout_evaluated": n_evaluated, "holdout_r2": holdout_r2,
            "verdict": verdict, "bucket_median": bucket_median,
        }

    return {"bucket": bucket, "results": results}


# Previous session's absolute-multiple hold-out validation results (log-linear
# model), for the side-by-side comparison table — NOT recomputed here (that
# was already run; re-running it would just re-fetch the same 159 tickers for
# no new information). Numbers and feature sets as reported previously.
ABSOLUTE_BASELINE = {
    ("semiconductors", "ev_sales"): {
        "features": ["trailing_growth", "gross_margin", "ebitda_margin", "roic"],
        "training_cv_r2": 0.716, "holdout_r2": 0.259, "verdict": "Overfit",
    },
    ("semiconductors", "ev_ebitda"): {
        "features": ["r_and_d_intensity", "trailing_growth", "ebitda_margin", "gross_margin", "operating_margin"],
        "training_cv_r2": 0.717, "holdout_r2": 0.421, "verdict": "Overfit",
    },
    ("mature_tech", "ev_sales"): {
        "features": ["log_revenue"],
        "training_cv_r2": 0.083, "holdout_r2": 0.284, "verdict": "Confirmed",
    },
    ("mature_tech", "ev_ebitda"): {
        "features": ["operating_margin"],
        "training_cv_r2": 0.674, "holdout_r2": 0.117, "verdict": "Overfit",
    },
}


def print_comparison_table(all_results):
    print(f"\n{'=' * 130}")
    print("ABSOLUTE vs RELATIVE MULTIPLE — HOLD-OUT VALIDATION COMPARISON")
    print(f"{'=' * 130}")
    header = f"{'Bucket':<16}{'Multiple':<11}{'Formulation':<12}{'Feature set':<48}{'Train CV R²':>12}{'Holdout R²':>11}  Verdict"
    print(header)
    print("-" * 130)

    confirmed_relative = []
    for r in all_results:
        bucket = r["bucket"]
        for multiple_type, res in r["results"].items():
            baseline = ABSOLUTE_BASELINE[(bucket, multiple_type)]
            b_feats = ", ".join(baseline["features"])
            print(f"{bucket:<16}{multiple_type:<11}{'Absolute (prev)':<12}{b_feats:<48}{baseline['training_cv_r2']:>12.3f}{baseline['holdout_r2']:>11.3f}  {baseline['verdict']}")

            r_feats = ", ".join(res["selected_features"]) if res["selected_features"] else "(none)"
            tcv = f"{res['training_cv_r2']:.3f}" if res["training_cv_r2"] is not None else "N/A"
            hr2 = f"{res['holdout_r2']:.3f}" if res["holdout_r2"] is not None else "N/A"
            print(f"{bucket:<16}{multiple_type:<11}{'Relative':<12}{r_feats:<48}{tcv:>12}{hr2:>11}  {res['verdict']}")
            print()

            if res["verdict"] == "Confirmed":
                confirmed_relative.append((bucket, multiple_type))

    print("-" * 130)
    if confirmed_relative:
        print(f"\nRelative formulation CONFIRMED for: {confirmed_relative} — worth pursuing per the task's success bar.")
    else:
        print("\nRelative formulation did NOT produce a single Confirmed result — hypothesis not supported by this data.")

    print("\nFeature-set comparison (absolute vs relative, training-only selection):")
    for r in all_results:
        bucket = r["bucket"]
        for multiple_type, res in r["results"].items():
            baseline_feats = set(ABSOLUTE_BASELINE[(bucket, multiple_type)]["features"])
            relative_feats = set(res["selected_features"])
            overlap = baseline_feats & relative_feats
            print(f"  {bucket}/{multiple_type}: absolute={sorted(baseline_feats)}  relative={sorted(relative_feats)}  overlap={sorted(overlap)}")


def run_relative_regression_diagnostic():
    all_results = [run_relative_holdout_validation_for_bucket(b) for b in ("semiconductors", "mature_tech")]
    print_comparison_table(all_results)
    return all_results


if __name__ == "__main__":
    run_relative_regression_diagnostic()
