"""
Final methodology test in this diagnostic chain: does Ridge regression
(L2-penalized) stabilize the comps regression, where OLS — with or without
forward-selected features — has consistently overfit the tech universe
(see holdout_validation.py, relative_regression.py)?

Fits RidgeCV (scikit-learn) for all 4 bucket x multiple combinations, two
ways: the fixed original 4-feature set (trailing_growth, operating_margin,
roic, log_revenue) and the full 10-candidate pool (same pool as
feature_selection.py, filtered the same way — drop any feature missing
for >20% of the TRAINING set). Log-transforms the multiple, matching the
rest of this codebase's methodology (relative_regression.py was the one
exception, using a linear target). Compares against a freshly-computed
OLS 4-feature baseline (same fixed feature set, evaluated the same
80/20-split way) so all three models in the final table are apples-to-
apples on data and evaluation procedure.

Implementation note not in the task's literal spec but necessary for
Ridge to behave correctly: features are standardized (zero mean, unit
variance, scaler fit on TRAINING data only) before fitting. Ridge's
penalty operates on raw coefficient magnitude, so without standardizing,
a large-scale feature like log_revenue (~20-27) would be unfairly cheap
to keep and a small-scale feature like operating_margin (~-0.5 to 0.6)
unfairly expensive — the comparison would mostly reflect feature scale,
not genuine regularization behavior.

Metric-scale caveat (flagged again in the final report): "Train CV R²" for
Ridge is RidgeCV's own built-in 5-fold CV score (sklearn's R² on the LOG-
transformed target, per the task's explicit instruction to report "CV R²
from RidgeCV's built-in CV"). "Hold-out R²" is corr(predicted, actual)**2
on the LEVEL (exponentiated) scale, matching every other diagnostic in
this chain. These are not the same metric on the same scale — a caveat
worth remembering when reading the verdict, not a bug.

Reuses: compute_extended_features_for_ticker, filter_available_features
(feature_selection.py); split_universe, interpret (holdout_validation.py
— same seed=42, same tickers as every prior comparison); winsorize,
MultipleObservation, MIN_OBSERVATIONS, _feature_vector_complete,
FEATURE_ORDER, fit_regression, predict_multiple (regression.py);
loo_cross_validate (diagnostics.py).

Read-only diagnostic. Run directly:
python3 -m valuation.comps.ridge_regression
"""

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from valuation.comps.multiples import winsorize, compute_multiples_for_ticker
from valuation.comps.regression import (
    MultipleObservation, MIN_OBSERVATIONS, FEATURE_ORDER,
    fit_regression, predict_multiple, _feature_vector_complete,
)
from valuation.comps.diagnostics import loo_cross_validate
from valuation.comps.feature_selection import compute_extended_features_for_ticker, filter_available_features
from valuation.comps.holdout_validation import split_universe, interpret, SPLIT_SEED
from valuation.comps.universe import COMPS_UNIVERSE

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
CV_FOLDS = 5


@dataclass
class RidgeRegressionResult:
    multiple_type: str
    n_observations: int
    alpha: float | None
    r_squared: float | None      # in-sample R², log scale (sklearn .score())
    cv_r2: float | None          # RidgeCV's built-in CV score at the optimal alpha, log scale
    coefficients: dict
    intercept: float | None
    data_flags: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)
    _model: object = None
    _scaler: object = None


def fit_ridge_regression(observations, multiple_type, feature_names, alphas=ALPHAS, cv_folds=CV_FOLDS):
    """
    RidgeCV on log(multiple) ~ standardized features. Winsorizes the
    multiple and each feature independently at [1%, 99%] before fitting,
    same as fit_regression(). Requires an explicit feature_names list.
    """
    if not feature_names:
        raise ValueError("fit_ridge_regression requires an explicit feature_names list")
    feature_names = list(feature_names)
    data_flags = []

    usable = [
        obs for obs in observations
        if obs.multiple_type == multiple_type
        and _feature_vector_complete(obs.features, feature_names)
        and obs.multiple_value is not None and obs.multiple_value > 0
    ]
    dropped = len(observations) - len(usable)
    if dropped:
        data_flags.append(f"Dropped {dropped} of {len(observations)} observations (missing features or non-positive multiple)")

    min_needed = max(MIN_OBSERVATIONS, cv_folds)
    if len(usable) < min_needed:
        data_flags.append(f"Only {len(usable)} usable observations (< {min_needed} minimum for {cv_folds}-fold CV) — Ridge not fit")
        return RidgeRegressionResult(
            multiple_type=multiple_type, n_observations=len(usable), alpha=None,
            r_squared=None, cv_r2=None, coefficients={}, intercept=None,
            data_flags=data_flags, feature_names=feature_names,
        )

    X_raw = np.array([[getattr(obs.features, f) for f in feature_names] for obs in usable], dtype=float)
    y_raw = np.array([obs.multiple_value for obs in usable], dtype=float)

    y_wins = winsorize(y_raw)
    X_wins = np.column_stack([winsorize(X_raw[:, i]) for i in range(X_raw.shape[1])])
    y_log = np.log(y_wins)
    data_flags.append("Winsorized multiple and each feature at [1%, 99%]; log-transformed the multiple")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_wins)
    data_flags.append("Standardized features (fit on this training data only) before Ridge fitting")

    effective_cv = min(cv_folds, len(usable))
    if effective_cv < cv_folds:
        data_flags.append(f"Reduced CV folds from {cv_folds} to {effective_cv} (fewer usable observations than requested folds)")
    if effective_cv < 2:
        data_flags.append("Fewer than 2 usable observations for CV — Ridge not fit")
        return RidgeRegressionResult(
            multiple_type=multiple_type, n_observations=len(usable), alpha=None,
            r_squared=None, cv_r2=None, coefficients={}, intercept=None,
            data_flags=data_flags, feature_names=feature_names,
        )

    model = RidgeCV(alphas=alphas, cv=effective_cv)
    model.fit(X_scaled, y_log)

    coefficients = {name: float(c) for name, c in zip(feature_names, model.coef_)}
    r_squared = float(model.score(X_scaled, y_log))
    cv_r2 = float(model.best_score_)

    data_flags.append(f"alpha={model.alpha_:g}, in-sample R²={r_squared:.3f}, RidgeCV built-in {effective_cv}-fold CV R²={cv_r2:.3f}")

    return RidgeRegressionResult(
        multiple_type=multiple_type, n_observations=len(usable), alpha=float(model.alpha_),
        r_squared=r_squared, cv_r2=cv_r2, coefficients=coefficients, intercept=float(model.intercept_),
        data_flags=data_flags, feature_names=feature_names, _model=model, _scaler=scaler,
    )


def predict_ridge(regression, features):
    """
    Predict the multiple (reversing log + standardization). Returns
    (predicted_multiple, None) — matches fit_fn/predict_fn's (value, se)
    calling convention used by loo_cross_validate()/forward_selection(),
    even though Ridge has no analytic per-prediction SE the way OLS's
    get_prediction() does.
    """
    if regression is None or regression._model is None:
        return None, None
    if not _feature_vector_complete(features, regression.feature_names):
        return None, None
    x_raw = np.array([[getattr(features, f) for f in regression.feature_names]])
    x_scaled = regression._scaler.transform(x_raw)
    predicted_log = float(regression._model.predict(x_scaled)[0])
    return float(np.exp(predicted_log)), None


def build_observations_for_tickers(features_by_ticker, multiples_by_ticker, tickers, multiple_type):
    obs = []
    for t in tickers:
        f = features_by_ticker.get(t)
        m = multiples_by_ticker.get(t)
        if f is None or m is None:
            continue
        value = m.get(multiple_type)
        if value is not None:
            obs.append(MultipleObservation(t, multiple_type, value, f))
    return obs


def evaluate_holdout(predict_fn, regression, holdout_obs):
    """
    Fit-once-predict-many on genuinely unseen tickers. Returns
    (n_evaluated, holdout_r2 = corr(predicted, actual)**2 on the LEVEL
    scale) — same definition used throughout this diagnostic chain.
    """
    predicted_list, actual_list = [], []
    for obs in holdout_obs:
        predicted, _se = predict_fn(regression, obs.features)
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
    r2 = float(corr ** 2)
    if math.isnan(r2) or abs(r2) > 1.0:
        print(f"  [ANOMALY] holdout R²={r2} is NaN or |value|>1.0 — impossible for a squared correlation, investigate")
    return len(predicted_list), r2


def run_ols_4feature_baseline(train_obs, holdout_obs, multiple_type):
    """
    Fresh OLS baseline on the FIXED original 4 features (no forward
    selection), evaluated on this same 80/20 split — this exact
    combination (fixed 4 features x this split) hasn't been run before,
    so it's computed here for a genuine apples-to-apples row against
    Ridge, rather than reusing the forward-selected-feature numbers from
    the earlier holdout_validation.py session (a different model).
    """
    reg = fit_regression(train_obs, multiple_type, feature_names=FEATURE_ORDER)
    _loo, train_cv_r2 = loo_cross_validate(train_obs, multiple_type, feature_names=FEATURE_ORDER)
    n_evaluated, holdout_r2 = evaluate_holdout(predict_multiple, reg, holdout_obs)
    return {
        "alpha": None, "n_train": reg.n_observations, "train_cv_r2": train_cv_r2,
        "n_holdout_evaluated": n_evaluated, "holdout_r2": holdout_r2,
    }


def run_ridge_validation_for_bucket(bucket):
    tickers = COMPS_UNIVERSE[bucket]
    train_tickers, holdout_tickers = split_universe(tickers)  # seed=42 — same split as every prior comparison

    print(f"\n{'=' * 90}\nRIDGE REGRESSION VALIDATION: {bucket}\n{'=' * 90}")
    print(f"  Universe: {len(tickers)} -> train={len(train_tickers)}, holdout={len(holdout_tickers)} (seed={SPLIT_SEED})")

    print("  Computing extended features + multiples for full universe...")
    features_by_ticker = {t: compute_extended_features_for_ticker(t) for t in tickers}
    multiples_by_ticker = {t: compute_multiples_for_ticker(t) for t in tickers}

    train_features_list = [features_by_ticker[t] for t in train_tickers]
    kept_10_features, dropped_features = filter_available_features(train_features_list)
    print(f"  10-feature pool after >20% missing filter (TRAINING set only): {kept_10_features}")
    if dropped_features:
        print(f"  Dropped: { {k: f'{v * 100:.0f}%' for k, v in dropped_features.items()} }")

    results = {}
    for multiple_type in ("ev_sales", "ev_ebitda"):
        train_obs = build_observations_for_tickers(features_by_ticker, multiples_by_ticker, train_tickers, multiple_type)
        holdout_obs = build_observations_for_tickers(features_by_ticker, multiples_by_ticker, holdout_tickers, multiple_type)

        print(f"\n  --- {multiple_type} ---")

        ols = run_ols_4feature_baseline(train_obs, holdout_obs, multiple_type)
        ols["verdict"] = interpret(ols["train_cv_r2"], ols["holdout_r2"])
        print(f"  OLS 4-feature:   train_cv_r2={ols['train_cv_r2']}  holdout_r2={ols['holdout_r2']}  verdict={ols['verdict']}")

        ridge4_reg = fit_ridge_regression(train_obs, multiple_type, FEATURE_ORDER)
        n_eval4, holdout_r2_4 = evaluate_holdout(predict_ridge, ridge4_reg, holdout_obs)
        ridge4 = {
            "alpha": ridge4_reg.alpha, "n_train": ridge4_reg.n_observations, "train_cv_r2": ridge4_reg.cv_r2,
            "n_holdout_evaluated": n_eval4, "holdout_r2": holdout_r2_4,
            "verdict": interpret(ridge4_reg.cv_r2, holdout_r2_4),
        }
        print(f"  Ridge 4-feature: alpha={ridge4['alpha']}  train_cv_r2={ridge4['train_cv_r2']}  holdout_r2={ridge4['holdout_r2']}  verdict={ridge4['verdict']}")

        ridge10_reg = fit_ridge_regression(train_obs, multiple_type, kept_10_features)
        n_eval10, holdout_r2_10 = evaluate_holdout(predict_ridge, ridge10_reg, holdout_obs)
        ridge10 = {
            "alpha": ridge10_reg.alpha, "n_train": ridge10_reg.n_observations, "train_cv_r2": ridge10_reg.cv_r2,
            "n_holdout_evaluated": n_eval10, "holdout_r2": holdout_r2_10,
            "verdict": interpret(ridge10_reg.cv_r2, holdout_r2_10),
            "coefficients": ridge10_reg.coefficients,
        }
        print(f"  Ridge 10-feature: alpha={ridge10['alpha']}  train_cv_r2={ridge10['train_cv_r2']}  holdout_r2={ridge10['holdout_r2']}  verdict={ridge10['verdict']}")

        results[multiple_type] = {"ols_4feature": ols, "ridge_4feature": ridge4, "ridge_10feature": ridge10}

    return {"bucket": bucket, "results": results}


def print_comparison_table(all_results):
    print(f"\n{'=' * 128}")
    print("FINAL COMPARISON — OLS vs RIDGE (4-feature) vs RIDGE (10-feature)")
    print(f"{'=' * 128}")
    print("NOTE: 'Train CV R²' for Ridge rows is RidgeCV's own built-in CV score (log scale); for OLS it's LOO corr² (level")
    print("scale). 'Hold-out R²' is corr(predicted, actual)² on the level scale for all rows. Not the same metric/scale for")
    print("Ridge's train-vs-holdout columns — see module docstring. Verdicts are still computed per the task's literal rule.")
    header = f"{'Bucket':<16}{'Multiple':<11}{'Model':<18}{'Alpha':>8}{'Train CV R²':>13}{'Holdout R²':>12}  Verdict"
    print(header)
    print("-" * 128)
    any_confirmed_ridge = False
    for r in all_results:
        bucket = r["bucket"]
        for multiple_type, res in r["results"].items():
            for model_name, key in (("OLS 4-feature", "ols_4feature"), ("Ridge 4-feature", "ridge_4feature"), ("Ridge 10-feature", "ridge_10feature")):
                m = res[key]
                alpha = f"{m['alpha']:g}" if m["alpha"] is not None else "-"
                tcv = f"{m['train_cv_r2']:.3f}" if m["train_cv_r2"] is not None else "N/A"
                hr2 = f"{m['holdout_r2']:.3f}" if m["holdout_r2"] is not None else "N/A"
                print(f"{bucket:<16}{multiple_type:<11}{model_name:<18}{alpha:>8}{tcv:>13}{hr2:>12}  {m['verdict']}")
                if model_name.startswith("Ridge") and m["verdict"] == "Confirmed":
                    any_confirmed_ridge = True
            print()
    print("-" * 128)
    if any_confirmed_ridge:
        print("\nAt least one Ridge model Confirmed — Ridge produced at least one stable, generalizing result.")
    else:
        print("\nNo Ridge model Confirmed across any bucket x multiple combination.")


def run_ridge_regression_diagnostic():
    all_results = [run_ridge_validation_for_bucket(b) for b in ("semiconductors", "mature_tech")]
    print_comparison_table(all_results)
    return all_results


if __name__ == "__main__":
    run_ridge_regression_diagnostic()
