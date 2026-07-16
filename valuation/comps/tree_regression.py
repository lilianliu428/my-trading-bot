"""
Test whether non-linear tree-based models (RandomForest, GradientBoosting)
capture patterns in tech comps multiples that linear regression misses —
or whether they just overfit harder. Every linear variant tried earlier in
this diagnostic chain (forward-selected OLS, sector-relative OLS, Ridge)
has struggled against the same constraint: too few tickers (43-76 in
training) relative to how much flexibility the model/search is given.
Trees have much MORE flexibility than a 4-10 term linear model, so if
sample size is really the binding constraint, this should overfit worse,
not better.

Fits RandomForestRegressor (max_depth searched over [3, 5, 7] via 5-fold
CV, matching the spirit of RidgeCV's alpha search from the prior session)
and GradientBoostingRegressor (fixed hyperparameters per the task spec —
no search) on log(multiple) ~ standardized 10-feature pool.

Standardizing features is a no-op for tree splits (axis-aligned threshold
splits are invariant to monotonic per-feature rescaling — a tree finds the
same split points and the same feature_importances_ with or without it).
Done anyway per the literal task instruction and for pipeline consistency
with ridge_regression.py; flagged here so it's not mistaken for something
that matters to trees the way it did for Ridge.

OLS 4-feature and Ridge 4-feature baseline numbers are REUSED from the
prior ridge_regression.py session (identical seed=42 80/20 split, same
fixed 4 features) rather than recomputed — re-fetching the same 159
tickers again for numbers already on record would add ~15 minutes for no
new information.

Reuses: compute_extended_features_for_ticker, filter_available_features
(feature_selection.py); split_universe, interpret (holdout_validation.py);
build_observations_for_tickers (ridge_regression.py); MultipleObservation,
MIN_OBSERVATIONS, _feature_vector_complete, winsorize.

Read-only diagnostic. Run directly:
python3 -m valuation.comps.tree_regression
"""

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from valuation.comps.multiples import winsorize, compute_multiples_for_ticker
from valuation.comps.regression import MIN_OBSERVATIONS, _feature_vector_complete
from valuation.comps.feature_selection import compute_extended_features_for_ticker, filter_available_features
from valuation.comps.holdout_validation import split_universe, interpret, SPLIT_SEED
from valuation.comps.ridge_regression import build_observations_for_tickers
from valuation.comps.universe import COMPS_UNIVERSE

CV_FOLDS = 5
RANDOM_STATE = 42

RF_PARAM_GRID = {"max_depth": [3, 5, 7]}
RF_FIXED_PARAMS = dict(n_estimators=100, min_samples_split=5, min_samples_leaf=3, random_state=RANDOM_STATE)
GB_FIXED_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE)


@dataclass
class TreeRegressionResult:
    model_name: str
    multiple_type: str
    n_observations: int
    r_squared: float | None       # in-sample R², log scale
    cv_r2: float | None           # 5-fold CV R², log scale
    feature_importances: dict
    top_features: list            # top 3 by importance
    best_max_depth: int | None    # RandomForest only
    data_flags: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)
    _model: object = None
    _scaler: object = None


def _prepare_data(observations, multiple_type, feature_names):
    """Filter to usable observations, winsorize, log-transform target, standardize features."""
    usable = [
        obs for obs in observations
        if obs.multiple_type == multiple_type
        and _feature_vector_complete(obs.features, feature_names)
        and obs.multiple_value is not None and obs.multiple_value > 0
    ]
    if len(usable) < max(MIN_OBSERVATIONS, CV_FOLDS):
        return None, None, usable, None

    X_raw = np.array([[getattr(obs.features, f) for f in feature_names] for obs in usable], dtype=float)
    y_raw = np.array([obs.multiple_value for obs in usable], dtype=float)
    y_wins = winsorize(y_raw)
    X_wins = np.column_stack([winsorize(X_raw[:, i]) for i in range(X_raw.shape[1])])
    y_log = np.log(y_wins)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_wins)
    return X_scaled, y_log, usable, scaler


def fit_random_forest(observations, multiple_type, feature_names):
    feature_names = list(feature_names)
    data_flags = []
    X_scaled, y_log, usable, scaler = _prepare_data(observations, multiple_type, feature_names)
    dropped = len(observations) - len(usable)
    if dropped:
        data_flags.append(f"Dropped {dropped} of {len(observations)} observations (missing features or non-positive multiple)")
    if X_scaled is None:
        data_flags.append(f"Only {len(usable)} usable observations (< {max(MIN_OBSERVATIONS, CV_FOLDS)} minimum) — RandomForest not fit")
        return TreeRegressionResult("RandomForest", multiple_type, len(usable), None, None, {}, [], None, data_flags, feature_names)

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(RandomForestRegressor(**RF_FIXED_PARAMS), RF_PARAM_GRID, cv=cv, scoring="r2")
    grid.fit(X_scaled, y_log)

    model = grid.best_estimator_
    r_squared = float(model.score(X_scaled, y_log))
    cv_r2 = float(grid.best_score_)
    importances = {name: float(imp) for name, imp in zip(feature_names, model.feature_importances_)}
    top3 = [f for f, _ in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]]

    data_flags.append(f"best max_depth={grid.best_params_['max_depth']}, in-sample R²={r_squared:.3f}, 5-fold CV R²={cv_r2:.3f}")

    return TreeRegressionResult(
        model_name="RandomForest", multiple_type=multiple_type, n_observations=len(usable),
        r_squared=r_squared, cv_r2=cv_r2, feature_importances=importances, top_features=top3,
        best_max_depth=grid.best_params_["max_depth"], data_flags=data_flags,
        feature_names=feature_names, _model=model, _scaler=scaler,
    )


def fit_gradient_boosting(observations, multiple_type, feature_names):
    feature_names = list(feature_names)
    data_flags = []
    X_scaled, y_log, usable, scaler = _prepare_data(observations, multiple_type, feature_names)
    dropped = len(observations) - len(usable)
    if dropped:
        data_flags.append(f"Dropped {dropped} of {len(observations)} observations (missing features or non-positive multiple)")
    if X_scaled is None:
        data_flags.append(f"Only {len(usable)} usable observations (< {max(MIN_OBSERVATIONS, CV_FOLDS)} minimum) — GradientBoosting not fit")
        return TreeRegressionResult("GradientBoosting", multiple_type, len(usable), None, None, {}, [], None, data_flags, feature_names)

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(GradientBoostingRegressor(**GB_FIXED_PARAMS), X_scaled, y_log, cv=cv, scoring="r2")
    cv_r2 = float(np.mean(cv_scores))

    model = GradientBoostingRegressor(**GB_FIXED_PARAMS)
    model.fit(X_scaled, y_log)
    r_squared = float(model.score(X_scaled, y_log))
    importances = {name: float(imp) for name, imp in zip(feature_names, model.feature_importances_)}
    top3 = [f for f, _ in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]]

    data_flags.append(f"in-sample R²={r_squared:.3f}, 5-fold CV R²={cv_r2:.3f}")

    return TreeRegressionResult(
        model_name="GradientBoosting", multiple_type=multiple_type, n_observations=len(usable),
        r_squared=r_squared, cv_r2=cv_r2, feature_importances=importances, top_features=top3,
        best_max_depth=None, data_flags=data_flags,
        feature_names=feature_names, _model=model, _scaler=scaler,
    )


def predict_tree(regression, features):
    """Predict the multiple (reversing log + standardization). Returns (predicted_multiple, None)."""
    if regression is None or regression._model is None:
        return None, None
    if not _feature_vector_complete(features, regression.feature_names):
        return None, None
    x_raw = np.array([[getattr(features, f) for f in regression.feature_names]])
    x_scaled = regression._scaler.transform(x_raw)
    predicted_log = float(regression._model.predict(x_scaled)[0])
    return float(np.exp(predicted_log)), None


def evaluate_holdout(regression, holdout_obs):
    """Fit-once-predict-many. Returns (n_evaluated, holdout_r2 = corr(predicted, actual)**2, level scale)."""
    predicted_list, actual_list = [], []
    for obs in holdout_obs:
        predicted, _se = predict_tree(regression, obs.features)
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


def check_classic_overfit(model_name, r_squared, holdout_r2):
    """Task constraint: explicit flag if in-sample R² > 0.8 and holdout R² < 0.2."""
    if r_squared is not None and holdout_r2 is not None and r_squared > 0.8 and holdout_r2 < 0.2:
        print(f"  [CLASSIC TREE OVERFITTING] {model_name}: in-sample R²={r_squared:.3f} (>0.8) but holdout R²={holdout_r2:.3f} (<0.2)")
        return True
    return False


# Reused from the prior ridge_regression.py session (same seed=42 split, same
# fixed 4 features) — see that module's ABSOLUTE OLS/Ridge numbers.
OLS_4FEATURE_BASELINE = {
    ("semiconductors", "ev_sales"): {"train_cv_r2": 0.382, "holdout_r2": 0.003, "verdict": "Overfit"},
    ("semiconductors", "ev_ebitda"): {"train_cv_r2": 0.362, "holdout_r2": 0.000, "verdict": "Overfit"},
    ("mature_tech", "ev_sales"): {"train_cv_r2": 0.012, "holdout_r2": 0.772, "verdict": "Confirmed"},
    ("mature_tech", "ev_ebitda"): {"train_cv_r2": 0.000, "holdout_r2": 0.000, "verdict": "Confirmed"},
}
RIDGE_4FEATURE_BASELINE = {
    ("semiconductors", "ev_sales"): {"alpha": 1, "train_cv_r2": 0.161, "holdout_r2": 0.003, "verdict": "Suspicious"},
    ("semiconductors", "ev_ebitda"): {"alpha": 10, "train_cv_r2": 0.140, "holdout_r2": 0.003, "verdict": "Suspicious"},
    ("mature_tech", "ev_sales"): {"alpha": 10, "train_cv_r2": 0.042, "holdout_r2": 0.767, "verdict": "Confirmed"},
    ("mature_tech", "ev_ebitda"): {"alpha": 100, "train_cv_r2": -0.046, "holdout_r2": 0.002, "verdict": "Confirmed"},
}
# Forward-selected features from feature_selection.py / holdout_validation.py
# (adaptive search, NOT the fixed 4) — for the feature-importance comparison
# in step 6, since "features that OLS selected" most naturally refers to
# that adaptive search, not this session's fixed baseline.
OLS_SELECTED_FEATURES = {
    ("semiconductors", "ev_sales"): ["trailing_growth", "gross_margin", "ebitda_margin", "roic"],
    ("semiconductors", "ev_ebitda"): ["r_and_d_intensity", "trailing_growth", "ebitda_margin", "gross_margin", "operating_margin"],
    ("mature_tech", "ev_sales"): ["log_revenue"],
    ("mature_tech", "ev_ebitda"): ["operating_margin"],
}


def run_tree_validation_for_bucket(bucket):
    tickers = COMPS_UNIVERSE[bucket]
    train_tickers, holdout_tickers = split_universe(tickers)  # seed=42 — same split as every prior comparison

    print(f"\n{'=' * 90}\nTREE REGRESSION VALIDATION: {bucket}\n{'=' * 90}")
    print(f"  Universe: {len(tickers)} -> train={len(train_tickers)}, holdout={len(holdout_tickers)} (seed={SPLIT_SEED})")

    print("  Computing extended features + multiples for full universe...")
    features_by_ticker = {t: compute_extended_features_for_ticker(t) for t in tickers}
    multiples_by_ticker = {t: compute_multiples_for_ticker(t) for t in tickers}

    train_features_list = [features_by_ticker[t] for t in train_tickers]
    kept_features, dropped_features = filter_available_features(train_features_list)
    print(f"  10-feature pool after >20% missing filter (TRAINING set only): {kept_features}")
    if dropped_features:
        print(f"  Dropped: { {k: f'{v * 100:.0f}%' for k, v in dropped_features.items()} }")

    results = {}
    for multiple_type in ("ev_sales", "ev_ebitda"):
        train_obs = build_observations_for_tickers(features_by_ticker, multiples_by_ticker, train_tickers, multiple_type)
        holdout_obs = build_observations_for_tickers(features_by_ticker, multiples_by_ticker, holdout_tickers, multiple_type)

        print(f"\n  --- {multiple_type} ---")

        rf_reg = fit_random_forest(train_obs, multiple_type, kept_features)
        n_eval_rf, holdout_r2_rf = evaluate_holdout(rf_reg, holdout_obs)
        rf_verdict = interpret(rf_reg.cv_r2, holdout_r2_rf)
        rf_classic_overfit = check_classic_overfit("RandomForest", rf_reg.r_squared, holdout_r2_rf)
        print(f"  RandomForest:     depth={rf_reg.best_max_depth}  in_sample_r2={rf_reg.r_squared}  train_cv_r2={rf_reg.cv_r2}  holdout_r2={holdout_r2_rf}  verdict={rf_verdict}")
        print(f"    top3 features: {rf_reg.top_features}")

        gb_reg = fit_gradient_boosting(train_obs, multiple_type, kept_features)
        n_eval_gb, holdout_r2_gb = evaluate_holdout(gb_reg, holdout_obs)
        gb_verdict = interpret(gb_reg.cv_r2, holdout_r2_gb)
        gb_classic_overfit = check_classic_overfit("GradientBoosting", gb_reg.r_squared, holdout_r2_gb)
        print(f"  GradientBoosting: in_sample_r2={gb_reg.r_squared}  train_cv_r2={gb_reg.cv_r2}  holdout_r2={holdout_r2_gb}  verdict={gb_verdict}")
        print(f"    top3 features: {gb_reg.top_features}")

        results[multiple_type] = {
            "random_forest": {
                "in_sample_r2": rf_reg.r_squared, "train_cv_r2": rf_reg.cv_r2, "n_holdout_evaluated": n_eval_rf,
                "holdout_r2": holdout_r2_rf, "verdict": rf_verdict, "classic_overfit": rf_classic_overfit,
                "top_features": rf_reg.top_features, "importances": rf_reg.feature_importances, "best_max_depth": rf_reg.best_max_depth,
            },
            "gradient_boosting": {
                "in_sample_r2": gb_reg.r_squared, "train_cv_r2": gb_reg.cv_r2, "n_holdout_evaluated": n_eval_gb,
                "holdout_r2": holdout_r2_gb, "verdict": gb_verdict, "classic_overfit": gb_classic_overfit,
                "top_features": gb_reg.top_features, "importances": gb_reg.feature_importances,
            },
        }

    return {"bucket": bucket, "results": results}


def print_comparison_table(all_results):
    print(f"\n{'=' * 128}")
    print("FINAL COMPARISON — OLS vs RIDGE (4-feature, reused) vs RandomForest vs GradientBoosting (10-feature)")
    print(f"{'=' * 128}")
    print("NOTE: OLS/Ridge rows reused verbatim from ridge_regression.py's session (same seed=42 split). Tree 'Train CV R²'")
    print("is 5-fold CV R² (log scale, sklearn scoring). 'Hold-out R²' is corr(predicted, actual)² on the level scale for all rows.")
    header = f"{'Bucket':<16}{'Multiple':<11}{'Model':<20}{'Train CV R²':>13}{'Holdout R²':>12}  Verdict"
    print(header)
    print("-" * 128)
    any_confirmed_tree = False
    any_classic_overfit = False
    for r in all_results:
        bucket = r["bucket"]
        for multiple_type, res in r["results"].items():
            ols = OLS_4FEATURE_BASELINE[(bucket, multiple_type)]
            print(f"{bucket:<16}{multiple_type:<11}{'OLS 4-feature':<20}{ols['train_cv_r2']:>13.3f}{ols['holdout_r2']:>12.3f}  {ols['verdict']}")
            ridge = RIDGE_4FEATURE_BASELINE[(bucket, multiple_type)]
            print(f"{bucket:<16}{multiple_type:<11}{'Ridge 4-feature':<20}{ridge['train_cv_r2']:>13.3f}{ridge['holdout_r2']:>12.3f}  {ridge['verdict']}")

            for model_name, key in (("RandomForest 10-feature", "random_forest"), ("GradientBoosting 10-feature", "gradient_boosting")):
                m = res[key]
                tcv = f"{m['train_cv_r2']:.3f}" if m["train_cv_r2"] is not None else "N/A"
                hr2 = f"{m['holdout_r2']:.3f}" if m["holdout_r2"] is not None else "N/A"
                print(f"{bucket:<16}{multiple_type:<11}{model_name:<20}{tcv:>13}{hr2:>12}  {m['verdict']}" + ("  [CLASSIC OVERFIT]" if m["classic_overfit"] else ""))
                if m["verdict"] == "Confirmed":
                    any_confirmed_tree = True
                if m["classic_overfit"]:
                    any_classic_overfit = True
            print()
    print("-" * 128)
    if any_confirmed_tree:
        print("\nAt least one tree model Confirmed — trees produced at least one stable, generalizing result.")
    else:
        print("\nNo tree model Confirmed across any bucket x multiple combination.")
    if any_classic_overfit:
        print("At least one tree model showed classic tree-overfitting (in-sample R²>0.8, holdout R²<0.2).")

    print("\nFeature importance (top 3) vs OLS-selected features (adaptive forward selection, prior session):")
    for r in all_results:
        bucket = r["bucket"]
        for multiple_type, res in r["results"].items():
            ols_selected = set(OLS_SELECTED_FEATURES[(bucket, multiple_type)])
            for model_name, key in (("RandomForest", "random_forest"), ("GradientBoosting", "gradient_boosting")):
                top3 = set(res[key]["top_features"])
                overlap = top3 & ols_selected
                print(f"  {bucket}/{multiple_type} {model_name}: top3={sorted(top3)}  OLS-selected={sorted(ols_selected)}  overlap={sorted(overlap)}")


def run_tree_regression_diagnostic():
    all_results = [run_tree_validation_for_bucket(b) for b in ("semiconductors", "mature_tech")]
    print_comparison_table(all_results)
    return all_results


if __name__ == "__main__":
    run_tree_regression_diagnostic()
