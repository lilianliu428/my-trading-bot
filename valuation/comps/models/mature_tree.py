"""
Session 12: does a tree-based model (RandomForest / GradientBoosting)
provide more STABLE mature_tech EV/EBITDA predictions than the current
linear model, even if headline R² is lower?

Motivation: Session 9 (tree_regression.py) dismissed tree models on
mature_tech due to a negative-training-CV-R²-paired-with-strong-holdout-R²
pattern that looked like a small-sample artifact. Session 11 found the
shipped LINEAR model's hold-out R² swings roughly 6x (0.437 -> 0.074)
between data pulls on the IDENTICAL split/code, purely from live market
data drift. Since a RandomForest averages over 100 bootstrap trees (and
its leaf predictions are bounded by the observed training range, unlike
OLS's unbounded linear extrapolation), it may be more robust to exactly
this kind of day-to-day noise. This session tests that directly, with a
genuine multi-seed stability check rather than relying on a single split.

Isolated to mature_tech EV/EBITDA. Does not touch semi_forest.py or
peer_display.py. Only switches mature_relative.py if the decision logic
finds Scenario A (see run_decision()), preserving its public API
(fit_mature_relative, predict_mature_relative, MatureRelativeModel,
MATURE_RELATIVE_FEATURES, VALIDATED_HOLDOUT_R2) so final_engine.py needs
no changes.

Run directly: python3 -m valuation.comps.models.mature_tree
"""

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from valuation.comps.multiples import winsorize, compute_multiples_for_ticker
from valuation.comps.regression import MIN_OBSERVATIONS, _feature_vector_complete, MultipleObservation
from valuation.comps.feature_selection import compute_extended_features_for_ticker, filter_available_features
from valuation.comps.holdout_validation import split_universe, interpret, SPLIT_SEED
from valuation.comps.universe import COMPS_UNIVERSE, get_universe

CV_FOLDS = 5
RANDOM_STATE = 42
RF_PARAM_GRID = {"max_depth": [3, 5, 7]}
RF_FIXED = dict(n_estimators=100, min_samples_split=5, min_samples_leaf=3, random_state=RANDOM_STATE)
GB_PARAM_GRID = {"max_depth": [3, 5]}
GB_FIXED = dict(n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE)
STABILITY_SEEDS = [42, 43, 44, 45, 46]


@dataclass
class TreeMatureResult:
    model_type: str          # "RandomForest" or "GradientBoosting"
    formulation: str         # "absolute" (log-transformed) or "relative" (untransformed)
    n_observations: int
    r_squared: float | None      # in-sample, native scale
    cv_r2: float | None          # 5-fold, native scale (log for absolute, raw ratio for relative)
    best_max_depth: int | None
    bucket_median: float | None  # relative only
    feature_names: list = field(default_factory=list)
    data_flags: list = field(default_factory=list)
    _model: object = None
    _scaler: object = None


def _grid_search(model_type):
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if model_type == "RandomForest":
        return GridSearchCV(RandomForestRegressor(**RF_FIXED), RF_PARAM_GRID, cv=cv, scoring="r2")
    return GridSearchCV(GradientBoostingRegressor(**GB_FIXED), GB_PARAM_GRID, cv=cv, scoring="r2")


def fit_tree_absolute(observations, multiple_type, feature_names, model_type):
    """RandomForest/GradientBoosting on log(multiple) ~ standardized features."""
    data_flags = []
    usable = [
        obs for obs in observations
        if obs.multiple_type == multiple_type and obs.multiple_value is not None and obs.multiple_value > 0
        and _feature_vector_complete(obs.features, feature_names)
    ]
    dropped = len(observations) - len(usable)
    if dropped:
        data_flags.append(f"Dropped {dropped} of {len(observations)} (missing features or non-positive multiple)")

    min_needed = max(MIN_OBSERVATIONS, CV_FOLDS)
    if len(usable) < min_needed:
        data_flags.append(f"Only {len(usable)} usable (< {min_needed}) — not fit")
        return TreeMatureResult(model_type, "absolute", len(usable), None, None, None, None, feature_names, data_flags)

    X_raw = np.array([[getattr(obs.features, f) for f in feature_names] for obs in usable], dtype=float)
    y_raw = np.array([obs.multiple_value for obs in usable], dtype=float)
    y_wins = winsorize(y_raw)
    X_wins = np.column_stack([winsorize(X_raw[:, i]) for i in range(X_raw.shape[1])])
    y_log = np.log(y_wins)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_wins)

    grid = _grid_search(model_type)
    grid.fit(X_scaled, y_log)
    model = grid.best_estimator_
    r_squared = float(model.score(X_scaled, y_log))
    cv_r2 = float(grid.best_score_)
    best_depth = grid.best_params_["max_depth"]

    data_flags.append(f"Fit on {len(usable)} obs, max_depth={best_depth}, in-sample R²={r_squared:.3f}, 5-fold CV R²={cv_r2:.3f}")
    return TreeMatureResult(model_type, "absolute", len(usable), r_squared, cv_r2, best_depth, None, feature_names, data_flags, model, scaler)


def fit_tree_relative(observations, multiple_type, feature_names, model_type):
    """
    RandomForest/GradientBoosting on relative_multiple = (multiple -
    bucket_median) / bucket_median ~ standardized features. No log
    transform needed — trees don't require target positivity, unlike OLS.
    """
    data_flags = []
    usable_raw = [
        obs.multiple_value for obs in observations
        if obs.multiple_type == multiple_type and obs.multiple_value is not None and obs.multiple_value > 0
    ]
    if len(usable_raw) < 2:
        data_flags.append("Fewer than 2 valid observations — cannot compute bucket median")
        return TreeMatureResult(model_type, "relative", 0, None, None, None, None, feature_names, data_flags)
    bucket_median = float(np.median(usable_raw))

    rows, targets = [], []
    for obs in observations:
        if obs.multiple_type != multiple_type or obs.multiple_value is None or obs.multiple_value <= 0:
            continue
        if not _feature_vector_complete(obs.features, feature_names):
            continue
        rows.append([getattr(obs.features, f) for f in feature_names])
        targets.append((obs.multiple_value - bucket_median) / bucket_median)

    min_needed = max(MIN_OBSERVATIONS, CV_FOLDS)
    if len(rows) < min_needed:
        data_flags.append(f"Only {len(rows)} usable (< {min_needed}) — not fit")
        return TreeMatureResult(model_type, "relative", len(rows), None, None, None, bucket_median, feature_names, data_flags)

    X_raw = np.array(rows, dtype=float)
    y_raw = np.array(targets, dtype=float)
    X_wins = np.column_stack([winsorize(X_raw[:, i]) for i in range(X_raw.shape[1])])
    y_wins = winsorize(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_wins)

    grid = _grid_search(model_type)
    grid.fit(X_scaled, y_wins)
    model = grid.best_estimator_
    r_squared = float(model.score(X_scaled, y_wins))
    cv_r2 = float(grid.best_score_)
    best_depth = grid.best_params_["max_depth"]

    data_flags.append(f"Fit on {len(rows)} obs, bucket_median={bucket_median:.3f}, max_depth={best_depth}, in-sample R²={r_squared:.3f}, 5-fold CV R²={cv_r2:.3f}")
    return TreeMatureResult(model_type, "relative", len(rows), r_squared, cv_r2, best_depth, bucket_median, feature_names, data_flags, model, scaler)


def predict_tree_absolute(result, features):
    if result is None or result._model is None or not _feature_vector_complete(features, result.feature_names):
        return None, None
    x_scaled = result._scaler.transform(np.array([[getattr(features, f) for f in result.feature_names]]))
    predicted_log = float(result._model.predict(x_scaled)[0])
    return float(np.exp(predicted_log)), None


def predict_tree_relative(result, features):
    if result is None or result._model is None or result.bucket_median is None or not _feature_vector_complete(features, result.feature_names):
        return None, None
    x_scaled = result._scaler.transform(np.array([[getattr(features, f) for f in result.feature_names]]))
    predicted_relative = float(result._model.predict(x_scaled)[0])
    predicted_multiple = result.bucket_median * (1 + predicted_relative)
    if predicted_multiple <= 0:
        return None, None
    return predicted_multiple, None


def evaluate_holdout(predict_fn, result, holdout_obs):
    predicted, actual = [], []
    for obs in holdout_obs:
        p, _ = predict_fn(result, obs.features)
        if p is not None:
            predicted.append(p)
            actual.append(obs.multiple_value)
    if len(predicted) < 2:
        return len(predicted), None
    p_arr, a_arr = np.array(predicted), np.array(actual)
    if np.std(p_arr) == 0 or np.std(a_arr) == 0:
        return len(predicted), None
    return len(predicted), float(np.corrcoef(p_arr, a_arr)[0, 1] ** 2)


def _build_obs(subset, features_by_ticker, multiples_by_ticker, multiple_type="ev_ebitda"):
    obs = []
    for t in subset:
        f, m = features_by_ticker.get(t), multiples_by_ticker.get(t)
        if f is None or m is None or m.get(multiple_type) is None:
            continue
        obs.append(MultipleObservation(t, multiple_type, m[multiple_type], f))
    return obs


MODEL_COMBOS = [
    ("RandomForest", "absolute", fit_tree_absolute, predict_tree_absolute),
    ("RandomForest", "relative", fit_tree_relative, predict_tree_relative),
    ("GradientBoosting", "absolute", fit_tree_absolute, predict_tree_absolute),
    ("GradientBoosting", "relative", fit_tree_relative, predict_tree_relative),
]


def run_part1(features_by_ticker, multiples_by_ticker, kept_features):
    print(f"\n{'=' * 100}\nPART 1: FIT 4 MODEL COMBOS (5-fold CV on full universe + 80/20 hold-out)\n{'=' * 100}")

    bucket = "mature_tech"
    tickers = COMPS_UNIVERSE[bucket]
    all_obs = _build_obs(tickers, features_by_ticker, multiples_by_ticker)
    train_tickers, holdout_tickers = split_universe(tickers)  # seed=42
    train_obs = _build_obs(train_tickers, features_by_ticker, multiples_by_ticker)
    holdout_obs = _build_obs(holdout_tickers, features_by_ticker, multiples_by_ticker)

    results = {}
    for model_type, formulation, fit_fn, predict_fn in MODEL_COMBOS:
        print(f"\n--- {model_type} / {formulation} ---")
        full_fit = fit_fn(all_obs, "ev_ebitda", kept_features, model_type)
        print(f"  Full-universe fit: {full_fit.data_flags}")
        negative_cv = full_fit.cv_r2 is not None and full_fit.cv_r2 < 0
        if negative_cv:
            print(f"  [NEGATIVE 5-FOLD CV R² = {full_fit.cv_r2:.3f}] — Session 9's artifact signature; checking hold-out below")

        train_fit = fit_fn(train_obs, "ev_ebitda", kept_features, model_type)
        n_eval, holdout_r2 = evaluate_holdout(predict_fn, train_fit, holdout_obs)
        verdict = interpret(train_fit.cv_r2, holdout_r2) if train_fit.cv_r2 is not None else "N/A (not fit)"

        print(f"  5-fold CV R² (full universe, native scale): {full_fit.cv_r2}")
        print(f"  80/20 hold-out R² (multiple scale, seed={SPLIT_SEED}): {holdout_r2}  (n_eval={n_eval})")
        print(f"  Verdict: {verdict}")

        is_artifact_pattern = negative_cv and holdout_r2 is not None and holdout_r2 > 0.3
        if is_artifact_pattern:
            print(f"  [ARTIFACT-PATTERN CHECK] negative training CV ({full_fit.cv_r2:.3f}) + strong hold-out ({holdout_r2:.3f}) "
                  f"— same shape as Session 9's dismissed result. With n_eval={n_eval} hold-out points, a few outlier "
                  f"names dominating the Pearson correlation remains the more likely explanation than genuine skill "
                  f"the training CV somehow failed to detect — but reported explicitly rather than silently excluded.")

        results[(model_type, formulation)] = {
            "full_fit": full_fit, "train_fit": train_fit, "cv_r2": full_fit.cv_r2,
            "holdout_r2": holdout_r2, "n_eval": n_eval, "verdict": verdict,
            "negative_cv": negative_cv, "artifact_pattern": is_artifact_pattern,
            "fit_fn": fit_fn, "predict_fn": predict_fn,
        }

    return results


def run_part2(results, features_by_ticker, multiples_by_ticker, kept_features):
    from valuation.comps.models.mature_relative import fit_mature_relative, predict_mature_relative

    print(f"\n{'=' * 100}\nPART 2: OUTLIER TICKER PREDICTIONS\n{'=' * 100}")
    special_tickers = ["GOOGL", "AAPL", "MSFT", "META", "CSCO", "ORCL", "CRM", "ADBE"]
    bucket = "mature_tech"

    eligible = [(k, v) for k, v in results.items() if v["cv_r2"] is not None and v["cv_r2"] >= 0]
    print(f"Eligible combos (non-negative 5-fold CV R²): {[k for k, _ in eligible]}")
    if not eligible:
        print("No tree combo has non-negative CV R² — Part 2 predictions limited to linear comparison only.")

    ticker_predictions = {}
    for ticker in special_tickers:
        target_features = features_by_ticker.get(ticker)
        target_multiples = multiples_by_ticker.get(ticker)
        actual = target_multiples.get("ev_ebitda") if target_multiples else None
        print(f"\n{ticker}: actual={actual}")

        universe_tickers = get_universe(bucket, exclude_ticker=ticker)
        loo_obs = _build_obs(universe_tickers, features_by_ticker, multiples_by_ticker)

        linear_model, _ = fit_mature_relative(loo_obs)
        linear_pred, _m, _f = predict_mature_relative(linear_model, target_features)
        print(f"  Linear (shipped, Session 7/10/11):        {linear_pred}")

        preds_this_ticker = {"actual": actual, "linear": linear_pred}
        for (model_type, formulation), v in eligible:
            fit_fn, predict_fn = v["fit_fn"], v["predict_fn"]
            m = fit_fn(loo_obs, "ev_ebitda", kept_features, model_type)
            pred, _ = predict_fn(m, target_features)
            flag = ""
            if pred is not None and actual is not None and actual > 0 and pred > 3 * actual:
                flag = "  [CONCERNING: >3x actual]"
            print(f"  {model_type}/{formulation}: {pred}{flag}")
            preds_this_ticker[f"{model_type}/{formulation}"] = pred

        ticker_predictions[ticker] = preds_this_ticker

    return ticker_predictions


def run_part3(results, features_by_ticker, multiples_by_ticker, kept_features):
    from valuation.comps.models.mature_relative import fit_mature_relative, predict_mature_relative

    print(f"\n{'=' * 100}\nPART 3: STABILITY CHECK ACROSS 5 SPLITS (seeds {STABILITY_SEEDS})\n{'=' * 100}")

    eligible = [(k, v) for k, v in results.items() if v["cv_r2"] is not None and v["cv_r2"] >= 0 and v["holdout_r2"] is not None]
    if not eligible:
        print("No eligible tree model (all negative CV R² or no hold-out result) — cannot run stability check.")
        return None

    best_key, best_val = max(eligible, key=lambda kv: kv[1]["holdout_r2"])
    model_type, formulation = best_key
    fit_fn, predict_fn = best_val["fit_fn"], best_val["predict_fn"]
    print(f"Best model (by seed=42 hold-out R²={best_val['holdout_r2']:.3f}): {model_type} / {formulation}")

    bucket = "mature_tech"
    tickers = COMPS_UNIVERSE[bucket]

    tree_r2s, linear_r2s = [], []
    for seed in STABILITY_SEEDS:
        train_t, holdout_t = split_universe(tickers, seed=seed)
        train_obs = _build_obs(train_t, features_by_ticker, multiples_by_ticker)
        holdout_obs = _build_obs(holdout_t, features_by_ticker, multiples_by_ticker)

        tree_fit = fit_fn(train_obs, "ev_ebitda", kept_features, model_type)
        _n, tree_r2 = evaluate_holdout(predict_fn, tree_fit, holdout_obs)
        tree_r2s.append(tree_r2)

        linear_model, _ = fit_mature_relative(train_obs)
        lin_pred, lin_act = [], []
        for obs in holdout_obs:
            p, _m, _f = predict_mature_relative(linear_model, obs.features)
            if p is not None:
                lin_pred.append(p)
                lin_act.append(obs.multiple_value)
        linear_r2 = None
        if len(lin_pred) >= 2 and np.std(lin_pred) > 0 and np.std(lin_act) > 0:
            linear_r2 = float(np.corrcoef(lin_pred, lin_act)[0, 1] ** 2)
        linear_r2s.append(linear_r2)

        print(f"  seed={seed}: {model_type}/{formulation} R²={tree_r2}   linear R²={linear_r2}")

    valid_tree = [r for r in tree_r2s if r is not None]
    valid_linear = [r for r in linear_r2s if r is not None]
    tree_mean = float(np.mean(valid_tree)) if valid_tree else None
    tree_std = float(np.std(valid_tree)) if valid_tree else None
    linear_mean = float(np.mean(valid_linear)) if valid_linear else None
    linear_std = float(np.std(valid_linear)) if valid_linear else None

    print(f"\nTree ({model_type}/{formulation}): mean={tree_mean}, std={tree_std}  (n={len(valid_tree)}/{len(STABILITY_SEEDS)} valid)")
    print(f"Linear (shipped):           mean={linear_mean}, std={linear_std}  (n={len(valid_linear)}/{len(STABILITY_SEEDS)} valid)")
    if tree_std is not None and linear_std is not None:
        print(f"Tree {'MORE' if tree_std < linear_std else 'LESS'} stable than linear across these 5 splits (lower std = more stable).")

    return {
        "model_type": model_type, "formulation": formulation, "fit_fn": fit_fn, "predict_fn": predict_fn,
        "tree_r2s": tree_r2s, "linear_r2s": linear_r2s,
        "tree_mean": tree_mean, "tree_std": tree_std, "linear_mean": linear_mean, "linear_std": linear_std,
    }


def run_decision(stability_result):
    print(f"\n{'=' * 100}\nPART 4: DECISION\n{'=' * 100}")
    if stability_result is None:
        print("No eligible tree model reached the stability check.")
        print("SCENARIO B: no methodology reliably predicts mature_tech EV/EBITDA via trees.")
        print("Recommendation: keep current linear implementation (with defensive guard).")
        return "B", None

    mean_r2, std_r2 = stability_result["tree_mean"], stability_result["tree_std"]
    print(f"Best tree model: {stability_result['model_type']} / {stability_result['formulation']}")
    print(f"Mean hold-out R² across {len(STABILITY_SEEDS)} splits: {mean_r2}")
    print(f"Std hold-out R² across {len(STABILITY_SEEDS)} splits: {std_r2}")

    if mean_r2 is None or mean_r2 <= 0.20:
        scenario = "B"
        reason = f"mean hold-out R² ({mean_r2}) does not exceed 0.20"
    elif std_r2 is not None and std_r2 < 0.10:
        scenario = "A"
        reason = f"mean R²={mean_r2:.3f} > 0.20 AND std={std_r2:.3f} < 0.10"
    else:
        # std >= 0.10, including the task's explicit >0.15 case and the
        # 0.10-0.15 gap the task's literal thresholds don't cover — treated
        # conservatively as "not stable enough to ship" either way.
        scenario = "C"
        reason = f"mean R²={mean_r2:.3f} > 0.20 but std={std_r2:.3f} >= 0.10 (not reproducible enough)"

    print(f"\nScenario {scenario}: {reason}")
    if scenario == "A":
        print("Recommend SWITCHING mature_tech EV/EBITDA to the tree model.")
    else:
        print("Recommendation: KEEP current linear implementation (with defensive guard) — "
              "same outcome as Scenario B per the task's explicit instruction for Scenario C.")

    return scenario, stability_result


def run_full_diagnostic():
    bucket = "mature_tech"
    tickers = COMPS_UNIVERSE[bucket]
    print(f"Universe: {len(tickers)} tickers (mature_tech, per Session 4 expansion)")
    print("Computing extended features + multiples for full universe...")
    features_by_ticker = {t: compute_extended_features_for_ticker(t) for t in tickers}
    multiples_by_ticker = {t: compute_multiples_for_ticker(t) for t in tickers}

    kept_features, dropped_features = filter_available_features(list(features_by_ticker.values()))
    print(f"10-feature pool after >20% missing filter (full universe): {kept_features}")
    if dropped_features:
        print(f"Dropped: { {k: f'{v * 100:.0f}%' for k, v in dropped_features.items()} }")

    part1_results = run_part1(features_by_ticker, multiples_by_ticker, kept_features)
    part2_results = run_part2(part1_results, features_by_ticker, multiples_by_ticker, kept_features)
    part3_result = run_part3(part1_results, features_by_ticker, multiples_by_ticker, kept_features)
    scenario, stability_result = run_decision(part3_result)

    return {
        "part1": part1_results, "part2": part2_results, "part3": part3_result,
        "scenario": scenario, "kept_features": kept_features,
        "features_by_ticker": features_by_ticker, "multiples_by_ticker": multiples_by_ticker,
    }


if __name__ == "__main__":
    run_full_diagnostic()
