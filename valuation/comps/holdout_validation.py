"""
Hold-out validation of the forward-selection feature sets — checks whether
the training-set leave-one-out CV R² reported by feature_selection.py is
itself inflated by selection bias. Testing many feature combinations and
reporting the best LOO CV score is a form of overfitting to the available
data even though no single observation leaks across train/predict within
LOO — the selection PROCEDURE itself can find a combination that happens
to fit well by chance, which won't generalize.

Procedure per bucket: split the universe 80/20 (train/hold-out, ticker-
level, seed=42, shared across both multiple types), run forward selection
on the training set ONLY (same procedure/candidates as feature_selection.py),
fit the winning feature set on the full training set once, and evaluate
genuinely out-of-sample on the untouched hold-out tickers.

Read-only diagnostic. Run directly:
python3 -m valuation.comps.holdout_validation
"""

import random

import numpy as np

from valuation.comps.universe import COMPS_UNIVERSE
from valuation.comps.feature_selection import (
    compute_extended_features_for_ticker,
    filter_available_features,
    forward_selection,
)
from valuation.comps.multiples import compute_multiples_for_ticker
from valuation.comps.regression import MultipleObservation, fit_regression, predict_multiple

SPLIT_SEED = 42
TRAIN_FRACTION = 0.8


def split_universe(tickers, seed=SPLIT_SEED, train_fraction=TRAIN_FRACTION):
    """Ticker-level 80/20 split, reproducible via seed. Shared across both multiple types for a bucket."""
    shuffled = list(tickers)
    random.Random(seed).shuffle(shuffled)
    n_train = round(len(shuffled) * train_fraction)
    return shuffled[:n_train], shuffled[n_train:]


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


def evaluate_holdout(regression, holdout_obs):
    """
    Fit-once (on training set), predict-many (on hold-out) evaluation.
    Returns (n_evaluated, holdout_r2) where holdout_r2 is
    corr(predicted, actual)**2 — same definition as loo_cross_validate's
    cv_r2, for direct comparability with the training CV R².
    """
    predicted_list, actual_list = [], []
    for obs in holdout_obs:
        predicted, _se = predict_multiple(regression, obs.features)
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


def interpret(training_cv_r2, holdout_r2):
    """
    diff = training - holdout. Any diff <= 0.10 is Confirmed (this also
    covers the hold-out matching or beating training — only a training
    score that DOESN'T generalize is the failure mode being tested for).
    """
    if training_cv_r2 is None or holdout_r2 is None:
        return "N/A (insufficient data)"
    diff = training_cv_r2 - holdout_r2
    if diff <= 0.10:
        return "Confirmed"
    elif diff <= 0.20:
        return "Suspicious"
    else:
        return "Overfit"


def run_holdout_validation_for_bucket(bucket):
    tickers = COMPS_UNIVERSE[bucket]
    train_tickers, holdout_tickers = split_universe(tickers)

    print(f"\n{'=' * 90}\nHOLD-OUT VALIDATION: {bucket}\n{'=' * 90}")
    print(f"  Universe: {len(tickers)} tickers -> train={len(train_tickers)}, holdout={len(holdout_tickers)} (seed={SPLIT_SEED})")
    print(f"  Holdout tickers: {sorted(holdout_tickers)}")

    print("  Computing extended features + multiples for full universe...")
    features_by_ticker = {t: compute_extended_features_for_ticker(t) for t in tickers}
    multiples_by_ticker = {t: compute_multiples_for_ticker(t) for t in tickers}

    train_features_list = [features_by_ticker[t] for t in train_tickers]
    kept_features, dropped_features = filter_available_features(train_features_list)
    print(f"  Candidate features kept after >20% missing filter (computed on TRAINING set only): {kept_features}")
    if dropped_features:
        print(f"  Dropped: { {k: f'{v * 100:.0f}%' for k, v in dropped_features.items()} }")

    results = {}
    for multiple_type in ("ev_sales", "ev_ebitda"):
        train_obs = build_observations_for_tickers(features_by_ticker, multiples_by_ticker, train_tickers, multiple_type)
        holdout_obs = build_observations_for_tickers(features_by_ticker, multiples_by_ticker, holdout_tickers, multiple_type)

        fs_result = forward_selection(train_obs, multiple_type, kept_features)
        selected = fs_result["selected_features"]
        training_cv_r2 = fs_result["final_cv_r2"]

        if selected:
            final_reg = fit_regression(train_obs, multiple_type, feature_names=selected)
            n_evaluated, holdout_r2 = evaluate_holdout(final_reg, holdout_obs)
            n_train = final_reg.n_observations
        else:
            n_evaluated, holdout_r2, n_train = 0, None, 0

        verdict = interpret(training_cv_r2, holdout_r2)

        print(f"\n  --- {multiple_type} ---")
        print(f"  Winning feature set (training-only forward selection): {selected}")
        print(f"  Training CV R² (LOO, on training set): {training_cv_r2}")
        print(f"  Fitted on {n_train} training obs; evaluated on {n_evaluated} of {len(holdout_tickers)} holdout tickers")
        print(f"  Hold-out R² (corr(predicted, actual)²): {holdout_r2}")
        print(f"  Verdict: {verdict}")

        results[multiple_type] = {
            "selected_features": selected,
            "training_cv_r2": training_cv_r2,
            "n_train": n_train,
            "n_holdout_evaluated": n_evaluated,
            "holdout_r2": holdout_r2,
            "verdict": verdict,
        }

    return {"bucket": bucket, "train_tickers": train_tickers, "holdout_tickers": holdout_tickers, "results": results}


def print_summary_table(all_results):
    print(f"\n{'=' * 112}")
    print("HOLD-OUT VALIDATION — SUMMARY")
    print(f"{'=' * 112}")
    header = f"{'Bucket':<16}{'Multiple':<12}{'Feature set':<55}{'Train CV R²':>12}{'Holdout R²':>11}  Interpretation"
    print(header)
    print("-" * 112)
    any_overfit = False
    for r in all_results:
        for multiple_type, res in r["results"].items():
            tcv = f"{res['training_cv_r2']:.3f}" if res["training_cv_r2"] is not None else "N/A"
            hr2 = f"{res['holdout_r2']:.3f}" if res["holdout_r2"] is not None else "N/A"
            feats = ", ".join(res["selected_features"]) if res["selected_features"] else "(none)"
            print(f"{r['bucket']:<16}{multiple_type:<12}{feats:<55}{tcv:>12}{hr2:>11}  {res['verdict']}")
            if res["verdict"] == "Overfit":
                any_overfit = True
    print("-" * 112)
    if any_overfit:
        print("\n>=1 combination flagged OVERFIT — the corresponding feature-selection result should NOT be trusted.")
    else:
        print("\nNo combination flagged OVERFIT.")


def run_holdout_validation():
    all_results = [run_holdout_validation_for_bucket(b) for b in ("semiconductors", "mature_tech")]
    print_summary_table(all_results)
    return all_results


if __name__ == "__main__":
    run_holdout_validation()
