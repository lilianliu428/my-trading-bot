"""
test_alpha_operators.py

Unit tests for valuation/alpha/operators.py on synthetic data. No network
calls. Run directly: python3 test_alpha_operators.py
"""

import numpy as np
import pandas as pd

from valuation.alpha.operators import rank, zscore, neutralize

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_rank_ties():
    print("\n--- rank() ---")
    df = pd.DataFrame({
        "date": ["d1"] * 5,
        "val": [10.0, 20.0, 20.0, 30.0, np.nan],
    })
    r = rank(df, "val")

    check("NaN stays NaN", pd.isna(r.iloc[4]))
    check("all non-NaN ranks in (0, 1]", ((r.dropna() > 0) & (r.dropna() <= 1)).all())
    check("tied values (20, 20) get equal ranks", r.iloc[1] == r.iloc[2],
          detail=f"{r.iloc[1]} vs {r.iloc[2]}")
    # average-rank convention: ranks 2 and 3 (of 4 valid) average to 2.5 -> 2.5/4
    expected_tie_rank = 2.5 / 4
    check("tie rank uses averaged-rank convention", abs(r.iloc[1] - expected_tie_rank) < 1e-9,
          detail=f"got {r.iloc[1]}, expected {expected_tie_rank}")
    check("highest value gets rank 1.0 (top of (0,1])", r.iloc[3] == 1.0)

    # Multiple dates: ranks computed independently per date
    df2 = pd.DataFrame({
        "date": ["d1", "d1", "d2", "d2"],
        "val": [1.0, 2.0, 100.0, 200.0],
    })
    r2 = rank(df2, "val")
    check("rank is per-date, not pooled", r2.iloc[0] == r2.iloc[2],
          detail=f"{r2.iloc[0]} vs {r2.iloc[2]} (both should be the low-of-2 rank in their date)")


def test_zscore_zero_variance():
    print("\n--- zscore() ---")
    df = pd.DataFrame({
        "date": ["d1"] * 4,
        "val": [10.0, 20.0, 30.0, np.nan],
    })
    z = zscore(df, "val")
    check("NaN stays NaN", pd.isna(z.iloc[3]))
    check("mean of z-scores is ~0", abs(z.dropna().mean()) < 1e-9, detail=f"{z.dropna().mean()}")

    # Zero-variance date: all valid values identical
    df_flat = pd.DataFrame({
        "date": ["d1"] * 4,
        "val": [5.0, 5.0, 5.0, np.nan],
    })
    z_flat = zscore(df_flat, "val")
    check("zero-variance date does not raise / produce inf or nan for valid rows",
          (z_flat.iloc[:3] == 0.0).all(), detail=f"{z_flat.tolist()}")
    check("zero-variance date still keeps NaN as NaN", pd.isna(z_flat.iloc[3]))

    # Single valid observation on a date (std undefined, not literally 0)
    df_single = pd.DataFrame({
        "date": ["d1"],
        "val": [42.0],
    })
    z_single = zscore(df_single, "val")
    check("single-observation date does not raise / returns 0 rather than NaN/inf",
          z_single.iloc[0] == 0.0, detail=f"{z_single.iloc[0]}")

    # Multiple dates computed independently
    df2 = pd.DataFrame({
        "date": ["d1", "d1", "d2", "d2"],
        "val": [10.0, 20.0, 1000.0, 2000.0],
    })
    z2 = zscore(df2, "val")
    check("z-score is per-date, not pooled", abs(z2.iloc[0] - z2.iloc[2]) < 1e-9,
          detail=f"{z2.iloc[0]} vs {z2.iloc[2]} (should match: both low-of-2 in their date)")


def test_neutralize_group_means():
    print("\n--- neutralize() ---")
    df = pd.DataFrame({
        "date": ["d1"] * 6,
        "bucket": ["semi", "semi", "semi", "mature", "mature", "mature"],
        "val": [10.0, 20.0, 30.0, 100.0, 200.0, 600.0],
    })
    n = neutralize(df, "val", "bucket")

    semi_mask = df["bucket"] == "semi"
    mature_mask = df["bucket"] == "mature"
    check("semi group mean ~0 after neutralize", abs(n[semi_mask].mean()) < 1e-9,
          detail=f"{n[semi_mask].mean()}")
    check("mature group mean ~0 after neutralize", abs(n[mature_mask].mean()) < 1e-9,
          detail=f"{n[mature_mask].mean()}")
    check("cross-group spread is preserved (not fully centered pooled)",
          n[mature_mask].std() > n[semi_mask].std() * 2,
          detail="mature group had much larger raw spread than semi group")

    # Per-date-and-group: same bucket, different dates, should neutralize independently
    df2 = pd.DataFrame({
        "date": ["d1", "d1", "d2", "d2"],
        "bucket": ["semi", "semi", "semi", "semi"],
        "val": [10.0, 20.0, 1000.0, 2000.0],
    })
    n2 = neutralize(df2, "val", "bucket")
    check("neutralize is per-(date,group), not pooled across dates",
          abs(n2.iloc[0] - (-5.0)) < 1e-9, detail=f"{n2.iloc[0]}")

    # NaN preserved
    df3 = pd.DataFrame({
        "date": ["d1", "d1", "d1"],
        "bucket": ["semi", "semi", "semi"],
        "val": [10.0, 20.0, np.nan],
    })
    n3 = neutralize(df3, "val", "bucket")
    check("NaN input stays NaN after neutralize", pd.isna(n3.iloc[2]))


if __name__ == "__main__":
    test_rank_ties()
    test_zscore_zero_variance()
    test_neutralize_group_means()

    print(f"\n{'=' * 60}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")
