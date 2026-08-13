"""
Cross-sectional operators for the alpha research layer. Three functions —
the language for expressing alphas, not the deliverable. See
ALPHA_LAYER_SPEC.md "Operators".
"""

import numpy as np
import pandas as pd


def rank(df, col, date_col="date"):
    """
    Cross-sectional percentile rank within each date.
    Returns values in (0, 1]. Ties averaged. NaNs stay NaN.
    """
    return df.groupby(date_col)[col].rank(pct=True, method="average", na_option="keep")


def zscore(df, col, date_col="date"):
    """
    Cross-sectional z-score within each date: (x - mean) / std.
    NaNs stay NaN. If std == 0 (or undefined, e.g. a single valid
    observation) for a date, non-NaN values on that date become 0
    rather than dividing by zero.
    """
    def _z(s):
        std = s.std()
        if not std or pd.isna(std):
            return s.where(s.isna(), 0.0)
        return (s - s.mean()) / std

    return df.groupby(date_col)[col].transform(_z)


def neutralize(df, col, group_col, date_col="date"):
    """
    Subtract the group mean within each (date, group) cell.
    Used to remove bucket effects so the alpha compares semis to semis
    and mature tech to mature tech, rather than betting one sector
    against the other.
    """
    group_means = df.groupby([date_col, group_col])[col].transform("mean")
    return df[col] - group_means
