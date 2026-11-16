"""Statistical utilities: confidence intervals and rank correlation.

Choices (explained in docs/methodology.md):
  * For means of skewed, budget-censored quantities such as
    "steps to first complete cycle" we use PERCENTILE BOOTSTRAP confidence
    intervals instead of assuming normality.
  * For success rates (binomial proportions) we use the WILSON score
    interval, which behaves well even at 0% or 100% success.
  * Spearman rank correlation is computed manually (no scipy dependency).
"""

import math

import numpy as np
import pandas as pd

import config as C


def mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def median(values):
    return float(np.median(values)) if len(values) else float("nan")


def bootstrap_ci_mean(values, n_resamples=C.BOOTSTRAP_RESAMPLES,
                      seed=C.SEED_BOOTSTRAP, ci_level=0.95):
    """95% percentile-bootstrap CI for the mean.

    Resample the observed values with replacement ``n_resamples`` times,
    compute the mean of each resample, and take the 2.5 / 97.5 percentiles.
    Fixed ``seed`` keeps the analysis reproducible.  Requires at least one
    value; with very few values the interval is wide -- that is honest.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return (float(lo), float(hi))


def wilson_ci(successes, total, z=1.96, ci_level=None):
    """Wilson score interval for a binomial proportion.

    z = 1.96 corresponds to a 95% two-sided interval (the value is fixed
    here so the project needs no scipy dependency).

    The interval is clamped to [0, 1] and to contain the observed proportion
    p-hat (both are exact-arithmetic properties of the Wilson interval that
    floating-point round-off can violate by one ulp at the extremes: at
    p = 0 or p = 1 an endpoint can land infinitesimally outside the unit
    interval, e.g. -5e-17 which renders as -0.0, and at p = 1 the upper
    endpoint can land one ulp below 1.0).  The clamps keep the reported
    interval consistent with the exact interval.
    """
    if ci_level is not None and abs(ci_level - 0.95) > 1e-9:
        raise ValueError("this helper only implements the 95% interval")
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total
                         + z * z / (4.0 * total * total)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    # Guarantee containment of p-hat (one-ulp violations at the extremes).
    lo = min(lo, p)
    hi = max(hi, p)
    # Adding +0.0 normalises a possible IEEE -0.0 endpoint to +0.0.
    return (lo + 0.0, hi + 0.0)


def _ranks(values):
    """Average ranks for ties (pandas handles this correctly)."""
    return pd.Series(values).rank(method="average").to_numpy()


def spearman_rho(x, y):
    """Spearman rank correlation between two equal-length sequences.

    Implemented as Pearson correlation of the ranks; returns NaN when fewer
    than 2 paired values exist or when either variable is constant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or x.size != y.size:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
