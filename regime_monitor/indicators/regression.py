# src/indicators/regression.py

"""Rolling linear regression slope utilities.

Includes a small helper to compute slope and R² over a rolling window
and two convenience functions that return either the slope series or
both slope and R² as a DataFrame. The slope can optionally be
computed on the log of the input series and can be normalized by the
window mean.

All rolling computations are fully vectorized (``sliding_window_view`` +
array ops) — closed-form OLS, no Python-level ``rolling().apply()`` callbacks.
``_window_slope_and_r2`` is retained as the per-window reference the vectorized
paths are pinned to in the tests.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class RegSlopeConfig:
    length: int = 50          # window length
    use_log: bool = False     # use log(price) instead of price
    normalize: bool = True    # normalize slope by avg price in window


def _window_slope_and_r2(
    y: pd.Series,
    normalize: bool,
) -> tuple[float, float]:
    """
    Compute slope and R^2 of y against x = 0..N-1 for one window.

    Reference (non-vectorized) implementation; the vectorized paths below are
    kept numerically equivalent to this and pinned to it in the tests.
    """
    n = len(y)
    if n < 2:
        return np.nan, np.nan

    x = np.arange(n, dtype="float64")
    x_mean = x.mean()
    y_mean = y.mean()

    x_centered = x - x_mean
    y_centered = y - y_mean

    ss_x = np.sum(x_centered * x_centered)
    if ss_x == 0:
        return np.nan, np.nan

    # slope b = cov(x, y) / var(x)
    cov_xy = np.sum(x_centered * y_centered)
    slope = cov_xy / ss_x

    # Optional normalization (e.g. per-unit-of-price)
    if normalize and y_mean != 0:
        slope = slope / abs(y_mean)

    # R^2
    y_hat = slope * x_centered + y_mean
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_hat) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return float(slope), float(r2)


def _prepare_y(series: pd.Series, use_log: bool) -> np.ndarray:
    """Return the float64 value array the regression operates on."""
    if use_log:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(series.replace(0, np.nan).to_numpy(dtype="float64"))
    return series.to_numpy(dtype="float64")


def _rolling_slope_r2(
    y: np.ndarray,
    length: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized rolling slope & R^2 of *y* against x = 0..length-1.

    Numerically equivalent to applying :func:`_window_slope_and_r2` over each
    rolling window of size *length* (``min_periods == length``), but computed
    with a single set of array operations instead of a Python-level
    ``rolling().apply()``.  The first ``length - 1`` positions are ``NaN``.

    Notes
    -----
    The R^2 is intentionally computed from the *normalized* slope when
    ``normalize`` is True, matching the original per-window implementation.
    """
    n = y.shape[0]
    slope = np.full(n, np.nan, dtype="float64")
    r2 = np.full(n, np.nan, dtype="float64")
    if length < 2 or n < length:
        return slope, r2

    # x is the same 0..length-1 ramp for every window, so its terms are constants.
    x = np.arange(length, dtype="float64")
    x_centered = x - x.mean()
    ss_x = float(np.sum(x_centered * x_centered))  # > 0 for length >= 2

    windows = sliding_window_view(y, length)        # (n - length + 1, length)
    y_mean = windows.mean(axis=1)                    # (m,)
    y_centered = windows - y_mean[:, None]
    cov_xy = y_centered @ x_centered                 # sum(x_centered * y_centered, axis=1)
    slope_used = cov_xy / ss_x

    if normalize:
        denom = np.abs(y_mean)
        # Where y_mean == 0 the original keeps the un-normalized slope; guard the
        # divisor so those positions don't raise/inf before np.where selects.
        safe_denom = np.where(denom == 0.0, 1.0, denom)
        slope_used = np.where(y_mean != 0.0, slope_used / safe_denom, slope_used)

    # R^2 from the (possibly normalized) slope — preserves original semantics.
    y_hat = slope_used[:, None] * x_centered[None, :] + y_mean[:, None]
    ss_tot = np.sum(y_centered * y_centered, axis=1)
    resid = windows - y_hat
    ss_res = np.sum(resid * resid, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_win = np.where(ss_tot == 0.0, np.nan, 1.0 - ss_res / ss_tot)

    slope[length - 1:] = slope_used
    r2[length - 1:] = r2_win
    return slope, r2


def _rolling_ols_forecast(
    y: np.ndarray,
    length: int,
    project_at: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized rolling OLS forecast + in-window residual std of *y*.

    For each trailing window of size *length* (``min_periods == length``), fit
    ``y`` against x = 0..length-1 (oldest→newest), then evaluate the fitted line
    at ``x = project_at`` (e.g. ``length`` for a 1-step-ahead projection or
    ``length - 1`` for the in-window endpoint). Returns:

        forecast   — fitted value at ``x = project_at``
        resid_std  — population std of the in-window residuals (matches np.std)

    Closed-form OLS with no Python-level ``rolling().apply()``. The first
    ``length - 1`` positions are ``NaN``. NaNs inside a window propagate to that
    window's outputs, matching the original ``raw=True`` apply behavior.
    """
    n = y.shape[0]
    forecast = np.full(n, np.nan, dtype="float64")
    resid_std = np.full(n, np.nan, dtype="float64")
    if length < 2 or n < length:
        return forecast, resid_std

    x = np.arange(length, dtype="float64")
    x_mean = x.mean()
    x_centered = x - x_mean
    ss_x = float(np.sum(x_centered * x_centered))  # > 0 for length >= 2

    windows = sliding_window_view(y, length)        # (m, length)
    y_mean = windows.mean(axis=1)
    y_centered = windows - y_mean[:, None]
    cov_xy = y_centered @ x_centered                 # sum(x_centered * y_centered, axis=1)
    slope = cov_xy / ss_x
    intercept = y_mean - slope * x_mean

    fc = intercept + slope * project_at

    fitted = slope[:, None] * x[None, :] + intercept[:, None]
    resid = windows - fitted
    rstd = np.sqrt(np.mean(resid * resid, axis=1))

    forecast[length - 1:] = fc
    resid_std[length - 1:] = rstd
    return forecast, resid_std


def rolling_ols_forecast(
    series: pd.Series,
    length: int,
    project_at: float,
) -> tuple[pd.Series, pd.Series]:
    """Series wrapper around :func:`_rolling_ols_forecast`.

    ``series`` should already be in the space the fit operates in (e.g. log
    price if the caller wants a log-space forecast). Returns ``(forecast,
    resid_std)`` aligned to ``series.index``.
    """
    y = series.to_numpy(dtype="float64")
    fc, rstd = _rolling_ols_forecast(y, int(length), float(project_at))
    idx = series.index
    return pd.Series(fc, index=idx), pd.Series(rstd, index=idx)


def compute_regression_slope(
    series: pd.Series,
    cfg: RegSlopeConfig = RegSlopeConfig(),
) -> pd.Series:
    """
    Compute linear regression slope over a rolling window.

    Slope is in 'units per bar' (optionally normalized by avg price).

    Parameters
    ----------
    series : pd.Series
        Price series (e.g., close).
    cfg : RegSlopeConfig, optional
        length, use_log, normalize.

    Returns
    -------
    pd.Series
        Rolling slope values.
    """
    y = _prepare_y(series, cfg.use_log)
    slope, _ = _rolling_slope_r2(y, cfg.length, cfg.normalize)
    return pd.Series(slope, index=series.index, name="reg_slope")


def compute_regression_slope_with_r2(
    series: pd.Series,
    cfg: RegSlopeConfig = RegSlopeConfig(),
) -> pd.DataFrame:
    """
    Compute both slope and R^2 of price vs time over a rolling window.

    Returns a DataFrame with:
        - 'slope'
        - 'r2'
    """
    y = _prepare_y(series, cfg.use_log)
    slope, r2 = _rolling_slope_r2(y, cfg.length, cfg.normalize)
    return pd.DataFrame(
        {"slope": slope, "r2": r2},
        index=series.index,
    )
