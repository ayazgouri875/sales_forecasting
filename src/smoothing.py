"""
Retail Demand Forecasting & Seasonal Intelligence
Module: Exponential Smoothing Methods (SES, Holt's Linear, Holt-Winters)
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fit_simple_exponential_smoothing(
    train_series: pd.Series,
    forecast_steps: int = 90,
    smoothing_level: Optional[float] = None,
    optimized: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit Simple Exponential Smoothing (SES):
    Suitable for data with no distinct trend or seasonal pattern.
    y_hat_{t+1|t} = alpha * y_t + (1 - alpha) * y_hat_{t|t-1}
    """
    start_time = time.time()
    model = SimpleExpSmoothing(train_series, initialization_method="estimated")
    fit = model.fit(smoothing_level=smoothing_level, optimized=optimized)
    forecast = fit.forecast(steps=forecast_steps)
    elapsed = time.time() - start_time

    params = {
        "model": "Simple Exponential Smoothing",
        "alpha": float(fit.params.get("smoothing_level", np.nan)),
        "aic": float(fit.aic) if hasattr(fit, "aic") else np.nan,
        "bic": float(fit.bic) if hasattr(fit, "bic") else np.nan,
        "fit_time_sec": elapsed,
    }
    logger.info("Fitted SES (alpha=%.4f) in %.3fs", params["alpha"], elapsed)
    return np.asarray(forecast), params


def fit_holts_linear_trend(
    train_series: pd.Series,
    forecast_steps: int = 90,
    damped_trend: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit Holt's Linear Trend model:
    Captures level and linear trend.
    l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
    b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
    y_hat_{t+h|t} = l_t + h * b_t
    """
    start_time = time.time()
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        damped_trend=damped_trend,
        seasonal=None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(steps=forecast_steps)
    elapsed = time.time() - start_time

    params = {
        "model": "Holt's Linear Trend (Damped)" if damped_trend else "Holt's Linear Trend",
        "alpha": float(fit.params.get("smoothing_level", np.nan)),
        "beta": float(fit.params.get("smoothing_trend", np.nan)),
        "phi": float(fit.params.get("damping_trend", np.nan)) if damped_trend else 1.0,
        "aic": float(fit.aic) if hasattr(fit, "aic") else np.nan,
        "fit_time_sec": elapsed,
    }
    logger.info("Fitted Holt's Trend in %.3fs", elapsed)
    return np.asarray(forecast), params


def fit_holt_winters(
    train_series: pd.Series,
    forecast_steps: int = 90,
    seasonal_periods: int = 7,
    trend: str = "add",
    seasonal: str = "add",
    damped_trend: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit Holt-Winters Exponential Smoothing:
    Captures level, trend, and seasonal oscillations.
    For daily retail sales, seasonal_periods=7 represents the weekly cycle.
    """
    start_time = time.time()
    model = ExponentialSmoothing(
        train_series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(steps=forecast_steps)
    elapsed = time.time() - start_time

    params = {
        "model": f"Holt-Winters ({trend.capitalize()} Trend, {seasonal.capitalize()} Seasonal, Period={seasonal_periods})",
        "alpha": float(fit.params.get("smoothing_level", np.nan)),
        "beta": float(fit.params.get("smoothing_trend", np.nan)),
        "gamma": float(fit.params.get("smoothing_seasonal", np.nan)),
        "aic": float(fit.aic) if hasattr(fit, "aic") else np.nan,
        "bic": float(fit.bic) if hasattr(fit, "bic") else np.nan,
        "fit_time_sec": elapsed,
    }
    logger.info("Fitted Holt-Winters in %.3fs", elapsed)
    return np.asarray(forecast), params
