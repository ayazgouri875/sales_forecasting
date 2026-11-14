"""
Retail Demand Forecasting & Seasonal Intelligence
Module: Forecasting Models, Feature Engineering, Multi-Series Strategy & Recursive Multi-Step Engine
"""

import copy
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.smoothing import fit_holt_winters, fit_holts_linear_trend, fit_simple_exponential_smoothing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# SECTION 1: FEATURE ENGINEERING (STRICTLY NO DATA LEAKAGE)
# =====================================================================

def extract_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Generate calendar features from a date column.
    """
    data = df.copy()
    dt = pd.to_datetime(data[date_col])

    data["year"] = dt.dt.year
    data["month"] = dt.dt.month
    data["day"] = dt.dt.day
    data["day_of_week"] = dt.dt.dayofweek
    data["day_of_year"] = dt.dt.dayofyear
    data["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    data["quarter"] = dt.dt.quarter
    data["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    data["is_month_start"] = dt.dt.is_month_start.astype(int)
    data["is_month_end"] = dt.dt.is_month_end.astype(int)

    # Cyclical representations
    data["sin_dow"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["cos_dow"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
    data["sin_doy"] = np.sin(2 * np.pi * data["day_of_year"] / 365.25)
    data["cos_doy"] = np.cos(2 * np.pi * data["day_of_year"] / 365.25)

    return data


def generate_lag_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: Optional[List[str]] = None,
    lags: List[int] = [1, 7, 14, 28, 30, 90, 365],
    windows: List[int] = [7, 14, 28],
) -> pd.DataFrame:
    """
    Generate lag and rolling statistics features strictly without future leakage.
    All rolling statistics are applied AFTER a shift(1), guaranteeing that the current
    day's observation is never included in the feature set.
    """
    data = df.copy()
    n_rows = len(data)
    # Only keep lags and windows that fit within available series length
    effective_lags = [l for l in lags if l < n_rows]
    effective_windows = [w for w in windows if w < n_rows]

    if group_cols:
        grouped = data.groupby(group_cols)[target_col]
        # Lags
        for lag in effective_lags:
            data[f"lag_{lag}"] = grouped.shift(lag)

        # Rolling features (shifted by 1 to prevent current-step target leakage)
        for w in effective_windows:
            data[f"rolling_mean_{w}"] = grouped.transform(lambda s: s.shift(1).rolling(w).mean())
            data[f"rolling_std_{w}"] = grouped.transform(lambda s: s.shift(1).rolling(w).std())
    else:
        # Univariate series
        for lag in effective_lags:
            data[f"lag_{lag}"] = data[target_col].shift(lag)

        for w in effective_windows:
            data[f"rolling_mean_{w}"] = data[target_col].shift(1).rolling(w).mean()
            data[f"rolling_std_{w}"] = data[target_col].shift(1).rolling(w).std()

    return data


def build_ml_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "sales",
    date_col: str = "date",
    group_cols: Optional[List[str]] = None,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build complete feature matrix X, target y, and list of feature column names.
    """
    data = extract_calendar_features(df, date_col=date_col)
    data = generate_lag_rolling_features(data, target_col=target_col, group_cols=group_cols)

    exclude_cols = {date_col, target_col}
    feature_cols = [c for c in data.columns if c not in exclude_cols]

    if drop_na:
        data = data.dropna().reset_index(drop=True)

    X = data[feature_cols]
    y = data[target_col]
    return X, y, feature_cols


# =====================================================================
# SECTION 2: BASELINE FORECASTING MODELS
# =====================================================================

def forecast_naive(train_series: Union[pd.Series, np.ndarray], steps: int) -> np.ndarray:
    """Naive Baseline: y_hat_{t+h} = y_t (last observed value)."""
    last_val = np.asarray(train_series)[-1]
    return np.full(steps, last_val, dtype=float)


def forecast_seasonal_naive(
    train_series: Union[pd.Series, np.ndarray],
    steps: int,
    period: int = 7,
) -> np.ndarray:
    """
    Seasonal Naive Baseline: y_hat_{t+h} = y_{t+h - period}.
    Repeats the last observed seasonal cycle of length `period`.
    """
    arr = np.asarray(train_series)
    if len(arr) < period:
        raise ValueError(f"Training series length ({len(arr)}) must be >= seasonal period ({period}).")
    last_cycle = arr[-period:]
    repeats = int(np.ceil(steps / period))
    forecast = np.tile(last_cycle, repeats)[:steps]
    return np.asarray(forecast, dtype=float)


def forecast_mean(train_series: Union[pd.Series, np.ndarray], steps: int) -> np.ndarray:
    """Mean Baseline: y_hat_{t+h} = mean(y_train)."""
    mean_val = float(np.mean(train_series))
    return np.full(steps, mean_val, dtype=float)


def forecast_moving_average(
    train_series: Union[pd.Series, np.ndarray],
    steps: int,
    window: int = 30,
) -> np.ndarray:
    """Moving Average Baseline: y_hat_{t+h} = mean of last `window` observations."""
    arr = np.asarray(train_series)
    window_actual = min(len(arr), window)
    ma_val = float(np.mean(arr[-window_actual:]))
    return np.full(steps, ma_val, dtype=float)


def forecast_drift(train_series: Union[pd.Series, np.ndarray], steps: int) -> np.ndarray:
    """
    Drift Baseline: Draws a straight line between the first and last observation:
    y_hat_{t+h} = y_t + h * (y_t - y_1) / (T - 1)
    """
    arr = np.asarray(train_series)
    T = len(arr)
    if T < 2:
        return np.full(steps, arr[-1], dtype=float)
    slope = (arr[-1] - arr[0]) / (T - 1)
    h = np.arange(1, steps + 1)
    return np.asarray(arr[-1] + h * slope, dtype=float)


# =====================================================================
# SECTION 3: STATISTICAL & TIME SERIES MODELS (ARIMA, SARIMA)
# =====================================================================

def fit_arima_forecast(
    train_series: pd.Series,
    steps: int = 90,
    order: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit ARIMA(p, d, q) model and generate out-of-sample forecasts."""
    start_time = time.time()
    model = SARIMAX(train_series, order=order, seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.forecast(steps=steps)
    elapsed = time.time() - start_time

    params = {
        "model": f"ARIMA{order}",
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "fit_time_sec": elapsed,
    }
    logger.info("Fitted ARIMA%s in %.3fs (AIC=%.1f)", order, elapsed, fit.aic)
    return np.asarray(fc, dtype=float), params


def fit_sarima_forecast(
    train_series: pd.Series,
    steps: int = 90,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit SARIMA(p, d, q)x(P, D, Q)_s model with weekly seasonal period (s=7)."""
    start_time = time.time()
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.forecast(steps=steps)
    elapsed = time.time() - start_time

    params = {
        "model": f"SARIMA{order}x{seasonal_order}",
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "fit_time_sec": elapsed,
    }
    logger.info("Fitted SARIMA%sx%s in %.3fs (AIC=%.1f)", order, seasonal_order, elapsed, fit.aic)
    return np.asarray(fc, dtype=float), params


# =====================================================================
# SECTION 4: MACHINE LEARNING MODELS & RECURSIVE MULTI-STEP ENGINE
# =====================================================================

def get_ml_model_instances() -> Dict[str, Any]:
    """Instantiate standard scikit-learn regressors with fixed random seed."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=100.0, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, max_depth=8, learning_rate=0.08, random_state=42),
    }


def recursive_ml_forecast(
    model: Any,
    full_history_df: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    feature_cols: List[str],
    target_col: str = "sales",
    date_col: str = "date",
    lags: List[int] = [1, 7, 14, 28, 30, 90, 365],
    windows: List[int] = [7, 14, 28],
) -> np.ndarray:
    """
    Perform rigorous recursive multi-step forecasting for machine learning models.

    At each step h in 1..H:
    1. Construct features for forecast_date[h] using only past actuals and prior predictions.
    2. Predict y_hat_{t+h}.
    3. Append predicted y_hat_{t+h} to the running history buffer.
    4. Proceed to step h+1.
    Guarantees absolute zero future data leakage.
    """
    history = full_history_df[[date_col, target_col]].copy()
    history[date_col] = pd.to_datetime(history[date_col])
    predictions = []

    for f_date in forecast_dates:
        # 1. Calendar features
        day_of_week = f_date.dayofweek
        day_of_year = f_date.dayofyear
        row = {
            "year": f_date.year,
            "month": f_date.month,
            "day": f_date.day,
            "day_of_week": day_of_week,
            "day_of_year": day_of_year,
            "week_of_year": int(f_date.isocalendar().week),
            "quarter": f_date.quarter,
            "is_weekend": int(day_of_week in [5, 6]),
            "is_month_start": int(f_date.is_month_start),
            "is_month_end": int(f_date.is_month_end),
            "sin_dow": np.sin(2 * np.pi * day_of_week / 7.0),
            "cos_dow": np.cos(2 * np.pi * day_of_week / 7.0),
            "sin_doy": np.sin(2 * np.pi * day_of_year / 365.25),
            "cos_doy": np.cos(2 * np.pi * day_of_year / 365.25),
        }

        # 2. Lag features from running history
        hist_len = len(history)
        sales_arr = history[target_col].values
        for lag in lags:
            val = sales_arr[-lag] if hist_len >= lag else sales_arr[0]
            row[f"lag_{lag}"] = val

        # 3. Rolling statistics from running history (strictly past values)
        for w in windows:
            slice_vals = sales_arr[-w:] if hist_len >= w else sales_arr
            row[f"rolling_mean_{w}"] = float(np.mean(slice_vals))
            row[f"rolling_std_{w}"] = float(np.std(slice_vals)) if len(slice_vals) > 1 else 0.0

        # Construct single feature row matching training columns
        x_row = pd.DataFrame([row])[feature_cols]

        pred = float(model.predict(x_row)[0])
        pred = max(0.0, pred)  # unit demand cannot be negative
        predictions.append(pred)

        # Append prediction to history buffer for subsequent steps
        new_row = pd.DataFrame({date_col: [f_date], target_col: [pred]})
        history = pd.concat([history, new_row], ignore_index=True)

    return np.asarray(predictions, dtype=float)


# =====================================================================
# SECTION 5: MODEL COMPARISON & VALIDATION RUNNER
# =====================================================================

def run_model_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "sales",
    date_col: str = "date",
    seasonality_period: int = 7,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Run baseline, statistical, and machine learning models on a train/validation split.
    Calculates MAE, RMSE, sMAPE, WAPE, MAPE, MASE, and records training execution time.
    """
    from src.evaluation import calculate_all_metrics

    train_series = train_df.set_index(date_col)[target_col].asfreq("D")
    y_test = test_df[target_col].values
    test_dates = pd.DatetimeIndex(test_df[date_col])
    steps = len(test_df)
    results = []
    predictions_dict = {}

    # 1. BASELINE MODELS
    logger.info("Evaluating Baseline Models...")

    # Naive
    t0 = time.time()
    p_naive = forecast_naive(train_series, steps)
    results.append(calculate_all_metrics(y_test, p_naive, train_series.values, seasonality_period, "Naive Baseline", time.time() - t0))
    predictions_dict["Naive Baseline"] = p_naive

    # Seasonal Naive
    t0 = time.time()
    p_snaive = forecast_seasonal_naive(train_series, steps, period=seasonality_period)
    results.append(calculate_all_metrics(y_test, p_snaive, train_series.values, seasonality_period, "Seasonal Naive (s=7)", time.time() - t0))
    predictions_dict["Seasonal Naive (s=7)"] = p_snaive

    # Mean
    t0 = time.time()
    p_mean = forecast_mean(train_series, steps)
    results.append(calculate_all_metrics(y_test, p_mean, train_series.values, seasonality_period, "Mean Baseline", time.time() - t0))
    predictions_dict["Mean Baseline"] = p_mean

    # Moving Average (30d)
    t0 = time.time()
    p_ma = forecast_moving_average(train_series, steps, window=30)
    results.append(calculate_all_metrics(y_test, p_ma, train_series.values, seasonality_period, "Moving Average (30d)", time.time() - t0))
    predictions_dict["Moving Average (30d)"] = p_ma

    # Drift
    t0 = time.time()
    p_drift = forecast_drift(train_series, steps)
    results.append(calculate_all_metrics(y_test, p_drift, train_series.values, seasonality_period, "Drift Baseline", time.time() - t0))
    predictions_dict["Drift Baseline"] = p_drift

    # 2. STATISTICAL MODELS
    logger.info("Evaluating Statistical Models...")

    # SES
    try:
        t0 = time.time()
        p_ses, _ = fit_simple_exponential_smoothing(train_series, steps)
        results.append(calculate_all_metrics(y_test, p_ses, train_series.values, seasonality_period, "Simple Exp Smoothing", time.time() - t0))
        predictions_dict["Simple Exp Smoothing"] = p_ses
    except Exception as e:
        logger.warning("SES failed: %s", e)

    # Holt's Linear
    try:
        t0 = time.time()
        p_holt, _ = fit_holts_linear_trend(train_series, steps)
        results.append(calculate_all_metrics(y_test, p_holt, train_series.values, seasonality_period, "Holt's Linear Trend", time.time() - t0))
        predictions_dict["Holt's Linear Trend"] = p_holt
    except Exception as e:
        logger.warning("Holt's Linear failed: %s", e)

    # Holt-Winters Additive
    try:
        t0 = time.time()
        p_hw, _ = fit_holt_winters(train_series, steps, seasonal_periods=7, trend="add", seasonal="add")
        results.append(calculate_all_metrics(y_test, p_hw, train_series.values, seasonality_period, "Holt-Winters (Additive)", time.time() - t0))
        predictions_dict["Holt-Winters (Additive)"] = p_hw
    except Exception as e:
        logger.warning("Holt-Winters failed: %s", e)

    # ARIMA(1, 1, 1)
    try:
        t0 = time.time()
        p_arima, _ = fit_arima_forecast(train_series, steps, order=(1, 1, 1))
        results.append(calculate_all_metrics(y_test, p_arima, train_series.values, seasonality_period, "ARIMA(1,1,1)", time.time() - t0))
        predictions_dict["ARIMA(1,1,1)"] = p_arima
    except Exception as e:
        logger.warning("ARIMA failed: %s", e)

    # SARIMA(1, 1, 1)x(1, 1, 1)_7
    try:
        t0 = time.time()
        p_sarima, _ = fit_sarima_forecast(train_series, steps, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        results.append(calculate_all_metrics(y_test, p_sarima, train_series.values, seasonality_period, "SARIMA(1,1,1)(1,1,1)[7]", time.time() - t0))
        predictions_dict["SARIMA(1,1,1)(1,1,1)[7]"] = p_sarima
    except Exception as e:
        logger.warning("SARIMA failed: %s", e)

    # 3. MACHINE LEARNING MODELS
    logger.info("Evaluating Machine Learning Models...")
    X_train, y_tr, feature_cols = build_ml_feature_matrix(train_df, target_col=target_col, date_col=date_col)
    ml_models = get_ml_model_instances()

    for name, model in ml_models.items():
        try:
            t0 = time.time()
            model.fit(X_train, y_tr)
            fit_time = time.time() - t0

            # Recursive out-of-sample forecast
            p_ml = recursive_ml_forecast(
                model=model,
                full_history_df=train_df,
                forecast_dates=test_dates,
                feature_cols=feature_cols,
                target_col=target_col,
                date_col=date_col,
            )
            total_time = fit_time + (time.time() - t0)
            results.append(calculate_all_metrics(y_test, p_ml, train_series.values, seasonality_period, name, total_time))
            predictions_dict[name] = p_ml
            logger.info("Evaluated ML model %s: RMSE=%.2f, sMAPE=%.2f%%", name, results[-1]["RMSE"], results[-1]["sMAPE"])
        except Exception as e:
            logger.warning("ML Model %s failed: %s", name, e)

    comparison_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return comparison_df, predictions_dict
