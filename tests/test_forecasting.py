"""
Unit tests for forecasting models, feature engineering, and baselines.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.forecasting import (
    build_ml_feature_matrix,
    extract_calendar_features,
    forecast_drift,
    forecast_mean,
    forecast_moving_average,
    forecast_naive,
    forecast_seasonal_naive,
    generate_lag_rolling_features,
    recursive_ml_forecast,
)


@pytest.fixture
def daily_series():
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    sales = np.arange(100) + 10.0
    return pd.DataFrame({"date": dates, "sales": sales})


def test_calendar_features(daily_series):
    feats = extract_calendar_features(daily_series)
    assert "day_of_week" in feats.columns
    assert "is_weekend" in feats.columns
    assert "sin_dow" in feats.columns
    assert "cos_dow" in feats.columns


def test_lag_rolling_no_future_leakage(daily_series):
    feats = generate_lag_rolling_features(daily_series, target_col="sales", lags=[1, 7], windows=[7])
    # lag_1 for index 1 should be sales at index 0 (10.0)
    assert feats.loc[1, "lag_1"] == 10.0
    # rolling_mean_7 at index 7 should be mean of sales from index 0..6 (10..16) -> 13.0
    # and MUST NOT include index 7
    assert feats.loc[7, "rolling_mean_7"] == pytest.approx(13.0)


def test_baselines(daily_series):
    train_s = daily_series["sales"].values
    steps = 14

    p_naive = forecast_naive(train_s, steps)
    assert len(p_naive) == steps
    assert p_naive[0] == train_s[-1]

    p_snaive = forecast_seasonal_naive(train_s, steps, period=7)
    assert len(p_snaive) == steps
    assert np.array_equal(p_snaive[:7], train_s[-7:])
    assert np.array_equal(p_snaive[7:14], train_s[-7:])

    p_mean = forecast_mean(train_s, steps)
    assert len(p_mean) == steps
    assert p_mean[0] == pytest.approx(float(np.mean(train_s)))

    p_drift = forecast_drift(train_s, steps)
    assert len(p_drift) == steps
    assert p_drift[0] > train_s[-1]  # positive slope


def test_recursive_ml_forecast_non_negative(daily_series):
    X, y, feat_cols = build_ml_feature_matrix(daily_series, target_col="sales", date_col="date")
    model = LinearRegression()
    model.fit(X, y)

    future_dates = pd.date_range("2021-04-11", periods=10, freq="D")
    preds = recursive_ml_forecast(
        model=model,
        full_history_df=daily_series,
        forecast_dates=future_dates,
        feature_cols=feat_cols,
        target_col="sales",
        date_col="date",
        lags=[1, 7, 14, 28, 30, 90],
        windows=[7, 14, 28],
    )
    assert len(preds) == 10
    assert (preds >= 0).all()
