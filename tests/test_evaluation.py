"""
Unit tests for forecast evaluation metrics.
"""

import numpy as np
import pytest

from src.evaluation import (
    calculate_all_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
)


def test_error_metrics_perfect_prediction():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])

    assert mean_absolute_error(y_true, y_pred) == 0.0
    assert root_mean_squared_error(y_true, y_pred) == 0.0
    assert mean_absolute_percentage_error(y_true, y_pred) == 0.0
    assert symmetric_mean_absolute_percentage_error(y_true, y_pred) == 0.0
    assert weighted_absolute_percentage_error(y_true, y_pred) == 0.0


def test_error_metrics_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])  # errors: +2, -2, +3 -> abs errors: 2, 2, 3

    assert round(mean_absolute_error(y_true, y_pred), 4) == round(7.0 / 3.0, 4)
    # RMSE: sqrt((4 + 4 + 9) / 3) = sqrt(17 / 3) = 2.38047
    assert round(root_mean_squared_error(y_true, y_pred), 4) == round(np.sqrt(17.0 / 3.0), 4)
    # WAPE: sum(abs error) / sum(y_true) = 7 / 60 = 11.6667%
    assert round(weighted_absolute_percentage_error(y_true, y_pred), 2) == 11.67


def test_mase_benchmark():
    y_train = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0])
    y_true = np.array([30.0, 32.0])
    y_pred = np.array([31.0, 33.0])

    mase = mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality_period=2)
    assert mase > 0.0
    assert isinstance(mase, float)


def test_non_negative_clipping_in_calculate_all():
    y_true = np.array([10.0, 20.0])
    y_pred_negative = np.array([-5.0, 20.0])

    # After clipping, pred becomes [0, 20]
    metrics = calculate_all_metrics(y_true, y_pred_negative)
    assert metrics["MAE"] == 5.0
