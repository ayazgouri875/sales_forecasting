"""
Unit tests for stationarity tests and transformations.
"""

import numpy as np
import pandas as pd
import pytest

from src.stationarity import (
    adf_test,
    apply_first_difference,
    apply_log_transform,
    apply_seasonal_difference,
    kpss_test,
)


def test_stationarity_white_noise():
    np.random.seed(42)
    white_noise = pd.Series(np.random.normal(0, 1, 500))
    adf = adf_test(white_noise)
    assert adf["is_stationary"] is True


def test_stationarity_random_walk_differenced():
    np.random.seed(42)
    # Random walk: non-stationary
    rw = pd.Series(np.cumsum(np.random.normal(0, 1, 500)))
    diff = apply_first_difference(rw)

    adf_diff = adf_test(diff)
    assert adf_diff["is_stationary"] is True


def test_transformations():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    d1 = apply_first_difference(s)
    assert len(d1) == 4
    assert (d1 == 10.0).all()

    log_s = apply_log_transform(s)
    assert len(log_s) == 5
    assert np.isclose(log_s.iloc[0], np.log(10.0))
