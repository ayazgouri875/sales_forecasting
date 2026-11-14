"""
Unit tests for data preprocessing and validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    aggregate_daily_sales,
    aggregate_item_sales,
    aggregate_store_sales,
    get_series,
    load_raw_data,
    split_time_series,
    validate_raw_data,
)


@pytest.fixture
def sample_sales_df():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = []
    for d in dates:
        for s in [1, 2]:
            for i in [1, 2]:
                data.append({"date": d, "store": s, "item": i, "sales": 10 + s + i})
    return pd.DataFrame(data)


def test_validate_raw_data(sample_sales_df):
    summary = validate_raw_data(sample_sales_df)
    assert summary["total_records"] == 400
    assert summary["missing_values"] == 0
    assert summary["duplicate_records"] == 0
    assert summary["unique_stores"] == 2
    assert summary["unique_items"] == 2
    assert summary["total_store_item_combinations"] == 4
    assert summary["is_continuous"] is True


def test_aggregations(sample_sales_df):
    daily = aggregate_daily_sales(sample_sales_df)
    assert len(daily) == 100
    assert "date" in daily.columns
    assert "sales" in daily.columns

    store_daily = aggregate_store_sales(sample_sales_df)
    assert len(store_daily) == 200

    item_daily = aggregate_item_sales(sample_sales_df)
    assert len(item_daily) == 200


def test_get_series(sample_sales_df):
    # Aggregate
    agg_s = get_series(sample_sales_df)
    assert len(agg_s) == 100
    assert isinstance(agg_s.index, pd.DatetimeIndex)

    # Store
    s1 = get_series(sample_sales_df, store=1)
    assert len(s1) == 100

    # Store-Item
    s1_i1 = get_series(sample_sales_df, store=1, item=1)
    assert len(s1_i1) == 100
    assert (s1_i1 == 12).all()


def test_split_time_series_no_leakage(sample_sales_df):
    daily = aggregate_daily_sales(sample_sales_df)
    train, val, test = split_time_series(daily, val_days=20, test_days=20, date_col="date")

    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20

    # Strict chronological boundaries
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
