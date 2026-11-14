"""
Retail Demand Forecasting & Seasonal Intelligence
Module: Data Preprocessing
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(data_path: Union[str, Path] = "data/raw/train.csv") -> pd.DataFrame:
    """
    Load raw historical sales data, parse dates, and sort chronologically.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned, chronologically sorted DataFrame with columns:
        ['date', 'store', 'item', 'sales'].
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")

    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)

    expected_cols = {"date", "store", "item", "sales"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns. Expected {expected_cols}, got {set(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "store", "item"]).reset_index(drop=True)
    logger.info("Loaded %d records spanning from %s to %s", len(df), df["date"].min().date(), df["date"].max().date())
    return df


def validate_raw_data(df: pd.DataFrame) -> Dict[str, Union[int, float, bool, str]]:
    """
    Perform thorough validation checks on the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or preprocessed sales DataFrame.

    Returns
    -------
    dict
        Dictionary containing validation metrics.
    """
    total_records = len(df)
    missing_count = int(df.isnull().sum().sum())
    duplicate_records = int(df.duplicated().sum())
    duplicate_series_keys = int(df.duplicated(subset=["date", "store", "item"]).sum())

    unique_dates = df["date"].nunique()
    min_date = df["date"].min()
    max_date = df["date"].max()
    expected_days = (max_date - min_date).days + 1

    unique_stores = int(df["store"].nunique())
    unique_items = int(df["item"].nunique())
    total_combinations = int(df.groupby(["store", "item"]).ngroups)

    # Check complete observations for all combinations
    obs_per_combo = df.groupby(["store", "item"]).size()
    min_obs_per_combo = int(obs_per_combo.min())
    max_obs_per_combo = int(obs_per_combo.max())

    is_continuous = (unique_dates == expected_days) and (min_obs_per_combo == expected_days) and (max_obs_per_combo == expected_days)

    summary = {
        "total_records": total_records,
        "missing_values": missing_count,
        "duplicate_records": duplicate_records,
        "duplicate_series_keys": duplicate_series_keys,
        "unique_dates": unique_dates,
        "expected_dates": expected_days,
        "min_date": str(min_date.date()),
        "max_date": str(max_date.date()),
        "unique_stores": unique_stores,
        "unique_items": unique_items,
        "total_store_item_combinations": total_combinations,
        "min_obs_per_combo": min_obs_per_combo,
        "max_obs_per_combo": max_obs_per_combo,
        "is_continuous": is_continuous,
        "zero_sales_count": int((df["sales"] == 0).sum()),
        "negative_sales_count": int((df["sales"] < 0).sum()),
        "min_sales": float(df["sales"].min()),
        "max_sales": float(df["sales"].max()),
        "mean_sales": float(df["sales"].mean()),
        "median_sales": float(df["sales"].median()),
    }
    logger.info("Validation complete: continuous=%s, missing=%d, duplicates=%d", is_continuous, missing_count, duplicate_records)
    return summary


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sales to total daily retail sales across all stores and items.

    Returns
    -------
    pd.DataFrame with 'date' column and 'sales' column, sorted by date.
    """
    daily = df.groupby("date", as_index=False)["sales"].sum()
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def aggregate_store_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales by date and store.
    """
    store_daily = df.groupby(["date", "store"], as_index=False)["sales"].sum()
    return store_daily.sort_values(["store", "date"]).reset_index(drop=True)


def aggregate_item_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales by date and item.
    """
    item_daily = df.groupby(["date", "item"], as_index=False)["sales"].sum()
    return item_daily.sort_values(["item", "date"]).reset_index(drop=True)


def get_series(
    df: pd.DataFrame,
    store: Optional[int] = None,
    item: Optional[int] = None,
) -> pd.Series:
    """
    Extract a specific univariate daily time series from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    store : Optional[int]
        Store ID. If None and item is None, returns total aggregate daily sales.
        If None and item is set, returns total daily sales for that item across all stores.
    item : Optional[int]
        Item ID. If None and store is set, returns total daily sales for that store across all items.
        If both store and item are set, returns daily sales for that specific combination.

    Returns
    -------
    pd.Series
        Indexed by DatetimeIndex with daily frequency 'D', values are unit sales.
    """
    subset = df
    if store is not None and item is not None:
        subset = subset[(subset["store"] == store) & (subset["item"] == item)]
        daily = subset.groupby("date")["sales"].sum()
    elif store is not None:
        subset = subset[subset["store"] == store]
        daily = subset.groupby("date")["sales"].sum()
    elif item is not None:
        subset = subset[subset["item"] == item]
        daily = subset.groupby("date")["sales"].sum()
    else:
        daily = subset.groupby("date")["sales"].sum()

    daily = daily.asfreq("D")
    daily.name = "sales"
    return daily


def split_time_series(
    df: Union[pd.DataFrame, pd.Series],
    val_days: int = 90,
    test_days: int = 90,
    date_col: Optional[str] = "date",
) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    """
    Chronologically split a time-series dataset into training, validation, and test partitions
    without future data leakage.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        Dataset with date column or DatetimeIndex.
    val_days : int
        Number of days in the validation window (default: 90).
    test_days : int
        Number of days in the test window (default: 90).
    date_col : Optional[str]
        Name of date column if DataFrame has no DatetimeIndex.

    Returns
    -------
    Tuple[train, val, test]
    """
    if isinstance(df, pd.Series):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Series must have a DatetimeIndex for chronological splitting.")
        dates = pd.Series(df.index, index=df.index)
        max_date = df.index.max()
        test_start = max_date - pd.Timedelta(days=test_days - 1)
        val_start = test_start - pd.Timedelta(days=val_days)

        train = df[df.index < val_start]
        val = df[(df.index >= val_start) & (df.index < test_start)]
        test = df[df.index >= test_start]
        return train, val, test

    if isinstance(df.index, pd.DatetimeIndex):
        max_date = df.index.max()
        test_start = max_date - pd.Timedelta(days=test_days - 1)
        val_start = test_start - pd.Timedelta(days=val_days)

        train = df[df.index < val_start]
        val = df[(df.index >= val_start) & (df.index < test_start)]
        test = df[df.index >= test_start]
        return train, val, test

    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")

    dates = pd.to_datetime(df[date_col])
    max_date = dates.max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = df[dates < val_start].copy()
    val = df[(dates >= val_start) & (dates < test_start)].copy()
    test = df[dates >= test_start].copy()

    return train, val, test


def run_preprocessing_pipeline(
    raw_path: Union[str, Path] = "data/raw/train.csv",
    processed_dir: Union[str, Path] = "data/processed",
) -> Dict[str, Union[str, int, float, bool]]:
    """
    Execute end-to-end preprocessing pipeline and save processed artifacts.
    """
    proc_dir = Path(processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_path)
    summary = validate_raw_data(df)

    # Save summary metadata
    meta_path = proc_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info("Saved dataset metadata to %s", meta_path)

    # Generate and save daily total sales
    daily_total = aggregate_daily_sales(df)
    daily_path = proc_dir / "daily_total_sales.csv"
    daily_total.to_csv(daily_path, index=False)
    logger.info("Saved aggregate daily sales to %s (%d rows)", daily_path, len(daily_total))

    # Generate and save store-level daily sales
    store_daily = aggregate_store_sales(df)
    store_path = proc_dir / "store_daily_sales.csv"
    store_daily.to_csv(store_path, index=False)
    logger.info("Saved store daily sales to %s (%d rows)", store_path, len(store_daily))

    # Generate and save item-level daily sales
    item_daily = aggregate_item_sales(df)
    item_path = proc_dir / "item_daily_sales.csv"
    item_daily.to_csv(item_path, index=False)
    logger.info("Saved item daily sales to %s (%d rows)", item_path, len(item_daily))

    return summary


if __name__ == "__main__":
    run_preprocessing_pipeline()
