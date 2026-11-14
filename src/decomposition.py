"""
Retail Demand Forecasting & Seasonal Intelligence
Module: Time Series Classical and STL Decomposition
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def decompose_additive(
    series: pd.Series,
    period: int = 12,
) -> Tuple[object, float]:
    """
    Perform classical additive decomposition: Y_t = Trend_t + Seasonal_t + Residual_t.

    Parameters
    ----------
    series : pd.Series
        Time series indexed with DatetimeIndex and defined frequency.
    period : int
        Seasonal period (e.g., 12 for monthly data, 7 for weekly pattern in daily data).

    Returns
    -------
    Tuple[DecomposeResult, float]
        Decomposition result object and in-sample reconstruction RMSE.
    """
    result = seasonal_decompose(series, model="additive", period=period)
    fitted = result.trend + result.seasonal
    valid_idx = ~(fitted.isna() | series.isna())
    reconstruction_rmse = float(np.sqrt(np.mean((series[valid_idx] - fitted[valid_idx]) ** 2)))

    logger.info("Additive decomposition (period=%d): Reconstruction RMSE = %.2f", period, reconstruction_rmse)
    return result, reconstruction_rmse


def decompose_multiplicative(
    series: pd.Series,
    period: int = 12,
) -> Tuple[object, float]:
    """
    Perform classical multiplicative decomposition: Y_t = Trend_t * Seasonal_t * Residual_t.

    Parameters
    ----------
    series : pd.Series
        Time series with strictly positive values.
    period : int
        Seasonal period (e.g. 12 for monthly, 7 for daily).

    Returns
    -------
    Tuple[DecomposeResult, float]
        Decomposition result object and in-sample reconstruction RMSE.
    """
    if (series <= 0).any():
        raise ValueError("Multiplicative decomposition requires strictly positive values.")

    result = seasonal_decompose(series, model="multiplicative", period=period)
    fitted = result.trend * result.seasonal
    valid_idx = ~(fitted.isna() | series.isna())
    reconstruction_rmse = float(np.sqrt(np.mean((series[valid_idx] - fitted[valid_idx]) ** 2)))

    logger.info("Multiplicative decomposition (period=%d): Reconstruction RMSE = %.2f", period, reconstruction_rmse)
    return result, reconstruction_rmse


def decompose_stl(
    series: pd.Series,
    period: int = 7,
    robust: bool = True,
) -> object:
    """
    Perform STL (Seasonal and Trend decomposition using Loess).
    Robust to outliers and allows seasonal patterns to change over time.

    Parameters
    ----------
    series : pd.Series
        Time series.
    period : int
        Seasonal cycle length (e.g., 7 for day-of-week seasonality, 365 for annual).
    robust : bool
        Whether to apply robust LOESS weights against outliers.

    Returns
    -------
    STLResult
    """
    stl = STL(series, period=period, robust=robust)
    result = stl.fit()
    logger.info("STL decomposition (period=%d, robust=%s) fitted successfully", period, robust)
    return result


def plot_decomposition(
    result: object,
    title: str = "Time Series Decomposition",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot 4 components: Observed, Trend, Seasonal, Residual.
    """
    fig = result.plot()
    fig.set_size_inches(14, 9)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    for ax in fig.axes:
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved decomposition plot to %s", save_path)

    return fig


def compare_decompositions(
    monthly_series: pd.Series,
    save_dir: Union[str, Path] = "outputs/plots",
) -> Dict[str, float]:
    """
    Run and compare Additive and Multiplicative decompositions on monthly series,
    reproducing historical notebook benchmark results.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    add_res, add_rmse = decompose_additive(monthly_series, period=12)
    mult_res, mult_rmse = decompose_multiplicative(monthly_series, period=12)

    plot_decomposition(add_res, title="Monthly Sales — Additive Decomposition (Period=12)", save_path=save_path / "decomposition_additive_monthly.png")
    plot_decomposition(mult_res, title="Monthly Sales — Multiplicative Decomposition (Period=12)", save_path=save_path / "decomposition_multiplicative_monthly.png")

    return {
        "additive_rmse": round(add_rmse, 2),
        "multiplicative_rmse": round(mult_rmse, 2),
    }


if __name__ == "__main__":
    from src.data_preprocessing import load_raw_data, aggregate_daily_sales

    df = load_raw_data()
    daily = aggregate_daily_sales(df).set_index("date")["sales"]
    monthly = daily.resample("ME").sum()

    results = compare_decompositions(monthly)
    print("Decomposition Comparison:", results)
