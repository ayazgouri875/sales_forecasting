"""
Retail Demand Forecasting & Seasonal Intelligence
Module: Forecast Evaluation Metrics & Diagnostic Visualizations
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE): (1/n) * sum(|y - y_hat|)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE): sqrt((1/n) * sum((y - y_hat)^2))."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE):
    (100/n) * sum(|(y - y_hat) / (y + epsilon)|).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE):
    (100/n) * sum( 2 * |y - y_hat| / (|y| + |y_hat| + epsilon) ).
    Ranges between 0% and 200%.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom < epsilon, epsilon, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def weighted_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate Weighted Absolute Percentage Error (WAPE):
    sum(|y - y_hat|) / sum(y + epsilon) * 100%.
    Robust to low sales values and heavily preferred in retail supply chains.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    total_sales = float(np.sum(y_true))
    if total_sales < epsilon:
        return 0.0
    return float((np.sum(np.abs(y_true - y_pred)) / total_sales) * 100.0)


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality_period: int = 7,
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE):
    MAE / (in-sample MAE of seasonal naive benchmark).
    If y_train is None, fallback uses non-seasonal lag-1 differencing on y_true.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mae_model = mean_absolute_error(y_true, y_pred)

    if y_train is not None and len(y_train) > seasonality_period:
        y_tr = np.asarray(y_train, dtype=float)
        scale = np.mean(np.abs(y_tr[seasonality_period:] - y_tr[:-seasonality_period]))
    else:
        if len(y_true) <= seasonality_period:
            scale = np.mean(np.abs(np.diff(y_true))) if len(y_true) > 1 else 1.0
        else:
            scale = np.mean(np.abs(y_true[seasonality_period:] - y_true[:-seasonality_period]))

    if scale < 1e-6:
        scale = 1.0
    return float(mae_model / scale)


def calculate_all_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    y_train: Optional[Union[np.ndarray, pd.Series, list]] = None,
    seasonality_period: int = 7,
    model_name: Optional[str] = None,
    training_time: Optional[float] = None,
) -> Dict[str, Union[str, float]]:
    """
    Calculate full evaluation metrics suite for time series predictions.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    y_tr = np.asarray(y_train, dtype=float) if y_train is not None else None

    # Handle negative predictions by clipping at 0 (sales cannot be negative)
    y_p_clipped = np.clip(y_p, 0, None)

    metrics = {
        "MAE": round(mean_absolute_error(y_t, y_p_clipped), 2),
        "RMSE": round(root_mean_squared_error(y_t, y_p_clipped), 2),
        "sMAPE": round(symmetric_mean_absolute_percentage_error(y_t, y_p_clipped), 2),
        "WAPE": round(weighted_absolute_percentage_error(y_t, y_p_clipped), 2),
        "MAPE": round(mean_absolute_percentage_error(y_t, y_p_clipped), 2),
        "MASE": round(mean_absolute_scaled_error(y_t, y_p_clipped, y_tr, seasonality_period), 3),
    }
    if model_name:
        metrics["Model"] = model_name
    if training_time is not None:
        metrics["Training_Time_Sec"] = round(training_time, 3)

    return metrics


def plot_forecast_vs_actual(
    dates: pd.DatetimeIndex,
    y_true: Union[pd.Series, np.ndarray],
    predictions_dict: Dict[str, Union[pd.Series, np.ndarray]],
    title: str = "Forecast vs Actual Sales",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot actual sales versus multiple model predictions over a time window.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(dates, y_true, label="Actual Sales", color="black", linewidth=2.0)

    palette = sns.color_palette("tab10", len(predictions_dict))
    for (name, preds), color in zip(predictions_dict.items(), palette):
        ax.plot(dates, preds, label=f"Forecast: {name}", linestyle="--", linewidth=1.75, color=color)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Unit Sales", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info("Saved forecast comparison plot to %s", save_path)

    return fig


def plot_residuals(
    dates: pd.DatetimeIndex,
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate diagnostic residual plots: Residuals over time and Residual Distribution.
    """
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Residuals over time
    axes[0].plot(dates, residuals, color="crimson", linewidth=1.2, label="Residuals (Actual - Predicted)")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"{model_name} — Residuals Over Time", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Residual", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Residual distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color="darkblue", bins=30)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero Error")
    axes[1].axvline(np.mean(residuals), color="orange", linestyle=":", linewidth=1.5, label=f"Mean Error: {np.mean(residuals):.2f}")
    axes[1].set_title(f"{model_name} — Residual Error Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Residual Error", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info("Saved residual diagnostic plot to %s", save_path)

    return fig
