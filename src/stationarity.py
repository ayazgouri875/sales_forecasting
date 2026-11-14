"""
Retail Demand Forecasting & Seasonal Intelligence
Module: Stationarity Testing, Differencing, and Statistical Diagnostics
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def adf_test(series: pd.Series, maxlag: Optional[int] = None) -> Dict[str, Union[float, int, bool, str]]:
    """
    Perform Augmented Dickey-Fuller (ADF) test.

    Null Hypothesis (H0): The series has a unit root (Non-Stationary).
    Alternative Hypothesis (H1): The series has no unit root (Stationary).

    Decision: If p-value < 0.05, reject H0 -> Stationary.
    """
    clean_series = series.dropna()
    res = adfuller(clean_series, maxlag=maxlag, autolag="AIC")

    test_stat = float(res[0])
    p_value = float(res[1])
    used_lag = int(res[2])
    n_obs = int(res[3])
    crit_vals = {k: float(v) for k, v in res[4].items()}

    is_stationary = p_value < 0.05

    result = {
        "test": "ADF",
        "test_statistic": round(test_stat, 4),
        "p_value": round(p_value, 5),
        "lags_used": used_lag,
        "n_observations": n_obs,
        "critical_values": crit_vals,
        "is_stationary": is_stationary,
        "interpretation": "Stationary (Reject H0)" if is_stationary else "Non-Stationary (Fail to Reject H0)",
    }
    logger.info("ADF Test: Stat=%.4f, p=%.5f => %s", test_stat, p_value, result["interpretation"])
    return result


def kpss_test(series: pd.Series, regression: str = "c") -> Dict[str, Union[float, int, bool, str]]:
    """
    Perform Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

    Null Hypothesis (H0): The series is stationary around a deterministic trend/level.
    Alternative Hypothesis (H1): The series has a unit root (Non-Stationary).

    Decision: If p-value < 0.05, reject H0 -> Non-Stationary.
    """
    clean_series = series.dropna()
    # statsmodels kpss returns (stat, p_value, lags, crit_dict)
    res = kpss(clean_series, regression=regression, nlags="auto")

    test_stat = float(res[0])
    p_value = float(res[1])
    used_lag = int(res[2])
    crit_vals = {k: float(v) for k, v in res[3].items()}

    # For KPSS, rejecting H0 means non-stationary!
    is_stationary = p_value >= 0.05

    result = {
        "test": "KPSS",
        "test_statistic": round(test_stat, 4),
        "p_value": round(p_value, 5),
        "lags_used": used_lag,
        "critical_values": crit_vals,
        "is_stationary": is_stationary,
        "interpretation": "Stationary (Fail to Reject H0)" if is_stationary else "Non-Stationary (Reject H0)",
    }
    logger.info("KPSS Test: Stat=%.4f, p=%.5f => %s", test_stat, p_value, result["interpretation"])
    return result


def comprehensive_stationarity_check(series: pd.Series, series_name: str = "Series") -> Dict[str, Union[str, float, bool]]:
    """
    Combine ADF and KPSS tests into a unified diagnostic verdict.

    Cases:
    1. ADF Stationary + KPSS Stationary -> Confirmed Stationary
    2. ADF Non-Stationary + KPSS Non-Stationary -> Confirmed Non-Stationary (Unit root)
    3. ADF Stationary + KPSS Non-Stationary -> Difference Stationary
    4. ADF Non-Stationary + KPSS Stationary -> Trend Stationary
    """
    adf_res = adf_test(series)
    kpss_res = kpss_test(series)

    if adf_res["is_stationary"] and kpss_res["is_stationary"]:
        verdict = "Strictly Stationary"
    elif not adf_res["is_stationary"] and not kpss_res["is_stationary"]:
        verdict = "Strictly Non-Stationary (Unit Root)"
    elif adf_res["is_stationary"] and not kpss_res["is_stationary"]:
        verdict = "Difference Stationary"
    else:
        verdict = "Trend Stationary"

    return {
        "series": series_name,
        "adf_stat": adf_res["test_statistic"],
        "adf_pvalue": adf_res["p_value"],
        "adf_result": "Stationary" if adf_res["is_stationary"] else "Non-Stationary",
        "kpss_stat": kpss_res["test_statistic"],
        "kpss_pvalue": kpss_res["p_value"],
        "kpss_result": "Stationary" if kpss_res["is_stationary"] else "Non-Stationary",
        "overall_verdict": verdict,
    }


def apply_first_difference(series: pd.Series) -> pd.Series:
    """Apply first-order differencing: Delta y_t = y_t - y_{t-1}."""
    diff = series.diff().dropna()
    diff.name = f"{series.name}_diff1" if series.name else "diff1"
    return diff


def apply_seasonal_difference(series: pd.Series, period: int = 7) -> pd.Series:
    """Apply seasonal differencing: Delta_s y_t = y_t - y_{t-s}."""
    diff = series.diff(period).dropna()
    diff.name = f"{series.name}_diff{period}" if series.name else f"diff{period}"
    return diff


def apply_log_transform(series: pd.Series) -> pd.Series:
    """Apply natural logarithm transform: ln(y_t). Stabilizes exponential variance."""
    if (series <= 0).any():
        raise ValueError("Log transformation requires strictly positive values.")
    logged = np.log(series)
    logged.name = f"{series.name}_log" if series.name else "log"
    return logged


def plot_stationarity_diagnostics(
    series: pd.Series,
    title: str = "Time Series Stationarity Diagnostics",
    window: int = 30,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Produce a 4-panel diagnostic figure:
    1. Series with rolling mean and rolling std
    2. Histogram & KDE
    3. Autocorrelation (ACF) plot
    4. Partial Autocorrelation (PACF) plot
    """
    clean_series = series.dropna()
    rolling_mean = clean_series.rolling(window=window).mean()
    rolling_std = clean_series.rolling(window=window).std()

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    # Panel 1: Series with rolling stats
    axes[0, 0].plot(clean_series.index, clean_series, label="Observations", color="steelblue", alpha=0.6, linewidth=1.0)
    axes[0, 0].plot(clean_series.index, rolling_mean, label=f"{window}-day Rolling Mean", color="crimson", linewidth=1.8)
    axes[0, 0].plot(clean_series.index, rolling_std, label=f"{window}-day Rolling Std", color="darkgreen", linewidth=1.5, linestyle="--")
    axes[0, 0].set_title(f"{title} — Rolling Statistics", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Sales", fontsize=10)
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Distribution
    axes[0, 1].hist(clean_series, bins=30, density=True, color="teal", alpha=0.7, edgecolor="black")
    clean_series.plot(kind="kde", ax=axes[0, 1], color="darkred", linewidth=1.8)
    axes[0, 1].set_title(f"{title} — Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Value", fontsize=10)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: ACF
    plot_acf(clean_series, lags=40, ax=axes[1, 0], color="navy", auto_ylims=True)
    axes[1, 0].set_title("Autocorrelation Function (ACF)", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 4: PACF
    plot_pacf(clean_series, lags=40, ax=axes[1, 1], color="darkviolet", method="ywm", auto_ylims=True)
    axes[1, 1].set_title("Partial Autocorrelation Function (PACF)", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info("Saved stationarity diagnostics plot to %s", save_path)

    return fig


def run_stationarity_analysis(
    series: pd.Series,
    save_dir: Union[str, Path] = "outputs/plots",
) -> pd.DataFrame:
    """
    Run stationarity analysis on raw series, first difference, seasonal difference (7),
    and combined (first + seasonal) difference.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Raw series
    plot_stationarity_diagnostics(series, title="Raw Daily Total Sales", save_path=save_path / "stationarity_raw.png")
    check_raw = comprehensive_stationarity_check(series, "Raw Sales")

    # 2. First difference
    diff1 = apply_first_difference(series)
    plot_stationarity_diagnostics(diff1, title="First Differenced Sales (Delta_1)", save_path=save_path / "stationarity_diff1.png")
    check_diff1 = comprehensive_stationarity_check(diff1, "First Difference (Delta_1)")

    # 3. Seasonal difference (7)
    diff7 = apply_seasonal_difference(series, period=7)
    plot_stationarity_diagnostics(diff7, title="Seasonal Differenced Sales (Delta_7)", save_path=save_path / "stationarity_diff7.png")
    check_diff7 = comprehensive_stationarity_check(diff7, "Seasonal Difference (Delta_7)")

    # 4. Combined first and seasonal difference
    diff_both = apply_first_difference(diff7)
    plot_stationarity_diagnostics(diff_both, title="Seasonal + First Differenced Sales (Delta_1 Delta_7)", save_path=save_path / "stationarity_diff1_diff7.png")
    check_both = comprehensive_stationarity_check(diff_both, "Combined (Delta_1 Delta_7)")

    summary_df = pd.DataFrame([check_raw, check_diff1, check_diff7, check_both])
    logger.info("Stationarity Summary:\n%s", summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    from src.data_preprocessing import load_raw_data, aggregate_daily_sales

    df = load_raw_data()
    daily_sales = aggregate_daily_sales(df).set_index("date")["sales"]
    summary = run_stationarity_analysis(daily_sales)
    print("\nStationarity Diagnostic Table:")
    print(summary)
