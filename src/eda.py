"""
Retail Demand Forecasting & Seasonal Intelligence
Module: Exploratory Data Analysis & Automated Plot Generation
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_distribution_stats(series: pd.Series) -> Dict[str, float]:
    """
    Compute statistical distribution metrics including skewness, kurtosis,
    and Tukey IQR outlier bounds.
    """
    clean = series.dropna()
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_count = int(((clean < lower_bound) | (clean > upper_bound)).sum())

    stats = {
        "mean": round(float(clean.mean()), 2),
        "std": round(float(clean.std()), 2),
        "median": round(float(clean.median()), 2),
        "skewness": round(float(clean.skew()), 3),
        "kurtosis": round(float(clean.kurtosis()), 3),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_fence": round(lower_bound, 2),
        "upper_fence": round(upper_bound, 2),
        "outlier_count": outliers_count,
        "outlier_percentage": round((outliers_count / len(clean)) * 100, 2),
    }
    return stats


def plot_overall_sales_trend(
    daily_sales: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot aggregate daily retail sales trend over full 5-year timeline."""
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(daily_sales["date"], daily_sales["sales"], color="royalblue", linewidth=1.2, label="Daily Total Sales")

    # 30-day and 90-day moving averages
    ma30 = daily_sales["sales"].rolling(30).mean()
    ma90 = daily_sales["sales"].rolling(90).mean()
    ax.plot(daily_sales["date"], ma30, color="darkorange", linewidth=2.0, label="30-Day Moving Average")
    ax.plot(daily_sales["date"], ma90, color="crimson", linewidth=2.2, linestyle="--", label="90-Day Moving Average")

    ax.set_title("5-Year Total Retail Demand Trend (2013 - 2017)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Total Unit Sales", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info("Saved overall trend plot to %s", save_path)
    return fig


def plot_weekday_analysis(
    daily_sales: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Analyze sales distribution across days of the week."""
    df = daily_sales.copy()
    if "day_name" not in df.columns:
        df["day_name"] = df["date"].dt.day_name()
        df["day_of_week"] = df["date"].dt.dayofweek

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Boxplot
    sns.boxplot(data=df, x="day_name", y="sales", order=day_order, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Sales Distribution by Day of Week", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Day of Week", fontsize=11)
    axes[0].set_ylabel("Total Daily Sales", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Average Sales barplot
    day_avg = df.groupby("day_name")["sales"].mean().reindex(day_order).reset_index()
    sns.barplot(data=day_avg, x="day_name", y="sales", ax=axes[1], palette="Blues_r")
    axes[1].set_title("Average Daily Sales by Day of Week", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Day of Week", fontsize=11)
    axes[1].set_ylabel("Mean Sales", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                         ha="center", va="bottom", fontsize=10, xytext=(0, 4), textcoords="offset points")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info("Saved weekday analysis plot to %s", save_path)
    return fig


def plot_monthly_seasonality_heatmap(
    daily_sales: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Generate Year vs Month sales heatmap to illustrate annual seasonal cycles."""
    df = daily_sales.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    pivot = df.pivot_table(index="year", columns="month", values="sales", aggfunc="sum")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = month_names

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Monthly Total Sales"}, ax=ax)
    ax.set_title("Monthly Total Sales Heatmap by Year", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Year", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info("Saved monthly heatmap to %s", save_path)
    return fig


def plot_store_item_performance(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot aggregate sales comparisons across all 10 stores and top/bottom items."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Store Sales
    store_sales = df.groupby("store")["sales"].sum().reset_index().sort_values("sales", ascending=False)
    sns.barplot(data=store_sales, x="store", y="sales", order=store_sales["store"], ax=axes[0], palette="viridis")
    axes[0].set_title("Total Sales by Store (All 10 Stores)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Store ID", fontsize=11)
    axes[0].set_ylabel("Total Cumulative Sales", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Item Sales: Top 10 and Bottom 10
    item_sales = df.groupby("item")["sales"].sum().reset_index().sort_values("sales", ascending=False)
    top10 = item_sales.head(10).assign(Category="Top 10")
    bottom10 = item_sales.tail(10).assign(Category="Bottom 10")
    combined_items = pd.concat([top10, bottom10])

    sns.barplot(data=combined_items, x="item", y="sales", hue="Category", dodge=False, ax=axes[1],
                order=combined_items["item"], palette={"Top 10": "teal", "Bottom 10": "coral"})
    axes[1].set_title("Top 10 vs Bottom 10 Items by Total Sales", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Item ID", fontsize=11)
    axes[1].set_ylabel("Total Cumulative Sales", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info("Saved store/item performance plot to %s", save_path)
    return fig


def run_full_eda(
    raw_path: Union[str, Path] = "data/raw/train.csv",
    save_dir: Union[str, Path] = "outputs/plots",
) -> Dict[str, Union[dict, float, int]]:
    """Execute complete automated EDA and save all presentation plots."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.data_preprocessing import load_raw_data, aggregate_daily_sales

    df = load_raw_data(raw_path)
    daily_sales = aggregate_daily_sales(df)

    dist_stats = compute_distribution_stats(daily_sales["sales"])

    plot_overall_sales_trend(daily_sales, save_path=out_dir / "eda_overall_trend.png")
    plot_weekday_analysis(daily_sales, save_path=out_dir / "eda_weekday_analysis.png")
    plot_monthly_seasonality_heatmap(daily_sales, save_path=out_dir / "eda_monthly_heatmap.png")
    plot_store_item_performance(df, save_path=out_dir / "eda_store_item_performance.png")

    logger.info("Completed automated EDA pipeline. Distribution stats: %s", dist_stats)
    return dist_stats


if __name__ == "__main__":
    run_full_eda()
