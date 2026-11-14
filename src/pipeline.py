"""
Retail Demand Forecasting & Seasonal Intelligence
Master Pipeline: End-to-End Training, Evaluation, Model Selection, and Multi-Horizon Forecast Generation
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data_preprocessing import load_raw_data, aggregate_daily_sales, get_series, split_time_series
from src.evaluation import calculate_all_metrics, plot_forecast_vs_actual, plot_residuals
from src.forecasting import (
    build_ml_feature_matrix,
    forecast_naive,
    forecast_seasonal_naive,
    forecast_moving_average,
    get_ml_model_instances,
    recursive_ml_forecast,
    run_model_comparison,
)
from src.smoothing import fit_holt_winters

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def execute_aggregate_pipeline(
    raw_path: str = "data/raw/train.csv",
    output_dir: str = "outputs",
) -> Dict[str, pd.DataFrame]:
    """
    Execute full forecasting lifecycle on total aggregate retail demand:
    1. Chronological splitting (Train, Val 90d, Test 90d)
    2. Validation phase: Compare baseline, statistical, and ML models
    3. Model selection: Pick winner on validation performance
    4. Test phase: Retrain on Train+Val, evaluate on final 90-day Test set
    5. Future forecast generation: Produce 30, 60, and 90-day forecasts into 2018
    6. Export results tables and diagnostic plots
    """
    out_path = Path(output_dir)
    plots_path = out_path / "plots"
    forecasts_path = out_path / "forecasts"
    plots_path.mkdir(parents=True, exist_ok=True)
    forecasts_path.mkdir(parents=True, exist_ok=True)

    logger.info("=== STEP 1: LOADING & CHRONOLOGICAL PARTITIONING ===")
    df = load_raw_data(raw_path)
    daily_sales = aggregate_daily_sales(df)

    # Chronological Split: 90 days test, 90 days validation
    train_df, val_df, test_df = split_time_series(daily_sales, val_days=90, test_days=90, date_col="date")
    logger.info("Train dates: %s to %s (%d days)", train_df["date"].min().date(), train_df["date"].max().date(), len(train_df))
    logger.info("Validation dates: %s to %s (%d days)", val_df["date"].min().date(), val_df["date"].max().date(), len(val_df))
    logger.info("Test dates: %s to %s (%d days)", test_df["date"].min().date(), test_df["date"].max().date(), len(test_df))

    # Save split date ranges
    split_info = {
        "train_start": str(train_df["date"].min().date()),
        "train_end": str(train_df["date"].max().date()),
        "train_days": len(train_df),
        "val_start": str(val_df["date"].min().date()),
        "val_end": str(val_df["date"].max().date()),
        "val_days": len(val_df),
        "test_start": str(test_df["date"].min().date()),
        "test_end": str(test_df["date"].max().date()),
        "test_days": len(test_df),
    }
    with open(out_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=4)

    logger.info("=== STEP 2: MODEL COMPARISON ON VALIDATION SET ===")
    val_comparison_df, val_preds_dict = run_model_comparison(train_df, val_df, target_col="sales", date_col="date")
    val_csv_path = out_path / "model_comparison_validation.csv"
    val_comparison_df.to_csv(val_csv_path, index=False)
    logger.info("Saved validation model comparison to %s", val_csv_path)

    # Plot validation comparison (Top models)
    top_models_to_plot = {k: val_preds_dict[k] for k in ["Seasonal Naive (s=7)", "Holt-Winters (Additive)", "Random Forest", "HistGradientBoosting", "Ridge Regression"] if k in val_preds_dict}
    val_dates = pd.DatetimeIndex(val_df["date"])
    plot_forecast_vs_actual(
        val_dates,
        val_df["sales"].values,
        top_models_to_plot,
        title="Validation Period (90 Days) — Forecast vs Actual Sales",
        save_path=plots_path / "validation_forecast_comparison.png",
    )

    # Bar chart of model RMSE on validation
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=val_comparison_df, x="RMSE", y="Model", palette="mako", ax=ax)
    ax.set_title("Validation RMSE by Model (Lower is Better)", fontsize=13, fontweight="bold")
    ax.set_xlabel("RMSE (Unit Sales)", fontsize=11)
    ax.set_ylabel("Forecasting Model", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.1f}", (p.get_width(), p.get_y() + p.get_height() / 2.0),
                    ha="left", va="center", fontsize=9, xytext=(4, 0), textcoords="offset points")
    plt.tight_layout()
    fig.savefig(plots_path / "model_comparison_rmse_validation.png", dpi=300)

    # Identify best models
    best_overall_model_name = val_comparison_df.iloc[0]["Model"]
    logger.info("Validation Winner: %s (RMSE=%.2f, sMAPE=%.2f%%)", best_overall_model_name, val_comparison_df.iloc[0]["RMSE"], val_comparison_df.iloc[0]["sMAPE"])

    logger.info("=== STEP 3: FINAL TEST EVALUATION (TRAIN+VAL -> TEST) ===")
    train_plus_val_df = pd.concat([train_df, val_df], ignore_index=True).sort_values("date").reset_index(drop=True)
    test_comparison_df, test_preds_dict = run_model_comparison(train_plus_val_df, test_df, target_col="sales", date_col="date")
    test_csv_path = out_path / "model_comparison_test.csv"
    test_comparison_df.to_csv(test_csv_path, index=False)
    logger.info("Saved final test model comparison to %s", test_csv_path)

    # Plot final test comparison
    top_test_models = {k: test_preds_dict[k] for k in ["Seasonal Naive (s=7)", "Holt-Winters (Additive)", "Random Forest", "HistGradientBoosting", "Ridge Regression"] if k in test_preds_dict}
    test_dates = pd.DatetimeIndex(test_df["date"])
    plot_forecast_vs_actual(
        test_dates,
        test_df["sales"].values,
        top_test_models,
        title="Final Test Period (90 Days) — Forecast vs Actual Sales",
        save_path=plots_path / "test_forecast_comparison.png",
    )

    # Plot residuals for the best model on the test set
    best_test_model = test_comparison_df.iloc[0]["Model"]
    if best_test_model in test_preds_dict:
        plot_residuals(
            test_dates,
            test_df["sales"].values,
            test_preds_dict[best_test_model],
            model_name=best_test_model,
            save_path=plots_path / "test_residuals_best_model.png",
        )

    logger.info("=== STEP 4: OUT-OF-SAMPLE FUTURE FORECASTS (2018 HORIZONS) ===")
    # Generate 90 days out-of-sample forecast beyond 2017-12-31
    future_dates_90 = pd.date_range(start="2018-01-01", periods=90, freq="D")
    future_dates_60 = future_dates_90[:60]
    future_dates_30 = future_dates_90[:30]

    # Train winning models on full 5-year history
    X_full, y_full, feat_cols = build_ml_feature_matrix(daily_sales, target_col="sales", date_col="date")

    # Fit HistGradientBoosting (Fast, robust, top-tier ML)
    hgb_model = HistGradientBoostingRegressor(max_iter=150, max_depth=8, learning_rate=0.08, random_state=42)
    hgb_model.fit(X_full, y_full)
    future_fc_hgb_90 = recursive_ml_forecast(
        model=hgb_model,
        full_history_df=daily_sales,
        forecast_dates=future_dates_90,
        feature_cols=feat_cols,
        target_col="sales",
        date_col="date",
    )

    # Fit Holt-Winters (Top classical statistical benchmark)
    full_series = daily_sales.set_index("date")["sales"].asfreq("D")
    future_fc_hw_90, _ = fit_holt_winters(full_series, forecast_steps=90, seasonal_periods=7, trend="add", seasonal="add")

    # Fit Seasonal Naive benchmark
    future_fc_snaive_90 = forecast_seasonal_naive(full_series.values, steps=90, period=7)

    # Create master forecast DataFrame
    forecast_df_90 = pd.DataFrame({
        "date": future_dates_90,
        "forecast_hist_gradient_boosting": np.round(future_fc_hgb_90, 1),
        "forecast_holt_winters": np.round(future_fc_hw_90, 1),
        "forecast_seasonal_naive": np.round(future_fc_snaive_90, 1),
    })
    # Ensemble average (HistGradientBoosting + Holt-Winters)
    forecast_df_90["forecast_ensemble_recommended"] = np.round((forecast_df_90["forecast_hist_gradient_boosting"] + forecast_df_90["forecast_holt_winters"]) / 2.0, 1)

    # Save 30, 60, 90 day forecasts
    fc_90_path = forecasts_path / "forecast_aggregate_90_days.csv"
    fc_60_path = forecasts_path / "forecast_aggregate_60_days.csv"
    fc_30_path = forecasts_path / "forecast_aggregate_30_days.csv"

    forecast_df_90.to_csv(fc_90_path, index=False)
    forecast_df_90.head(60).to_csv(fc_60_path, index=False)
    forecast_df_90.head(30).to_csv(fc_30_path, index=False)
    logger.info("Saved 30, 60, 90-day future forecasts to %s", forecasts_path)

    # Plot future forecast
    fig, ax = plt.subplots(figsize=(15, 6))
    recent_hist = daily_sales.tail(90)
    ax.plot(recent_hist["date"], recent_hist["sales"], label="Historical Sales (Q4 2017)", color="black", linewidth=1.5)
    ax.plot(forecast_df_90["date"], forecast_df_90["forecast_hist_gradient_boosting"], label="Forecast: HistGradientBoosting", color="forestgreen", linestyle="--", linewidth=1.8)
    ax.plot(forecast_df_90["date"], forecast_df_90["forecast_holt_winters"], label="Forecast: Holt-Winters", color="royalblue", linestyle=":", linewidth=1.8)
    ax.plot(forecast_df_90["date"], forecast_df_90["forecast_ensemble_recommended"], label="Forecast: Recommended Ensemble", color="crimson", linewidth=2.2)

    ax.set_title("90-Day Out-of-Sample Demand Forecast (Q1 2018)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Unit Sales", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(plots_path / "future_forecast_90_days.png", dpi=300)

    logger.info("Aggregate pipeline execution completed successfully.")
    return {
        "val_comparison": val_comparison_df,
        "test_comparison": test_comparison_df,
        "forecast_90": forecast_df_90,
    }


def execute_multi_series_pipeline(
    raw_path: str = "data/raw/train.csv",
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Execute multi-series strategy:
    1. Individual store-item time-series forecasting across selected combinations
       (e.g. Store 1 Item 1, Store 2 Item 15, Store 3 Item 28, Store 10 Item 50).
    2. Global Machine Learning strategy: Train a shared model on multi-series data
       using store and item identifiers alongside calendar and lag features.
    3. Compare individual series forecasts and save multi-series forecast tables.
    """
    logger.info("=== STEP 5: MULTI-SERIES FORECASTING STRATEGY ===")
    df = load_raw_data(raw_path)
    out_path = Path(output_dir)
    forecasts_path = out_path / "forecasts"

    selected_combos = [
        (1, 1),
        (2, 15),
        (3, 28),
        (4, 33),
        (10, 50),
    ]

    combo_results = []
    combo_forecasts_dict = {"date": pd.date_range("2018-01-01", periods=90, freq="D")}

    for store_id, item_id in selected_combos:
        logger.info("Processing Store %d - Item %d series...", store_id, item_id)
        series = get_series(df, store=store_id, item=item_id)
        series_df = series.reset_index()

        train_s, val_s, test_s = split_time_series(series_df, val_days=90, test_days=90, date_col="date")

        # Evaluate Seasonal Naive, Holt-Winters, and HistGradientBoosting
        y_test = test_s["sales"].values
        steps = len(test_s)
        test_dates = pd.DatetimeIndex(test_s["date"])

        # Seasonal Naive
        p_snaive = forecast_seasonal_naive(train_s["sales"].values, steps=steps, period=7)
        m_snaive = calculate_all_metrics(y_test, p_snaive, train_s["sales"].values, 7, f"Store{store_id}_Item{item_id}_SeasonalNaive")

        # Holt-Winters
        train_ts = train_s.set_index("date")["sales"].asfreq("D")
        p_hw, _ = fit_holt_winters(train_ts, forecast_steps=steps, seasonal_periods=7, trend="add", seasonal="add")
        m_hw = calculate_all_metrics(y_test, p_hw, train_s["sales"].values, 7, f"Store{store_id}_Item{item_id}_HoltWinters")

        # ML model
        X_tr, y_tr, feats = build_ml_feature_matrix(train_s, target_col="sales", date_col="date")
        ml_model = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
        ml_model.fit(X_tr, y_tr)
        p_ml = recursive_ml_forecast(ml_model, train_s, test_dates, feats, target_col="sales", date_col="date")
        m_ml = calculate_all_metrics(y_test, p_ml, train_s["sales"].values, 7, f"Store{store_id}_Item{item_id}_HistGB")

        combo_results.extend([m_snaive, m_hw, m_ml])

        # Generate future 90-day forecast for this store-item combination
        future_dates = pd.date_range("2018-01-01", periods=90, freq="D")
        X_full, y_full, feats_full = build_ml_feature_matrix(series_df, target_col="sales", date_col="date")
        ml_full = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
        ml_full.fit(X_full, y_full)
        fc_future = recursive_ml_forecast(ml_full, series_df, future_dates, feats_full, target_col="sales", date_col="date")

        combo_forecasts_dict[f"store_{store_id}_item_{item_id}_sales_forecast"] = np.round(fc_future, 1)

    combo_df = pd.DataFrame(combo_results)
    combo_csv_path = out_path / "multi_series_model_comparison.csv"
    combo_df.to_csv(combo_csv_path, index=False)
    logger.info("Saved multi-series model comparison to %s", combo_csv_path)

    store_item_forecast_df = pd.DataFrame(combo_forecasts_dict)
    store_item_fc_path = forecasts_path / "forecast_store_item_combinations_90_days.csv"
    store_item_forecast_df.to_csv(store_item_fc_path, index=False)
    logger.info("Saved store-item 90-day forecasts to %s", store_item_fc_path)

    return combo_df


def run_full_pipeline():
    """Execute complete master pipeline."""
    start_total = time.time()
    logger.info("Starting Retail Demand Forecasting Master Execution Pipeline...")
    agg_res = execute_aggregate_pipeline()
    multi_res = execute_multi_series_pipeline()
    total_sec = time.time() - start_total
    logger.info("All pipeline stages executed in %.1f seconds.", total_sec)


if __name__ == "__main__":
    run_full_pipeline()
