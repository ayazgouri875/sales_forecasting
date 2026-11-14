# 08. Project Architecture & Codebase Design

## 1. Directory Tree & Module Map

```
retail-demand-forecasting/
├── README.md                                # Root comprehensive portfolio README
├── requirements.txt                         # Certified production dependencies
├── app.py                                   # Interactive multi-page Streamlit web application
│
├── data/
│   ├── raw/
│   │   └── train.csv                        # Raw 913,000-row Kaggle dataset
│   └── processed/
│       ├── daily_total_sales.csv            # Aggregate daily sales (1,826 days)
│       ├── store_daily_sales.csv            # Store-level daily sales (18,260 rows)
│       ├── item_daily_sales.csv             # Item-level daily sales (91,300 rows)
│       └── metadata.json                    # Automated validation metadata
│
├── outputs/
│   ├── model_comparison_validation.csv      # Validation leaderboard metrics
│   ├── model_comparison_test.csv            # Final test evaluation metrics
│   ├── multi_series_model_comparison.csv    # Individual store-item metrics
│   ├── split_info.json                      # Exact train/val/test date boundaries
│   ├── forecasts/
│   │   ├── forecast_aggregate_30_days.csv   # 30-day out-of-sample forecast
│   │   ├── forecast_aggregate_60_days.csv   # 60-day out-of-sample forecast
│   │   ├── forecast_aggregate_90_days.csv   # 90-day out-of-sample forecast
│   │   └── forecast_store_item_combinations_90_days.csv
│   └── plots/                               # 15 high-resolution diagnostic charts
│
├── src/
│   ├── data_preprocessing.py                # Data loading, sorting, validation & splits
│   ├── decomposition.py                     # Additive, multiplicative & STL decomposition
│   ├── eda.py                               # Trend, weekday, heatmap & distribution routines
│   ├── evaluation.py                        # Metric calculations (MAE, RMSE, sMAPE, WAPE, MASE)
│   ├── forecasting.py                       # Feature pipeline, baselines, ML & recursive forecasting
│   ├── pipeline.py                          # Master execution pipeline
│   ├── smoothing.py                         # SES, Holt's linear & Holt-Winters smoothing
│   └── stationarity.py                      # ADF, KPSS, differencing & ACF/PACF diagnostics
│
├── notebooks/
│   ├── 01_Data_Inspection.ipynb             # Initial dataset inspection
│   ├── 02_EDA.ipynb                         # Exploratory data analysis
│   ├── 03_Decomposition.ipynb               # Time-series decomposition
│   ├── 04_Stationarity_and_Differencing.ipynb# Unit root testing & differencing
│   └── 05_Forecasting_and_Model_Evaluation.ipynb# Model benchmarking & selection
│
├── tests/
│   ├── test_preprocessing.py                # Tests for data loading, validation & splits
│   ├── test_evaluation.py                   # Tests for metric formulas & edge cases
│   ├── test_forecasting.py                  # Tests for feature shifts & baselines
│   └── test_stationarity.py                 # Tests for ADF, KPSS & transformations
│
└── docs/                                    # 14 technical & interview documentation guides
```

---

## 2. Core Python Modules & Responsibilities

### `src/data_preprocessing.py`
- `load_raw_data(data_path)`: Loads CSV, parses `date` as datetime, verifies required columns, and sorts chronologically.
- `validate_raw_data(df)`: Computes row counts, null counts, duplicate checks, date continuity, and store/item completeness.
- `aggregate_daily_sales(df)`: Resamples to daily total unit sales across all 500 series.
- `split_time_series(df, val_days, test_days)`: Performs strict chronological out-of-sample partitioning without data leakage.

### `src/stationarity.py`
- `adf_test(series)`: Computes Augmented Dickey-Fuller test statistic, p-value, and critical values.
- `kpss_test(series)`: Computes Kwiatkowski-Phillips-Schmidt-Shin test for trend/level stationarity.
- `comprehensive_stationarity_check(series)`: Produces joint stationarity diagnoses.
- `apply_first_difference(series)`: Calculates $\Delta y_t = y_t - y_{t-1}$.
- `apply_seasonal_difference(series, period)`: Calculates $\Delta_s y_t = y_t - y_{t-s}$.

### `src/forecasting.py`
- `extract_calendar_features(df)`: Generates day of week, day of year, quarter, month, weekend flags, and cyclical sin/cos terms.
- `generate_lag_rolling_features(df)`: Generates shifted lags ($1, 7, 14, 28, 30, 90, 365$) and shifted rolling statistics ($7, 14, 28$) strictly avoiding current-step leakage.
- `forecast_naive`, `forecast_seasonal_naive`, `forecast_drift`: Baseline generators.
- `recursive_ml_forecast()`: Dynamically predicts $H$ steps into the future, dynamically updating lags and rolling statistics with previous predictions.

### `src/evaluation.py`
- `mean_absolute_error`, `root_mean_squared_error`, `symmetric_mean_absolute_percentage_error`, `weighted_absolute_percentage_error`, `mean_absolute_scaled_error`.
- `calculate_all_metrics(y_true, y_pred)`: Returns complete evaluation dictionary with non-negative prediction clipping.

---

## 3. End-to-End Execution Commands

### Run Full Master Pipeline (Train all models & generate forecasts)
```bash
python -m src.pipeline
```

### Run Automated Unit Test Suite
```bash
pytest tests/
```

### Launch Interactive Streamlit Web Application
```bash
streamlit run app.py
```
