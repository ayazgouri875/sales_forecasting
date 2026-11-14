# 13. Project Setup, Environment & Execution Guide

## 1. Environment & Prerequisites
- **Operating System**: macOS Monterey / Ventura / Sonoma / Sequoia (Intel x86_64 or Apple Silicon ARM64) or Linux (Ubuntu 20.04+).
- **Python Version**: Python 3.10 to 3.13 (Verified on Python 3.13.3).
- **Virtual Environment**: Clean virtual environment (`venv`).

---

## 2. Installation & Setup Instructions

### Step 1: Clone Repository & Create Virtual Environment
```bash
cd retail-demand-forecasting

# Create virtual environment if not already present
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 2: Install Certified Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Raw Dataset
Ensure the raw Kaggle dataset is located at:
```bash
data/raw/train.csv
```
File size should be approximately 17.3 MB with 913,000 records.

---

## 3. Execution Commands (Verified & Tested)

### A. Run Data Preprocessing Pipeline
Processes raw transaction records, performs data integrity audits, and creates hierarchical aggregations:
```bash
python -m src.data_preprocessing
```
*Outputs generated*:
- `data/processed/daily_total_sales.csv`
- `data/processed/store_daily_sales.csv`
- `data/processed/item_daily_sales.csv`
- `data/processed/metadata.json`

### B. Run Stationarity & Unit Root Tests
Executes ADF, KPSS, and differencing diagnostics:
```bash
python -m src.stationarity
```
*Outputs generated*:
- `outputs/plots/stationarity_raw.png`
- `outputs/plots/stationarity_diff1.png`
- `outputs/plots/stationarity_diff7.png`
- `outputs/plots/stationarity_diff1_diff7.png`

### C. Run Exploratory Data Analysis Pipeline
Generates distribution metrics, trend plots, heatmaps, and weekday distributions:
```bash
python -m src.eda
```
*Outputs generated*:
- `outputs/plots/eda_overall_trend.png`
- `outputs/plots/eda_weekday_analysis.png`
- `outputs/plots/eda_monthly_heatmap.png`
- `outputs/plots/eda_store_item_performance.png`

### D. Run Master Forecasting & Evaluation Pipeline
Trains and benchmarks all 16 models across Validation and Final Test sets, generates 30/60/90-day out-of-sample forecasts, and evaluates multi-series combinations:
```bash
python -m src.pipeline
```
*Execution Time*: $\approx 50$ seconds on modern multi-core CPU.
*Outputs generated*:
- `outputs/model_comparison_validation.csv`
- `outputs/model_comparison_test.csv`
- `outputs/multi_series_model_comparison.csv`
- `outputs/forecasts/forecast_aggregate_30_days.csv`
- `outputs/forecasts/forecast_aggregate_60_days.csv`
- `outputs/forecasts/forecast_aggregate_90_days.csv`
- `outputs/forecasts/forecast_store_item_combinations_90_days.csv`
- Diagnostic comparison and residual plots in `outputs/plots/`

### E. Run Automated Test Suite
Executes all 15 automated pytest tests:
```bash
pytest tests/
```
*Expected Output*: `15 passed in ~4.5s`.

### F. Launch Interactive Streamlit Web Application
Starts the multi-page executive dashboard:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.
