# 02. Dataset and Preprocessing Pipeline

## 1. Dataset Origin & Physical Properties
The underlying dataset originates from the Kaggle Store Item Demand Forecasting Challenge. It represents five complete calendar years of retail transaction history.

### Dimensional Statistics
- **Total Rows**: 913,000 observations
- **Total Columns**: 4 columns (`date`, `store`, `item`, `sales`)
- **Memory Footprint**: ~28 MB in uncompressed memory
- **Temporal Span**: January 1, 2013 to December 31, 2017 ($1,826$ calendar days)
- **Entities**: 10 Stores, 50 Items, generating 500 distinct univariate daily time-series ($10 \times 50 \times 1,826 = 913,000$).

---

## 2. Comprehensive Data Quality Audit
Prior to model experimentation, a strict data integrity audit was conducted (`validate_raw_data()` in `src/data_preprocessing.py`).

| Quality Metric | Measured Value | Threshold / Requirement | Status |
|---|---|---|---|
| Missing Values (`NaN`/Nulls) | 0 | 0 | PASSED |
| Duplicate Rows | 0 | 0 | PASSED |
| Key Uniqueness (`date`, `store`, `item`) | 0 duplicates | 0 | PASSED |
| Time Continuity | 1,826 consecutive days | 1,826 days | PASSED |
| Series Completeness | 500 series $\times$ 1,826 days | 100% complete | PASSED |
| Zero Unit Sales | 0 records | Allowed if present | Verified |
| Negative Unit Sales | 0 records | Strict 0 (cannot be negative) | PASSED |
| Minimum Unit Sales | 0 units | $\ge 0$ | PASSED |
| Maximum Unit Sales | 231 units | Realistic physical limit | PASSED |
| Mean Unit Sales | 52.26 units | N/A | Verified |
| Median Unit Sales | 47.00 units | N/A | Verified |

---

## 3. Data Transformations & Preprocessing Decisions

### A. Date Parsing and Chronological Sorting
- **Action**: The `date` column is parsed into `pd.Timestamp` (`datetime64[ns]`) and the entire DataFrame is sorted by `['date', 'store', 'item']`.
- **Rationale**: Time series models require strictly monotonic temporal ordering. Random splits or out-of-order records cause lookahead leakage and invalidate recursive feature generation.

### B. Regularization of Frequency
- **Action**: For all aggregated series (daily total, store-level, item-level, and store-item combinations), frequency is explicitly set to `'D'` (calendar day) using `.asfreq('D')`.
- **Rationale**: Econometric algorithms (ARIMA, SARIMA, Holt-Winters, STL) in `statsmodels` require an explicit, un-interrupted datetime frequency index to infer seasonal periodicity and compute out-of-sample forecast timestamps.

### C. Hierarchical Aggregation Pipeline
To support diverse organizational planning horizons, the raw observations are processed into four distinct hierarchical levels:
1. **Aggregate Daily Total Sales** (`data/processed/daily_total_sales.csv`):
   $$\text{TotalSales}_t = \sum_{s=1}^{10} \sum_{i=1}^{50} \text{Sales}_{s, i, t}$$
   - **Dimensions**: 1,826 rows $\times$ 2 columns (`date`, `sales`).
   - **Use Case**: Macro-level executive forecasting, overall warehouse capacity planning, financial budgeting.
2. **Store-Level Daily Sales** (`data/processed/store_daily_sales.csv`):
   $$\text{StoreSales}_{s, t} = \sum_{i=1}^{50} \text{Sales}_{s, i, t}$$
   - **Dimensions**: 18,260 rows $\times$ 3 columns (`date`, `store`, `sales`).
   - **Use Case**: Store manager staffing, local distribution logistics, regional replenishment.
3. **Item-Level Daily Sales** (`data/processed/item_daily_sales.csv`):
   $$\text{ItemSales}_{i, t} = \sum_{s=1}^{10} \text{Sales}_{s, i, t}$$
   - **Dimensions**: 91,300 rows $\times$ 3 columns (`date`, `item`, `sales`).
   - **Use Case**: Vendor procurement, central supplier ordering, catalog lifecycle management.
4. **Individual Store-Item Time Series**:
   $$\text{Sales}_{s, i, t} \quad \text{for } s \in [1, 10], i \in [1, 50]$$
   - **Dimensions**: 500 series of 1,826 daily records each.
   - **Use Case**: Automated retail store shelf replenishment.

---

## 4. Chronological Splitting Architecture (Zero Data Leakage)
A primary flaw in many retail forecasting projects is the use of `train_test_split(shuffle=True)` or cross-validation with random k-folds. Shuffling time series data leaks future observations into past training, producing artificially high accuracy metrics ($R^2 > 0.99$) that collapse completely in production.

We enforce strict **chronological out-of-sample partitioning**:

```
[================= TRAINING SET =================] [== VALIDATION ==] [==== TEST ====]
2013-01-01                              2017-07-04 2017-07-05 2017-10-02 2017-10-03 2017-12-31
(1,646 Days / 90.1% of History)                  (90 Days / 4.9%)  (90 Days / 4.9%)
```

- **Training Window**: 2013-01-01 to 2017-07-04 ($1,646$ days). Used to fit all model parameters, learn coefficients, and construct feature distributions.
- **Validation Window**: 2017-07-05 to 2017-10-02 ($90$ days / Q3 2017). Dedicated exclusively to model comparison, hyperparameter tuning, and selecting the champion architecture.
- **Final Test Window**: 2017-10-03 to 2017-12-31 ($90$ days / Q4 2017). Held out completely untouched until champion selection is final. Evaluated by retraining models on `Train + Validation` ($1,736$ days) and predicting the unseen final quarter.
- **Future Forecasting Horizon**: 2018-01-01 to 2018-03-31 ($90$ days / Q1 2018). True out-of-sample recursive forecasting into the future.
