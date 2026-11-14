# 12. Project Revision Cheatsheet & Quick Reference

## 1. Quick Project Numbers (Memorize for Interviews!)
- **Dataset**: Kaggle Store Item Demand Forecasting Challenge.
- **Total Records**: $913,000$ rows ($100\%$ complete, $0$ nulls, $0$ duplicates).
- **Entities**: $10$ Stores, $50$ Items $\to 500$ distinct time-series.
- **Time Span**: 2013-01-01 to 2017-12-31 ($1,826$ days / 5 full years).
- **Partitioning**:
  - Training: 2013-01-01 to 2017-07-04 ($1,646$ days / $90.1\%$).
  - Validation: 2017-07-05 to 2017-10-02 ($90$ days / $4.9\%$).
  - Final Test: 2017-10-03 to 2017-12-31 ($90$ days / $4.9\%$).
- **Decomposition Reconstruction RMSE**:
  - Additive: $15,338.05$
  - Multiplicative: $8,855.21$ ($42\%$ lower error).
- **Validation Champion**: **Gradient Boosting** (RMSE: $823.66$, sMAPE: $2.00\%$, WAPE: $1.92\%$).
- **Final Test Champion**: **HistGradientBoosting** (RMSE: $890.44$, sMAPE: $2.14\%$, WAPE: $2.02\%$, MASE: $0.710$).
- **Baseline Comparison**: Outperformed Seasonal Naive (RMSE: $5,050.29$) by **82.4%**.

---

## 2. Core Mathematical Formulas

| Concept | Mathematical Formula | Key Property |
|---|---|---|
| **MAE** | $\frac{1}{n} \sum \|y - \hat{y}\|$ | Average physical unit error |
| **RMSE** | $\sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$ | Heavily penalizes large outlier errors |
| **sMAPE** | $\frac{100\%}{n} \sum \frac{2\|y - \hat{y}\|}{\|y\| + \|\hat{y}\|}$ | Percentage error bounded $[0\%, 200\%]$ |
| **WAPE** | $\frac{\sum \|y - \hat{y}\|}{\sum y} \times 100\%$ | Supply chain volume-weighted error |
| **MASE** | $\frac{\text{MAE}}{\frac{1}{T-s}\sum_{t=s+1}^T \|y_t - y_{t-s}\|}$ | $< 1.0 \implies$ beats Seasonal Naive |
| **ADF Test** | $\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \dots$ | $H_0: \gamma=0$ (Unit root / Non-stationary) |
| **KPSS Test** | $y_t = \xi t + r_t + \epsilon_t$ | $H_0: \sigma_u^2=0$ (Stationary around trend) |
| **First Diff** | $\Delta y_t = y_t - y_{t-1}$ | Removes linear stochastic trend |
| **Seasonal Diff**| $\Delta_7 y_t = y_t - y_{t-7}$ | Removes day-of-week weekly seasonality |

---

## 3. High-Impact Interview Traps to Avoid

1. ❌ **Don't say you used `train_test_split(shuffle=True)`**: That is a fatal error in time series. Always emphasize **chronological out-of-sample partitioning**.
2. ❌ **Don't claim the dataset has pricing or promotions**: The dataset strictly has `date`, `store`, `item`, and `sales`. Explicitly state that pricing, promotions, and stockout labels are absent.
3. ❌ **Don't say lower decomposition RMSE means better forecasting**: Decomposition RMSE evaluates in-sample curve fitting, not out-of-sample generalization.
4. ❌ **Don't compute rolling statistics without shifting**: Always mention that rolling means and stds were shifted by 1 day (`shift(1)`) to guarantee zero target leakage.
5. ❌ **Don't rely only on MAPE**: Standard MAPE explodes when sales are near zero. Use **WAPE** and **sMAPE** for supply chain credibility.

---

## 4. 20 Rapid-Fire Interview Questions & Answers

1. **What dataset did you use?** Store Item Demand Forecasting (10 stores, 50 items, 913k records).
2. **What was your primary evaluation metric?** WAPE (2.02%) and sMAPE (2.14%).
3. **Why not just RMSE?** RMSE is in squared units; WAPE gives the percentage volume error meaningful to inventory managers.
4. **Did the data have missing values?** Zero missing values; all 500 combinations had complete daily records.
5. **What was the best statistical model?** Holt-Winters Additive (sMAPE 11.9% on test).
6. **What was the best machine learning model?** HistGradientBoosting (sMAPE 2.14% on test).
7. **What was the seasonal naive test error?** sMAPE 13.35%, RMSE 5,050.
8. **Why did multiplicative decomposition beat additive?** Seasonal amplitude expanded proportionally as total store volume grew over the 5 years.
9. **Was the raw sales series stationary?** No, it was difference-stationary (ADF rejected weakly, but KPSS rejected stationarity with $p=0.01$).
10. **How did you make it stationary?** First differencing $\Delta_1$ and seasonal differencing $\Delta_7$.
11. **What lags did you engineer?** Lags 1, 7, 14, 28, 30, 90, 365.
12. **Why use cyclical sin/cos features?** To map continuous adjacency between Sunday (6) and Monday (0).
13. **How did you avoid data leakage?** Chronological splitting, shifted rolling statistics, and recursive multi-step forecasting.
14. **What was the peak sales day?** Sunday (+43.5% higher volume than Monday).
15. **What was the peak sales season?** June and July (nearly 2x January volume).
16. **Which store had the highest sales?** Store 2 (6.15M units), followed closely by Store 8.
17. **Which store had the lowest sales?** Store 7 (3.13M units).
18. **How fast did HistGradientBoosting train?** 1.12 seconds on CPU.
19. **What was the MASE of your winning model?** 0.710 (29% better than seasonal naive).
20. **How do you deploy this model?** As an automated nightly batch inference pipeline feeding forecasts into ERP/replenishment tables with Streamlit for human-in-the-loop inspection.
