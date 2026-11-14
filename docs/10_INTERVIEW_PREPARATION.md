# 10. Comprehensive Interview Preparation Guide

## Section A: Project Introductions

### 30-Second Elevator Pitch
"I built an enterprise-grade retail demand forecasting and seasonal intelligence system using five years of store-item sales data across 10 stores and 50 products—totaling 913,000 records. I implemented a zero-leakage chronological validation architecture, evaluated 16 models across classical statistical methods and gradient-boosted decision trees, and deployed an interactive Streamlit dashboard. My final HistGradientBoosting model achieved an sMAPE of 2.14% and a WAPE of 2.02%, outperforming the seasonal naive baseline by over 80%."

### 1-Minute Executive Pitch
"In retail, poor demand forecasting leads directly to stockouts or bloated working capital. I developed an end-to-end forecasting pipeline analyzing 500 store-item combinations. First, I verified data integrity and performed classical multiplicative and STL decompositions, finding that seasonal amplitude expands with sales scale. I tested stationarity using both ADF and KPSS, applied seasonal differencing, and engineered shifted lag and rolling features to guarantee zero data leakage. I benchmarked classical baselines, ARIMA, SARIMA, and Holt-Winters against machine learning models. Using a strict 90-day chronological holdout, HistGradientBoosting and Gradient Boosting emerged as champions with an RMSE under 900 units and ~2.1% error, reducing baseline error by 80%. The solution includes an interactive Streamlit dashboard supporting multi-horizon projections and supply chain inventory planning heuristics."

### 3-Minute Technical Deep-Dive
"The goal of this project was to build a production-ready forecasting system that balances econometric rigor with machine learning scalability. The dataset comprises 913,000 continuous observations spanning 2013 to 2017 without missing dates. 

During EDA, I identified two dominant seasonal cycles: an intra-week pattern peaking on Sunday (+43% vs Monday) and an annual summer surge peaking in July at nearly double January volumes. Classical decomposition confirmed a multiplicative relationship (reconstruction RMSE 8,855 vs 15,338 for additive). Stationarity analysis revealed the raw series was difference-stationary; ADF rejected the unit root only weakly, while KPSS strongly rejected stationarity, proving differencing was mandatory.

For modeling, I enforced chronological partitioning: 1,646 days for training, a 90-day validation window (Q3 2017) for model selection, and a final 90-day holdout (Q4 2017). To prevent target leakage, all rolling statistics were computed on shifted series, and future multi-step forecasts used a recursive engine. 

I benchmarked 16 models. While Holt-Winters provided a strong statistical baseline (sMAPE 11.9%), Gradient Boosting models delivered superior accuracy (sMAPE 2.14%, WAPE 2.02%, MASE 0.710). On individual store-item time series, machine learning reduced RMSE by ~60% compared to Holt-Winters. Finally, I built a Streamlit application with scenario selection, 95% uncertainty intervals, and automated reporting."

---

## Section B: Core Business & Supply Chain Questions

### 1. What business problem does this project solve?
**Answer**: It solves the supply chain trade-off between inventory holding costs and stockout penalties. By improving forecast accuracy from 13.3% error (baseline) down to 2.1%, retailers can lower safety stock buffers by ~30% while maintaining a 95%+ customer service level.

### 2. How can forecasts support inventory replenishment?
**Answer**: By integrating predicted demand into Reorder Point (ROP) equations: $\text{ROP} = (\bar{d} \times L) + (Z \times \sigma_{\text{forecast}} \times \sqrt{L})$. When inventory drops to ROP, an order of size EOQ (Economic Order Quantity) is placed.

### 3. What are the business implications of under-forecasting vs. over-forecasting?
**Answer**: Under-forecasting results in empty shelves, immediate lost revenue, and customer defection. Over-forecasting ties up capital, requires extra warehouse space, and leads to margin-eroding clearance markdowns.

---

## Section C: Dataset & Preprocessing Questions

### 4. Why is random splitting dangerous in time series?
**Answer**: Random splitting (`shuffle=True`) causes future data leakage. The model trains on tomorrow's and next month's sales to predict yesterday's sales. It produces artificially inflated training accuracy ($R^2 > 0.99$) that collapses in production.

### 5. How did you verify dataset integrity?
**Answer**: I built an automated validation function checking null values ($0$), duplicate rows ($0$), key uniqueness across `date-store-item`, and time continuity (all 1,826 consecutive dates present for all 500 store-item combinations).

---

## Section D: Exploratory Data Analysis Questions

### 6. What was the most significant finding from the day-of-week analysis?
**Answer**: Sales build gradually from Monday to Friday, peaking on Sunday at $43.5\%$ higher volume than Monday. This requires logistics replenishment to shift to Thursdays and Fridays.

### 7. What did the rolling statistics tell you?
**Answer**: The 30-day and 90-day moving averages filtered out day-of-week noise, highlighting an annual compound growth rate of ~7-9% alongside consistent July peaks and January troughs.

---

## Section E: Time Series Econometrics & Decomposition

### 8. Explain the difference between Additive and Multiplicative decomposition.
**Answer**: Additive assumes seasonal fluctuations have constant absolute amplitude ($y_t = T_t + S_t + R_t$). Multiplicative assumes seasonal fluctuations grow proportionally with the trend ($y_t = T_t \times S_t \times R_t$). In our project, multiplicative achieved an RMSE of 8,855 vs 15,338 for additive.

### 9. Why use both ADF and KPSS tests?
**Answer**: ADF tests the null hypothesis of a unit root (non-stationary). KPSS tests the null hypothesis of stationarity. Using both avoids false conclusions: if ADF rejects and KPSS fails to reject, the series is strictly stationary; if both reject, the series is difference-stationary.

### 10. What is the difference between ACF and PACF?
**Answer**: ACF measures correlation between $y_t$ and $y_{t-k}$ including all intermediate effects. PACF measures the direct correlation between $y_t$ and $y_{t-k}$ after partialling out the linear influence of lags $1$ to $k-1$.

---

## Section F: Machine Learning & Feature Engineering

### 11. How do you prevent data leakage in rolling window features?
**Answer**: By applying `.shift(1)` before computing `.rolling(w)`. This ensures that day $t$'s feature only contains data up to day $t-1$.

### 12. Why did Gradient Boosting outperform Linear Regression?
**Answer**: Linear regression cannot naturally model non-linear interactions without explicit polynomial terms. Gradient boosting partitions feature space into piecewise regions, naturally capturing interactions like "if month is July AND day_of_week is Sunday, apply an exponential boost."

---

## Section G: Evaluation & Benchmarking

### 13. Why use WAPE over MAPE?
**Answer**: Standard MAPE divides error by actual sales. On low-sales days (e.g. 1 unit), an error of 2 units yields a 200% error, distorting the average. WAPE divides total absolute error by total actual sales, naturally weighting high-volume items and remaining numerically stable.

### 14. What does a MASE of 0.710 indicate?
**Answer**: MASE scales error against an in-sample seasonal naive benchmark. A score of 0.710 means the model's absolute forecast error is $29\%$ smaller than the seasonal naive benchmark.

---

## Section H: Practical Deployment & Operational Edge Cases

### 15. How does the system prevent negative sales forecasts?
**Answer**: In retail demand, negative sales are physically impossible. The recursive engine applies $\max(0, \hat{y})$ clipping to all model outputs.

### 16. How would you handle a newly introduced product with zero sales history (Cold Start)?
**Answer**: Use category-level clustering or transfer learning: borrow lag features and seasonal profiles from similar products in the same hierarchy or parent store.

---

## Section I: 50 Project-Specific Questions & Answers

1. **What is the shape of the raw dataset?**
   - 913,000 rows by 4 columns.
2. **What date range does it cover?**
   - 2013-01-01 to 2017-12-31 (1,826 days).
3. **How many stores and items exist?**
   - 10 stores and 50 items (500 series).
4. **Were there missing dates?**
   - Zero missing dates.
5. **What was the date of the validation split?**
   - 2017-07-05 to 2017-10-02 (90 days).
6. **What was the date of the final test split?**
   - 2017-10-03 to 2017-12-31 (90 days).
7. **What was the in-sample reconstruction RMSE for Additive decomposition?**
   - 15,338.05 units.
8. **What was the in-sample reconstruction RMSE for Multiplicative decomposition?**
   - 8,855.21 units.
9. **What seasonal period was used for monthly decomposition?**
   - Period = 12.
10. **What seasonal period was used for daily STL?**
    - Period = 7 (weekly cycle).
11. **What was the ADF p-value on raw sales?**
    - $p = 0.02964$ (ADF stat: -3.0602).
12. **What was the KPSS p-value on raw sales?**
    - $p = 0.01000$ (KPSS stat: 2.0010, rejecting stationarity).
13. **What happened to the ADF statistic after first differencing?**
    - Dropped to -8.9190 ($p < 10^{-5}$).
14. **What happened to the KPSS statistic after first differencing?**
    - Dropped to 0.0511 ($p = 0.10$, confirming stationarity).
15. **What was the Naive baseline validation RMSE?**
    - 5,130.58 units.
16. **What was the Seasonal Naive (s=7) validation RMSE?**
    - 4,372.86 units.
17. **What was the Holt-Winters validation RMSE?**
    - 5,597.02 units.
18. **What was the ARIMA(1,1,1) validation RMSE?**
    - 5,486.89 units.
19. **What was the SARIMA validation RMSE?**
    - 9,063.46 units (due to optimization instability).
20. **What was the Ridge Regression validation RMSE?**
    - 2,819.17 units.
21. **What was the Random Forest validation RMSE?**
    - 2,358.47 units.
22. **What was the Gradient Boosting validation RMSE?**
    - 823.66 units (Validation Champion).
23. **What was the HistGradientBoosting validation RMSE?**
    - 1,459.91 units.
24. **Which model won on the final test set (Q4 2017)?**
    - HistGradientBoosting (RMSE: 890.44, sMAPE: 2.14%).
25. **What was the Gradient Boosting test RMSE?**
    - 992.94 units.
26. **What was the Holt-Winters test RMSE?**
    - 4,423.11 units.
27. **What was the Seasonal Naive test RMSE?**
    - 5,050.29 units.
28. **What was the percentage improvement of HistGB over Seasonal Naive on the test set?**
    - $82.4\%$ reduction in RMSE (890 vs 5,050).
29. **What was the WAPE of HistGB on the test set?**
    - 2.02%.
30. **What was the MASE of HistGB on the test set?**
    - 0.710.
31. **What was the fastest ML model to train?**
    - Ridge Regression (0.34s) followed by HistGradientBoosting (1.12s).
32. **Why was HistGradientBoosting faster than standard Random Forest?**
    - It discretizes continuous values into 256 histogram bins, reducing split evaluation from $O(N \log N)$ to $O(K)$ where $K=256$.
33. **What lags were generated in the feature pipeline?**
    - Lags 1, 7, 14, 28, 30, 90, 365.
34. **What rolling windows were generated?**
    - 7, 14, and 28 days for rolling mean and rolling std.
35. **Why were sin and cos cyclical encodings included?**
    - To map Sunday and Monday as continuous neighbors on a circular domain.
36. **How was multi-step forecasting implemented?**
    - A recursive engine updating the history buffer step-by-step.
37. **What were the highest-volume stores?**
    - Store 2 (6.15M units) and Store 8 (6.05M units).
38. **What were the lowest-volume stores?**
    - Store 7 (3.13M units) and Store 6 (3.34M units).
39. **What was the highest-volume product?**
    - Item 15 (1.70M units).
40. **What was the lowest-volume product?**
    - Item 1 (435k units).
41. **How did HistGB perform on Store 1 Item 1?**
    - RMSE 4.59 vs 10.99 for Holt-Winters (58% improvement).
42. **How did HistGB perform on Store 2 Item 15?**
    - RMSE 22.12 vs 57.72 for Holt-Winters (62% improvement).
43. **How did HistGB perform on Store 3 Item 28?**
    - RMSE 15.72 vs 44.10 for Holt-Winters (64% improvement).
44. **What framework was used for the web dashboard?**
    - Streamlit (modular multi-page application).
45. **What test framework was used for verification?**
    - Pytest (15 unit tests covering preprocessing, evaluation, forecasting, stationarity).
46. **What is the formula for sMAPE?**
    - $(100/n) \sum 2|y - \hat{y}| / (|y| + |\hat{y}|)$.
47. **What is the formula for WAPE?**
    - $\sum |y - \hat{y}| / \sum y \times 100\%$.
48. **How were 95% uncertainty intervals computed?**
    - $\hat{y} \pm 1.96 \times \sigma_{\text{residual}}$.
49. **What forecast horizons were generated?**
    - 30, 60, and 90 days out-of-sample into Q1 2018.
50. **Can this pipeline be automated?**
    - Yes, via `python -m src.pipeline` which executes all steps in ~50 seconds.

---

## Section J: Difficult Follow-Up Questions

### Follow-Up 1: *"Your HistGB achieved 2.14% sMAPE. Isn't that suspiciously high accuracy? How do you prove there was no leakage?"*
**Answer**: "I verified leakage prevention at three distinct levels:
1. **Partitioning**: Train, validation, and test were split strictly chronologically. The test set dates (Oct-Dec 2017) were never seen during model training.
2. **Feature Engineering**: Every rolling mean and standard deviation was computed with `.shift(1)`. For test observations, the model never accesses same-day actuals.
3. **Recursive Out-of-Sample Engine**: For day $t+2$ through $t+90$, future lags were populated using previous *predictions*, not actuals. The low error is genuine because aggregate retail sales across 10 stores and 50 items exhibit smooth macroeconomic momentum and strong, predictable weekly and seasonal regularities."

### Follow-Up 2: *"Why did SARIMA perform so poorly (RMSE 9,063 on validation) compared to Holt-Winters?"*
**Answer**: "SARIMA on daily data with seasonal period $s=7$ involves estimating non-seasonal and seasonal AR/MA polynomials simultaneously. The seasonal autoregressive roots were close to the unit circle, causing the BFGS numerical optimizer to fail convergence within standard iteration limits. In contrast, Holt-Winters uses direct exponential smoothing recursion which converged reliably in 0.3 seconds."
