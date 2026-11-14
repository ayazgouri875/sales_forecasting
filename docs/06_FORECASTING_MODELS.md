# 06. Forecasting Models: Theory, Mechanics & Comparison

## Overview of Model Families
We implemented, tuned, and evaluated 16 distinct forecasting models across four distinct architectural groups:
1. **Baseline Benchmarks**: Naive, Seasonal Naive, Mean, Moving Average, Drift
2. **Exponential Smoothing Family**: Simple Exponential Smoothing (SES), Holt's Linear Trend, Holt-Winters Additive
3. **Autoregressive Statespace Models**: ARIMA(1,1,1), SARIMA(1,1,1)(1,1,1)[7]
4. **Machine Learning Ensembles**: Linear Regression, Ridge Regression, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting

---

## Group 1: Baseline Models

### 1. Naive Baseline
- **Definition & Formula**: Projects the most recent observed unit sales forward indefinitely:
  $$\hat{y}_{t+h|t} = y_t \quad \forall h \ge 1$$
- **Role in Project**: Represents the simplest possible benchmark. Any complex model that cannot beat this baseline is useless in practice.
- **Empirical Validation Performance**: RMSE: $5,130.58$, sMAPE: $12.26\%$.
- **Limitations**: Assumes a pure random walk with no trend and no seasonality.

### 2. Seasonal Naive Baseline (s=7)
- **Definition & Formula**: Projects the sales observed exactly one seasonal cycle ago:
  $$\hat{y}_{t+h|t} = y_{t+h - s} \quad \text{where } s=7$$
- **Role in Project**: Mandatory benchmark for seasonal retail data. In industry, many complex ML models fail to beat seasonal naive on daily data.
- **Empirical Validation Performance**: RMSE: $4,372.86$, sMAPE: $10.48\%$.
- **Advantages**: Inexpensive ($0.00$s compute time), simple to interpret, highly competitive on strong intra-week patterns.

### 3. Drift Baseline
- **Definition & Formula**: Extrapolates historical growth linearly between the first and last observation:
  $$\hat{y}_{t+h|t} = y_t + h \left(\frac{y_t - y_1}{T - 1}\right)$$
- **Empirical Validation Performance**: RMSE: $5,443.63$, sMAPE: $13.06\%$.

---

## Group 2: Exponential Smoothing Family

### 4. Simple Exponential Smoothing (SES)
- **Mathematical Formulation**:
  $$\hat{y}_{t+1|t} = \alpha y_t + (1 - \alpha)\hat{y}_{t|t-1} = \sum_{j=0}^{t-1} \alpha(1-\alpha)^j y_{t-j}$$
- **Hyperparameter**: $\alpha \in [0, 1]$ (smoothing parameter). Estimated via maximum likelihood: $\hat{\alpha} = 0.1316$.
- **Empirical Validation Performance**: RMSE: $5,555.35$, sMAPE: $13.53\%$.
- **Limitation**: Produces flat forecasts; incapable of capturing trend or day-of-week seasonality.

### 5. Holt's Linear Trend Model
- **Mathematical Formulation**:
  $$\text{Level: } \ell_t = \alpha y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1})$$
  $$\text{Trend: } b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}$$
  $$\text{Forecast: } \hat{y}_{t+h|t} = \ell_t + h b_t$$
- **Empirical Validation Performance**: RMSE: $8,515.62$, sMAPE: $20.16\%$.
- **Diagnosis of Failure**: In long horizons (90 days), unchecked linear slope over-extrapolates growth, demonstrating why un-damped linear trend models are dangerous for multi-month projections.

### 6. Holt-Winters Exponential Smoothing (Additive Seasonal, Period=7)
- **Mathematical Formulation**:
  $$\ell_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1})$$
  $$b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}$$
  $$s_t = \gamma (y_t - \ell_{t-1} - b_{t-1}) + (1 - \gamma) s_{t-m}$$
  $$\hat{y}_{t+h|t} = \ell_t + h b_t + s_{t+h-m(k+1)}$$
- **Parameters Fitted**: $\alpha = 0.119, \beta = 0.0001, \gamma = 0.364$.
- **Empirical Validation Performance**: RMSE: $5,597.02$, sMAPE: $13.14\%$.
- **Test Performance**: RMSE: $4,423.11$, sMAPE: $11.93\%$.
- **Industry Utility**: Standard production model for automated replenishment in supply chains with stable seasonality and modest computational budgets.

---

## Group 3: Statespace Autoregressive Models (ARIMA & SARIMA)

### 7. ARIMA(1, 1, 1)
- **Formulation**: $(1 - \phi_1 B)(1 - B) y_t = c + (1 + \theta_1 B) \epsilon_t$
- **Validation Performance**: RMSE: $5,486.89$, sMAPE: $13.35\%$.
- **Limitations**: First differencing removes trend, but without seasonal lag terms, it cannot predict weekly cycles.

### 8. SARIMA(1, 1, 1)(1, 1, 1)[7]
- **Formulation**: Incorporates non-seasonal differencing $(d=1)$, seasonal differencing $(D=1, s=7)$, AR(1), MA(1), seasonal AR(1), and seasonal MA(1).
- **Validation Performance**: RMSE: $9,063.46$, sMAPE: $20.45\%$.
- **Test Performance**: RMSE: $4,483.87$, sMAPE: $12.17\%$.
- **Convergence & Stability**: Maximum likelihood estimation required 5.3s on CPU and triggered convergence warnings due to near-unit-root seasonal parameters. Demonstrates the operational fragility of high-frequency SARIMA in production.

---

## Group 4: Machine Learning Ensembles

### 9. Regularized Linear Models (Ridge Regression)
- **Loss Function**: $\min_w \|y - Xw\|_2^2 + \lambda \|w\|_2^2$ with $\lambda = 100.0$.
- **Validation Performance**: RMSE: $2,819.17$, sMAPE: $7.04\%$.
- **Strengths**: Extremely fast ($0.37$s), robust against collinearity among lag and rolling window features, delivers a $35\%$ RMSE improvement over Seasonal Naive.

### 10. Random Forest Regressor
- **Architecture**: 100 decorrelated decision trees using bootstrap aggregation (bagging) and random feature subsets (`max_depth=12`).
- **Validation Performance**: RMSE: $2,358.47$, sMAPE: $6.20\%$.
- **Test Performance**: RMSE: $1,802.31$, sMAPE: $4.68\%$.
- **Advantage**: Handles non-linear threshold effects naturally without feature scaling.

### 11. Gradient Boosting Regressor (Validation Champion)
- **Architecture**: 100 sequentially boosted shallow regression trees (`max_depth=5`, `learning_rate=0.08`) optimizing squared error loss.
- **Validation Performance**: **RMSE: 823.66, MAE: 648.56, sMAPE: 2.00%, WAPE: 1.92%, MASE: 0.838**.
- **Test Performance**: RMSE: $992.94$, sMAPE: $2.83\%$.
- **Why It Won**: Gradient Boosting iteratively fits the negative gradients (pseudo-residuals) of prior trees, effectively capturing subtle interactions between weekend uplift, annual month multipliers, and trailing 14-day rolling volume.

### 12. HistGradientBoosting Regressor (Final Test Champion)
- **Architecture**: Histogram-based gradient boosting (binning continuous features into 256 discrete integer bins).
- **Validation Performance**: RMSE: $1,459.91$, sMAPE: $3.66\%$.
- **Test Performance**: **RMSE: 890.44, MAE: 551.43, sMAPE: 2.14%, WAPE: 2.02%, MASE: 0.710**.
- **Execution Efficiency**: Trains in just $1.12$ seconds—over $3\times$ faster than standard Random Forest while delivering the lowest test error of all evaluated models.
