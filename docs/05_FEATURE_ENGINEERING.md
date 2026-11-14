# 05. Feature Engineering & Leakage Prevention

## 1. Feature Engineering Philosophy
Machine learning models (such as Gradient Boosting, Random Forest, and Ridge Regression) cannot natively understand time. They treat rows of tabular data as exchangeable samples unless temporal dynamics are explicitly translated into numerical feature representations.

Our feature pipeline in `src/forecasting.py` generates 27 distinct engineered signals organized into three main categories:
1. **Calendar & Deterministic Signals**
2. **Lag Features (Shifted Autoregressive Signals)**
3. **Rolling Window Statistics (Shifted Moving Averages and Volatility)**

---

## 2. Complete Feature Inventory & Formulations

### Category A: Calendar & Cyclical Features
Captures deterministic calendar patterns without relying on past sales history:
- `year`: Linear trend indicator ($2013 \dots 2017$).
- `month`: Annual seasonality indicator ($1 \dots 12$).
- `day`: Day of the month ($1 \dots 31$), capturing payroll/salary disbursement cycles.
- `day_of_week`: Intra-week calendar index ($0 = \text{Monday} \dots 6 = \text{Sunday}$).
- `day_of_year`: Day index within the year ($1 \dots 365$).
- `week_of_year`: ISO calendar week ($1 \dots 52$).
- `quarter`: Fiscal quarter indicator ($1 \dots 4$).
- `is_weekend`: Binary flag ($1$ for Saturday/Sunday, $0$ otherwise).
- `is_month_start`: Binary flag for the 1st of the month.
- `is_month_end`: Binary flag for month-end retail cycles.

#### Cyclical Encodings
Ordinary integer representations of `day_of_week` introduce an artificial jump between Sunday (6) and Monday (0), even though they are temporally adjacent. We map cyclical dimensions onto unit circles via trigonometric projection:
$$\text{sin\_dow} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \text{cos\_dow} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$$
$$\text{sin\_doy} = \sin\left(\frac{2\pi \cdot \text{day\_of\_year}}{365.25}\right), \quad \text{cos\_doy} = \cos\left(\frac{2\pi \cdot \text{day\_of\_year}}{365.25}\right)$$

---

### Category B: Shifted Lag Features
Lags represent direct historical sales from past time intervals:
- `lag_1`: Sales volume on day $t-1$. Captures immediate day-to-day momentum.
- `lag_7`: Sales volume on day $t-7$. Captures same-day-of-week sales from exactly one week ago.
- `lag_14`: Sales volume on day $t-14$. Two-week seasonal benchmark.
- `lag_28`: Sales volume on day $t-28$. Four-week monthly cycle benchmark.
- `lag_30`: Sales volume on day $t-30$. Monthly calendar benchmark.
- `lag_90`: Sales volume on day $t-90$. Prior quarterly benchmark.
- `lag_365`: Sales volume on day $t-365$. Same-day annual benchmark from one year prior.

---

### Category C: Shifted Rolling Window Statistics
Rolling statistics smooth high-frequency noise and capture local demand momentum:
- `rolling_mean_7`: 7-day moving average.
- `rolling_mean_14`: 14-day moving average.
- `rolling_mean_28`: 28-day moving average.
- `rolling_std_7`: 7-day rolling standard deviation (short-term demand volatility).
- `rolling_std_14`: 14-day rolling standard deviation.
- `rolling_std_28`: 28-day rolling standard deviation (monthly demand volatility).

---

## 3. Strict Prevention of Target Data Leakage

### The Danger of Naive Rolling Windows
In standard pandas, computing:
```python
# SEVERE TARGET LEAKAGE BUG
df["rolling_mean_7"] = df["sales"].rolling(7).mean()
```
includes the current day's sales $y_t$ inside the window:
$$\text{rolling\_mean\_7}(t) = \frac{y_t + y_{t-1} + \dots + y_{t-6}}{7}$$
If this feature is fed to a model predicting $y_t$, the model learns to isolate $y_t$ directly from the feature, achieving an unrealistically near-perfect training score that fails catastrophically when predicting future days where $y_t$ is unknown.

### Our Leakage-Free Implementation
We enforce strict antecedent shifting on all rolling statistics:
```python
# LEAKAGE-FREE PRODUCTION IMPLEMENTATION
df["rolling_mean_7"] = df["sales"].shift(1).rolling(7).mean()
df["rolling_std_7"]  = df["sales"].shift(1).rolling(7).std()
```
This guarantees that for any prediction at day $t$, only observations from day $t-1$ and earlier are ever accessed.

---

## 4. Multi-Step Recursive Forecasting Architecture
When forecasting 30, 60, or 90 days into the future, future actual sales are unavailable. A naive model cannot compute `lag_1` for day $t+2$ without knowing actual sales on day $t+1$.

To resolve this, we implemented `recursive_ml_forecast()`:
1. **Step 1**: Use historical actuals to compute features for day $t+1$. Model predicts $\hat{y}_{t+1}$.
2. **Buffer Update**: Append predicted $\hat{y}_{t+1}$ into the running history series.
3. **Step 2**: Dynamically compute `lag_1 = \hat{y}_{t+1}`, recompute rolling averages including $\hat{y}_{t+1}$, and predict $\hat{y}_{t+2}$.
4. **Iterate**: Repeat through day $t+H$ for any arbitrary forecast horizon $H$.
5. **Physical Validity**: All intermediate predictions are clipped at zero ($\max(0, \hat{y})$) to reflect physical inventory constraints.
