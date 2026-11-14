# 03. Exploratory Data Analysis & Business Intelligence

## 1. Overall Sales Trend & Growth Trajectory
- **What Was Calculated**: Daily total sales across all 500 store-item combinations, supplemented by 30-day and 90-day trailing rolling moving averages.
- **Why It Was Calculated**: To identify secular long-term trends, macro regime shifts, and structural demand shifts across the 5-year operating span.
- **How It Was Calculated**:
  $$\text{MA}_{30}(t) = \frac{1}{30} \sum_{i=0}^{29} y_{t-i}$$
- **Actual Numerical Findings**:
  - 2013 Annual Sales: $7,941,243$ units
  - 2014 Annual Sales: $9,125,636$ units ($+14.9\%$ growth)
  - 2015 Annual Sales: $9,583,011$ units ($+5.0\%$ growth)
  - 2016 Annual Sales: $10,357,160$ units ($+8.1\%$ growth)
  - 2017 Annual Sales: $10,733,741$ units ($+3.6\%$ growth)
  - Overall 5-year demand grew from $\approx 21,500$ units/day in early 2013 to over $\approx 29,500$ units/day in 2017 ($+37.2\%$ total expansion).
- **Business Insight**: Strong upward baseline growth demonstrates brand expansion or organic foot traffic gains. Models must include deterministic trend components or autoregressive trend features; static mean baselines will severely under-forecast.
- **Interview Question**: *"Why did you calculate rolling moving averages rather than only looking at daily raw sales?"*
  - **Answer**: Daily retail sales contain high-frequency noise from day-of-week shopping patterns. A 30-day rolling window acts as a low-pass filter, attenuating weekly noise to uncover true underlying macroeconomic momentum and seasonal peaks.

---

## 2. Intra-Week Seasonality (Day of Week Analysis)
- **What Was Calculated**: Average unit sales volume, distribution boxplots, and percentage uplift across the seven calendar days.
- **Actual Numerical Findings**:
  | Day of Week | Mean Daily Sales (Units) | Relative Volume vs Monday |
  |---|---|---|
  | Monday | 21,392 | Baseline ($1.00\times$) |
  | Tuesday | 25,053 | $+17.1\%$ |
  | Wednesday | 25,142 | $+17.5\%$ |
  | Thursday | 26,712 | $+24.9\%$ |
  | Friday | 28,458 | $+33.0\%$ |
  | Saturday | 30,223 | $+41.3\%$ |
  | Sunday | 30,695 | $+43.5\%$ (Peak Day) |
- **Business Insight**: Retail demand builds steadily through the workweek, peaking aggressively on Saturday and Sunday. Supply chains must schedule warehouse picking and delivery trucks on Thursday and Friday to guarantee shelf availability for the weekend rush.
- **Interview Question**: *"How would this day-of-week pattern impact your choice of time-series features?"*
  - **Answer**: It dictates that any classical model must specify a seasonal frequency of $s=7$. For machine learning models, we must provide explicit categorical day-of-week indicators, weekend flags, and lag-7 features to capture direct week-over-week auto-dependencies.

---

## 3. Annual Seasonality & Monthly Heatmaps
- **What Was Calculated**: Cross-tabulation pivot table and heatmap of total monthly unit sales across years 2013–2017.
- **Actual Numerical Findings**:
  - **Peak Months**: July (average $\approx 34,200$ units/day) and June ($\approx 32,500$ units/day).
  - **Trough Months**: January ($\approx 17,200$ units/day) and February ($\approx 19,400$ units/day).
  - **Seasonal Ratio**: July demand is **1.98x** higher than January demand.
- **Business Insight**: Massive summer seasonality. Retail procurement must begin ramp-up purchasing in April and May. Inventory buffers can be drawn down starting in late August to avoid holding excess inventory into the low-volume winter quarter.
- **Interview Question**: *"Did you notice any interaction between trend and seasonality in the heatmap?"*
  - **Answer**: Yes. As total annual sales grew from 2013 to 2017, the absolute magnitude of the summer peak expanded proportionally (from $\approx 850,000$ units in July 2013 to $\approx 1,170,000$ units in July 2017). This proportional expansion strongly suggests multiplicative rather than additive seasonality.

---

## 4. Store Performance & Heterogeneity Analysis
- **What Was Calculated**: Total cumulative unit sales and average daily velocity across all 10 stores.
- **Actual Numerical Findings**:
  | Store ID | Total 5-Year Sales | Market Share | Velocity Rank |
  |---|---|---|---|
  | Store 2 | 6,153,994 | $12.9\%$ | #1 (Highest Volume) |
  | Store 8 | 6,056,169 | $12.7\%$ | #2 |
  | Store 3 | 5,435,140 | $11.4\%$ | #3 |
  | Store 10 | 5,360,158 | $11.2\%$ | #4 |
  | Store 9 | 5,025,976 | $10.5\%$ | #5 |
  | Store 4 | 4,816,019 | $10.1\%$ | #6 |
  | Store 1 | 4,310,731 | $9.0\%$ | #7 |
  | Store 5 | 3,352,842 | $7.0\%$ | #8 |
  | Store 6 | 3,346,310 | $7.0\%$ | #9 |
  | Store 7 | 3,134,287 | $6.6\%$ | #10 (Lowest Volume) |
- **Business Insight**: Store 2 sells almost **double** the volume of Store 7 ($1.96\times$). Despite identical catalog offerings (all 50 items), store footprint, location density, or regional demographics drive substantial volume disparities. Inventory allocation cannot be uniform.
- **Interview Question**: *"How did you handle store differences when building global machine learning models?"*
  - **Answer**: We encoded Store ID as a categorical identifier and provided historical store-level rolling volume features, allowing the tree ensemble to learn store-specific intercepts and scaling factors within a single unified model.

---

## 5. Product Catalog Demand Dispersion (Pareto Analysis)
- **What Was Calculated**: Distribution of sales across the 50 product catalog items.
- **Actual Numerical Findings**:
  - **Top Velocity Items**: Item 15 ($1,699,776$ units), Item 28 ($1,688,055$ units), Item 18 ($1,614,406$ units).
  - **Low Velocity Items**: Item 5 ($443,013$ units), Item 1 ($435,424$ units).
  - **Item Volume Ratio**: Top-selling Item 15 sells **3.9x** more volume than Item 1.
- **Business Insight**: Confirms retail catalog Pareto dynamics. High-velocity items require stringent safety stock parameters and shorter reorder cycles, whereas low-velocity items can be replenished via reorder-point pull systems to conserve shelf space.
