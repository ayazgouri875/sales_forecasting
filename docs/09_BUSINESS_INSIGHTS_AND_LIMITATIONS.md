# 09. Business Insights, Inventory Planning & Limitations

## 1. Key Business Insights

### A. Weekend Demand Acceleration
- **Empirical Discovery**: Daily unit sales build steadily through the workweek (Tuesday through Friday), jumping dramatically on Saturday ($+41.3\%$) and Sunday ($+43.5\%$) relative to Monday.
- **Operational Application**:
  - Logistics and cross-dock replenishment must schedule primary store deliveries on Thursday and Friday.
  - Store staffing schedules should allocate $\approx 45\%$ of retail associate labor hours between Friday evening and Sunday close.

### B. Summer Holiday Demand Peak
- **Empirical Discovery**: Sales surge across June and July, reaching nearly **double** the sales volume observed in January and February ($1.98\times$ ratio).
- **Operational Application**:
  - Central warehouse procurement must submit vendor bulk purchase orders in March/April (accounting for standard 60-day manufacturing and shipping lead times).
  - Safety stock levels should be adjusted dynamically throughout the year rather than held constant.

### C. Store Footprint Disparities
- **Empirical Discovery**: Store 2 and Store 8 together drive over **25.6%** of total chain volume, while Store 6 and Store 7 account for only **13.6%**.
- **Operational Application**:
  - Space allocation and backroom storage in high-velocity stores (Store 2, 8) must be optimized to prevent stockouts during peak shopping hours.
  - Slower-velocity outlets (Store 6, 7) should use lower reorder points to avoid shelf obsolescence.

---

## 2. Inventory Planning Heuristics (Stated Assumptions)

Because the dataset does not include inventory or stockout flags, we apply classical inventory theory formulas based on forecast demand and explicit operational assumptions:

### Safety Stock Calculation
$$\text{Safety Stock} = Z \times \sigma_{\text{forecast}} \times \sqrt{L}$$
Where:
- $Z$: Service-level factor (e.g., $Z = 1.65$ for $95\%$ in-stock service level; $Z = 2.05$ for $98\%$).
- $\sigma_{\text{forecast}}$: Forecast error standard deviation (empirically derived from validation RMSE).
- $L$: Replenishment lead time in days (e.g., $L = 7$ days).

### Reorder Point (ROP) Formula
$$\text{ROP} = (\text{Average Daily Demand} \times L) + \text{Safety Stock}$$
- **Example Calculation (Aggregate Chain-Wide)**:
  - Average Daily Demand: $\approx 26,000$ units/day.
  - Lead Time ($L$): $7$ days.
  - Cycle Demand: $26,000 \times 7 = 182,000$ units.
  - $\sigma_{\text{error}}$ (HistGB): $\approx 890$ units.
  - Safety Stock ($95\%$ service): $1.65 \times 890 \times \sqrt{7} \approx 3,885$ units.
  - **Recommended ROP**: $182,000 + 3,885 = 185,885$ units.

---

## 3. Real-World Limitations & Explicit Assumptions

### Limitations
1. **Unconstrained vs. Constrained Demand**: We observe historical sales, not true consumer demand. If items were out of stock, recorded sales dropped to zero, potentially biasing forecasts downward.
2. **Missing Pricing & Elasticity**: Price changes, discounts, and markdown clearances are unrecorded. We cannot model price elasticity.
3. **Absence of Marketing Data**: Circulars, digital ad campaigns, loyalty promotions, and coupon drops are omitted.
4. **No Direct Supply Chain Costs**: Unit margins, holding costs, and transportation freight rates are unavailable.

### Assumptions
- Customer demand patterns are assumed to remain structurally consistent into early 2018.
- Seasonality is assumed to follow the 7-day weekly cycle and 365-day annual cycle identified in historical data.
- Inventory estimates assume stable supplier lead times and normally distributed forecast errors.
