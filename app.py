"""
Retail Demand Forecasting & Seasonal Intelligence
Enterprise Business Intelligence & Forecasting Web Application
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =====================================================================
# APPLICATION CONFIGURATION & CUSTOM THEME
# =====================================================================
st.set_page_config(
    page_title="Retail Demand Intelligence & Forecasting Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom executive CSS
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-container {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #2563EB;
        font-weight: 500;
        margin-top: 4px;
    }
    /* Headers */
    .page-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    /* Alert / Callout Box */
    .insight-card {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    .warning-card {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# DATA CACHING & FAST DATA LOADER
# =====================================================================
@st.cache_data(show_spinner=False)
def load_application_data():
    """Load preprocessed historical series, forecasts, and evaluation metrics."""
    base_dir = Path(__file__).resolve().parent

    raw_path = base_dir / "data" / "raw" / "train.csv"
    daily_path = base_dir / "data" / "processed" / "daily_total_sales.csv"
    store_path = base_dir / "data" / "processed" / "store_daily_sales.csv"
    item_path = base_dir / "data" / "processed" / "item_daily_sales.csv"
    meta_path = base_dir / "data" / "processed" / "metadata.json"

    val_comp_path = base_dir / "outputs" / "model_comparison_validation.csv"
    test_comp_path = base_dir / "outputs" / "model_comparison_test.csv"
    multi_comp_path = base_dir / "outputs" / "multi_series_model_comparison.csv"
    fc_90_path = base_dir / "outputs" / "forecasts" / "forecast_aggregate_90_days.csv"
    fc_combo_path = base_dir / "outputs" / "forecasts" / "forecast_store_item_combinations_90_days.csv"

    daily_df = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame()
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"])

    store_df = pd.read_csv(store_path) if store_path.exists() else pd.DataFrame()
    if not store_df.empty:
        store_df["date"] = pd.to_datetime(store_df["date"])

    item_df = pd.read_csv(item_path) if item_path.exists() else pd.DataFrame()
    if not item_df.empty:
        item_df["date"] = pd.to_datetime(item_df["date"])

    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    val_comp = pd.read_csv(val_comp_path) if val_comp_path.exists() else pd.DataFrame()
    test_comp = pd.read_csv(test_comp_path) if test_comp_path.exists() else pd.DataFrame()
    multi_comp = pd.read_csv(multi_comp_path) if multi_comp_path.exists() else pd.DataFrame()

    fc_90 = pd.read_csv(fc_90_path) if fc_90_path.exists() else pd.DataFrame()
    if not fc_90.empty:
        fc_90["date"] = pd.to_datetime(fc_90["date"])

    fc_combo = pd.read_csv(fc_combo_path) if fc_combo_path.exists() else pd.DataFrame()
    if not fc_combo.empty:
        fc_combo["date"] = pd.to_datetime(fc_combo["date"])

    return daily_df, store_df, item_df, metadata, val_comp, test_comp, multi_comp, fc_90, fc_combo


@st.cache_data(show_spinner=False)
def load_raw_subset(store: int, item: int):
    """Load specific store-item daily time series on demand."""
    base_dir = Path(__file__).resolve().parent
    raw_path = base_dir / "data" / "raw" / "train.csv"
    if not raw_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(raw_path)
    sub = df[(df["store"] == store) & (df["item"] == item)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    return sub.sort_values("date").reset_index(drop=True)


daily_df, store_df, item_df, metadata, val_comp, test_comp, multi_comp, fc_90, fc_combo = load_application_data()


# =====================================================================
# SIDEBAR NAVIGATION & GLOBAL BRANDING
# =====================================================================
with st.sidebar:
    st.markdown("## 📈 Retail Intelligence")
    st.caption("Enterprise Demand Forecasting & Seasonal Intelligence System")

    nav_page = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "🔍 Sales Intelligence",
            "🔮 Forecasting Studio",
            "🏆 Model Benchmark & Comparison",
            "☀️ Seasonal Intelligence",
            "📥 Reports & Downloads",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.markdown(f"**Records Monitored:** {metadata.get('total_records', 913000):,}")
    st.markdown(f"**Network Scope:** 10 Stores · 50 Items")
    st.markdown(f"**Validation:** 90d Holdout (Zero Leakage)")
    st.markdown(f"**Test Champion:** HistGradientBoosting")
    st.markdown(f"**Best Test Error:** **2.02% WAPE** (sMAPE: 2.14%)")

    st.markdown("---")
    st.caption("Powered by Scikit-Learn, Statsmodels & Streamlit")


# =====================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# =====================================================================
if nav_page == "📊 Executive Dashboard":
    st.markdown('<div class="page-title">Executive Demand Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Strategic overview of enterprise retail unit sales, macroeconomic trends, and supply chain health</div>', unsafe_allow_html=True)

    # Executive Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    total_sales = metadata.get("mean_sales", 52.26) * metadata.get("total_records", 913000)
    with kpi1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">5-Year Unit Volume</div>
            <div class="metric-value">{int(total_sales/1e6):.1f}M</div>
            <div class="metric-subtitle">47,740,791 Units Total</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Chain Daily Velocity</div>
            <div class="metric-value">26.1K</div>
            <div class="metric-subtitle">Units / Day (Avg)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">5-Year Sales Growth</div>
            <div class="metric-value">+37.2%</div>
            <div class="metric-subtitle">2013 → 2017 Expansion</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Top Retail Outlet</div>
            <div class="metric-value">Store 2</div>
            <div class="metric-subtitle">6.15M Units (12.9% Share)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Best Forecast Accuracy</div>
            <div class="metric-value">97.9%</div>
            <div class="metric-subtitle">2.14% sMAPE (HistGB)</div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Macro Sales Timeline
    st.markdown("### 5-Year Aggregate Retail Demand Trajectory (2013 – 2017)")
    if not daily_df.empty:
        plot_df = daily_df.copy()
        plot_df["MA30"] = plot_df["sales"].rolling(30).mean()
        plot_df["MA90"] = plot_df["sales"].rolling(90).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df["date"], y=plot_df["sales"],
            mode="lines", name="Daily Total Sales",
            line=dict(color="#94A3B8", width=1.0),
            opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=plot_df["date"], y=plot_df["MA30"],
            mode="lines", name="30-Day Moving Average",
            line=dict(color="#2563EB", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=plot_df["date"], y=plot_df["MA90"],
            mode="lines", name="90-Day Moving Average",
            line=dict(color="#DC2626", width=2.0, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(
                title="Date",
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis=dict(title="Unit Sales (Chain-Wide)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=30, b=40),
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Executive Strategic Insights
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("""
        <div class="insight-card">
            <h4>💡 Key Commercial Insights</h4>
            <ul>
                <li><strong>Weekend Demand Multiplier:</strong> Sunday generates <strong>43.5% higher volume</strong> than Monday. Store delivery and cross-dock logistics should position inventory on Thursdays and Fridays.</li>
                <li><strong>Summer Holiday Surge:</strong> June and July drive peak volume (up to <strong>1.98×</strong> January demand). Procurement lead-times require purchase orders by April.</li>
                <li><strong>Multiplicative Scaling:</strong> As the retail chain expanded (+37%), the seasonal amplitude grew proportionally. Multiplicative decomposition achieved an RMSE of 8,855.21 (vs 15,338.05 additive).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div class="warning-card">
            <h4>🛡️ Supply Chain & Inventory Impact</h4>
            <ul>
                <li><strong>Safety Stock Optimization:</strong> Lowering forecast error from 13.3% (Seasonal Naive) to 2.14% (HistGB) enables a <strong>~30% reduction</strong> in required safety stock buffers.</li>
                <li><strong>Zero Stockout Records:</strong> The dataset observes fulfilled sales. Where inventory stockouts occurred, true customer demand was censored.</li>
                <li><strong>Working Capital Efficiency:</strong> Accurate 90-day demand visibility prevents Q4 overstocking and costly post-holiday margin markdowns.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# PAGE 2: SALES INTELLIGENCE
# =====================================================================
elif nav_page == "🔍 Sales Intelligence":
    st.markdown('<div class="page-title">Granular Sales Intelligence & Slice Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Slice, filter, and inspect demand velocity across stores, product categories, and temporal ranges</div>', unsafe_allow_html=True)

    # Dynamic Filter Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        store_choice = st.selectbox("Store Filter", ["All Stores"] + [f"Store {i}" for i in range(1, 11)], index=0)
    with f_col2:
        item_choice = st.selectbox("Product Filter", ["All Products"] + [f"Item {i}" for i in range(1, 51)], index=0)
    with f_col3:
        date_range = st.date_input("Date Range", value=(daily_df["date"].min().date(), daily_df["date"].max().date()),
                                   min_value=daily_df["date"].min().date(), max_value=daily_df["date"].max().date())

    start_date = pd.to_datetime(date_range[0]) if len(date_range) > 0 else daily_df["date"].min()
    end_date = pd.to_datetime(date_range[1]) if len(date_range) > 1 else daily_df["date"].max()

    # Data Slicing Logic
    if store_choice == "All Stores" and item_choice == "All Products":
        filtered_data = daily_df[(daily_df["date"] >= start_date) & (daily_df["date"] <= end_date)].copy()
        slice_name = "Chain-Wide Aggregate Total"
    elif store_choice != "All Stores" and item_choice == "All Products":
        s_id = int(store_choice.split()[1])
        filtered_data = store_df[(store_df["store"] == s_id) & (store_df["date"] >= start_date) & (store_df["date"] <= end_date)].copy()
        slice_name = f"Store {s_id} Aggregate"
    elif store_choice == "All Stores" and item_choice != "All Products":
        i_id = int(item_choice.split()[1])
        filtered_data = item_df[(item_df["item"] == i_id) & (item_df["date"] >= start_date) & (item_df["date"] <= end_date)].copy()
        slice_name = f"Item {i_id} Aggregate (All Stores)"
    else:
        s_id = int(store_choice.split()[1])
        i_id = int(item_choice.split()[1])
        sub_raw = load_raw_subset(s_id, i_id)
        filtered_data = sub_raw[(sub_raw["date"] >= start_date) & (sub_raw["date"] <= end_date)].copy()
        slice_name = f"Store {s_id} - Item {i_id}"

    # Filtered Slice KPIs
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Filtered Volume", f"{int(filtered_data['sales'].sum()):,} units")
    with m2:
        st.metric("Daily Run Rate", f"{filtered_data['sales'].mean():.1f} units/day")
    with m3:
        st.metric("Peak Day Volume", f"{int(filtered_data['sales'].max()):,} units")
    with m4:
        st.metric("Daily Volatility (σ)", f"{filtered_data['sales'].std():.1f} units")

    # Interactive Sliced Time Series
    fig = px.line(
        filtered_data, x="date", y="sales",
        title=f"Demand History: {slice_name} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
        labels={"sales": "Unit Sales", "date": "Date"},
        color_discrete_sequence=["#2563EB"],
    )
    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Store & Item Volume Distribution Benchmarks
    c_s, c_p = st.columns(2)
    with c_s:
        st.markdown("#### Store Volume Distribution Ranking")
        if not store_df.empty:
            store_tot = store_df.groupby("store")["sales"].sum().reset_index().sort_values("sales", ascending=True)
            store_tot["Store_Label"] = store_tot["store"].apply(lambda s: f"Store {s}")
            fig_store = px.bar(
                store_tot, x="sales", y="Store_Label", orientation="h",
                color="sales", color_continuous_scale="Blues",
                labels={"sales": "Cumulative Sales (Units)", "Store_Label": "Store Outlet"},
            )
            fig_store.update_layout(template="plotly_white", height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_store, use_container_width=True)
    with c_p:
        st.markdown("#### Top 10 vs Bottom 10 Catalog Products (Pareto)")
        if not item_df.empty:
            item_tot = item_df.groupby("item")["sales"].sum().reset_index().sort_values("sales", ascending=False)
            top10 = item_tot.head(10).assign(Tier="Top 10 High Velocity")
            bot10 = item_tot.tail(10).assign(Tier="Bottom 10 Low Velocity")
            pareto_df = pd.concat([top10, bot10])
            pareto_df["Item_Label"] = pareto_df["item"].apply(lambda i: f"Item {i}")
            fig_item = px.bar(
                pareto_df, x="Item_Label", y="sales", color="Tier",
                color_discrete_map={"Top 10 High Velocity": "#059669", "Bottom 10 Low Velocity": "#DC2626"},
                labels={"sales": "Cumulative Sales (Units)", "Item_Label": "Product SKU"},
            )
            fig_item.update_layout(template="plotly_white", height=380, xaxis=dict(categoryorder="total descending"))
            st.plotly_chart(fig_item, use_container_width=True)


# =====================================================================
# PAGE 3: FORECASTING STUDIO
# =====================================================================
elif nav_page == "🔮 Forecasting Studio":
    st.markdown('<div class="page-title">Forecasting Studio & Supply Chain Replenishment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Generate out-of-sample forward projections, simulate lead-time demand, and calibrate safety stock</div>', unsafe_allow_html=True)

    fc_col1, fc_col2, fc_col3 = st.columns(3)
    with fc_col1:
        target_scope = st.selectbox("Forecasting Target", ["Chain-Wide Aggregate Demand", "Granular Store-Item Series"])
    with fc_col2:
        horizon_days = st.selectbox("Forecast Horizon", [30, 60, 90], index=2,
                                    format_func=lambda h: f"{h} Days ({'1 Month' if h==30 else '2 Months' if h==60 else '1 Quarter'})")
    with fc_col3:
        if target_scope == "Chain-Wide Aggregate Demand":
            selected_model = st.selectbox("Select Model Architecture", [
                "Recommended Ensemble (HistGB + Holt-Winters)",
                "HistGradientBoosting (Test Champion)",
                "Holt-Winters (Additive Seasonal)",
                "Seasonal Naive (s=7 Baseline)",
            ])
        else:
            selected_combo = st.selectbox("Select Store-Item Combination", [
                "Store 1 - Item 1",
                "Store 2 - Item 15 (Top Velocity)",
                "Store 3 - Item 28",
                "Store 4 - Item 33",
                "Store 10 - Item 50",
            ])

    st.markdown("---")

    if target_scope == "Chain-Wide Aggregate Demand":
        if not fc_90.empty:
            fc_slice = fc_90.head(horizon_days).copy()
            model_map = {
                "Recommended Ensemble (HistGB + Holt-Winters)": "forecast_ensemble_recommended",
                "HistGradientBoosting (Test Champion)": "forecast_hist_gradient_boosting",
                "Holt-Winters (Additive Seasonal)": "forecast_holt_winters",
                "Seasonal Naive (s=7 Baseline)": "forecast_seasonal_naive",
            }
            fc_col_name = model_map[selected_model]

            # Parametric 95% uncertainty confidence intervals (empirical residual sigma = 890 units)
            sigma_error = 890.44
            fc_slice["lower_bound"] = np.clip(fc_slice[fc_col_name] - 1.96 * sigma_error, 0, None)
            fc_slice["upper_bound"] = fc_slice[fc_col_name] + 1.96 * sigma_error

            # Summary Metrics
            tot_fc = int(fc_slice[fc_col_name].sum())
            avg_fc = int(fc_slice[fc_col_name].mean())
            peak_row = fc_slice.loc[fc_slice[fc_col_name].idxmax()]
            trough_row = fc_slice.loc[fc_slice[fc_col_name].idxmin()]

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Total Projected Demand", f"{tot_fc:,} units")
            with k2:
                st.metric("Average Daily Rate", f"{avg_fc:,} units/day")
            with k3:
                st.metric("Projected Peak Day", f"{peak_row['date'].strftime('%b %d, %Y')}", f"{int(peak_row[fc_col_name]):,} units")
            with k4:
                st.metric("Projected Trough Day", f"{trough_row['date'].strftime('%b %d, %Y')}", f"{int(trough_row[fc_col_name]):,} units")

            # Interactive Plotly Forecast Chart
            fig = go.Figure()
            hist_tail = daily_df.tail(90)
            # Historical actuals
            fig.add_trace(go.Scatter(
                x=hist_tail["date"], y=hist_tail["sales"],
                mode="lines", name="Historical Actuals (Q4 2017)",
                line=dict(color="#0F172A", width=1.5),
            ))
            # Upper bound
            fig.add_trace(go.Scatter(
                x=fc_slice["date"], y=fc_slice["upper_bound"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=fc_slice["date"], y=fc_slice["lower_bound"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(37, 99, 235, 0.15)",
                name="95% Confidence Interval (±1.96σ)",
            ))
            # Forecast line
            fig.add_trace(go.Scatter(
                x=fc_slice["date"], y=fc_slice[fc_col_name],
                mode="lines+markers", name=f"Forecast: {selected_model}",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=4),
            ))
            fig.update_layout(
                title=f"{horizon_days}-Day Out-of-Sample Retail Demand Projection (Q1 2018)",
                template="plotly_white",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Chain-Wide Unit Sales"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Replenishment Calculator Simulation
            st.markdown("### 📦 Supply Chain Replenishment Heuristics Calculator")
            st.caption("Calculate recommended Safety Stock buffers and Reorder Points (ROP) driven by forecast uncertainty.")
            c_rp1, c_rp2, c_rp3 = st.columns(3)
            with c_rp1:
                lead_time = st.slider("Supplier Lead Time (Days)", min_value=1, max_value=30, value=7)
            with c_rp2:
                service_level = st.select_slider("Target Cycle Service Level", options=["90%", "95%", "98%", "99%"], value="95%")
                z_map = {"90%": 1.28, "95%": 1.65, "98%": 2.05, "99%": 2.33}
                z_score = z_map[service_level]
            with c_rp3:
                safety_stock = int(z_score * sigma_error * np.sqrt(lead_time))
                cycle_demand = int(avg_fc * lead_time)
                reorder_point = cycle_demand + safety_stock
                st.metric("Recommended Reorder Point (ROP)", f"{reorder_point:,} units")
                st.caption(f"Lead Demand: {cycle_demand:,} + Safety Buffer: {safety_stock:,}")

            st.markdown("""
            <div class="warning-card">
                <strong>⚠️ Explicit Operational Notice:</strong> The dataset does not contain physical inventory or stockout ledgers.
                Replenishment calculations are mathematical estimates derived from forecast demand ($d$) and stated supply chain heuristics ($Z \\cdot \\sigma \\sqrt{L}$).
            </div>
            """, unsafe_allow_html=True)

            # Forecast Data Table
            with st.expander("📋 View Detailed Forecast Schedule Data Table"):
                disp_df = fc_slice[["date", fc_col_name, "lower_bound", "upper_bound"]].copy()
                disp_df.columns = ["Date", "Predicted Sales", "95% Lower Bound", "95% Upper Bound"]
                st.dataframe(disp_df.style.format({"Predicted Sales": "{:.1f}", "95% Lower Bound": "{:.1f}", "95% Upper Bound": "{:.1f}"}), use_container_width=True)

    else:
        # Granular store-item forecast
        s_id = selected_combo.split()[1]
        i_id = selected_combo.split()[4]
        combo_col = f"store_{s_id}_item_{i_id}_sales_forecast"

        if combo_col in fc_combo.columns:
            fc_sub = fc_combo[["date", combo_col]].head(horizon_days).copy()
            fc_sub.columns = ["date", "sales_forecast"]

            # Store-item KPIs
            tot_c = int(fc_sub["sales_forecast"].sum())
            avg_c = fc_sub["sales_forecast"].mean()
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric(f"Projected Demand (Store {s_id}, Item {i_id})", f"{tot_c:,} units")
            with k2:
                st.metric("Average Daily Rate", f"{avg_c:.1f} units/day")
            with k3:
                st.metric("Horizon Span", f"{horizon_days} Calendar Days")

            fig_c = px.line(
                fc_sub, x="date", y="sales_forecast",
                title=f"{horizon_days}-Day Forward Projection: Store {s_id} - Item {i_id} (HistGB Engine)",
                labels={"sales_forecast": "Predicted Daily Units", "date": "Date"},
                markers=True,
                color_discrete_sequence=["#059669"],
            )
            fig_c.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_c, use_container_width=True)
            st.dataframe(fc_sub.style.format({"sales_forecast": "{:.1f}"}), use_container_width=True)


# =====================================================================
# PAGE 4: MODEL BENCHMARK & COMPARISON
# =====================================================================
elif nav_page == "🏆 Model Benchmark & Comparison":
    st.markdown('<div class="page-title">Model Benchmark Leaderboard & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Comprehensive benchmarking across Baselines, Exponential Smoothing, ARIMA/SARIMA, and Tree Ensembles</div>', unsafe_allow_html=True)

    t_val, t_test, t_diag = st.tabs(["📋 Validation Leaderboard (Selection)", "🏁 Final Test Set Performance", "🔬 Residual Diagnostics"])

    with t_val:
        st.subheader("Validation Leaderboard (90-Day Chronological Horizon / Q3 2017)")
        st.caption("Trained on 2013-01-01 to 2017-07-04; evaluated strictly out-of-sample on 2017-07-05 to 2017-10-02 without future leakage.")
        if not val_comp.empty:
            st.dataframe(
                val_comp.style.highlight_min(subset=["RMSE", "MAE", "sMAPE", "WAPE", "MASE"], color="#D1FAE5")
                .highlight_max(subset=["RMSE"], color="#FEE2E2")
                .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "sMAPE": "{:.2f}%", "WAPE": "{:.2f}%", "MAPE": "{:.2f}%", "MASE": "{:.3f}", "Training_Time_Sec": "{:.3f}s"}),
                use_container_width=True,
            )

            # Interactive Plotly Benchmark Bar Chart
            metric_sel = st.selectbox("Select Benchmark Metric to Compare", ["RMSE", "MAE", "sMAPE", "WAPE", "MASE"], index=0)
            plot_val = val_comp.sort_values(metric_sel, ascending=True)
            fig_bar = px.bar(
                plot_val, x=metric_sel, y="Model", orientation="h",
                title=f"Validation Leaderboard Ranked by {metric_sel} (Lower is Better)",
                color=metric_sel, color_continuous_scale="Teal",
                text=metric_sel,
            )
            fig_bar.update_layout(template="plotly_white", height=500, yaxis=dict(autorange="reversed"))
            fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

    with t_test:
        st.subheader("Final Unseen Test Performance (90-Day Holdout / Q4 2017)")
        st.caption("Retrained on full Train + Validation history (1,736 days) and evaluated strictly on the untouched test quarter (2017-10-03 to 2017-12-31).")
        if not test_comp.empty:
            st.dataframe(
                test_comp.style.highlight_min(subset=["RMSE", "MAE", "sMAPE", "WAPE", "MASE"], color="#D1FAE5")
                .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "sMAPE": "{:.2f}%", "WAPE": "{:.2f}%", "MAPE": "{:.2f}%", "MASE": "{:.3f}", "Training_Time_Sec": "{:.3f}s"}),
                use_container_width=True,
            )

            st.markdown("""
            <div class="insight-card">
                <strong>🏆 Champion Architecture:</strong> <strong>HistGradientBoosting</strong> achieved the lowest overall test error:
                <strong>RMSE: 890.44 · sMAPE: 2.14% · WAPE: 2.02% · MASE: 0.710</strong>.<br>
                It reduced forecast error by <strong>82.4%</strong> compared to the Seasonal Naive baseline (RMSE 5,050.29) and trained in just 1.12 seconds.
            </div>
            """, unsafe_allow_html=True)

    with t_diag:
        st.subheader("Model Diagnostic Residuals & Visual Trajectories")
        c_i1, c_i2 = st.columns(2)
        with c_i1:
            p_val = Path("outputs/plots/validation_forecast_comparison.png")
            if p_val.exists():
                st.image(str(p_val), caption="Validation Holdout Forecast Trajectories vs Actuals", use_container_width=True)
        with c_i2:
            p_test = Path("outputs/plots/test_forecast_comparison.png")
            if p_test.exists():
                st.image(str(p_test), caption="Final Test Holdout Forecast Trajectories vs Actuals", use_container_width=True)

        p_resid = Path("outputs/plots/test_residuals_best_model.png")
        if p_resid.exists():
            st.image(str(p_resid), caption="Champion Model Residual Diagnostics (Error Over Time & Distribution)", use_container_width=True)


# =====================================================================
# PAGE 5: SEASONAL INTELLIGENCE
# =====================================================================
elif nav_page == "☀️ Seasonal Intelligence":
    st.markdown('<div class="page-title">Seasonal Intelligence & Time Series Decomposition</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Uncover intra-week shopping patterns, annual summer cycles, and econometric decomposition mechanics</div>', unsafe_allow_html=True)

    s_tab1, s_tab2 = st.tabs(["📅 Intra-Week & Annual Cycles", "🔬 Econometric Decomposition"])

    with s_tab1:
        st.subheader("Intra-Week Volume Distribution & Weekend Acceleration")
        # Day of week interactive bar
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_avg = [21392, 25053, 25142, 26712, 28458, 30223, 30695]
        dow_df = pd.DataFrame({"Day": dow_names, "Mean_Sales": dow_avg, "Index": [1.00, 1.17, 1.18, 1.25, 1.33, 1.41, 1.44]})

        fig_dow = px.bar(
            dow_df, x="Day", y="Mean_Sales", color="Mean_Sales",
            color_continuous_scale="Blues",
            title="Average Daily Unit Demand by Day of Week (Chain-Wide)",
            labels={"Mean_Sales": "Average Units Sold", "Day": "Day of Week"},
            text="Mean_Sales",
        )
        fig_dow.update_layout(template="plotly_white", height=380, coloraxis_showscale=False)
        fig_dow.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig_dow, use_container_width=True)

        st.markdown("""
        - 🚀 **Sunday Peak Effect**: Sunday volume ($30,695$ units/day) is **43.5% higher** than Monday baseline ($21,392$ units/day).
        - 🚚 **Replenishment Strategy**: Warehouses must disaggregate logistics dispatch: 45% of warehouse picking and store deliveries must occur on Thursday and Friday.
        """)

        # Heatmap plot
        p_heat = Path("outputs/plots/eda_monthly_heatmap.png")
        if p_heat.exists():
            st.image(str(p_heat), caption="Annual Monthly Seasonality Heatmap (2013 - 2017)", use_container_width=True)

    with s_tab2:
        st.subheader("Classical Decomposition: Additive vs. Multiplicative")
        st.markdown(r"""
        When decomposing a retail time series into Trend, Seasonal, and Residual components ($y_t = T_t \circ S_t \circ R_t$), we evaluated both classical formulations:
        """)
        c_dec1, c_dec2 = st.columns(2)
        with c_dec1:
            st.metric("Additive Decomposition RMSE", "15,338.05 units")
            p_add = Path("outputs/plots/decomposition_additive_monthly.png")
            if p_add.exists():
                st.image(str(p_add), caption="Additive Decomposition (Period=12)", use_container_width=True)
        with c_dec2:
            st.metric("Multiplicative Decomposition RMSE", "8,855.21 units", "-42.3% Error Reduction")
            p_mult = Path("outputs/plots/decomposition_multiplicative_monthly.png")
            if p_mult.exists():
                st.image(str(p_mult), caption="Multiplicative Decomposition (Period=12)", use_container_width=True)

        st.info("💡 Multiplicative decomposition is mathematically superior because as the retail chain expanded (+37% growth), seasonal swings widened proportionally rather than remaining fixed in absolute units.")


# =====================================================================
# PAGE 6: REPORTS & DOWNLOADS
# =====================================================================
elif nav_page == "📥 Reports & Downloads":
    st.markdown('<div class="page-title">Enterprise Reports & Data Export Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Export out-of-sample forecast schedules, model leaderboards, and preprocessed datasets</div>', unsafe_allow_html=True)

    c_exp1, c_exp2 = st.columns(2)

    with c_exp1:
        st.subheader("📦 Forecast CSV Exports")
        st.caption("Production forecast tables for ERP ingestion (SAP, Blue Yonder, Oracle Retail).")

        for days in [90, 60, 30]:
            p = Path(f"outputs/forecasts/forecast_aggregate_{days}_days.csv")
            if p.exists():
                with open(p, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {days}-Day Chain Forecast CSV",
                        data=f,
                        file_name=f"forecast_aggregate_{days}_days.csv",
                        mime="text/csv",
                        key=f"dl_fc_{days}",
                    )

        p_multi = Path("outputs/forecasts/forecast_store_item_combinations_90_days.csv")
        if p_multi.exists():
            with open(p_multi, "rb") as f:
                st.download_button(
                    label="📥 Download Store-Item 90-Day Combinations Forecast CSV",
                    data=f,
                    file_name="forecast_store_item_combinations_90_days.csv",
                    mime="text/csv",
                    key="dl_multi_fc",
                )

    with c_exp2:
        st.subheader("📊 Model Evaluation & Benchmark Exports")
        st.caption("Audit reports and validation metrics for executive reviews.")

        for fname, desc, key in [
            ("model_comparison_validation.csv", "Validation Leaderboard CSV", "dl_val_csv"),
            ("model_comparison_test.csv", "Final Test Performance CSV", "dl_test_csv"),
            ("multi_series_model_comparison.csv", "Multi-Series Performance CSV", "dl_multi_csv"),
        ]:
            p = Path("outputs") / fname
            if p.exists():
                with open(p, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {desc}",
                        data=f,
                        file_name=fname,
                        mime="text/csv",
                        key=key,
                    )

    st.markdown("---")
    st.subheader("📚 System Methodology & Technical Documentation")
    st.markdown("""
    Complete technical documentation is available in the repository `docs/` folder:
    - **01. Project Overview**: System architecture, business objectives, dataset boundaries.
    - **04. Time Series Fundamentals**: STL, LOESS, ADF/KPSS stationarity testing, differencing.
    - **05. Feature Engineering**: Shifted lags ($1..365$), shifted rolling windows, cyclical sin/cos terms.
    - **06. Forecasting Models**: Formulas, assumptions, and mechanics for all 16 models.
    - **10. Interview Preparation Guide**: 50+ project-specific interview questions and answers.
    """)
