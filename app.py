"""
PharmaGuard AI – Pharmaceutical Demand Intelligence
Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaGuard AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
DRUGS = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
DRUG_NAMES = {
    'M01AB': 'M01AB – Anti-inflammatories',
    'M01AE': 'M01AE – Ibuprofen Group',
    'N02BA': 'N02BA – Aspirin Group',
    'N02BE': 'N02BE – Paracetamol Group',
    'N05B':  'N05B – Anxiolytics',
    'N05C':  'N05C – Hypnotics/Sedatives',
    'R03':   'R03 – Airway Drugs',
    'R06':   'R06 – Antihistamines',
}
COLORS = px.colors.qualitative.D3

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    daily   = pd.read_csv("IBMDataset/salesdaily.csv",  parse_dates=["datum"])
    hourly  = pd.read_csv("IBMDataset/saleshourly.csv", parse_dates=["datum"])
    weekly  = pd.read_csv("IBMDataset/salesweekly.csv", parse_dates=["datum"])
    monthly = pd.read_csv("IBMDataset/salesmonthly.csv", parse_dates=["datum"])

    # Clean: remove all-zero rows (data gap Jan 2017)
    daily   = daily[daily[DRUGS].sum(axis=1) > 0].copy()
    monthly = monthly[monthly[DRUGS].sum(axis=1) > 0].copy()
    weekly  = weekly[weekly[DRUGS].sum(axis=1) > 0].copy()
    hourly  = hourly[hourly[DRUGS].sum(axis=1) > 0].copy()

    # Derived columns
    monthly["Year"]  = monthly["datum"].dt.year
    monthly["Month"] = monthly["datum"].dt.month
    daily["Year"]    = daily["datum"].dt.year
    daily["Month"]   = daily["datum"].dt.month

    # Day-of-week ordering
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    if "Weekday Name" in daily.columns:
        daily["Weekday Name"] = pd.Categorical(
            daily["Weekday Name"], categories=dow_order, ordered=True
        )
    return daily, hourly, weekly, monthly

daily, hourly, weekly, monthly = load_data()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=80,
    )
    st.title("PharmaGuard AI")
    st.caption("Pharmaceutical Demand Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Overview & KPIs",
         "📈 Trends & Seasonality",
         "⏱ Hourly & Weekly Patterns",
         "🔍 Anomaly Detection",
         "🔮 Demand Forecasting",
         "⚠️ Risk Analysis",
         "💡 Opportunities & Actions",
         "📋 Raw Data Explorer"],
        index=0,
    )

    st.divider()
    selected_drugs = st.multiselect(
        "Filter Drug Categories",
        options=DRUGS,
        default=DRUGS,
        format_func=lambda x: DRUG_NAMES[x],
    )
    if not selected_drugs:
        selected_drugs = DRUGS

    year_range = st.slider(
        "Year Range",
        int(monthly["Year"].min()),
        int(monthly["Year"].max()),
        (int(monthly["Year"].min()), int(monthly["Year"].max())),
    )

    monthly_f = monthly[
        (monthly["Year"] >= year_range[0]) &
        (monthly["Year"] <= year_range[1])
    ]

    st.divider()
    st.caption("Data: Pharmacy Sales 2014–2019")
    st.caption("Built with IBM Bob · Streamlit")


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def metric_card(label, value, delta=None, suffix=""):
    if delta is not None:
        st.metric(label, f"{value}{suffix}", delta=f"{delta:+.1f}%")
    else:
        st.metric(label, f"{value}{suffix}")


# ══════════════════════════════════════════════
# PAGE 1 – OVERVIEW & KPIs
# ══════════════════════════════════════════════
if page == "🏠 Overview & KPIs":
    st.title("💊 PharmaGuard AI – Pharmaceutical Demand Intelligence")
    st.markdown(
        "> **Framework:** KPIs → Trends → Drivers → Risk → Opportunity → Action  \n"
        "> **Data source:** Real pharmacy sales data (Slovenia, 2014–2019)  \n"
        "> **Drug categories:** M01AB · M01AE · N02BA · N02BE · N05B · N05C · R03 · R06"
    )
    st.divider()

    # KPI row
    totals     = monthly_f[selected_drugs].sum()
    avgs       = monthly_f[selected_drugs].mean()
    top_drug   = totals.idxmax()
    total_all  = totals.sum()
    months_cnt = monthly_f["datum"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Units Sold", f"{total_all:,.0f}", help="Sum across all selected drugs and years")
    with c2:
        st.metric("Months Covered", months_cnt)
    with c3:
        st.metric("Top Drug", top_drug, help=DRUG_NAMES[top_drug])
    with c4:
        st.metric("Avg Monthly (Top)", f"{avgs[top_drug]:.1f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Market Share by Drug Category")
        ms = (totals / totals.sum() * 100).round(2).sort_values(ascending=False)
        fig = px.pie(
            values=ms.values,
            names=ms.index,
            color_discrete_sequence=COLORS,
            hole=0.35,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=True, height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Total Sales by Drug Category")
        fig2 = px.bar(
            x=totals.sort_values(ascending=False).index,
            y=totals.sort_values(ascending=False).values,
            color=totals.sort_values(ascending=False).index,
            color_discrete_sequence=COLORS,
            labels={"x": "Drug", "y": "Total Units"},
        )
        fig2.update_layout(showlegend=False, height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("KPI Summary Table")
    cv = (monthly_f[selected_drugs].std() / monthly_f[selected_drugs].mean() * 100).round(2)
    kpi_table = pd.DataFrame({
        "Drug": selected_drugs,
        "Total Sales": [totals[d] for d in selected_drugs],
        "Monthly Avg": [avgs[d].round(2) for d in selected_drugs],
        "Monthly Max": [monthly_f[d].max() for d in selected_drugs],
        "Monthly Min": [monthly_f[d].min() for d in selected_drugs],
        "CV % (Volatility)": [cv[d] for d in selected_drugs],
        "Market Share %": [(totals[d] / total_all * 100).round(2) for d in selected_drugs],
    }).set_index("Drug")
    st.dataframe(kpi_table.style.background_gradient(subset=["Total Sales"], cmap="Blues")
                              .format({"Total Sales": "{:.1f}", "Monthly Avg": "{:.2f}",
                                       "CV % (Volatility)": "{:.1f}%", "Market Share %": "{:.2f}%"}),
                 use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2 – TRENDS & SEASONALITY
# ══════════════════════════════════════════════
elif page == "📈 Trends & Seasonality":
    st.title("📈 Trends & Seasonality Analysis")

    st.subheader("Monthly Sales Trends Over Time")
    fig = go.Figure()
    for drug in selected_drugs:
        fig.add_trace(go.Scatter(
            x=monthly_f["datum"], y=monthly_f[drug],
            name=drug, mode="lines",
            line=dict(width=2),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Units Sold",
        height=420, legend=dict(orientation="h", yanchor="bottom", y=1.01),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Annual Sales by Year")
        annual = monthly_f.groupby("Year")[selected_drugs].sum().reset_index()
        fig2 = px.bar(
            annual.melt(id_vars="Year", value_vars=selected_drugs,
                        var_name="Drug", value_name="Units"),
            x="Year", y="Units", color="Drug",
            barmode="group",
            color_discrete_sequence=COLORS,
            height=380,
        )
        fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Year-over-Year Growth (%)")
        annual_yoy = monthly.groupby("Year")[selected_drugs].sum()
        full_yrs = annual_yoy[annual_yoy.index.isin([2014, 2015, 2016, 2017, 2018])]
        yoy = full_yrs.pct_change().dropna() * 100
        fig3 = px.imshow(
            yoy.round(1).T,
            text_auto=True,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels=dict(color="YoY %"),
            height=380,
        )
        fig3.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Seasonal Demand Patterns (Average by Month)")
    seasonal = monthly_f.groupby("Month")[selected_drugs].mean().reset_index()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    seasonal["Month_Name"] = seasonal["Month"].apply(lambda x: month_labels[x-1])

    fig4 = go.Figure()
    for i, drug in enumerate(selected_drugs):
        fig4.add_trace(go.Scatter(
            x=seasonal["Month_Name"], y=seasonal[drug],
            name=drug, mode="lines+markers",
            line=dict(width=2, color=COLORS[i % len(COLORS)]),
            marker=dict(size=7),
        ))
    fig4.update_layout(
        xaxis_title="Month", yaxis_title="Avg Units",
        height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Peak Demand Month per Drug")
    peak = monthly_f.groupby("Month")[selected_drugs].mean().idxmax()
    peak_df = pd.DataFrame({
        "Drug": peak.index,
        "Peak Month": peak.map(lambda x: month_labels[x-1]),
        "Avg Units at Peak": [monthly_f.groupby("Month")[d].mean().max().round(2) for d in selected_drugs],
    }).set_index("Drug")
    st.dataframe(peak_df, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 3 – HOURLY & WEEKLY PATTERNS
# ══════════════════════════════════════════════
elif page == "⏱ Hourly & Weekly Patterns":
    st.title("⏱ Hourly & Weekly Demand Patterns")

    st.subheader("Average Hourly Demand Profile")
    hourly_clean = hourly[hourly[DRUGS].sum(axis=1) > 0].copy()
    hourly_avg = hourly_clean.groupby("Hour")[selected_drugs].mean().reset_index()

    fig = go.Figure()
    for i, drug in enumerate(selected_drugs):
        fig.add_trace(go.Scatter(
            x=hourly_avg["Hour"], y=hourly_avg[drug],
            name=drug, mode="lines+markers",
            line=dict(width=2, color=COLORS[i % len(COLORS)]),
        ))
    fig.update_layout(
        xaxis_title="Hour of Day", yaxis_title="Avg Units",
        height=400, hovermode="x unified",
        xaxis=dict(tickmode="linear", tick0=8, dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Insight:** Demand peaks typically occur between 10:00–13:00 (morning rush) with a secondary peak after 16:00 (afternoon). Minimal activity before 9:00 and after 19:00.")

    st.divider()
    st.subheader("Average Daily Sales by Day of Week")
    if "Weekday Name" in daily.columns:
        dow_avg = daily.groupby("Weekday Name")[selected_drugs].mean().reset_index()
        fig2 = px.bar(
            dow_avg.melt(id_vars="Weekday Name", value_vars=selected_drugs,
                         var_name="Drug", value_name="Avg Units"),
            x="Weekday Name", y="Avg Units", color="Drug",
            barmode="group",
            color_discrete_sequence=COLORS,
            height=400,
        )
        fig2.update_layout(
            xaxis_title="Day of Week", yaxis_title="Avg Daily Units",
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Weekly Sales Heatmap (Top Drug: N02BE)")
    weekly_drug = "N02BE"
    weekly_data = weekly[["datum", weekly_drug]].copy()
    weekly_data["Week"] = weekly_data["datum"].dt.isocalendar().week.astype(int)
    weekly_data["Year"] = weekly_data["datum"].dt.year
    pivot = weekly_data.pivot_table(index="Year", columns="Week", values=weekly_drug, aggfunc="mean")

    fig3 = px.imshow(
        pivot,
        color_continuous_scale="YlOrRd",
        labels=dict(color="Units", x="Week of Year", y="Year"),
        aspect="auto",
        height=320,
        title=f"Weekly N02BE Sales Heatmap (Paracetamol)",
    )
    fig3.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 4 – ANOMALY DETECTION
# ══════════════════════════════════════════════
elif page == "🔍 Anomaly Detection":
    st.title("🔍 Anomaly Detection")
    st.markdown("Using **Isolation Forest** (ML-based) and **Z-Score** (statistical) methods to identify unusual demand events.")

    # Isolation Forest on monthly data
    scaler = StandardScaler()
    X = scaler.fit_transform(monthly[DRUGS])
    iso = IsolationForest(contamination=0.05, random_state=42)
    monthly_ad = monthly.copy()
    monthly_ad["Anomaly"] = iso.fit_predict(X)
    monthly_ad["Anomaly_Label"] = monthly_ad["Anomaly"].map({1: "Normal", -1: "🚨 Anomaly"})
    anomalies = monthly_ad[monthly_ad["Anomaly"] == -1]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Months Analyzed", len(monthly_ad))
    with c2:
        st.metric("Anomalies Detected", len(anomalies), help="Isolation Forest (5% contamination)")
    with c3:
        st.metric("Anomaly Rate", f"{len(anomalies)/len(monthly_ad)*100:.1f}%")

    st.divider()

    drug_for_anomaly = st.selectbox("Select drug for anomaly visualization", selected_drugs, index=3)

    fig = go.Figure()
    normal_m = monthly_ad[monthly_ad["Anomaly"] == 1]
    anom_m   = monthly_ad[monthly_ad["Anomaly"] == -1]

    fig.add_trace(go.Scatter(
        x=monthly_ad["datum"], y=monthly_ad[drug_for_anomaly],
        mode="lines", name="Sales", line=dict(color="steelblue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=anom_m["datum"], y=anom_m[drug_for_anomaly],
        mode="markers", name="Anomaly",
        marker=dict(color="red", size=12, symbol="x"),
    ))
    fig.update_layout(
        title=f"{drug_for_anomaly} – Anomaly Detection (Isolation Forest)",
        xaxis_title="Date", yaxis_title="Units",
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomalous Months")
    show_cols = ["datum", "Anomaly_Label"] + DRUGS
    st.dataframe(
        anomalies[show_cols].rename(columns={"datum": "Date"}).round(2).reset_index(drop=True),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Daily Z-Score Anomaly Detection")
    daily_z = daily.copy()
    daily_z["Total"] = daily_z[DRUGS].sum(axis=1)
    mean_d, std_d = daily_z["Total"].mean(), daily_z["Total"].std()
    daily_z["Z_Score"] = (daily_z["Total"] - mean_d) / std_d
    daily_z["Z_Anomaly"] = daily_z["Z_Score"].abs() > 3

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=daily_z["datum"], y=daily_z["Total"],
        mode="lines", name="Daily Total", line=dict(color="steelblue", width=1),
    ))
    fig2.add_trace(go.Scatter(
        x=daily_z[daily_z["Z_Anomaly"]]["datum"],
        y=daily_z[daily_z["Z_Anomaly"]]["Total"],
        mode="markers", name="Z-Score Anomaly (>3σ)",
        marker=dict(color="red", size=10, symbol="circle"),
    ))
    fig2.update_layout(
        title="Daily Total Sales – Z-Score Anomalies (|Z| > 3)",
        xaxis_title="Date", yaxis_title="Total Daily Units",
        height=380, hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Z-Score anomalies detected: {daily_z['Z_Anomaly'].sum()} days out of {len(daily_z)}")


# ══════════════════════════════════════════════
# PAGE 5 – DEMAND FORECASTING
# ══════════════════════════════════════════════
elif page == "🔮 Demand Forecasting":
    st.title("🔮 Demand Forecasting")
    st.markdown("Linear trend model with train/test split for demand projection.")

    drug_fc = st.selectbox("Select drug to forecast", selected_drugs, index=3)

    df_fc = monthly[["datum", drug_fc]].copy().reset_index(drop=True)
    df_fc["t"] = np.arange(len(df_fc))

    test_months = st.slider("Holdout months (test set)", 3, 12, 6)
    train_df = df_fc.iloc[:-test_months]
    test_df  = df_fc.iloc[-test_months:]

    model = LinearRegression()
    model.fit(train_df[["t"]], train_df[drug_fc])
    pred_train = model.predict(train_df[["t"]])
    pred_test  = model.predict(test_df[["t"]])

    mae = mean_absolute_error(test_df[drug_fc], pred_test)
    slope = model.coef_[0]
    trend_dir = "📈 Upward" if slope > 0 else "📉 Downward"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Trend Direction", trend_dir)
    with c2:
        st.metric("Monthly Slope", f"{slope:+.2f} units/month")
    with c3:
        st.metric("MAE (Test)", f"{mae:.2f} units")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_fc["datum"], y=df_fc[drug_fc],
        mode="lines", name="Actual", line=dict(color="steelblue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=train_df["datum"], y=pred_train,
        mode="lines", name="Trend (train)", line=dict(color="green", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=test_df["datum"], y=pred_test,
        mode="lines+markers", name="Forecast (test)", line=dict(color="orange", width=2),
        marker=dict(size=8),
    ))
    fig.update_layout(
        title=f"{drug_fc} – Linear Demand Forecast",
        xaxis_title="Date", yaxis_title="Units",
        height=420, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 6-month future projection
    st.subheader("6-Month Future Projection")
    last_t = df_fc["t"].max()
    future_t = np.arange(last_t + 1, last_t + 7).reshape(-1, 1)
    future_preds = model.predict(future_t)
    last_date = df_fc["datum"].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq="MS")

    future_df = pd.DataFrame({"Date": future_dates, "Forecast": future_preds.round(2)})
    col1, col2 = st.columns([2, 1])
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_fc["datum"], y=df_fc[drug_fc],
            mode="lines", name="Historical", line=dict(color="steelblue"),
        ))
        fig2.add_trace(go.Scatter(
            x=future_df["Date"], y=future_df["Forecast"],
            mode="lines+markers", name="6M Projection", line=dict(color="red", dash="dot"),
            marker=dict(size=9),
        ))
        fig2.update_layout(height=320, hovermode="x unified", margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.dataframe(future_df.set_index("Date").style.format("{:.2f}"), use_container_width=True)

    # 3-month rolling MA
    st.subheader("3-Month Rolling Moving Average")
    ma_drug = st.selectbox("Drug for MA", selected_drugs, index=0, key="ma_drug")
    ma_df = monthly[["datum", ma_drug]].copy()
    ma_df["MA3"] = ma_df[ma_drug].rolling(3).mean()
    ma_df["MA6"] = ma_df[ma_drug].rolling(6).mean()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ma_df["datum"], y=ma_df[ma_drug], mode="lines",
                              name="Actual", line=dict(color="lightblue", width=1)))
    fig3.add_trace(go.Scatter(x=ma_df["datum"], y=ma_df["MA3"], mode="lines",
                              name="3M MA", line=dict(color="orange", width=2)))
    fig3.add_trace(go.Scatter(x=ma_df["datum"], y=ma_df["MA6"], mode="lines",
                              name="6M MA", line=dict(color="red", width=2, dash="dot")))
    fig3.update_layout(height=340, hovermode="x unified",
                       xaxis_title="Date", yaxis_title="Units")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 6 – RISK ANALYSIS
# ══════════════════════════════════════════════
elif page == "⚠️ Risk Analysis":
    st.title("⚠️ Risk Analysis")
    st.markdown("Supply chain risk via **Coefficient of Variation (CV%)**, trend reversals, and demand volatility.")

    # CV Analysis
    cv = (monthly_f[selected_drugs].std() / monthly_f[selected_drugs].mean() * 100).round(2)
    cv_df = cv.sort_values(ascending=False).reset_index()
    cv_df.columns = ["Drug", "CV (%)"]
    cv_df["Risk Level"] = cv_df["CV (%)"].apply(
        lambda x: "🔴 High" if x > 30 else ("🟠 Medium" if x > 20 else "🟢 Low")
    )

    c1, c2, c3 = st.columns(3)
    high_risk = cv_df[cv_df["CV (%)"] > 30]
    with c1:
        st.metric("High Risk Drugs (CV>30%)", len(high_risk))
    with c2:
        st.metric("Most Volatile Drug", cv_df.iloc[0]["Drug"])
    with c3:
        st.metric("Most Stable Drug", cv_df.iloc[-1]["Drug"])

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Supply Chain Risk (CV %)")
        fig = px.bar(
            cv_df, x="CV (%)", y="Drug", orientation="h",
            color="Risk Level",
            color_discrete_map={"🔴 High": "red", "🟠 Medium": "orange", "🟢 Low": "green"},
            height=380,
        )
        fig.add_vline(x=20, line_dash="dash", line_color="orange", annotation_text="Medium risk")
        fig.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="High risk")
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Summary Table")
        st.dataframe(cv_df.set_index("Drug"), use_container_width=True)

    st.divider()
    st.subheader("Month-over-Month Volatility")
    mom_drug = st.selectbox("Drug for MoM volatility", selected_drugs, key="mom")
    mom = monthly_f[["datum", mom_drug]].copy()
    mom["MoM_Change"] = mom[mom_drug].pct_change() * 100

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=mom["datum"], y=mom["MoM_Change"],
        marker_color=["red" if v < 0 else "green" for v in mom["MoM_Change"].fillna(0)],
        name="MoM Change %",
    ))
    fig2.update_layout(
        title=f"{mom_drug} – Month-over-Month Change (%)",
        xaxis_title="Date", yaxis_title="% Change",
        height=350, hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Correlation Risk Matrix")
    st.caption("Highly correlated drugs face concurrent demand spikes – compounding supply risk.")
    corr = monthly_f[selected_drugs].corr()
    fig3 = px.imshow(
        corr.round(2),
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 7 – OPPORTUNITIES & ACTIONS
# ══════════════════════════════════════════════
elif page == "💡 Opportunities & Actions":
    st.title("💡 Opportunities & Strategic Actions")

    # Growth analysis
    avg_start = monthly[monthly["Year"] == 2014][selected_drugs].mean()
    avg_end   = monthly[monthly["Year"] == 2018][selected_drugs].mean()
    growth    = ((avg_end - avg_start) / avg_start * 100).round(2)

    opp_df = pd.DataFrame({
        "Drug": selected_drugs,
        "Avg 2014": [avg_start[d].round(2) for d in selected_drugs],
        "Avg 2018": [avg_end[d].round(2) for d in selected_drugs],
        "Growth %": [growth[d] for d in selected_drugs],
    }).sort_values("Growth %", ascending=False).set_index("Drug")

    c1, c2, c3 = st.columns(3)
    with c1:
        growers = opp_df[opp_df["Growth %"] > 0]
        st.metric("Growing Categories", len(growers))
    with c2:
        st.metric("Top Growth Drug", opp_df.index[0], f"{opp_df['Growth %'].iloc[0]:+.1f}%")
    with c3:
        decliners = opp_df[opp_df["Growth %"] < 0]
        st.metric("Declining Categories", len(decliners))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Growth: 2014 vs 2018 (%)")
        fig = px.bar(
            opp_df.reset_index(), x="Growth %", y="Drug", orientation="h",
            color="Growth %",
            color_continuous_scale="RdYlGn",
            height=380,
        )
        fig.add_vline(x=0, line_color="black", line_width=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("2014 vs 2018 Average Monthly Units")
        compare = opp_df.reset_index()[["Drug","Avg 2014","Avg 2018"]].melt(
            id_vars="Drug", var_name="Period", value_name="Units"
        )
        fig2 = px.bar(
            compare, x="Drug", y="Units", color="Period",
            barmode="group",
            color_discrete_map={"Avg 2014": "steelblue", "Avg 2018": "orange"},
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("📋 Strategic Action Plan")

    action_data = [
        ("🟢 Opportunity", "N02BE (Paracetamol)", "Highest volume drug with October demand spike (cold/flu season). Pre-stock 20–30% above monthly average by Sep.", "High"),
        ("🟢 Opportunity", "R06 (Antihistamines)", "Strong spring/summer peak (allergy season). Launch Q1 pre-season promotions and bulk procurement in Feb.", "High"),
        ("🟢 Opportunity", "M01AB / M01AE", "Steady upward trend 2014→2018. Expand shelf space and supplier contracts for anti-inflammatory category.", "High"),
        ("🟠 Monitor", "N05B (Anxiolytics)", "Consistent demand with moderate growth. Monitor for regulatory shifts. Maintain 2-month safety stock.", "Medium"),
        ("🟠 Monitor", "R03 (Airway Drugs)", "Bimodal seasonality (spring + autumn). Align replenishment cycles with influenza season forecasts.", "Medium"),
        ("🔴 Risk Action", "N05C (Hypnotics)", "Lowest volume, most erratic (high CV%). Reduce overstock risk. Move to JIT (just-in-time) ordering.", "Low"),
        ("🔴 Risk Action", "N02BA (Aspirin)", "Gradual decline observed 2016–2019. Investigate substitution by paracetamol. Adjust stock downward.", "Low"),
        ("🔴 Alert", "Jan 2017 Data Gap", "Complete zero sales recorded. Investigate data quality or actual pharmacy closure. Exclude from trend models.", "Critical"),
    ]
    action_df = pd.DataFrame(action_data, columns=["Type", "Target", "Action", "Priority"])
    st.dataframe(action_df, use_container_width=True, height=340)

    st.divider()
    st.subheader("🤖 AI Demand Clustering (K-Means by Seasonal Profile)")
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans

    seasonal_profile = monthly.groupby("Month")[DRUGS].mean().T
    scaler_km = MinMaxScaler()
    profile_scaled = scaler_km.fit_transform(seasonal_profile)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(profile_scaled)

    cluster_map = {0: "Cluster A – High Volume / Seasonal Spike",
                   1: "Cluster B – Steady / Moderate Volume",
                   2: "Cluster C – Low Volume / Erratic"}
    cluster_df = pd.DataFrame({
        "Drug": DRUGS,
        "Cluster": [cluster_map.get(c, f"Cluster {c}") for c in clusters],
        "Total Sales": monthly[DRUGS].sum().values.round(0),
        "CV%": (monthly[DRUGS].std() / monthly[DRUGS].mean() * 100).round(1).values,
    }).set_index("Drug")

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(cluster_df, use_container_width=True)
    with col2:
        fig3 = px.scatter(
            cluster_df.reset_index(),
            x="Total Sales", y="CV%",
            color="Cluster", text="Drug",
            size="Total Sales",
            height=350,
            color_discrete_sequence=COLORS,
        )
        fig3.update_traces(textposition="top center")
        fig3.update_layout(showlegend=True, margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 8 – RAW DATA EXPLORER
# ══════════════════════════════════════════════
elif page == "📋 Raw Data Explorer":
    st.title("📋 Raw Data Explorer")

    dataset = st.selectbox("Select Dataset", ["Monthly", "Weekly", "Daily", "Hourly"])
    ds_map = {"Monthly": monthly, "Weekly": weekly, "Daily": daily, "Hourly": hourly}
    df_show = ds_map[dataset].copy()

    if "Year" in df_show.columns:
        df_show = df_show[
            (df_show["Year"] >= year_range[0]) & (df_show["Year"] <= year_range[1])
        ]

    st.caption(f"Rows: {len(df_show):,}  |  Columns: {df_show.shape[1]}")
    st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=420)

    st.divider()
    st.subheader("Dataset Statistics")
    st.dataframe(df_show[DRUGS].describe().round(3), use_container_width=True)

    st.divider()
    st.subheader("Missing Values")
    nulls = df_show.isnull().sum()
    st.dataframe(nulls[nulls > 0].rename("Missing Count").to_frame()
                 if nulls.sum() > 0 else pd.DataFrame({"Status": ["No missing values"]}),
                 use_container_width=True)
