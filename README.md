# 💊 PharmaGuard AI – Pharmaceutical Demand Intelligence

> **AI-powered pharmaceutical demand analytics platform** built with Python, Streamlit, and scikit-learn.  
> Analyzes real pharmacy sales data (2014–2019) across 8 WHO drug categories to deliver actionable KPIs, trend intelligence, anomaly alerts, demand forecasting, and strategic recommendations.

---

## 📌 Project Overview

| Field | Details |
|---|---|
| **Project Name** | PharmaGuard AI – Pharmaceutical Demand Intelligence |
| **Domain** | Healthcare / Pharmaceutical Analytics |
| **Data Period** | January 2014 – October 2019 |
| **Drug Categories** | 8 (WHO ATC classification) |
| **Analysis Framework** | KPIs → Trends → Drivers → Risk → Opportunity → Action |

---

## 🧪 Drug Categories (WHO ATC Codes)

| Code | Category | Example Drug |
|---|---|---|
| **M01AB** | Anti-inflammatory / antirheumatic | Diclofenac |
| **M01AE** | Propionic acid derivatives | Ibuprofen |
| **N02BA** | Salicylic acid derivatives | Aspirin |
| **N02BE** | Anilides | Paracetamol *(49.4% market share – dominant)* |
| **N05B** | Anxiolytics | Diazepam |
| **N05C** | Hypnotics and sedatives | Zolpidem |
| **R03** | Drugs for obstructive airway diseases | Salbutamol |
| **R06** | Antihistamines | Loratadine |

---

## 📁 Project Structure

```
IBMPROJECT/
├── app.py                          # Streamlit dashboard (8-page interactive app)
├── run_analysis.py                 # Headless execution script → outputs/
├── PharmaGuard_AI_Analysis.ipynb   # Jupyter EDA notebook (executed, with embedded outputs)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── PharmaGuard_AI_Report.docx      # Detailed project report
├── IBMDataset/
│   ├── salesdaily.csv              # 2,080 daily records
│   ├── saleshourly.csv             # 26,588 hourly records
│   ├── salesweekly.csv             # 302 weekly records
│   └── salesmonthly.csv            # 70 monthly records
└── outputs/                        # All generated charts and result CSVs
    ├── 01_market_share.png
    ├── 02_monthly_trends.png
    ├── 03_annual_sales.png
    ├── 04_seasonality.png
    ├── 05_dow_pattern.png
    ├── 06_hourly_pattern.png
    ├── 07_correlation.png
    ├── 08_anomaly_n02be.png
    ├── 09_zscore_anomalies.png
    ├── 10_forecast_all_drugs.png
    ├── 11_moving_average_n02be.png
    ├── 12_risk_cv.png
    ├── 13_yoy_growth.png
    ├── 14_growth_opportunity.png
    ├── 15_drug_clusters.png
    ├── 16_seasonal_by_cluster.png
    ├── 17_weekly_heatmap_n02be.png
    ├── kpi_summary.csv
    ├── forecast_metrics.csv
    ├── risk_cv.csv
    ├── anomalies_monthly.csv
    └── key_findings.txt
```

---

## 📊 Dataset Summary

Dataset link:https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data/data
| Dataset | Records | Granularity | Date Range |
|---|---|---|---|
| `salesdaily.csv` | 2,080 rows | Per day | Jan 2014 – Oct 2019 |
| `saleshourly.csv` | 26,588 rows | Per hour | Jan 2014 – Oct 2019 |
| `salesweekly.csv` | 302 rows | Per week (Sunday) | Jan 2014 – Oct 2019 |
| `salesmonthly.csv` | 70 rows | Per month (EOM) | Jan 2014 – Oct 2019 |

> **Note:** The all-zero January 2017 row in `salesmonthly.csv` is a confirmed data gap and is excluded from all analyses.

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full analysis (generates all charts & CSVs)

```bash
python run_analysis.py
```

### 3. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 4. Open the Jupyter notebook (pre-executed with outputs)

```bash
jupyter notebook PharmaGuard_AI_Analysis.ipynb
```

---



## 💡 Key Verified Findings (from executed analysis)

### KPIs
| Drug | Total Sales | Monthly Avg | CV% | Market Share |
|---|---|---|---|---|
| **N02BE** | 62,478 | 892.5 | 38.0% | **49.4%** |
| **N05B** | 18,348 | 262.1 | 32.5% | 14.5% |
| **R03** | 11,737 | 167.7 | 48.8% | 9.3% |
| **M01AB** | 10,499 | 150.0 | 21.0% | 8.3% |
| **M01AE** | 8,156 | 116.5 | 23.9% | 6.4% |
| **N02BA** | 8,051 | 115.0 | 27.2% | 6.4% |
| **R06** | 6,066 | 86.7 | 52.9% | 4.8% |
| **N05C** | 1,249 | 17.8 | 47.5% | 1.0% |

### Detected Anomalies (Isolation Forest)
| Month | N02BE | Notes |
|---|---|---|
| **Oct 2014** | 1,856.8 | ~2× avg – largest spike in dataset |
| **Jan 2017** | 0.0 | Data gap – exclude from models |
| **Jan 2019** | 1,660.6 | Seasonal surge + M01AE high |
| **Oct 2019** | 295.2 | Partial month – incomplete data |

### Forecast Metrics (Linear Trend, 6-month holdout)
| Drug | Slope (units/mo) | R² | MAE | Trend |
|---|---|---|---|---|
| R03 | +2.41 | 0.304 | 124.5 | ↑ Upward |
| M01AB | +0.30 | 0.035 | 29.3 | ↑ Upward |
| R06 | +0.49 | 0.039 | 32.7 | ↑ Upward |
| N02BA | -1.04 | 0.426 | 21.0 | ↓ Downward |
| N05B | -1.38 | 0.090 | 43.7 | ↓ Downward |

### Growth: 2014 vs 2018
| Drug | Growth % |
|---|---|
| **R03** | **+116.5%** |
| R06 | +44.5% |
| M01AB | +18.3% |
| N02BA | -33.2% |
| N05B | -25.0% |

---

## 🗂 Dashboard Pages

| Page | Content |
|---|---|
| 🏠 **Overview & KPIs** | Total sales, market share, KPI table with CV% |
| 📈 **Trends & Seasonality** | Monthly/annual trends, YoY heatmap, peak months |
| ⏱ **Hourly & Weekly Patterns** | Intra-day demand, day-of-week, weekly heatmap |
| 🔍 **Anomaly Detection** | Isolation Forest + Z-Score visualization |
| 🔮 **Demand Forecasting** | Linear trend model, MAE metrics, 6-month projection |
| ⚠️ **Risk Analysis** | CV% supply risk, MoM volatility, correlation matrix |
| 💡 **Opportunities & Actions** | Growth analysis, action table, K-Means clustering |
| 📋 **Raw Data Explorer** | Interactive dataset browser with statistics |

---

## 🔬 Analysis Methods

| Technique | Purpose |
|---|---|
| Descriptive statistics | KPI computation and baseline |
| Time-series trend analysis | Temporal demand pattern extraction |
| **Isolation Forest** (scikit-learn) | Unsupervised monthly anomaly detection |
| **Z-Score (3σ rule)** | Statistical daily anomaly detection |
| **Linear Regression** | Demand trend forecasting with train/test split |
| **Rolling Moving Average (3M/6M)** | Smoothed demand projections |
| **K-Means Clustering** | Drug grouping by seasonal demand profile |
| Coefficient of Variation | Supply chain risk quantification |
| Pearson Correlation | Drug demand co-movement analysis |

---

## ⚙️ Requirements

```
Python >= 3.9
streamlit, pandas, numpy, matplotlib, seaborn, plotly
scikit-learn, statsmodels, openpyxl, python-docx
```
Full versions in `requirements.txt`.

---

## 📄 License & Attribution

- **Data**: Pharmaceutical sales dataset from a real Slovenian pharmacy (anonymized, publicly available)
- **Built with**: IBM Bob AI · Python · Streamlit · Plotly · scikit-learn · Matplotlib
