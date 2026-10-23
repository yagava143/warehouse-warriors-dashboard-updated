# Warehouse Warriors | Enterprise Commerce Intelligence Suite

A comprehensive Data Warehouse & Data Mining analytical platform built with **Streamlit**, featuring dimensional modeling, multi-dimensional OLAP analysis, customer/seller segmentation, predictive modeling, and market basket mining.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
Open a terminal in the `WarehouseWarriors` directory and install the requirements:

```bash
pip install -r dashboard/requirements.txt
```

### 2. Launch Dashboard
Run Streamlit from the `WarehouseWarriors` directory:

```bash
python -m streamlit run dashboard/app.py
```

Or from the repository root:

```bash
python -m streamlit run WarehouseWarriors/dashboard/app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🌙 High-Contrast Dark Theme Design System

The dashboard interface uses a synchronized **High-Contrast Dark Executive Theme** designed for comfortable readability and optimal cursor contrast:

- **Enhanced Pointer & Mouse Contrast:** Deep obsidian canvas (`#0B0F19`) and charcoal slate cards (`#111827`) ensure that white mouse pointers stand out with crystal-clear visibility anywhere on the screen.
- **Executive Colorway:** Glowing Electric Cobalt (`#3B82F6`), Cyan (`#06B6D4`), and Emerald (`#10B981`) accents against dark card backgrounds.
- **Synchronized Plotly Charts:** Built on `plotly_dark` with custom dark slate hover cards (`#030712`), subtle grid lines, and vivid colorways.
- **Dark Mode Star Schema Diagram:** Custom styled Graphviz architecture with dark slate nodes and cobalt central fact highlight.
- **Executive KPI Ribbon:** Modern top-level metrics featuring real-time data integrity indicators, total GMV volume, seller counts, and geographic coverage.
- **Synchronized Navigation & Cards:** Segmented pill tabs, bordered container cards (`st.container(border=True)`), and an informative workspace sidebar.

---

## 🔬 Interactive Analytical Modules

1. **🏛️ Data Warehouse:**
   - Interactive relational Star Schema architecture diagram (`Dim_Time`, `Dim_Store`, `Dim_Product`, `Dim_Customer`, `Fact_Store_Sales`).
   - Table catalogue with exact row counts across the warehouse.
   - Interactive table data inspector with customizable row limits and CSV export.

2. **📈 OLAP Operations:**
   - Multi-dimensional query workbench: **Roll-up**, **Drill-down**, **Slice**, **Dice**, and **Pivot**.
   - Reactive parameter controls and formatted SQL query inspection.
   - Top-15 dimensional revenue distribution chart and downloadable query outputs.

3. **🧩 Clustering Lab (Unsupervised Mining):**
   - Interactive feature selection and cluster count tuning ($K = 2$ to $10$).
   - Side-by-side **Elbow Method (WCSS)** and **Silhouette Score** diagnostic curves with chosen $K$ reference markers.
   - Comparative evaluation across **K-Means**, **Agglomerative Clustering**, and **DBSCAN**.
   - Cluster size distributions, feature profile averages, and downloadable assignments.

4. **🎯 Classification Lab (Supervised Mining):**
   - Multi-model benchmarking: **Decision Tree**, **Random Forest**, **Logistic Regression**, and **KNN**.
   - Hyperparameter sliders: Test split proportion and tree depth.
   - Comparative grouped metric chart (Accuracy, Precision, Recall, F1-Score).
   - Interactive Confusion Matrix heatmap with high-contrast dark blue palette.

5. **🛒 Association Rules (Market Basket Mining):**
   - Data source selector: Built-in Demo Retail Baskets, Olist Brazilian E-Commerce, or custom transaction CSV/Excel upload.
   - Customizable support, confidence, and lift thresholds with FP-Growth mining.
   - Basket size distribution histogram and interactive Support vs. Confidence bubble scatter plot.

6. **💹 Revenue Prediction (Regression):**
   - Evaluates **Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**, and **Extra Trees Regressor**.
   - Error metrics comparison (MAE, RMSE, $R^2$).
   - Interactive Actual vs. Predicted scatter plot with dashed $45^\circ$ reference line ($y = x$) and error sample tables.

7. **📚 Data & Methodology:**
   - Documented ETL pipeline, star schema dimensional grain, model formulations, and methodological limitations.
   - Full dataset preview and one-click raw CSV download.

---

## 📦 Requirements

- `streamlit >= 1.40`
- `pandas >= 2.0`
- `numpy >= 1.24`
- `scikit-learn >= 1.3`
- `plotly >= 5.18`
- `mlxtend >= 0.23`
- `openpyxl >= 3.1`
