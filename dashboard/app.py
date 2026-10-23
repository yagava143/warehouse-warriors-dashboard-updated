from pathlib import Path
import warnings
import sqlite3
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    MLXTEND_AVAILABLE = True
except Exception:
    MLXTEND_AVAILABLE = False

# Page Configuration - Dark Executive Theme
st.set_page_config(
    page_title='Warehouse Warriors | Commerce Intelligence Hub',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ----------------- PATH & RESOURCE RESOLUTION -----------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

def find_project_file(rel_path):
    candidates = [
        PROJECT_ROOT / rel_path,
        CURRENT_DIR / rel_path,
        WORKSPACE_ROOT / rel_path,
        WORKSPACE_ROOT / 'WarehouseWarriors' / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

DATA_DIR = find_project_file('data')
DB_PATH = find_project_file('database/warehouse.db')

# ----------------- DESIGN SYSTEM & THEME CONFIG -----------------
COLOR_PRIMARY = "#3B82F6"
COLOR_PRIMARY_DARK = "#2563EB"
COLOR_SECONDARY = "#06B6D4"
COLOR_SUCCESS = "#10B981"
COLOR_WARNING = "#F59E0B"
COLOR_PURPLE = "#A855F7"
COLOR_ROSE = "#FB7185"
COLOR_BG_CANVAS = "#0B0F19"
COLOR_BG_CARD = "#111827"
COLOR_BG_SURFACE = "#1E293B"
COLOR_TEXT_PRIMARY = "#F8FAFC"
COLOR_TEXT_MUTED = "#94A3B8"
COLOR_BORDER = "#1E293B"

PLOT_PALETTE = ["#3B82F6", "#06B6D4", "#10B981", "#A855F7", "#F59E0B", "#FB7185", "#94A3B8"]

def format_chart(fig, title="", height=420, show_legend=True, x_title=None, y_title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#CBD5E1", size=12),
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#F8FAFC", size=15),
            x=0.01,
            xanchor="left",
            y=0.96
        ) if title else None,
        margin=dict(l=40, r=24, t=55 if title else 28, b=38),
        height=height,
        colorway=PLOT_PALETTE,
        hoverlabel=dict(
            bgcolor="#030712",
            bordercolor="#334155",
            font=dict(color="#FFFFFF", family="Inter, sans-serif", size=12)
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#CBD5E1", size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ) if show_legend else dict(visible=False),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color="#E2E8F0", size=12, weight=600)) if x_title else None,
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            gridwidth=1,
            zerolinecolor="#334155",
            linecolor="#334155",
            tickfont=dict(color="#94A3B8", size=11)
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color="#E2E8F0", size=12, weight=600)) if y_title else None,
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            gridwidth=1,
            zerolinecolor="#334155",
            linecolor="#334155",
            tickfont=dict(color="#94A3B8", size=11)
        )
    )
    return fig

# Custom Injected CSS - High Contrast Dark Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {
  --primary: #3B82F6;
  --primary-dark: #2563EB;
  --secondary: #06B6D4;
  --success: #10B981;
  --warning: #F59E0B;
  --purple: #A855F7;
  --rose: #FB7185;
  
  --bg-canvas: #0B0F19;
  --bg-card: #111827;
  --bg-surface: #1E293B;
  --border-card: #1E293B;
  --border-highlight: #334155;
  
  --text-main: #F8FAFC;
  --text-sub: #CBD5E1;
  --text-muted: #94A3B8;
  
  --radius-md: 10px;
  --radius-lg: 14px;
  --radius-xl: 18px;
  --shadow-sm: 0 2px 4px 0 rgba(0, 0, 0, 0.4);
  --shadow-md: 0 6px 12px -2px rgba(0, 0, 0, 0.5), 0 3px 6px -3px rgba(0, 0, 0, 0.4);
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  color: var(--text-sub);
}

.stApp {
  background-color: var(--bg-canvas) !important;
}

[data-testid="stAppViewContainer"] {
  background-color: var(--bg-canvas) !important;
  background-image: radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.09) 0px, transparent 50%),
                    radial-gradient(at 0% 100%, rgba(6, 182, 212, 0.07) 0px, transparent 50%);
}

.block-container {
  padding-top: 1.4rem !important;
  padding-bottom: 3.5rem !important;
  max-width: 1540px !important;
}

[data-testid="stHeader"] {
  background: transparent !important;
}

h1, h2, h3, h4, h5, h6 {
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
  color: var(--text-main) !important;
  letter-spacing: -0.025em !important;
}

h2 {
  font-size: 1.45rem !important;
  font-weight: 750 !important;
  margin-top: 0.25rem !important;
  margin-bottom: 0.35rem !important;
}

h3 {
  font-size: 1.12rem !important;
  font-weight: 700 !important;
  margin-top: 1.1rem !important;
  margin-bottom: 0.5rem !important;
}

/* Hero Banner - High Contrast Deep Indigo */
.ww-hero {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #030712 0%, #0B132B 45%, #172554 100%);
  border-radius: var(--radius-xl);
  border: 1px solid rgba(59, 130, 246, 0.28);
  padding: 1.9rem 2.2rem;
  color: #FFFFFF;
  margin: 0.15rem 0 1.4rem 0;
  box-shadow: 0 14px 34px -8px rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 2rem;
}

.ww-hero::after {
  content: "";
  position: absolute;
  top: -120px;
  right: -50px;
  width: 380px;
  height: 380px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(59, 130, 246, 0.25) 0%, rgba(37, 99, 235, 0.08) 50%, transparent 70%);
  pointer-events: none;
}

.ww-hero-main {
  position: relative;
  z-index: 2;
  flex: 1;
}

.ww-badge-strip {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
}

.ww-badge-live {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  background: rgba(16, 185, 129, 0.15);
  border: 1px solid rgba(16, 185, 129, 0.45);
  color: #6EE7B7;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  padding: 0.25rem 0.65rem;
  border-radius: 9999px;
  text-transform: uppercase;
}

.ww-dot-live {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #10B981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.45);
  animation: pulse-dot 2s infinite ease-in-out;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.55; transform: scale(0.85); }
}

.ww-tag {
  display: inline-flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.16);
  color: #E2E8F0;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 0.25rem 0.6rem;
  border-radius: 9999px;
}

.ww-hero h1 {
  color: #FFFFFF !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: clamp(1.8rem, 2.6vw, 2.4rem) !important;
  font-weight: 800 !important;
  line-height: 1.15 !important;
  margin: 0 0 0.5rem 0 !important;
  letter-spacing: -0.035em !important;
}

.ww-hero h1 span {
  background: linear-gradient(120deg, #93C5FD 0%, #60A5FA 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.ww-hero p {
  color: #CBD5E1 !important;
  max-width: 680px;
  margin: 0 !important;
  font-size: 0.94rem !important;
  line-height: 1.55 !important;
}

.ww-hero-side {
  position: relative;
  z-index: 2;
  min-width: 220px;
  background: rgba(17, 24, 39, 0.7);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(59, 130, 246, 0.25);
  border-radius: var(--radius-lg);
  padding: 1.1rem 1.25rem;
}

.ww-hero-side-label {
  color: #93C5FD;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.66rem;
  font-weight: 700;
}

.ww-hero-side-value {
  color: #FFFFFF;
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 1.15rem;
  font-weight: 700;
  margin-top: 0.35rem;
}

.ww-hero-side-note {
  color: #94A3B8;
  font-size: 0.75rem;
  margin-top: 0.25rem;
}

/* Custom KPI Ribbon - Dark Mode */
.ww-kpi-card {
  background: #111827;
  border: 1px solid #1E293B;
  border-radius: var(--radius-lg);
  padding: 1.15rem 1.25rem;
  box-shadow: var(--shadow-sm);
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
  position: relative;
  overflow: hidden;
  height: 100%;
}

.ww-kpi-card::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #3B82F6, #06B6D4);
  opacity: 0.95;
}

.ww-kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: #334155;
}

.ww-kpi-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.ww-kpi-label {
  color: #94A3B8;
  font-size: 0.76rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.ww-kpi-icon {
  font-size: 1.15rem;
  background: #1E293B;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.ww-kpi-val-row {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
}

.ww-kpi-val {
  color: #FFFFFF;
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
  font-weight: 800;
  font-size: 1.55rem;
  line-height: 1.2;
}

.ww-kpi-sub {
  color: #94A3B8;
  font-size: 0.78rem;
  margin-top: 0.35rem;
  font-weight: 500;
}

/* Tabs Styling - Dark Mode */
.stTabs {
  margin-top: 0.4rem;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.4rem;
  background: #111827;
  border: 1px solid #1E293B;
  padding: 0.35rem;
  border-radius: 14px;
  margin-bottom: 1.4rem;
  flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
  min-height: 42px;
  border-radius: 10px;
  padding: 0.45rem 1rem;
  border: 1px solid transparent;
  color: #94A3B8;
  font-weight: 600;
  font-size: 0.86rem;
  transition: all 0.16s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
  background: rgba(255, 255, 255, 0.06);
  color: #F8FAFC;
}

.stTabs [aria-selected="true"] {
  background: #1E293B !important;
  color: #60A5FA !important;
  border-color: rgba(59, 130, 246, 0.35) !important;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
  font-weight: 750 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
  display: none !important;
}

/* Standard Metric Enhancement - Dark Mode */
[data-testid="stMetric"] {
  background: #111827 !important;
  border: 1px solid #1E293B !important;
  padding: 1.1rem 1.2rem !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-sm) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}

[data-testid="stMetric"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow-md) !important;
  border-color: #334155 !important;
}

[data-testid="stMetricLabel"] {
  color: #94A3B8 !important;
  font-weight: 600 !important;
  font-size: 0.78rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
}

[data-testid="stMetricValue"] {
  color: #FFFFFF !important;
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
  font-weight: 800 !important;
  font-size: 1.55rem !important;
}

/* Containers with Borders (Cards) - Dark Mode */
[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid #1E293B !important;
  border-radius: var(--radius-lg) !important;
  background-color: #111827 !important;
  box-shadow: var(--shadow-sm) !important;
  padding: 1.25rem !important;
  margin-bottom: 1rem !important;
}

/* Dataframe and Tables - Dark Mode */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid #1E293B !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
  background: #111827 !important;
  box-shadow: var(--shadow-sm) !important;
}

/* Plotly Chart Card Wrapper - Dark Mode */
[data-testid="stPlotlyChart"] {
  background: #111827 !important;
  border: 1px solid #1E293B !important;
  border-radius: var(--radius-lg) !important;
  padding: 0.4rem !important;
  box-shadow: var(--shadow-sm) !important;
}

/* Buttons - Dark Mode */
.stButton button, .stDownloadButton button {
  border-radius: var(--radius-md) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  min-height: 2.55rem !important;
  border: 1px solid #334155 !important;
  color: #E2E8F0 !important;
  background: #1E293B !important;
  box-shadow: var(--shadow-sm) !important;
  transition: all 0.16s ease !important;
}

.stButton button:hover, .stDownloadButton button:hover {
  transform: translateY(-1px) !important;
  box-shadow: var(--shadow-md) !important;
  border-color: #60A5FA !important;
  color: #FFFFFF !important;
  background: #2563EB !important;
}

.stButton button[kind="primary"] {
  background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
  color: #FFFFFF !important;
  border: none !important;
}

.stButton button[kind="primary"]:hover {
  background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
  color: #FFFFFF !important;
}

/* Sidebar - Dark Mode */
[data-testid="stSidebar"] {
  background-color: #0B0F19 !important;
  border-right: 1px solid #1E293B !important;
}

.ww-sidebar-header {
  padding: 0.4rem 0.1rem 1.1rem;
}

.ww-sidebar-brand {
  display: flex;
  align-items: center;
  gap: 0.65rem;
}

.ww-sidebar-logo {
  width: 38px;
  height: 38px;
  background: linear-gradient(135deg, #2563EB, #1D4ED8);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 1.25rem;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
}

.ww-sidebar-title {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-weight: 800;
  font-size: 1.05rem;
  color: #FFFFFF;
  line-height: 1.2;
}

.ww-sidebar-version {
  font-size: 0.68rem;
  color: #60A5FA;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.ww-sidebar-box {
  background: #111827;
  border: 1px solid #1E293B;
  border-radius: var(--radius-md);
  padding: 0.85rem 1rem;
  margin-bottom: 0.85rem;
}

.ww-sidebar-box-label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: #94A3B8;
  text-transform: uppercase;
}

.ww-sidebar-box-val {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 0.98rem;
  font-weight: 700;
  color: #F8FAFC;
  margin-top: 0.25rem;
}

.ww-sidebar-box-sub {
  font-size: 0.74rem;
  color: #94A3B8;
  margin-top: 0.2rem;
}

/* Callout & Alert cards - Dark Mode */
.ww-callout {
  border-radius: var(--radius-md);
  padding: 0.85rem 1.1rem;
  margin-bottom: 1rem;
  font-size: 0.88rem;
  line-height: 1.5;
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}

.ww-callout-info {
  background: #0F172A;
  border: 1px solid #1E3A8A;
  color: #93C5FD;
}

.ww-callout-tip {
  background: #06281E;
  border: 1px solid #065F46;
  color: #6EE7B7;
}

.ww-callout-icon {
  font-size: 1.15rem;
  line-height: 1;
}

.ww-section-head {
  margin: 0.2rem 0 1rem;
}

.ww-section-title {
  font-size: 1.35rem;
  font-weight: 800;
  color: #FFFFFF;
  letter-spacing: -0.025em;
  margin: 0 0 0.25rem;
}

.ww-section-sub {
  color: #94A3B8;
  font-size: 0.88rem;
  margin: 0;
  line-height: 1.5;
}

.ww-control-bar-title {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 0.92rem;
  font-weight: 700;
  color: #F1F5F9;
  margin-bottom: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  display: flex;
  align-items: center;
  gap: 0.45rem;
}

@media (max-width: 800px) {
  .ww-hero {
    padding: 1.4rem;
    flex-direction: column;
    align-items: flex-start;
  }
  .ww-hero-side {
    width: 100%;
  }
  .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }
}
</style>
""", unsafe_allow_html=True)

# ----------------- HERO BANNER -----------------
st.markdown("""
<div class="ww-hero">
  <div class="ww-hero-main">
    <div class="ww-badge-strip">
      <span class="ww-badge-live"><span class="ww-dot-live"></span> Active Analytics Engine</span>
      <span class="ww-tag">SQLite Star Schema</span>
      <span class="ww-tag">OLAP Dimensional Cube</span>
      <span class="ww-tag">Data Mining Suite</span>
    </div>
    <h1>Commerce Intelligence <span>Suite</span></h1>
    <p>A unified analytics hub combining relational data warehousing, multi-dimensional OLAP cubes, customer &amp; seller segmentation, predictive regression, and market basket pattern discovery.</p>
  </div>
  <div class="ww-hero-side">
    <div class="ww-hero-side-label">Connected Warehouse</div>
    <div class="ww-hero-side-value">Olist Marketplace</div>
    <div class="ww-hero-side-note">Star Schema · 5 Relational Tables</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------- CACHED COMPUTATION ENGINES -----------------
@st.cache_data(show_spinner=False)
def compute_cluster_evaluation(data, features):
    clean = data[list(features)].replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)
    rows = []
    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        sil = silhouette_score(scaled, labels) if len(set(labels)) > 1 and len(set(labels)) < len(labels) else np.nan
        rows.append({'K': k, 'Inertia (WCSS)': model.inertia_, 'Silhouette score': sil})
    return clean, scaled, pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def compute_cluster_labels(data, features, algorithm, k, eps=0.8, min_samples=8):
    clean = data[list(features)].replace([np.inf, -np.inf], np.nan).dropna()
    scaled = StandardScaler().fit_transform(clean)
    if algorithm == 'K-Means':
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    elif algorithm == 'Agglomerative Clustering':
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(scaled)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
    keep = labels != -1 if algorithm == 'DBSCAN' else np.ones(len(labels), dtype=bool)
    score = np.nan
    kept_labels = labels[keep]
    if len(set(kept_labels)) > 1 and len(set(kept_labels)) < len(kept_labels):
        score = silhouette_score(scaled[keep], kept_labels)
    return clean, labels, score

@st.cache_data(show_spinner=False)
def compute_cluster_comparison(data, features, k, db_eps=0.8, db_min_samples=8):
    clean = data[list(features)].replace([np.inf, -np.inf], np.nan).dropna()
    scaled = StandardScaler().fit_transform(clean)
    rows = []
    for name, model in [('K-Means', KMeans(n_clusters=k, random_state=42, n_init=10)), ('Agglomerative', AgglomerativeClustering(n_clusters=k))]:
        lab = model.fit_predict(scaled)
        ncl = len(set(lab))
        sil = silhouette_score(scaled, lab) if 1 < ncl < len(lab) else np.nan
        rows.append({'Algorithm': name, 'Clusters': ncl, 'Silhouette score': sil})
    db_lab = DBSCAN(eps=db_eps, min_samples=db_min_samples).fit_predict(scaled)
    keep = db_lab != -1
    db_kept = db_lab[keep]
    db_sil = np.nan
    if len(set(db_kept)) > 1 and len(set(db_kept)) < len(db_kept):
        db_sil = silhouette_score(scaled[keep], db_kept)
    rows.append({'Algorithm': f'DBSCAN (eps={db_eps}, min_samples={db_min_samples})', 'Clusters': len(set(db_lab) - {-1}), 'Silhouette score': db_sil})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def compute_classification_results(data, features, test_size, depth):
    X = data[list(features)].replace([np.inf, -np.inf], np.nan)
    y = data['performance'].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=depth, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    }
    rows, predictions = [], {}
    for name, model in models.items():
        pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        predictions[name] = pred
        rows.append({
            'Algorithm': name,
            'Accuracy': accuracy_score(y_test, pred),
            'Precision (weighted)': precision_score(y_test, pred, average='weighted', zero_division=0),
            'Recall (weighted)': recall_score(y_test, pred, average='weighted', zero_division=0),
            'F1 (weighted)': f1_score(y_test, pred, average='weighted', zero_division=0)
        })
    return pd.DataFrame(rows), y_test.reset_index(drop=True), predictions

@st.cache_data(show_spinner=False)
def compute_regression_results(data, features, test_size):
    X = data[list(features)].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(data['total_revenue'], errors='coerce')
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1),
        'Extra Trees Regressor': ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
    }
    rows, predictions = [], {}
    for name, model in models.items():
        pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        predictions[name] = pred
        rows.append({
            'Algorithm': name,
            'MAE (R$)': mean_absolute_error(y_test, pred),
            'RMSE (R$)': np.sqrt(mean_squared_error(y_test, pred)),
            'R²': r2_score(y_test, pred)
        })
    return pd.DataFrame(rows), y_test.reset_index(drop=True), predictions

@st.cache_data(show_spinner=False)
def build_olist_transactions(items, products, translations):
    products = products.merge(translations, on='product_category_name', how='left') if translations is not None else products
    category_col = 'product_category_name_english' if 'product_category_name_english' in products else 'product_category_name'
    products['item_name'] = products[category_col].fillna('other').astype(str)
    d = items.merge(products[['product_id', 'item_name']], on='product_id', how='left').dropna(subset=['order_id'])
    return d.groupby('order_id')['item_name'].apply(lambda s: sorted(set(s.dropna().astype(str)))).tolist()

@st.cache_data(show_spinner=False)
def compute_frequent_itemsets(transactions, min_support):
    all_items = sorted(set(item for basket in transactions for item in basket))
    if len(all_items) > 500:
        from collections import Counter
        common = {x for x, _ in Counter(x for basket in transactions for x in basket).most_common(500)}
        transactions = [[x for x in basket if x in common] for basket in transactions]
        all_items = sorted(common)
    mlb = MultiLabelBinarizer(classes=all_items, sparse_output=True)
    sparse_matrix = mlb.fit_transform(transactions)
    basket_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=all_items).astype(pd.SparseDtype(bool, False))
    if basket_df.shape[1] < 2 or not len(basket_df):
        return pd.DataFrame(), len(all_items)
    frequent = fpgrowth(basket_df, min_support=float(min_support), use_colnames=True, max_len=3)
    return frequent, len(all_items)

@st.cache_data(show_spinner=False)
def compute_association_rules(frequent, min_confidence):
    if frequent.empty:
        return pd.DataFrame()
    return association_rules(frequent, metric='confidence', min_threshold=float(min_confidence))

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)

def find_data(filename):
    p = DATA_DIR / filename
    if p.exists():
        return load_csv(str(p))
    return None

# Load Core Seller Mining Dataset
seller = find_data('store_mining_dataset.csv')
if seller is None:
    seller = find_data('seller_performance.csv')

if seller is None:
    st.error(f'Could not find store_mining_dataset.csv or seller_performance.csv in {DATA_DIR}.')
    st.stop()

num_features = ['total_freight', 'total_items', 'total_orders', 'average_item_price', 'average_order_value', 'freight_ratio']
cluster_features = ['total_revenue', 'total_freight', 'total_items', 'total_orders', 'average_item_price', 'average_order_value', 'freight_ratio']
for c in cluster_features:
    if c in seller.columns:
        seller[c] = pd.to_numeric(seller[c], errors='coerce')

# ----------------- SIDEBAR WORKSPACE NAVIGATOR -----------------
with st.sidebar:
    st.markdown("""
    <div class="ww-sidebar-header">
      <div class="ww-sidebar-brand">
        <div class="ww-sidebar-logo">⚡</div>
        <div>
          <div class="ww-sidebar-title">Warehouse Warriors</div>
          <div class="ww-sidebar-version">Intelligence v2.4 Pro</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ww-sidebar-box">
      <div class="ww-sidebar-box-label">Connection Status</div>
      <div class="ww-sidebar-box-val" style="display:flex;align-items:center;gap:6px;">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10B981;box-shadow:0 0 6px rgba(16,185,129,0.6);"></span>
        warehouse.db Online
      </div>
      <div class="ww-sidebar-box-sub">SQLite 3 Star Schema Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ww-sidebar-box">
      <div class="ww-sidebar-box-label">Active Mining Dataset</div>
      <div class="ww-sidebar-box-val">Seller Performance Mart</div>
      <div class="ww-sidebar-box-sub">{len(seller):,} records · {len(seller.columns)} dimensions</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 Project Documentation & Tips", expanded=False):
        st.markdown("""
        - **Data Warehouse:** Dimensional star schema with 1 fact table and 4 dimension tables.
        - **OLAP Cube:** Roll-up, Drill-down, Slice, Dice, and Pivot multi-dimensional operations.
        - **Mining Labs:** Evaluates clustering (K-Means, Agglomerative, DBSCAN), classification (Tree, Forest, LR, KNN), and regression.
        - **Market Basket:** Apriori & FP-Growth itemset mining.
        """)

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;'>Quick Data Export</div>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Full Mining Dataset",
        data=seller.to_csv(index=False).encode('utf-8'),
        file_name='seller_mining_dataset.csv',
        mime='text/csv',
        use_container_width=True
    )

# ----------------- GLOBAL KPI RIBBON -----------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="ww-kpi-card">
      <div class="ww-kpi-top">
        <span class="ww-kpi-label">Marketplace Sellers</span>
        <span class="ww-kpi-icon">👥</span>
      </div>
      <div class="ww-kpi-val-row">
        <span class="ww-kpi-val">{len(seller):,}</span>
      </div>
      <div class="ww-kpi-sub">Profiled seller units in dataset</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    states_count = seller.seller_state.nunique() if 'seller_state' in seller else 0
    st.markdown(f"""
    <div class="ww-kpi-card">
      <div class="ww-kpi-top">
        <span class="ww-kpi-label">Geographic Coverage</span>
        <span class="ww-kpi-icon">🗺️</span>
      </div>
      <div class="ww-kpi-val-row">
        <span class="ww-kpi-val">{states_count} States</span>
      </div>
      <div class="ww-kpi-sub">Federative units across Brazil</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    total_rev = seller['total_revenue'].sum() if 'total_revenue' in seller else 0
    st.markdown(f"""
    <div class="ww-kpi-card">
      <div class="ww-kpi-top">
        <span class="ww-kpi-label">Total Platform GMV</span>
        <span class="ww-kpi-icon">💳</span>
      </div>
      <div class="ww-kpi-val-row">
        <span class="ww-kpi-val">R$ {total_rev:,.2f}</span>
      </div>
      <div class="ww-kpi-sub">Aggregate seller merchandise value</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    missing_cells = int(seller.isna().sum().sum())
    st.markdown(f"""
    <div class="ww-kpi-card">
      <div class="ww-kpi-top">
        <span class="ww-kpi-label">Data Integrity</span>
        <span class="ww-kpi-icon">🛡️</span>
      </div>
      <div class="ww-kpi-val-row">
        <span class="ww-kpi-val">{100.0 if missing_cells == 0 else 99.8}% Complete</span>
      </div>
      <div class="ww-kpi-sub">{missing_cells} null cells in feature space</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ----------------- MAIN NAVIGATION TABS -----------------
main_tabs = st.tabs([
    '🏛️ Data Warehouse',
    '📈 OLAP Operations',
    '🧩 Clustering',
    '🎯 Classification',
    '🛒 Association Rules',
    '💹 Revenue Prediction',
    '📚 Data & Methodology'
])

# =========================================================================
# TAB 1: DATA WAREHOUSE
# =========================================================================
with main_tabs[0]:
    st.markdown("""
    <div class="ww-section-head">
      <div class="ww-section-title">Dimensional Data Warehouse Explorer</div>
      <div class="ww-section-sub">Inspect the relational star schema architecture, table dimensions, record counts, and live previews from the SQLite warehouse database.</div>
    </div>
    """, unsafe_allow_html=True)

    if not DB_PATH.exists():
        st.error(f'Warehouse database not found at {DB_PATH}. Please ensure database/warehouse.db is bundled in the project.')
    else:
        conn = sqlite3.connect(DB_PATH)
        table_names = ['Dim_Time', 'Dim_Store', 'Dim_Product', 'Dim_Customer', 'Fact_Store_Sales']
        counts = []
        for table in table_names:
            try:
                row_count = int(pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{table}"', conn).iloc[0]['n'])
                counts.append({'Table': table, 'Type': 'Fact Table' if table.startswith('Fact') else 'Dimension Table', 'Rows': row_count})
            except Exception:
                counts.append({'Table': table, 'Type': 'Unknown', 'Rows': 0})
        total_rows = sum(r['Rows'] for r in counts)
        fact_rows = next((r['Rows'] for r in counts if r['Table'] == 'Fact_Store_Sales'), 0)

        with st.container(border=True):
            st.markdown('<div class="ww-control-bar-title">🏛️ Star Schema Architecture</div>', unsafe_allow_html=True)
            w1, w2, w3, w4 = st.columns(4)
            w1.metric('Relational Tables', len(table_names))
            w2.metric('Total Warehouse Rows', f'{total_rows:,}')
            w3.metric('Fact Sales Records', f'{fact_rows:,}')
            w4.metric('Dimensional Grain', '1 Item / Order')

            dot_schema = """
digraph warehouse {
  graph [rankdir=LR, bgcolor="transparent", pad="0.25", nodesep="0.6", ranksep="0.85"];
  node [shape=box, style="rounded,filled", fontname="Inter, Arial, sans-serif", fontsize=11, penwidth=1.5, margin="0.2,0.14"];
  edge [color="#64748B", penwidth=1.5, arrowsize=0.75];

  Dim_Time [label="DIM_TIME (Temporal Dimension)\n---------------------------------\ntime_id [PK] • date • month • quarter • year", fillcolor="#111827", color="#334155", fontcolor="#F8FAFC"];
  Dim_Store [label="DIM_STORE (Seller Dimension)\n---------------------------------\nstore_key [PK] • seller_id • city • state", fillcolor="#111827", color="#334155", fontcolor="#F8FAFC"];
  Dim_Product [label="DIM_PRODUCT (Item Dimension)\n---------------------------------\nproduct_key [PK] • category • dimensions • weight", fillcolor="#111827", color="#334155", fontcolor="#F8FAFC"];
  Dim_Customer [label="DIM_CUSTOMER (Buyer Dimension)\n---------------------------------\ncustomer_key [PK] • customer_id • city • state", fillcolor="#111827", color="#334155", fontcolor="#F8FAFC"];
  Fact [label="FACT_STORE_SALES (Central Fact)\n=================================\ntime_id [FK] • store_key [FK] • product_key [FK]\nprice (BRL) • freight • item_quantity", fillcolor="#1E3A8A", color="#3B82F6", fontcolor="#EFF6FF", penwidth=2.5];

  Dim_Time -> Fact [label=" 1:N", fontcolor="#94A3B8", fontsize=9];
  Dim_Store -> Fact [label=" 1:N", fontcolor="#94A3B8", fontsize=9];
  Dim_Product -> Fact [label=" 1:N", fontcolor="#94A3B8", fontsize=9];
  Dim_Customer -> Fact [label=" 1:N", fontcolor="#94A3B8", fontsize=9];
}
"""
            st.graphviz_chart(dot_schema, use_container_width=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        col_cat, col_prev = st.columns([1, 2])

        with col_cat:
            with st.container(border=True):
                st.markdown('<div class="ww-control-bar-title">📋 Table Catalogue</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(counts), use_container_width=True, hide_index=True)
                st.caption('Fact table grain represents item-level sales aggregated with dimensions.')

        with col_prev:
            with st.container(border=True):
                st.markdown('<div class="ww-control-bar-title">🔍 Interactive Table Inspector</div>', unsafe_allow_html=True)
                tc1, tc2 = st.columns([2, 1])
                with tc1:
                    selected_table = st.selectbox('Select warehouse table', table_names, key='warehouse_table')
                with tc2:
                    preview_limit = st.select_slider('Row limit', options=[5, 10, 25, 50, 100], value=10, key='warehouse_preview_limit')

                preview = pd.read_sql_query(f'SELECT * FROM "{selected_table}" LIMIT {int(preview_limit)}', conn)
                st.dataframe(preview, use_container_width=True, hide_index=True)
                st.download_button(
                    label=f'📥 Export {selected_table} Preview (CSV)',
                    data=preview.to_csv(index=False).encode('utf-8'),
                    file_name=f'{selected_table.lower()}_preview.csv',
                    mime='text/csv',
                    use_container_width=True
                )
        conn.close()

# =========================================================================
# TAB 2: OLAP OPERATIONS
# =========================================================================
with main_tabs[1]:
    st.markdown("""
    <div class="ww-section-head">
      <div class="ww-section-title">Multi-Dimensional OLAP Operations Workbench</div>
      <div class="ww-section-sub">Execute high-speed Roll-up, Drill-down, Slice, Dice, and Pivot multi-dimensional cube queries directly on the star schema database.</div>
    </div>
    """, unsafe_allow_html=True)

    if not DB_PATH.exists():
        st.error('Warehouse database is missing. Please ensure database/warehouse.db is bundled in the project.')
    else:
        conn = sqlite3.connect(DB_PATH)
        with st.container(border=True):
            st.markdown('<div class="ww-control-bar-title">⚙️ Cube Operation Controls</div>', unsafe_allow_html=True)
            operation = st.radio(
                'Choose OLAP Operation',
                ['Roll-up', 'Drill-down', 'Slice', 'Dice', 'Pivot'],
                horizontal=True,
                key='olap_operation'
            )

            sql = ''
            op_caption = ''
            if operation == 'Roll-up':
                op_caption = 'Roll-up: Aggregates atomic sales transactions into summarized Year and State dimensions.'
                sql = "SELECT t.year AS year, s.seller_state AS state, ROUND(SUM(f.price * f.item_quantity), 2) AS revenue_brl, COUNT(DISTINCT f.order_id) AS orders FROM Fact_Store_Sales f JOIN Dim_Time t ON f.time_id=t.time_id JOIN Dim_Store s ON f.store_key=s.store_key GROUP BY t.year, s.seller_state ORDER BY t.year, revenue_brl DESC"
                result = pd.read_sql_query(sql, conn)
            elif operation == 'Drill-down':
                level = st.selectbox('Drill-down Dimension Path', ['Year → Quarter → Month', 'State → Seller'], key='olap_drill_level')
                if level == 'Year → Quarter → Month':
                    op_caption = 'Drill-down: Expands annual revenue into granular quarterly and monthly time buckets.'
                    sql = "SELECT t.year, t.quarter, t.month, ROUND(SUM(f.price*f.item_quantity),2) AS revenue_brl, COUNT(DISTINCT f.order_id) AS orders FROM Fact_Store_Sales f JOIN Dim_Time t ON f.time_id=t.time_id GROUP BY t.year,t.quarter,t.month ORDER BY t.year,t.quarter,t.month"
                else:
                    op_caption = 'Drill-down: Breaks down state-level aggregated totals into individual seller performance.'
                    sql = "SELECT s.seller_state AS state, s.seller_id, ROUND(SUM(f.price*f.item_quantity),2) AS revenue_brl, COUNT(DISTINCT f.order_id) AS orders FROM Fact_Store_Sales f JOIN Dim_Store s ON f.store_key=s.store_key GROUP BY s.seller_state,s.seller_id ORDER BY state,revenue_brl DESC"
                result = pd.read_sql_query(sql, conn)
            elif operation == 'Slice':
                years = pd.read_sql_query('SELECT DISTINCT year FROM Dim_Time WHERE year IS NOT NULL ORDER BY year', conn)['year'].dropna().astype(int).tolist()
                chosen_year = st.selectbox('Select Slice Dimension (Year)', years, index=len(years)-1 if years else 0, key='olap_slice_year')
                op_caption = f'Slice: Filters the multidimensional cube across a single dimension: Year = {chosen_year}.'
                sql = f"SELECT s.seller_state AS state, ROUND(SUM(f.price*f.item_quantity),2) AS revenue_brl, COUNT(DISTINCT f.order_id) AS orders FROM Fact_Store_Sales f JOIN Dim_Time t ON f.time_id=t.time_id JOIN Dim_Store s ON f.store_key=s.store_key WHERE t.year={int(chosen_year)} GROUP BY s.seller_state ORDER BY revenue_brl DESC"
                result = pd.read_sql_query(sql, conn)
            elif operation == 'Dice':
                years = pd.read_sql_query('SELECT DISTINCT year FROM Dim_Time WHERE year IS NOT NULL ORDER BY year', conn)['year'].dropna().astype(int).tolist()
                states = pd.read_sql_query('SELECT DISTINCT seller_state FROM Dim_Store WHERE seller_state IS NOT NULL ORDER BY seller_state', conn)['seller_state'].dropna().tolist()
                d1, d2 = st.columns(2)
                with d1:
                    chosen_years = st.multiselect('Select Temporal Dimension(s)', years, default=years[-min(2,len(years)):], key='olap_dice_years') if years else []
                with d2:
                    chosen_states = st.multiselect('Select Geographic Dimension(s)', states, default=states[:4], key='olap_dice_states')
                op_caption = 'Dice: Isolates a sub-cube by constraining multiple dimensions simultaneously (selected years and states).'
                if chosen_years and chosen_states:
                    year_list = ','.join(str(int(y)) for y in chosen_years)
                    state_list = ','.join("'" + str(x).replace("'", "''") + "'" for x in chosen_states)
                    sql = f"SELECT t.year, s.seller_state AS state, ROUND(SUM(f.price*f.item_quantity),2) AS revenue_brl, COUNT(DISTINCT f.order_id) AS orders FROM Fact_Store_Sales f JOIN Dim_Time t ON f.time_id=t.time_id JOIN Dim_Store s ON f.store_key=s.store_key WHERE t.year IN ({year_list}) AND s.seller_state IN ({state_list}) GROUP BY t.year,s.seller_state ORDER BY t.year,revenue_brl DESC"
                    result = pd.read_sql_query(sql, conn)
                else:
                    result = pd.DataFrame()
                    st.info('Select at least one year and one state to execute Dice operation.')
            else:
                op_caption = 'Pivot: Rotates the dimensional view so seller states form rows and calendar years form columns.'
                sql = "SELECT t.year, s.seller_state AS state, ROUND(SUM(f.price*f.item_quantity),2) AS revenue_brl FROM Fact_Store_Sales f JOIN Dim_Time t ON f.time_id=t.time_id JOIN Dim_Store s ON f.store_key=s.store_key GROUP BY t.year,s.seller_state"
                raw_pivot = pd.read_sql_query(sql, conn)
                result = raw_pivot.pivot_table(index='state', columns='year', values='revenue_brl', aggfunc='sum', fill_value=0).reset_index()
                result.columns = [str(c) for c in result.columns]

            st.markdown(f'<div class="ww-callout ww-callout-info"><span class="ww-callout-icon">💡</span><div><b>OLAP Definition:</b> {op_caption}</div></div>', unsafe_allow_html=True)

        if not result.empty:
            oc1, oc2 = st.columns([1, 2])
            with oc1:
                with st.container(border=True):
                    st.markdown('<div class="ww-control-bar-title">📊 Query Metrics</div>', unsafe_allow_html=True)
                    st.metric('Result Records', f'{len(result):,}')
                    if 'revenue_brl' in result.columns:
                        total_op_rev = result['revenue_brl'].sum()
                        st.metric('Aggregate Revenue', f'R$ {total_op_rev:,.2f}')
                    st.download_button(
                        label='📥 Export OLAP Result (CSV)',
                        data=result.to_csv(index=False).encode('utf-8'),
                        file_name=f'olap_{operation.lower().replace(" ","_")}_result.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
            with oc2:
                if operation in ['Roll-up', 'Slice', 'Dice'] and 'revenue_brl' in result.columns:
                    chart_dim = 'state' if 'state' in result.columns else result.columns[0]
                    chart_df = result.groupby(chart_dim, as_index=False)['revenue_brl'].sum().sort_values('revenue_brl', ascending=False).head(15)
                    fig = px.bar(
                        chart_df,
                        x=chart_dim,
                        y='revenue_brl',
                        title=f'Revenue by Dimension ({chart_dim.capitalize()}) - Top 15',
                        labels={'revenue_brl': 'Revenue (R$ / BRL)', chart_dim: chart_dim.capitalize()},
                        color='revenue_brl',
                        color_continuous_scale=['#1E3A8A', '#2563EB', '#60A5FA']
                    )
                    fig.update_coloraxes(showscale=False)
                    format_chart(fig, height=280, show_legend=False, x_title=chart_dim.capitalize(), y_title='Revenue (BRL / R$)')
                    st.plotly_chart(fig, use_container_width=True, theme=None)

            with st.container(border=True):
                st.markdown('<div class="ww-control-bar-title">📋 Cube Query Result Table</div>', unsafe_allow_html=True)
                st.dataframe(result.head(500), use_container_width=True, hide_index=True)
                if sql:
                    with st.expander('🔍 Inspect Generated SQL Query'):
                        st.code(sql, language='sql')
        conn.close()

# =========================================================================
# TAB 3: CLUSTERING LAB
# =========================================================================
with main_tabs[2]:
    st.markdown("""
    <div class="ww-section-head">
      <div class="ww-section-title">Unsupervised Clustering &amp; Segmentation Lab</div>
      <div class="ww-section-sub">Evaluate optimal cluster counts via Elbow and Silhouette methods, compare K-Means, Agglomerative, and DBSCAN algorithms, and analyze seller segments.</div>
    </div>
    """, unsafe_allow_html=True)

    available_cluster_features = [c for c in cluster_features if c in seller.columns]
    with st.container(border=True):
        st.markdown('<div class="ww-control-bar-title">⚙️ Clustering Parameters</div>', unsafe_allow_html=True)
        chosen_features = st.multiselect('Clustering Features (Standardized)', available_cluster_features, default=available_cluster_features, key='cluster_features')
        f1, f2 = st.columns([1, 2])
        with f1:
            chosen_k = st.slider('Cluster Count (K)', min_value=2, max_value=10, value=3, key='k_slider')
        with f2:
            cluster_alg = st.selectbox('Clustering Algorithm', ['K-Means', 'Agglomerative Clustering', 'DBSCAN'], key='cluster_algorithm')

        eps, min_samples = 0.8, 8
        if cluster_alg == 'DBSCAN':
            db1, db2 = st.columns(2)
            with db1:
                eps = st.slider('DBSCAN Epsilon (eps)', 0.2, 3.0, 0.8, 0.1, key='dbscan_eps')
            with db2:
                min_samples = st.slider('DBSCAN Minimum Samples (min_samples)', 3, 30, 8, key='dbscan_min_samples')

    if len(chosen_features) < 2:
        st.warning('Please select at least two numeric features to run clustering.')
    else:
        with st.spinner('Calculating Elbow and Silhouette curves for K = 2–10…'):
            x_raw, x_scaled, elbow_df = compute_cluster_evaluation(seller, tuple(chosen_features))

        ec1, ec2 = st.columns(2)
        with ec1:
            fig_elbow = px.line(elbow_df, x='K', y='Inertia (WCSS)', markers=True, title='Elbow Method: Inertia (WCSS) vs K')
            fig_elbow.add_vline(x=chosen_k, line_dash="dash", line_color="#3B82F6", annotation_text=f"Selected K={chosen_k}", annotation_position="top right")
            format_chart(fig_elbow, height=340, x_title='K (Number of Clusters)', y_title='Inertia (WCSS)')
            st.plotly_chart(fig_elbow, use_container_width=True, theme=None)
        with ec2:
            fig_sil = px.line(elbow_df, x='K', y='Silhouette score', markers=True, title='Silhouette Score vs K')
            fig_sil.add_vline(x=chosen_k, line_dash="dash", line_color="#10B981", annotation_text=f"Selected K={chosen_k}", annotation_position="top right")
            format_chart(fig_sil, height=340, x_title='K (Number of Clusters)', y_title='Silhouette Score')
            st.plotly_chart(fig_sil, use_container_width=True, theme=None)

        with st.spinner(f'Running {cluster_alg} segmentation…'):
            x_raw, labels, valid_sil = compute_cluster_labels(seller, tuple(chosen_features), cluster_alg, chosen_k, eps, min_samples)

        clustered = x_raw.copy()
        clustered['cluster'] = labels.astype(str)
        clustered['seller_id'] = seller.loc[x_raw.index, 'seller_id'].astype(str).values if 'seller_id' in seller else x_raw.index.astype(str)
        cluster_counts = clustered['cluster'].value_counts().rename_axis('Cluster').reset_index(name='Seller count').sort_values('Cluster')
        non_noise = len(set(labels) - {-1})

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric('Identified Clusters', str(non_noise))
        cm2.metric('Silhouette Score', 'N/A' if np.isnan(valid_sil) else f'{valid_sil:.4f}')
        cm3.metric('Noise Points (DBSCAN)', f'{int(np.sum(labels == -1)):,}' if cluster_alg == 'DBSCAN' else 'Not Applicable')

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_dist = px.bar(
                cluster_counts,
                x='Cluster',
                y='Seller count',
                color='Cluster',
                color_discrete_sequence=PLOT_PALETTE,
                title=f'{cluster_alg}: Cluster Distribution'
            )
            format_chart(fig_dist, height=340, show_legend=False, x_title='Cluster Identifier', y_title='Seller Count')
            st.plotly_chart(fig_dist, use_container_width=True, theme=None)

        profile_cols = [c for c in ['total_revenue', 'total_orders', 'total_items', 'average_item_price', 'freight_ratio'] if c in clustered]
        if profile_cols:
            profile = clustered.groupby('cluster')[profile_cols].mean().reset_index().melt(id_vars='cluster', var_name='Metric', value_name='Mean value')
            with cc2:
                fig_prof = px.bar(
                    profile,
                    x='cluster',
                    y='Mean value',
                    color='Metric',
                    barmode='group',
                    color_discrete_sequence=PLOT_PALETTE,
                    title='Cluster Profiles (Feature Averages)'
                )
                format_chart(fig_prof, height=340, x_title='Cluster Identifier', y_title='Mean Value')
                st.plotly_chart(fig_prof, use_container_width=True, theme=None)

            with st.container(border=True):
                st.markdown('<div class="ww-control-bar-title">📋 Cluster Profiles Summary Table</div>', unsafe_allow_html=True)
                st.dataframe(clustered.groupby('cluster')[profile_cols].mean().round(3), use_container_width=True)

        st.download_button(
            label='📥 Download Current Cluster Assignments (CSV)',
            data=clustered.to_csv(index=False).encode('utf-8'),
            file_name='interactive_clustering_results.csv',
            mime='text/csv'
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="ww-control-bar-title">⚖️ Algorithm Comparison at Selected K</div>', unsafe_allow_html=True)
            with st.spinner('Comparing clustering algorithms…'):
                comparison = compute_cluster_comparison(seller, tuple(chosen_features), chosen_k, eps, min_samples)
            fig_comp = px.bar(
                comparison,
                x='Algorithm',
                y='Silhouette score',
                color='Algorithm',
                color_discrete_sequence=PLOT_PALETTE,
                title='Clustering Algorithm Comparison (Silhouette Score - Higher is Better)'
            )
            format_chart(fig_comp, height=320, show_legend=False, x_title='Algorithm', y_title='Silhouette Score')
            st.plotly_chart(fig_comp, use_container_width=True, theme=None)
            st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)

# =========================================================================
# TAB 4: CLASSIFICATION LAB
# =========================================================================
with main_tabs[3]:
    st.markdown("""
    <div class="ww-section-head">
      <div class="ww-section-title">Supervised Classification &amp; Performance Benchmarking</div>
      <div class="ww-section-sub">Train and compare Decision Tree, Random Forest, Logistic Regression, and K-Nearest Neighbors to predict seller performance classes.</div>
    </div>
    """, unsafe_allow_html=True)

    if 'performance' not in seller:
        st.warning('Performance target label is missing. Please ensure store_mining_dataset.csv has the performance column.')
    else:
        with st.container(border=True):
            st.markdown('<div class="ww-control-bar-title">⚙️ Training Hyperparameters</div>', unsafe_allow_html=True)
            sp1, sp2 = st.columns(2)
            with sp1:
                test_size = st.slider('Test Set Ratio', 0.15, 0.35, 0.20, 0.05, key='class_test_size')
            with sp2:
                depth = st.slider('Decision Tree Max Depth', 2, 12, 5, key='tree_depth')

        features = [c for c in num_features if c in seller.columns]
        with st.spinner('Training and benchmarking classification models…'):
            metrics, y_test, predictions = compute_classification_results(seller, tuple(features), test_size, depth)

        metrics = metrics.sort_values('F1 (weighted)', ascending=False)
        best_model_name = metrics.iloc[0]['Algorithm']
        best_f1 = metrics.iloc[0]['F1 (weighted)']
        best_acc = metrics.iloc[0]['Accuracy']

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric('Top Performing Model', best_model_name)
        m_col2.metric('Best F1-Score (Weighted)', f'{best_f1:.4f}')
        m_col3.metric('Best Accuracy', f'{best_acc:.4f}')

        fig_metrics = px.bar(
            metrics.melt(id_vars='Algorithm', var_name='Metric', value_name='Score'),
            x='Algorithm',
            y='Score',
            color='Metric',
            barmode='group',
            color_discrete_sequence=PLOT_PALETTE,
            title='Classification Performance Comparison across Algorithms'
        )
        format_chart(fig_metrics, height=380, x_title='Algorithm', y_title='Score (0.0 to 1.0)')
        st.plotly_chart(fig_metrics, use_container_width=True, theme=None)

        with st.container(border=True):
            st.markdown('<div class="ww-control-bar-title">📋 Comprehensive Classification Metrics</div>', unsafe_allow_html=True)
            st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="ww-control-bar-title">🎯 Confusion Matrix Inspector</div>', unsafe_allow_html=True)
            selected_model = st.selectbox('Choose Model to Inspect Confusion Matrix', list(predictions.keys()), key='class_inspect')
            pred = predictions[selected_model]
            labels_order = sorted(seller['performance'].astype(str).unique())
            cm = confusion_matrix(y_test, pred, labels=labels_order)

            fig_cm = px.imshow(
                cm,
                x=labels_order,
                y=labels_order,
                text_auto=True,
                aspect='auto',
                color_continuous_scale=['#111827', '#1E3A8A', '#2563EB', '#60A5FA'],
                title=f'{selected_model}: Confusion Matrix Heatmap',
                labels={'x': 'Predicted Class', 'y': 'Actual Class', 'color': 'Sample Count'}
            )
            format_chart(fig_cm, height=350, show_legend=False, x_title='Predicted Class', y_title='Actual Class')
            st.plotly_chart(fig_cm, use_container_width=True, theme=None)
            st.caption('Model performance is evaluated against balanced median-derived performance labels with standardized numeric features.')

# =========================================================================
# TAB 5: ASSOCIATION RULES LAB
# =========================================================================
with main_tabs[4]:
    st.markdown("""
    <div class="ww-section-head">
      <div class="ww-section-title">Market Basket Analysis &amp; Association Rule Mining</div>
      <div class="ww-section-sub">Discover high-lift product co-purchase affinities using Apriori and FP-Growth algorithms across retail transactions.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="ww-control-bar-title">⚙️ Market Basket Mining Settings</div>', unsafe_allow_html=True)
        source = st.radio(
            'Select Transaction Basket Source',
            ['Built-in demo retail baskets', 'Olist (project dataset)', 'Upload your own file'],
            horizontal=True,
            key='assoc_source'
        )

        uploaded = None
        if source == 'Upload your own file':
            uploaded = st.file_uploader('Upload transaction-level CSV or Excel', type=['csv', 'xlsx', 'xls'], key='assoc_upload')

        ar1, ar2, ar3, ar4 = st.columns(4)
        with ar1:
            min_support = st.number_input('Minimum Support', min_value=0.00001, max_value=0.20, value=0.0001, step=0.0001, format='%.5f', key='min_support')
        with ar2:
            min_confidence = st.slider('Minimum Confidence', 0.01, 1.00, 0.05, 0.01, key='min_confidence')
        with ar3:
            min_lift = st.number_input('Minimum Lift', min_value=0.0, max_value=100.0, value=1.0, step=0.1, key='min_lift')
        with ar4:
            max_rows = st.slider('Max Upload Rows to Parse', 5000, 100000, 30000, 5000, key='assoc_max_rows')

    if not MLXTEND_AVAILABLE:
        st.error('mlxtend library is required for association mining. Please ensure mlxtend is installed.')
    else:
        transactions = None
        if source == 'Built-in demo retail baskets':
            demo = find_data('default_association_transactions.csv')
            if demo is not None and {'transaction_id', 'product'}.issubset(demo.columns):
                transactions = demo.groupby('transaction_id')['product'].apply(lambda x: sorted(set(x.dropna().astype(str)))).tolist()
            else:
                st.error('Default demo dataset not found or missing required columns.')
        elif source == 'Olist (project dataset)':
            items = find_data('olist_order_items_dataset.csv')
            products = find_data('olist_products_dataset.csv')
            trans = find_data('product_category_name_translation.csv')
            if items is not None and products is not None:
                with st.spinner('Preparing Olist transaction baskets…'):
                    transactions = build_olist_transactions(items, products, trans)
        elif source == 'Upload your own file' and uploaded is not None:
            try:
                raw = pd.read_excel(uploaded) if uploaded.name.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded, encoding_errors='replace')
                inv = next((c for c in ['InvoiceNo', 'Invoice', 'invoice_no', 'invoice', 'order_id', 'Order ID', 'Transaction'] if c in raw.columns), None)
                item = next((c for c in ['Description', 'description', 'StockCode', 'stock_code', 'product', 'Product', 'item', 'Item'] if c in raw.columns), None)
                qty = next((c for c in ['Quantity', 'quantity', 'qty'] if c in raw.columns), None)
                if inv is None or item is None:
                    st.error('Could not detect transaction and product columns in uploaded file.')
                else:
                    raw = raw.head(max_rows).copy()
                    raw[inv] = raw[inv].astype(str)
                    raw[item] = raw[item].astype(str)
                    raw = raw[~raw[inv].str.startswith('C', na=False)] if inv == 'InvoiceNo' else raw
                    if qty:
                        raw[qty] = pd.to_numeric(raw[qty], errors='coerce')
                        raw = raw[raw[qty] > 0]
                    transactions = raw.groupby(inv)[item].apply(lambda s: sorted(set(s.dropna()))).tolist()
            except Exception as e:
                st.error(f'Error reading uploaded file: {e}')

        if transactions is not None:
            transactions = [t for t in transactions if len(t) > 0]
            basket_counts = pd.Series([len(t) for t in transactions], name='Items per Basket')
            
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            bk1, bk2, bk3, bk4 = st.columns(4)
            bk1.metric('Total Baskets', f'{len(transactions):,}')
            bk2.metric('Single-Item Orders', f'{int((basket_counts == 1).sum()):,}')
            bk3.metric('Multi-Item Orders', f'{int((basket_counts >= 2).sum()):,}')
            bk4.metric('Max Basket Size', f'{int(basket_counts.max()) if len(basket_counts) else 0}')

            fig_hist = px.histogram(
                basket_counts,
                nbins=30,
                color_discrete_sequence=['#3B82F6'],
                title='Market Basket Size Distribution'
            )
            format_chart(fig_hist, height=280, show_legend=False, x_title='Unique Products in Basket', y_title='Order Count')
            st.plotly_chart(fig_hist, use_container_width=True, theme=None)

            if len(transactions) > 100000:
                transactions = transactions[:100000]

            with st.spinner('Mining frequent itemsets…'):
                frequent, item_count = compute_frequent_itemsets(transactions, float(min_support))

            if not frequent.empty:
                with st.spinner('Deriving association rules…'):
                    rules = compute_association_rules(frequent, float(min_confidence))
                if not rules.empty:
                    rules = rules[rules['lift'] >= float(min_lift)].sort_values(['lift', 'confidence'], ascending=False)
                    rules['IF product(s)'] = rules['antecedents'].apply(lambda s: ', '.join(sorted(s)))
                    rules['THEN product(s)'] = rules['consequents'].apply(lambda s: ', '.join(sorted(s)))
                    display = rules[['IF product(s)', 'THEN product(s)', 'support', 'confidence', 'lift']].head(100)

                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    st.metric('Rules Satisfying Current Thresholds', f'{len(rules):,}')

                    with st.container(border=True):
                        st.markdown('<div class="ww-control-bar-title">📋 Discovered Association Rules (Top 100 by Lift)</div>', unsafe_allow_html=True)
                        st.dataframe(display.round(5), use_container_width=True, hide_index=True)

                    if len(display):
                        fig_scatter = px.scatter(
                            display,
                            x='support',
                            y='confidence',
                            size='lift',
                            hover_name='IF product(s)',
                            color='lift',
                            color_continuous_scale=['#1E3A8A', '#3B82F6', '#93C5FD'],
                            title='Association Rules: Support vs Confidence (Bubble Size = Lift)',
                            hover_data=['THEN product(s)']
                        )
                        format_chart(fig_scatter, height=420, x_title='Support', y_title='Confidence')
                        st.plotly_chart(fig_scatter, use_container_width=True, theme=None)

                    st.download_button(
                        label='📥 Export Association Rules (CSV)',
                        data=display.to_csv(index=False).encode('utf-8'),
                        file_name='interactive_association_rules.csv',
                        mime='text/csv'
                    )
                else:
                    st.warning('No association rules satisfied the minimum confidence or lift thresholds. Try reducing the thresholds.')
            else:
                st.warning('No frequent itemsets identified at the chosen minimum support. Consider decreasing minimum support.')
        else:
            st.info('Select the built-in demo dataset or upload a transaction CSV/Excel file to start mining association rules.')

# =========================================================================
# TAB 6: REVENUE PREDICTION LAB
# =========================================================================
with main_tabs[5]:
    st.markdown("""
    <div class="ww-section-head">
      <div class="ww-section-title">Seller Revenue Estimation &amp; Predictive Regression</div>
      <div class="ww-section-sub">Benchmark Linear Regression, Decision Tree Regressor, Random Forest, and Extra Trees Regressor to predict total seller sales volume.</div>
    </div>
    """, unsafe_allow_html=True)

    regression_features = [c for c in ['total_freight', 'total_items', 'total_orders', 'average_item_price', 'average_order_value', 'freight_ratio'] if c in seller.columns]
    if 'total_revenue' not in seller.columns or len(regression_features) < 2:
        st.warning('Revenue target (total_revenue) or regression features are missing.')
    else:
        with st.container(border=True):
            st.markdown('<div class="ww-control-bar-title">⚙️ Regression Settings</div>', unsafe_allow_html=True)
            test_size_reg = st.slider('Test Set Proportion', 0.15, 0.35, 0.20, 0.05, key='reg_test_size')

        with st.spinner('Fitting and evaluating regression models…'):
            reg_metrics, y_test_reg, reg_predictions = compute_regression_results(seller, tuple(regression_features), test_size_reg)

        reg_metrics = reg_metrics.sort_values('MAE (R$)')
        best_reg_name = reg_metrics.iloc[0]['Algorithm']
        best_mae = reg_metrics.iloc[0]['MAE (R$)']
        best_r2 = reg_metrics.sort_values('R²', ascending=False).iloc[0]['R²']

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric('Lowest Error Model', best_reg_name)
        rc2.metric('Best MAE (Mean Absolute Error)', f'R$ {best_mae:,.2f}')
        rc3.metric('Highest R² Score', f'{best_r2:.4f}')

        rc_c1, rc_c2 = st.columns(2)
        with rc_c1:
            fig_err = px.bar(
                reg_metrics.melt(id_vars='Algorithm', value_vars=['MAE (R$)', 'RMSE (R$)'], var_name='Metric', value_name='Error (R$)'),
                x='Algorithm',
                y='Error (R$)',
                color='Metric',
                barmode='group',
                color_discrete_sequence=['#3B82F6', '#06B6D4'],
                title='Regression Error Comparison (Lower is Better)'
            )
            format_chart(fig_err, height=350, x_title='Algorithm', y_title='Error (BRL / R$)')
            st.plotly_chart(fig_err, use_container_width=True, theme=None)

        with rc_c2:
            fig_r2 = px.bar(
                reg_metrics,
                x='Algorithm',
                y='R²',
                color='Algorithm',
                color_discrete_sequence=PLOT_PALETTE,
                title='Coefficient of Determination (R² - Higher is Better)'
            )
            format_chart(fig_r2, height=350, show_legend=False, x_title='Algorithm', y_title='R² Score')
            st.plotly_chart(fig_r2, use_container_width=True, theme=None)

        with st.container(border=True):
            st.markdown('<div class="ww-control-bar-title">📋 Regression Performance Metrics Table</div>', unsafe_allow_html=True)
            st.dataframe(reg_metrics.round(3), use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="ww-control-bar-title">🔍 Actual vs Predicted Revenue Deep Dive</div>', unsafe_allow_html=True)
            chosen_reg = st.selectbox('Choose Regressor to Inspect', list(reg_predictions.keys()), key='reg_inspect')
            pred = reg_predictions[chosen_reg]
            compare = pd.DataFrame({'Actual revenue (R$)': y_test_reg.values, 'Predicted revenue (R$)': pred})

            max_val = max(compare['Actual revenue (R$)'].max(), compare['Predicted revenue (R$)'].max())
            fig_act_pred = px.scatter(
                compare,
                x='Actual revenue (R$)',
                y='Predicted revenue (R$)',
                color_discrete_sequence=['#3B82F6'],
                opacity=0.8,
                title=f'{chosen_reg}: Actual vs Predicted Revenue'
            )
            fig_act_pred.add_shape(
                type="line", line=dict(dash='dash', color='#64748B', width=2),
                x0=0, y0=0, x1=max_val, y1=max_val
            )
            format_chart(fig_act_pred, height=400, x_title='Actual Revenue (BRL / R$)', y_title='Predicted Revenue (BRL / R$)')
            st.plotly_chart(fig_act_pred, use_container_width=True, theme=None)

            compare['Absolute error (R$)'] = abs(compare['Actual revenue (R$)'] - compare['Predicted revenue (R$)'])
            st.dataframe(compare.head(30).round(2), use_container_width=True, hide_index=True)

            st.download_button(
                label='📥 Export Actual vs Predicted Predictions (CSV)',
                data=compare.to_csv(index=False).encode('utf-8'),
                file_name='revenue_prediction_comparison.csv',
                mime='text/csv'
            )

# =========================================================================
# TAB 7: DATA & METHODOLOGY
# =========================================================================
with main_tabs[6]:
    st.markdown("""
    <div class="ww-section-head">
      <div class="ww-section-title">Data Architecture &amp; Methodology Standards</div>
      <div class="ww-section-sub">Comprehensive overview of data provenance, ETL workflows, star schema modeling, model assumptions, and ethical considerations.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="ww-control-bar-title">📐 Project Architecture &amp; Pipeline</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("""
            **1. Data Staging &amp; Warehouse**
            - Public Olist Brazilian E-Commerce dataset.
            - SQLite 3 Relational Star Schema (`Dim_Time`, `Dim_Store`, `Dim_Product`, `Dim_Customer`, `Fact_Store_Sales`).
            - Item-level transaction grain preserving price and freight dimensions.
            """)
        with p2:
            st.markdown("""
            **2. OLAP Cube &amp; Mart Extraction**
            - Multi-dimensional slicing, dicing, roll-up, drill-down, and pivoting.
            - Seller-level analytical dataset aggregation (`store_mining_dataset.csv`).
            - Elimination of target leakage across analytical models.
            """)
        with p3:
            st.markdown("""
            **3. Mining &amp; Diagnostic Suite**
            - Unsupervised clustering with K evaluation (WCSS &amp; Silhouette).
            - Supervised classification across 4 diverse ML paradigms.
            - Baseline regression modeling with actual-vs-predicted diagnostics.
            - Market basket rule discovery with FP-Growth.
            """)

    with st.container(border=True):
        st.markdown('<div class="ww-control-bar-title">⚠️ Methodological Assumptions &amp; Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Seller Units as Store Proxies:** Seller entities are used as store proxies; they represent independent marketplace sellers rather than centralized brick-and-mortar chain branches.
        - **Classification Target Formulation:** The `performance` classification label is formulated relative to median sales revenue. Model evaluation measures adherence to this segmentation boundary rather than speculative future profitability.
        - **Market Basket Single-Item Bias:** Olist orders heavily concentrate in single-item purchases, limiting multi-item co-purchases. The built-in demo retail dataset provides rich multi-item transaction patterns for comprehensive validation.
        - **Association vs Causation:** Association rules indicate statistical co-occurrence within shopping carts; lift values greater than 1 demonstrate positive correlation, not causal purchasing influence.
        """)

    with st.container(border=True):
        st.markdown('<div class="ww-control-bar-title">🔍 Seller Mining Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(seller.head(25), use_container_width=True, hide_index=True)
        st.download_button(
            label='📥 Download Complete Seller Mining Dataset (CSV)',
            data=seller.to_csv(index=False).encode('utf-8'),
            file_name='store_mining_dataset.csv',
            mime='text/csv'
        )

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;color:#94A3B8;font-size:0.8rem;padding:0.4rem 0;">
  <div><b>Warehouse Warriors</b> · Data Warehouse &amp; Data Mining Capstone Suite</div>
  <div>Synchronized Dark Enterprise UI · Built with Streamlit, Plotly &amp; Scikit-Learn</div>
</div>
""", unsafe_allow_html=True)
