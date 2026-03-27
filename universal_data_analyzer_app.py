"""
╔══════════════════════════════════════════════════════════════════╗
║         UNIVERSAL DATA ANALYZER — Streamlit App                  ║
║         Project: Universal Data Analyzer using Python            ║
║         Tools  : Streamlit, Pandas, NumPy, Matplotlib,           ║
║                  Seaborn, Scikit-learn                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import io

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Data Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Font import ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #ffffff;
    color: #0f172a !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    background-color: #ffffff;
    color: #0f172a !important;
}

/* ── Header banner ── */
.app-header {
    background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 50%, #94a3b8 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(148,163,184,0.3);
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(79,142,247,0.2) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header h1 {
    color: #0f172a;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: rgba(15,23,42,0.9) !important;
    font-size: 0.95rem;
    margin: 0;
    font-family: 'DM Mono', monospace;
}
.badge-row { display: flex; gap: 8px; margin-top: 1rem; flex-wrap: wrap; }
.badge {
    background: rgba(79,142,247,0.25);
    color: #7eb3ff;
    border: 1px solid rgba(79,142,247,0.4);
    border-radius: 99px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
}

/* ── Step cards ── */
.step-card {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #4f8ef7;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.step-card h4 { color: #0f172a !important; font-size: 0.95rem; font-weight: 600; margin: 0 0 0.3rem 0; }
.step-card p  { color: #334155 !important; font-size: 0.85rem; margin: 0; line-height: 1.6; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.75rem 1.25rem;
    background: linear-gradient(90deg, #e2e8f0, #cbd5e1);
    border-radius: 10px;
    border-left: 4px solid #4f8ef7;
    margin: 1.5rem 0 1rem 0;
}
.section-header h2 { color: #0f172a !important; font-size: 1.15rem; font-weight: 700; margin: 0; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 120px;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.07);
}
.metric-card .val { font-size: 1.6rem; font-weight: 700; color: #1e3a5f; font-family: 'DM Mono', monospace; }
.metric-card .lbl { font-size: 0.75rem; color: #334155 !important; margin-top: 2px; }

/* ── Insight box ── */
.insight-box {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}
.insight-box .ins-title { font-weight: 600; color: #1e3a5f !important; font-size: 0.95rem; margin-bottom: 0.3rem; }
.insight-box .ins-body  { color: #334155 !important; font-size: 0.875rem; line-height: 1.65; }

/* ── Code box ── */
.code-box {
    background: #f8fafc;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    margin: 0.5rem 0;
    white-space: pre;
    overflow-x: auto;
}

/* ── Viva card ── */
.viva-card {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #f6a623;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.viva-card .viva-q { font-weight: 600; color: #1e40af !important; font-size: 0.9rem; margin-bottom: 0.3rem; }
.viva-card .viva-a { color: #334155 !important; font-size: 0.85rem; line-height: 1.65; }

/* ── Upload area ── */
.upload-hint {
    background: #ffffff;
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: #0f172a !important;
    font-size: 0.875rem;
    margin-bottom: 1rem;
}

/* ── Interactive dataset cards ── */
.dataset-card {
    background: #ffffff;
    border: 2px solid #cbd5e1;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease-out;
}

.dataset-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
}

.dataset-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 100px;
    height: 100px;
    background: radial-gradient(circle, rgba(79,142,247,0.15) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 3s ease-in-out infinite;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

/* ── Success animation ── */
.success-animation {
    animation: successPulse 0.6s ease-out;
}

@keyframes successPulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

/* ── Tab style ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #e2e8f0;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 0.875rem;
    color: #0f172a !important;
    background: #f8fafc;
}
.stTabs [aria-selected="true"] {
    background: #4f8ef7 !important;
    color: white !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #cbd5e1;
    color: #0f172a !important;
}
section[data-testid="stSidebar"] *, section[data-testid="stSidebar"] .css-1d391kg, section[data-testid="stSidebar"] .css-1oe5b5c {
    color: #0f172a !important;
}

/* ── Streamlit component overrides ── */
.stSelectbox, .stFileUploader, .stButton, .stSlider, .stRadio {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
}

.stSelectbox div[data-baseweb="select"] div {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

.stFileUploader div {
    background-color: #ffffff !important;
    border: 2px dashed #4f8ef7 !important;
    color: #0f172a !important;
}

.stButton button {
    background: linear-gradient(135deg, #4f8ef7, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(79,142,247,0.3) !important;
}

/* ── Dataframe styling ── */
.stDataFrame {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

.stDataFrame table {
    color: #0f172a !important;
}

.stDataFrame th {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

.stDataFrame td {
    border-bottom: 1px solid #e2e8f0 !important;
}

/* ── Progress bar ── */
.stProgress div div {
    background: linear-gradient(90deg, #4f8ef7, #3b82f6) !important;
}

/* ── Text input and number input ── */
.stTextInput input, .stNumberInput input {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    color: #0f172a !important;
}

/* ── Multiselect ── */
.stMultiSelect div[data-baseweb="select"] div {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

/* ── Radio buttons ── */
.stRadio div[role="radiogroup"] label {
    color: #0f172a !important;
}

/* ── Info, success, warning, error messages ── */
.stAlert {
    background-color: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
}

.stSuccess {
    background-color: #dcfce7 !important;
    border-left: 4px solid #22c55e !important;
    color: #064e3b !important;
}

.stWarning {
    background-color: #fef3c7 !important;
    border-left: 4px solid #f59e0b !important;
    color: #92400e !important;
}

.stError {
    background-color: #fee2e2 !important;
    border-left: 4px solid #ef4444 !important;
    color: #991b1b !important;
}

.stInfo {
    background-color: #e0f2fe !important;
    border-left: 4px solid #3b82f6 !important;
    color: #1d4ed8 !important;
}

/* ── Plot styling ── */
.stPlotlyChart, .stPyplot {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────
def section(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.4rem">{icon}</span>
        <h2>{title}</h2>
    </div>""", unsafe_allow_html=True)

def metric_cards(items):
    """items = list of (value, label) tuples"""
    cols = st.columns(len(items))
    for col, (val, lbl) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

def insight(icon, title, body):
    st.markdown(f"""
    <div class="insight-box">
        <div class="ins-title">{icon} {title}</div>
        <div class="ins-body">{body}</div>
    </div>""", unsafe_allow_html=True)

def viva(q, a):
    st.markdown(f"""
    <div class="viva-card">
        <div class="viva-q">🎤 Q: {q}</div>
        <div class="viva-a">💬 {a}</div>
    </div>""", unsafe_allow_html=True)

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

def detect_cols(df):
    id_kw   = ['id','no','number','index','sl','serial','code']
    nm_kw   = ['name','id','code','number','email','phone']
    num = [c for c in df.select_dtypes(include=[np.number]).columns
           if not any(k in c.lower() for k in id_kw)]
    cat = [c for c in df.select_dtypes(include='object').columns
           if not any(k in c.lower() for k in nm_kw)]
    return num, cat

def plot_style():
    return {
        'facecolor': '#ffffff',
        'grid_color': '#f0f4ff',
        'text_color': '#1e3a5f',
        'accent': ['#4f8ef7','#34d399','#f87171','#fbbf24',
                   '#a78bfa','#22d3ee','#f472b6','#fb923c'],
    }


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Universal Data Analyzer")
    st.markdown("---")

    st.markdown("#### 🎯 Workflow Options")
    workflow_stage = st.radio(
        "Select a workflow stage:",
        [
            "Data Upload",
            "Data Understanding",
            "Data Cleaning",
            "Data Filtering",
            "EDA & Analysis",
            "Visualization",
            "ML Prediction",
            "Result Summary",
        ],
        index=0
    )
    st.markdown("<div style='margin-top:10px;font-size:0.8rem;color:#ffffff'>"
                f"Currently selected: <b style='color:#4f8ef7'>{workflow_stage}</b></div>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🎯 Project Info")
    st.markdown("""
    <div style='font-size:0.8rem;color:#ffffff;line-height:1.7'>
    <b>Title:</b> Universal Data Analyzer<br>
    <b>Tools:</b> Python · Pandas · Sklearn<br>
    <b>Type:</b> End-to-End DS Project<br>
    <b>Works with:</b> Any CSV dataset
    </div>""", unsafe_allow_html=True)

    st.markdown("---")


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🔬 Universal Data Analyzer</h1>
    <p>Upload any CSV → Auto Clean → Analyze → Visualize → Predict</p>
    <div class="badge-row">
        <span class="badge">Pandas</span>
        <span class="badge">NumPy</span>
        <span class="badge">Matplotlib</span>
        <span class="badge">Seaborn</span>
        <span class="badge">Scikit-learn</span>
        <span class="badge">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# STEP 1 — DATA UPLOAD
# ─────────────────────────────────────────────────────────────────
if workflow_stage == "Data Upload":
    section("📂", "Data Upload")

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def sample_student_data():
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        'StudentID'  : range(1, n+1),
        'Name'       : [f'Student_{i:03d}' for i in range(1, n+1)],
        'Math'       : np.random.randint(30, 100, n).astype(float),
        'Science'    : np.random.randint(25, 100, n).astype(float),
        'English'    : np.random.randint(20, 100, n).astype(float),
        'History'    : np.random.randint(35, 100, n).astype(float),
        'Attendance' : np.random.uniform(50, 100, n).round(1),
        'Grade'      : np.random.choice(['A','B','C','D','F'], n,
                                         p=[0.2,0.3,0.25,0.15,0.1])
    })
    for col in ['Math','Science','English','History']:
        df.loc[np.random.choice(n,10,replace=False), col] = np.nan
    df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)
    return df

@st.cache_data(show_spinner=False)
def sample_retail_data():
    np.random.seed(42)
    n = 220
    df = pd.DataFrame({
        'InvoiceID': range(1, n+1),
        'StoreID': np.random.choice(['A','B','C','D'], n),
        'Product' : np.random.choice(['Widget','Gadget','Doohickey','Thingamabob'], n),
        'Quantity': np.random.randint(1, 12, n),
        'Price'   : np.round(np.random.uniform(5, 120, n), 2),
        'Revenue' : lambda d: np.round(d.Quantity * d.Price, 2)
    })
    df['Revenue'] = df['Quantity'] * df['Price']
    return df

@st.cache_data(show_spinner=False)
def sample_heart_data():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        'Age': np.random.randint(29, 78, n),
        'Sex': np.random.choice(['M','F'], n),
        'Cholesterol': np.random.randint(150, 310, n),
        'RestBP': np.random.randint(90, 180, n),
        'MaxHR': np.random.randint(90, 210, n),
        'Oldpeak': np.round(np.random.uniform(0, 6, n), 1),
        'HeartDisease': np.random.choice([0,1], n, p=[0.65,0.35])
    })
    return df

col_up, col_hint = st.columns([1.6, 1])

dataset_preset_options = [
    "📘 student_marks.csv",
    "📗 retail_store.csv",
    "📕 heart_disease.csv",
    "📙 Custom CSV Upload"
]

with col_up:
    st.markdown("### 🎯 Dataset Selection")
    st.markdown("Choose from our curated datasets or upload your own for instant analysis! 🚀")

    dataset_choice = st.selectbox(
        "Choose your dataset to analyze:",
        dataset_preset_options,
        index=0,
        help="Select from built-in sample datasets or upload your own CSV"
    )

    uploaded = st.file_uploader(
        "📤 Upload your CSV dataset (optional)",
        type=["csv"],
        help="If you select Custom CSV Upload, use this to upload your own dataset."
    )


with col_hint:
    # Interactive dataset preview cards
    if dataset_choice == "📘 student_marks.csv":
        pass
    elif dataset_choice == "📗 retail_store.csv":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a2e1a, #2e1a2e); border: 2px solid #34d399; border-radius: 12px; padding: 1.5rem; text-align: center; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -20px; right: -20px; width: 60px; height: 60px; background: rgba(52,211,153,0.15); border-radius: 50%;"></div>
            <h4 style="color: #ffffff !important; margin: 0 0 0.5rem 0;">📗 Retail Store Dataset</h4>
            <p style="color: #ffffff !important; margin: 0; font-size: 0.9rem;">Sales data with product information, quantities, prices, and revenue calculations across different stores.</p>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #34d399 !important;">
                <b>📊 Stats:</b> 220 transactions • 6 columns • Revenue calculations<br>
                <b>🎯 Perfect for:</b> Sales forecasting & Store performance analysis<br>
                <b>🔍 Key features:</b> Multi-store comparison, product category analysis
            </div>
        </div>""", unsafe_allow_html=True)
    elif dataset_choice == "📕 heart_disease.csv":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2e1a1a, #1a2e2e); border: 2px solid #f87171; border-radius: 12px; padding: 1.5rem; text-align: center; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -20px; right: -20px; width: 60px; height: 60px; background: rgba(248,113,113,0.15); border-radius: 50%;"></div>
            <h4 style="color: #ffffff !important; margin: 0 0 0.5rem 0;">📕 Heart Disease Dataset</h4>
            <p style="color: #ffffff !important; margin: 0; font-size: 0.9rem;">Medical data with patient demographics, cholesterol levels, blood pressure, and heart disease diagnosis.</p>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #f87171 !important;">
                <b>📊 Stats:</b> 300 patients • 7 columns • Medical indicators<br>
                <b>🎯 Perfect for:</b> Medical prediction & Health risk assessment<br>
                <b>🔍 Key features:</b> Risk factor analysis, demographic correlations
            </div>
        </div>""", unsafe_allow_html=True)
    elif dataset_choice == "📙 Custom CSV Upload":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2e1a2e, #1a1a2e); border: 2px solid #a78bfa; border-radius: 12px; padding: 1.5rem; text-align: center; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -20px; right: -20px; width: 60px; height: 60px; background: rgba(167,139,250,0.15); border-radius: 50%;"></div>
            <h4 style="color: #ffffff !important; margin: 0 0 0.5rem 0;">📙 Custom Dataset Upload</h4>
            <p style="color: #ffffff !important; margin: 0; font-size: 0.9rem;">Upload your own CSV file for analysis. The system will automatically detect numeric and categorical columns.</p>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #a78bfa !important;">
                <b>📊 Stats:</b> Any size • Auto-detection • Flexible analysis<br>
                <b>🎯 Perfect for:</b> Custom research & Specialized datasets<br>
                <b>🔍 Key features:</b> Universal compatibility, intelligent column detection
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); border: 2px solid #4f8ef7; border-radius: 12px; padding: 1.5rem; text-align: center; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -20px; right: -20px; width: 60px; height: 60px; background: rgba(79,142,247,0.15); border-radius: 50%;"></div>
            <h4 style="color: #ffffff !important; margin: 0 0 0.5rem 0;">🎯 Choose Your Dataset</h4>
            <p style="color: #ffffff !important; margin: 0; font-size: 0.9rem;">Select from built-in sample datasets or upload your own CSV file to begin the analysis journey.</p>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #7eb3ff !important;">
                <b>📊 Ready to analyze:</b> Student data, Sales data, Medical data, or Your custom CSV<br>
                <b>🚀 Features:</b> Auto-cleaning, Smart visualization, ML predictions
            </div>
        </div>""", unsafe_allow_html=True)



if uploaded is not None:
    with st.spinner("🔄 Reading and processing your CSV file..."):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
        df_raw = load_data(uploaded.read(), uploaded.name)
        progress_bar.empty()
    st.success(f"✅ **{uploaded.name}** loaded successfully! — {df_raw.shape[0]} rows × {df_raw.shape[1]} columns", icon="🎉")
    st.balloons()
else:
    if dataset_choice == "📘 student_marks.csv":
        with st.spinner("🎓 Loading Student Marks dataset..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            df_raw = sample_student_data()
            progress_bar.empty()
        st.success(f"✅ Loaded sample: {dataset_choice} — {df_raw.shape[0]} rows × {df_raw.shape[1]} columns", icon="📚")
        st.balloons()
    elif dataset_choice == "📗 retail_store.csv":
        with st.spinner("🛒 Loading Retail Store dataset..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            df_raw = sample_retail_data()
            progress_bar.empty()
        st.success(f"✅ Loaded sample: {dataset_choice} — {df_raw.shape[0]} rows × {df_raw.shape[1]} columns", icon="💰")
        st.balloons()
    elif dataset_choice == "📕 heart_disease.csv":
        with st.spinner("❤️ Loading Heart Disease dataset..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            df_raw = sample_heart_data()
            progress_bar.empty()
        st.success(f"✅ Loaded sample: {dataset_choice} — {df_raw.shape[0]} rows × {df_raw.shape[1]} columns", icon="🏥")
        st.balloons()
    elif dataset_choice == "📙 Custom CSV Upload":
        st.warning("⚠️ Please upload a CSV file when Custom CSV Upload is selected.")
        st.stop()
    else:
        st.warning("⚠️ Please select a dataset option or upload a CSV file.")
        st.stop()

# ─────────────────────────────────────────────────────────────────
# STEP 2 — DATA UNDERSTANDING
# ─────────────────────────────────────────────────────────────────
null_total = int(df_raw.isnull().sum().sum())
dup_total  = int(df_raw.duplicated().sum())
num_cols_raw, cat_cols_raw = detect_cols(df_raw)

if workflow_stage == "Data Understanding":
    section("🔍", "Data Understanding")

    metric_cards([
        (df_raw.shape[0], "Total Rows"),
        (df_raw.shape[1], "Total Columns"),
        (null_total,      "Missing Values"),
        (dup_total,       "Duplicate Rows"),
        (len(num_cols_raw), "Numeric Cols"),
        (len(cat_cols_raw), "Categorical Cols"),
    ])

    tab_head, tab_info, tab_desc, tab_null = st.tabs(
        ["📋 Preview", "🗂 Info", "📊 Statistics", "⚠️ Missing Values"])

    with tab_head:
        n_preview = st.slider("Rows to preview", 5, min(50, len(df_raw)), 10, key="preview_slider")
        st.dataframe(df_raw.head(n_preview), use_container_width=True)

    with tab_info:
        info_df = pd.DataFrame({
            'Column'   : df_raw.columns,
            'Dtype'    : df_raw.dtypes.values,
            'Non-Null' : df_raw.count().values,
            'Null'     : df_raw.isnull().sum().values,
            'Null %'   : (df_raw.isnull().mean()*100).round(2).values,
            'Sample'   : [str(df_raw[c].dropna().iloc[0]) if df_raw[c].notna().any() else 'N/A'
                          for c in df_raw.columns]
        })
        st.dataframe(info_df, use_container_width=True)

    with tab_desc:
        st.dataframe(df_raw.describe().round(3), use_container_width=True)

    with tab_null:
        null_df = df_raw.isnull().sum().reset_index()
        null_df.columns = ['Column', 'Missing Count']
        null_df['Missing %'] = (null_df['Missing Count'] / len(df_raw) * 100).round(2)
        null_df = null_df[null_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        if null_df.empty:
            st.success("✅ No missing values found!")
        else:
            st.dataframe(null_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(null_df['Column'], null_df['Missing %'],
                    color='#f87171', edgecolor='white')
            ax.set_xlabel('Missing %')
            ax.set_title('Missing Values per Column', fontweight='bold', color='#1e3a5f')
            ax.set_facecolor('#f8faff')
            fig.patch.set_facecolor('#ffffff')
            st.pyplot(fig, use_container_width=True)
            plt.close()

    insight("🔍", "Data Understanding Complete",
            f"Dataset has <b>{df_raw.shape[0]} rows</b> and <b>{df_raw.shape[1]} columns</b>. "
            f"Found <b>{null_total} missing values</b> and <b>{dup_total} duplicate rows</b> that need cleaning. "
            f"Detected <b>{len(num_cols_raw)} numeric</b> and <b>{len(cat_cols_raw)} categorical</b> columns automatically.")

    viva("What did you check in Data Understanding?",
         "I checked shape (rows × columns), data types of each column, count of missing values, "
         "and duplicate rows. I also used describe() to get statistical summary like mean, std, min, max.")


# ─────────────────────────────────────────────────────────────────
# STEP 3 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────

df = df_raw.copy()
cleaning_log = []

# Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
dupes_removed = before - len(df)
cleaning_log.append(("✅ Duplicates Removed", f"{dupes_removed} duplicate rows dropped"))

# Fill numeric nulls with mean
num_fills = {}
for col in df.select_dtypes(include=[np.number]).columns:
    n_null = df[col].isnull().sum()
    if n_null > 0:
        m = df[col].mean()
        df[col].fillna(m, inplace=True)
        num_fills[col] = (n_null, round(m, 2))
        cleaning_log.append(("✅ Numeric Null Filled", f"[{col}] — {n_null} nulls → filled with mean = {round(m,2)}"))

# Fill categorical nulls with mode
cat_fills = {}
for col in df.select_dtypes(include='object').columns:
    n_null = df[col].isnull().sum()
    if n_null > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        cat_fills[col] = (n_null, mode_val)
        cleaning_log.append(("✅ Categorical Null Filled", f"[{col}] — {n_null} nulls → filled with mode = '{mode_val}'"))

df.reset_index(drop=True, inplace=True)

if workflow_stage == "Data Cleaning":
    section("🧹", "Data Cleaning")

    st.markdown("""
    <div class="step-card">
        <h4>Why Data Cleaning? → Garbage In = Garbage Out</h4>
        <p>Raw data contains errors. If we train a model on dirty data, the results will be wrong.
        We fix: Missing values (fill with mean/mode) · Duplicate rows (remove) · Invalid entries.</p>
    </div>""", unsafe_allow_html=True)

    col_log, col_after = st.columns([1.2, 1])
    with col_log:
        st.markdown("#### 🗒 Cleaning Log")
        for status, detail in cleaning_log:
            st.markdown(f"""
            <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
                        padding:8px 12px;margin-bottom:6px;font-size:0.84rem'>
                <b style='color:#166534'>{status}</b><br>
                <span style='color:#ffffff'>{detail}</span>
            </div>""", unsafe_allow_html=True)

    with col_after:
        st.markdown("#### 📊 After Cleaning")
        metric_cards([
            (len(df), "Clean Rows"),
            (int(df.isnull().sum().sum()), "Nulls Remaining"),
            (int(df.duplicated().sum()), "Duplicates Left"),
        ])
        st.dataframe(df.head(8), use_container_width=True)

    csv_clean = df.to_csv(index=False).encode()
    st.download_button("⬇️  Download Cleaned Dataset", csv_clean,
                       "cleaned_dataset.csv", "text/csv", key="dl_clean")

    insight("🧹", "Cleaning Complete",
            f"Removed <b>{dupes_removed} duplicate rows</b>. "
            f"Filled <b>{len(num_fills)} numeric columns</b> with column mean. "
            f"Dataset is now clean with <b>{len(df)} rows</b> ready for analysis.")

    viva("Why did you use mean to fill missing values?",
         "Mean is the best representative value for normally distributed numeric data. "
         "It does not distort the overall distribution. For categorical columns, I used mode (most frequent value).")


# ─────────────────────────────────────────────────────────────────
# STEP 4 — DATA FILTERING
# ─────────────────────────────────────────────────────────────────
num_cols, cat_cols = detect_cols(df)

df_filtered = df.copy()

if workflow_stage == "Data Filtering":
    section("🔎", "Data Filtering")

    st.markdown("""
    <div class="step-card">
        <h4>Purpose: Focus on Useful Data Only</h4>
        <p>Filtering helps us extract rows that meet a specific condition — e.g. Marks > 60, 
        Revenue > Average. This way we analyze only the relevant subset.</p>
    </div>""", unsafe_allow_html=True)

    if num_cols:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_col = st.selectbox("Select column to filter", num_cols, key="filter_col")
        with col_f2:
            filter_op  = st.selectbox("Condition", [">", ">=", "<", "<=", "==", "!="], key="filter_op")
        with col_f3:
            col_min = float(df[filter_col].min())
            col_max = float(df[filter_col].max())
            col_mean= float(df[filter_col].mean())
            filter_val = st.number_input("Value", value=round(col_mean, 2),
                                         min_value=col_min, max_value=col_max, key="filter_val")

        ops = {'>': df[filter_col] > filter_val,
               '>=': df[filter_col] >= filter_val,
               '<': df[filter_col] < filter_val,
               '<=': df[filter_col] <= filter_val,
               '==': df[filter_col] == filter_val,
               '!=': df[filter_col] != filter_val}

        df_filtered = df[ops[filter_op]].reset_index(drop=True)

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.info(f"📌 Filter: **{filter_col} {filter_op} {filter_val}**")
            metric_cards([
                (len(df), "Original Rows"),
                (len(df_filtered), "Filtered Rows"),
                (len(df)-len(df_filtered), "Rows Removed"),
            ])
        with col_res2:
            pct = len(df_filtered)/len(df)*100
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(['Kept','Removed'],
                    [len(df_filtered), len(df)-len(df_filtered)],
                    color=['#4f8ef7','#f87171'], edgecolor='white', height=0.5)
            ax.set_xlabel('Rows')
            ax.set_facecolor('#f8faff')
            fig.patch.set_facecolor('#ffffff')
            ax.set_title(f'{pct:.1f}% rows match the filter', color='#1e3a5f', fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.dataframe(df_filtered.head(10), use_container_width=True)
        csv_filtered = df_filtered.to_csv(index=False).encode()
        st.download_button("⬇️  Download Filtered Dataset", csv_filtered,
                           "filtered_dataset.csv", "text/csv", key="dl_filter")
    else:
        st.warning("No numeric columns found for filtering.")

    viva("What is Data Filtering?",
         "Data filtering is selecting only the rows that satisfy a specific condition. "
         "Example: students who scored more than 60 marks. It helps us focus on meaningful data.")



# ─────────────────────────────────────────────────────────────────
# STEP 5 — EDA & ANALYSIS
# ─────────────────────────────────────────────────────────────────
if workflow_stage == "EDA & Analysis":
    section("📊", "Exploratory Data Analysis (EDA)")

    st.markdown("""
    <div class="step-card">
        <h4>Purpose: Understand Patterns in Data</h4>
        <p>EDA calculates mean, sum, standard deviation, and correlation to reveal hidden patterns
        before building any model.</p>
    </div>""", unsafe_allow_html=True)

    if num_cols:
        tab_stat, tab_corr, tab_cat = st.tabs(
            ["📐 Statistics", "🔗 Correlation", "🏷 Category Counts"])

        with tab_stat:
            stats_df = pd.DataFrame({
                'Mean'   : df[num_cols].mean().round(3),
                'Median' : df[num_cols].median().round(3),
                'Std Dev': df[num_cols].std().round(3),
                'Min'    : df[num_cols].min().round(3),
                'Max'    : df[num_cols].max().round(3),
                'Sum'    : df[num_cols].sum().round(2),
            })
            st.dataframe(stats_df, use_container_width=True)

            for col in num_cols[:3]:
                m = df[col].mean()
                s = df[col].std()
                cv = (s/m*100) if m != 0 else 0
                insight("📐", f"Column: {col}",
                        f"Mean = <b>{m:.2f}</b> | Std Dev = <b>{s:.2f}</b> | "
                        f"CV = <b>{cv:.1f}%</b> — "
                        f"{'Low variance (stable data)' if cv < 20 else 'High variance (spread out data)'}")

        with tab_corr:
            if len(num_cols) >= 2:
                corr_matrix = df[num_cols].corr()
                st.dataframe(corr_matrix.round(3), use_container_width=True)

                corr_pairs = []
                for i in range(len(num_cols)):
                    for j in range(i+1, len(num_cols)):
                        corr_pairs.append((num_cols[i], num_cols[j],
                                           round(corr_matrix.iloc[i,j], 3)))
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                st.markdown("**Top Column Pairs by Correlation:**")
                for a, b, r in corr_pairs[:5]:
                    color = "#166534" if abs(r) > 0.7 else "#92400e" if abs(r) > 0.4 else "#4a5568"
                    direction = "Strong positive" if r > 0.7 else "Strong negative" if r < -0.7 else \
                                "Moderate" if abs(r) > 0.4 else "Weak"
                    st.markdown(f"""
                    <div style='padding:6px 12px;margin:4px 0;background:#f8faff;
                                border-radius:8px;border:1px solid #e2e8f8;font-size:0.84rem'>
                        <b style='color:{color}'>{a} ↔ {b}</b>: r = {r}
                        &nbsp;·&nbsp; <span style='color:{color}'>{direction}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Need at least 2 numeric columns for correlation.")

        with tab_cat:
            if cat_cols:
                for col in cat_cols[:3]:
                    vc = df[col].value_counts().reset_index()
                    vc.columns = [col, 'Count']
                    vc['%'] = (vc['Count']/len(df)*100).round(1)
                    st.markdown(f"**{col}**")
                    st.dataframe(vc, use_container_width=True)
            else:
                st.info("No categorical columns detected.")

    viva("What is EDA?",
         "EDA — Exploratory Data Analysis — is the process of summarizing dataset characteristics "
         "using statistics and charts before modelling. It reveals patterns, outliers, and relationships.")


# ─────────────────────────────────────────────────────────────────
# STEP 6 — VISUALIZATION
# ─────────────────────────────────────────────────────────────────
if workflow_stage == "Visualization":
    section("📈", "Visualization")

    st.markdown("""
    <div class="step-card">
        <h4>Purpose: See Patterns Visually</h4>
        <p>Charts make patterns obvious. We generate: Correlation Heatmap · Histogram ·
        Scatter Plot · Line Chart · Box Plot · Bar Chart — automatically.</p>
    </div>""", unsafe_allow_html=True)

    PALETTE = ['#4f8ef7','#34d399','#f87171','#fbbf24','#a78bfa','#22d3ee','#f472b6','#fb923c']

    chart_tabs = st.tabs(["🔥 Heatmap", "📦 Histogram", "🔵 Scatter",
                           "📉 Line Chart", "📊 Bar Chart", "🎁 Box Plot"])

    with chart_tabs[0]:
        if len(num_cols) >= 2:
            st.markdown("**Correlation Heatmap** — Shows relationship between all numeric columns.")
            fig, ax = plt.subplots(figsize=(min(12, len(num_cols)*1.5+3),
                                            min(9, len(num_cols)*1.2+2)))
            mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=bool))
            sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
                        cmap='RdYlBu_r', center=0, mask=mask,
                        linewidths=0.5, ax=ax,
                        annot_kws={'size':9, 'weight':'bold'})
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold',
                         color='#1e3a5f', pad=12)
            ax.tick_params(colors='#4a5568', labelsize=9)
            fig.patch.set_facecolor('#ffffff')
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.info("💡 Values close to **+1** = strong positive correlation | **-1** = strong negative | **0** = no relation")
        else:
            st.warning("Need ≥ 2 numeric columns.")

    with chart_tabs[1]:
        st.markdown("**Histogram** — Shows the distribution (spread) of each numeric column.")
        hist_col = st.selectbox("Select column", num_cols, key="hist_col")
        bins_n   = st.slider("Number of bins", 5, 50, 15, key="hist_bins")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#ffffff')

        axes[0].hist(df[hist_col].dropna(), bins=bins_n,
                     color='#4f8ef7', edgecolor='white', alpha=0.85)
        axes[0].axvline(df[hist_col].mean(), color='#f87171', linewidth=2,
                        linestyle='--', label=f'Mean = {df[hist_col].mean():.2f}')
        axes[0].axvline(df[hist_col].median(), color='#34d399', linewidth=2,
                        linestyle='--', label=f'Median = {df[hist_col].median():.2f}')
        axes[0].set_title(f'Distribution of {hist_col}', fontweight='bold', color='#1e3a5f')
        axes[0].set_xlabel(hist_col)
        axes[0].set_ylabel('Frequency')
        axes[0].set_facecolor('#f8faff')
        axes[0].legend(fontsize=9)

        vals = df[hist_col].dropna()
        axes[1].hist(vals, bins=bins_n, color='#4f8ef7', edgecolor='white',
                     alpha=0.4, density=True)
        vals.plot.kde(ax=axes[1], color='#1e3a5f', linewidth=2.5)
        axes[1].set_title(f'KDE — {hist_col}', fontweight='bold', color='#1e3a5f')
        axes[1].set_facecolor('#f8faff')
        axes[1].set_xlabel(hist_col)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with chart_tabs[2]:
        st.markdown("**Scatter Plot** — Shows correlation between two numeric columns.")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sc_x = st.selectbox("X-axis", num_cols, index=0, key="sc_x")
        with col_s2:
            sc_y = st.selectbox("Y-axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
        with col_s3:
            sc_hue = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="sc_hue")

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8faff')

        if sc_hue != "None" and sc_hue in df.columns:
            cats = df[sc_hue].unique()
            for i, cat in enumerate(cats):
                sub = df[df[sc_hue] == cat]
                ax.scatter(sub[sc_x], sub[sc_y], label=str(cat),
                           color=PALETTE[i % len(PALETTE)], alpha=0.7, s=55, edgecolors='white', linewidth=0.5)
            ax.legend(title=sc_hue, fontsize=8)
        else:
            ax.scatter(df[sc_x], df[sc_y], color='#4f8ef7', alpha=0.7,
                       s=55, edgecolors='white', linewidth=0.5)

        valid = df[[sc_x, sc_y]].dropna()
        if len(valid) > 2:
            m, b = np.polyfit(valid[sc_x], valid[sc_y], 1)
            x_line = np.linspace(valid[sc_x].min(), valid[sc_x].max(), 100)
            ax.plot(x_line, m*x_line+b, color='#f87171', linewidth=2, linestyle='--', label='Trend line')

        r_val = df[[sc_x, sc_y]].dropna().corr().iloc[0,1]
        ax.set_title(f'Scatter: {sc_x} vs {sc_y}  (r = {r_val:.3f})',
                     fontweight='bold', color='#1e3a5f')
        ax.set_xlabel(sc_x)
        ax.set_ylabel(sc_y)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with chart_tabs[3]:
        st.markdown("**Line Chart** — Shows trends over records/index.")
        line_cols = st.multiselect("Select columns to plot", num_cols,
                                    default=num_cols[:min(3, len(num_cols))], key="line_cols")
        line_max  = st.slider("Max rows", 20, min(200, len(df)), min(100, len(df)), key="line_max")

        if line_cols:
            fig, ax = plt.subplots(figsize=(11, 4))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8faff')
            for i, col in enumerate(line_cols):
                ax.plot(df[col].head(line_max).values,
                        color=PALETTE[i % len(PALETTE)],
                        linewidth=2, label=col, alpha=0.85)
            ax.set_title('Line Chart — Trends', fontweight='bold', color='#1e3a5f')
            ax.set_xlabel('Record Index')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with chart_tabs[4]:
        st.markdown("**Bar Chart** — Compare numeric values across categories.")
        if cat_cols and num_cols:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                bar_cat = st.selectbox("Category (X)", cat_cols, key="bar_cat")
            with col_b2:
                bar_num = st.selectbox("Value (Y)", num_cols, key="bar_num")

            bar_agg = st.radio("Aggregate by", ["Mean", "Sum", "Count"], horizontal=True, key="bar_agg")
            agg_map = {"Mean": "mean", "Sum": "sum", "Count": "count"}
            grouped = df.groupby(bar_cat)[bar_num].agg(agg_map[bar_agg]).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(max(7, len(grouped)*0.9), 4))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8faff')
            bars = ax.bar(grouped.index.astype(str), grouped.values,
                          color=[PALETTE[i % len(PALETTE)] for i in range(len(grouped))],
                          edgecolor='white', width=0.65)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+grouped.max()*0.01,
                        f'{bar.get_height():.1f}',
                        ha='center', va='bottom', fontsize=8, color='#4a5568')
            ax.set_title(f'{bar_agg} of {bar_num} by {bar_cat}',
                         fontweight='bold', color='#1e3a5f')
            ax.set_xlabel(bar_cat)
            ax.set_ylabel(f'{bar_agg} of {bar_num}')
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Need at least one categorical and one numeric column.")

    with chart_tabs[5]:
        st.markdown("**Box Plot** — Shows distribution, median, and outliers.")
        box_cols = st.multiselect("Select columns", num_cols,
                                   default=num_cols[:min(4, len(num_cols))], key="box_cols")
        if box_cols:
            fig, ax = plt.subplots(figsize=(max(7, len(box_cols)*1.5), 5))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8faff')
            bp = ax.boxplot([df[c].dropna() for c in box_cols],
                            labels=box_cols, patch_artist=True,
                            medianprops=dict(color='#f87171', linewidth=2))
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(PALETTE[i % len(PALETTE)] + '55')
                patch.set_edgecolor(PALETTE[i % len(PALETTE)])
            ax.set_title('Box Plot — Outlier Detection', fontweight='bold', color='#1e3a5f')
            ax.tick_params(axis='x', rotation=30)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.info("💡 Dots outside whiskers = **Outliers**. The line inside box = **Median**.")

    viva("What charts did you create and why?",
         "I created: Heatmap (to see correlations), Histogram (distribution), Scatter Plot (relationship "
         "between two variables with trend line), Line Chart (trends), Bar Chart (category comparison), "
         "Box Plot (outlier detection). Each chart serves a specific analytical purpose.")


# ─────────────────────────────────────────────────────────────────
# STEP 7 — ML PREDICTION
# ─────────────────────────────────────────────────────────────────
if workflow_stage == "ML Prediction":
    section("🤖", "ML Prediction")

    st.markdown("""
    <div class="step-card">
        <h4>ML Models Applied Automatically</h4>
        <p><b>Linear Regression</b> → Predicts a numeric value (e.g., predict Science marks from Math)<br>
        <b>Logistic Regression</b> → Classifies into categories (e.g., predict Grade A/B/C/D/F)</p>
    </div>""", unsafe_allow_html=True)

    ml_tabs = st.tabs(["📈 Linear Regression", "🏷 Logistic Regression"])

    # dataset-specific default columns for ML models
    default_lr_x, default_lr_y = None, None
    default_lo_target, default_lo_feats = None, None

    if dataset_choice == "📘 student_marks.csv":
        default_lr_x, default_lr_y = "Math", "Science"
        default_lo_target = "Grade"
        default_lo_feats = [c for c in num_cols if c not in ['StudentID']]

    elif dataset_choice == "📗 retail_store.csv":
        default_lr_x, default_lr_y = "Quantity", "Revenue"
        # no native categorical target; logistic can target StoreID/Product if needed
        default_lo_target = "StoreID" if 'StoreID' in cat_cols else None
        default_lo_feats = [c for c in num_cols if c not in ['Revenue', 'InvoiceID']]

    elif dataset_choice == "📕 heart_disease.csv":
        default_lr_x, default_lr_y = "Age", "Cholesterol"
        if 'HeartDisease' in num_cols and 'HeartDisease' not in cat_cols:
            cat_cols.append('HeartDisease')
        default_lo_target = "HeartDisease"
        default_lo_feats = [c for c in num_cols if c != 'HeartDisease']

    else:
        # Custom dataset: choices will be user-selected from existing columns
        default_lr_x, default_lr_y = (num_cols[0], num_cols[1]) if len(num_cols) >= 2 else (None, None)
        default_lo_target = cat_cols[0] if cat_cols else None
        default_lo_feats = num_cols[:min(3, len(num_cols))]

if workflow_stage == "ML Prediction":
    # ── 7.1 LINEAR REGRESSION ────────────────────────────────────
    with ml_tabs[0]:
        if len(num_cols) >= 2:
            col_lr1, col_lr2 = st.columns(2)
            with col_lr1:
                x_index = 0
                if default_lr_x in num_cols:
                    x_index = num_cols.index(default_lr_x)
                lr_x = st.selectbox("Feature (X)", num_cols, index=x_index, key="lr_x")
            with col_lr2:
                lr_y_options = [c for c in num_cols if c != lr_x]
                y_index = 0
                if default_lr_y in lr_y_options:
                    y_index = lr_y_options.index(default_lr_y)
                lr_y = st.selectbox("Target (Y) — what to predict", lr_y_options,
                                    index=y_index, key="lr_y")
            test_size = st.slider("Test set size (%)", 10, 40, 20, key="lr_test") / 100

            if st.button("▶  Run Linear Regression", type="primary", key="run_lr"):
                with st.spinner("Training model..."):
                    valid_df = df[[lr_x, lr_y]].dropna()
                    X = valid_df[[lr_x]].values
                    y = valid_df[lr_y].values

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42)

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse  = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2   = r2_score(y_test, y_pred)

                metric_cards([
                    (f"{r2:.4f}", "R² Score"),
                    (f"{mse:.3f}", "MSE"),
                    (f"{rmse:.3f}", "RMSE"),
                    (f"{model.coef_[0]:.4f}", "Coefficient"),
                    (f"{model.intercept_:.4f}", "Intercept"),
                ])

                if r2 >= 0.8:
                    r2_msg = "🟢 Excellent model fit!"
                elif r2 >= 0.6:
                    r2_msg = "🟡 Good model fit."
                elif r2 >= 0.4:
                    r2_msg = "🟠 Moderate fit — try more features."
                else:
                    r2_msg = "🔴 Weak fit — columns may not be linearly related."
                st.info(f"**R² Interpretation:** {r2_msg}")

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.patch.set_facecolor('#ffffff')

                ax = axes[0]
                ax.scatter(X_test, y_test, color='#4f8ef7', s=55, alpha=0.7,
                           edgecolors='white', label='Actual')
                ax.scatter(X_test, y_pred, color='#f87171', s=55, alpha=0.7,
                           edgecolors='white', marker='^', label='Predicted')
                x_line = np.linspace(X.min(), X.max(), 200)
                ax.plot(x_line, model.predict(x_line.reshape(-1,1)),
                        color='#1e3a5f', linewidth=2.5, linestyle='--', label='Regression Line')
                ax.set_xlabel(lr_x)
                ax.set_ylabel(lr_y)
                ax.set_title(f'Linear Regression\n{lr_x} → {lr_y}',
                             fontweight='bold', color='#1e3a5f')
                ax.legend(fontsize=8)
                ax.set_facecolor('#f8faff')

                ax2 = axes[1]
                residuals = y_test - y_pred
                ax2.scatter(y_pred, residuals, color='#a78bfa', alpha=0.7,
                            s=55, edgecolors='white')
                ax2.axhline(0, color='#f87171', linestyle='--', linewidth=2)
                ax2.set_xlabel('Predicted Values')
                ax2.set_ylabel('Residuals (Actual - Predicted)')
                ax2.set_title('Residual Plot', fontweight='bold', color='#1e3a5f')
                ax2.set_facecolor('#f8faff')

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.markdown(f"""
                <div class="code-box">Regression Equation:
{lr_y}  =  {model.coef_[0]:.4f} × {lr_x}  +  {model.intercept_:.4f}

Train samples : {len(X_train)}
Test  samples : {len(X_test)}
R² Score      : {r2:.4f}  →  Model explains {r2*100:.1f}% of variance</div>""",
                unsafe_allow_html=True)

                st.markdown("#### 🔮 Make a Prediction")
                user_val = st.number_input(
                    f"Enter {lr_x} value to predict {lr_y}",
                    value=float(df[lr_x].mean()), key="lr_predict_val")
                pred_result = model.predict([[user_val]])[0]
                st.success(f"If **{lr_x} = {user_val}** → Predicted **{lr_y} = {pred_result:.2f}**")
        else:
            st.warning("Need at least 2 numeric columns for Linear Regression.")

    # ── 7.2 LOGISTIC REGRESSION ──────────────────────────────────
    with ml_tabs[1]:
        if len(num_cols) >= 1:
            col_lo1, col_lo2 = st.columns(2)
            with col_lo1:
                lo_target_options = list(cat_cols)
                if default_lo_target and default_lo_target in df.columns and default_lo_target not in lo_target_options:
                    lo_target_options.append(default_lo_target)
                if not lo_target_options:
                    st.warning("No categorical or discrete target available for Logistic Regression on this dataset.")
                    lo_target = None
                else:
                    default_target_index = 0
                    if default_lo_target in lo_target_options:
                        default_target_index = lo_target_options.index(default_lo_target)
                    lo_target = st.selectbox("Target column (categorical)", lo_target_options,
                                              index=default_target_index, key="lo_target")

            with col_lo2:
                default_lo_feats = [c for c in (default_lo_feats or []) if c in num_cols]
                if not default_lo_feats:
                    default_lo_feats = num_cols[:min(3, len(num_cols))]
                lo_feats = st.multiselect("Feature columns (numeric)",
                                           num_cols, default=default_lo_feats,
                                           key="lo_feats")
            lo_test = st.slider("Test set size (%)", 10, 40, 20, key="lo_test") / 100

            if lo_target and lo_feats and st.button("▶  Run Logistic Regression", type="primary", key="run_lo"):
                with st.spinner("Training classifier..."):
                    sub = df[lo_feats + [lo_target]].dropna()
                    le  = LabelEncoder()
                    y   = le.fit_transform(sub[lo_target])
                    X   = sub[lo_feats].values

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=lo_test, random_state=42)

                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc    = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred,
                                                   target_names=le.classes_, zero_division=0)
                    cm     = confusion_matrix(y_test, y_pred)

                metric_cards([
                    (f"{acc*100:.2f}%", "Accuracy"),
                    (len(X_train), "Train Samples"),
                    (len(X_test), "Test Samples"),
                    (len(le.classes_), "Classes"),
                ])

                if acc >= 0.85:
                    st.success(f"🟢 High accuracy — {acc*100:.1f}%! Model classifies well.")
                elif acc >= 0.65:
                    st.warning(f"🟡 Moderate accuracy — {acc*100:.1f}%. Acceptable for this dataset.")
                else:
                    st.error(f"🔴 Low accuracy — {acc*100:.1f}%. Try adding more features.")

                col_cm, col_rep = st.columns(2)

                with col_cm:
                    st.markdown("**Confusion Matrix**")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    fig.patch.set_facecolor('#ffffff')
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=le.classes_, yticklabels=le.classes_,
                                ax=ax, linewidths=0.5)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix', fontweight='bold', color='#1e3a5f')
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                    st.caption("Diagonal = Correct predictions | Off-diagonal = Errors")

                with col_rep:
                    st.markdown("**Classification Report**")
                    st.code(report, language=None)
        else:
            st.warning("Need at least one categorical column and one numeric column.")

    viva("What ML models did you use and what is the difference?",
     "I used Linear Regression to predict continuous values (e.g., marks) and Logistic Regression "
     "to classify into categories (e.g., Grade A/B/C). Linear uses R² to evaluate; "
     "Logistic uses Accuracy and Confusion Matrix. Both use 80/20 train-test split.")


# ─────────────────────────────────────────────────────────────────
# STEP 8 — RESULT SUMMARY
# ─────────────────────────────────────────────────────────────────
if workflow_stage == "Result Summary":
    section("🧾", "Result Summary & Conclusion")

    col_sum1, col_sum2 = st.columns(2)

    with col_sum1:
        st.markdown("#### 📊 Dataset Summary")
        summary_data = {
            'Metric': ['Original Rows', 'Original Columns', 'Missing Values (raw)',
                       'Duplicates Removed', 'Clean Rows', 'Numeric Columns', 'Categorical Columns'],
            'Value': [df_raw.shape[0], df_raw.shape[1], null_total,
                      dupes_removed, len(df),
                      len(num_cols), len(cat_cols)]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    with col_sum2:
        st.markdown("#### 📁 Downloads")
        st.download_button("⬇️  Download Cleaned Dataset (CSV)",
                           df.to_csv(index=False).encode(),
                           "cleaned_dataset.csv", "text/csv")

        if num_cols:
            stats_excel = df[num_cols].describe().round(3)
            st.download_button("⬇️  Download Statistics (CSV)",
                               stats_excel.to_csv().encode(),
                               "statistics_summary.csv", "text/csv")

        if 'df_filtered' in locals() and len(df_filtered) > 0:
            st.download_button("⬇️  Download Filtered Dataset (CSV)",
                               df_filtered.to_csv(index=False).encode(),
                               "filtered_dataset.csv", "text/csv")

    st.markdown("---")

    st.markdown("#### 💡 Automatic Insights")
    if num_cols:
        for col in num_cols[:4]:
            vals = df[col].dropna()
            mean, std = vals.mean(), vals.std()
            cv = std/mean*100 if mean != 0 else 0
            max_val, min_val = vals.max(), vals.min()
            insight("📌", f"{col} — Key Findings",
                    f"Range: <b>{min_val:.2f} – {max_val:.2f}</b> | "
                    f"Mean: <b>{mean:.2f}</b> | Std: <b>{std:.2f}</b> | "
                    f"Variation: <b>{cv:.1f}%</b> — "
                    f"{'Stable/consistent' if cv < 20 else 'High variation detected'}")

    st.markdown("---")
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e3a5f,#0f2340);
                border-radius:16px;padding:2rem 2.5rem;color:white;text-align:center;
                border:1px solid rgba(79,142,247,0.3)'>
        <h2 style='margin:0 0 0.75rem;font-size:1.4rem'>🎯 Conclusion</h2>
        <p style='color:rgba(255,255,255,0.8);font-size:1rem;line-height:1.8;max-width:700px;margin:0 auto'>
        This project — <b>Universal Data Analyzer</b> — automates the complete data science pipeline.<br>
        It accepts any CSV dataset and performs<br>
        <b>Data Cleaning → Filtering → EDA → Visualization → ML Prediction</b><br>
        without writing dataset-specific code.<br><br>
        <i>"A smart Python-based system that converts raw data into meaningful insights automatically."</i>
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🎤 Viva Preparation")
    viva("What is your project?",
         "My project is a Universal Data Analyzer built using Python and Streamlit. "
         "Any CSV dataset can be uploaded and the system automatically performs data cleaning, "
         "filtering, exploratory data analysis, visualization using Matplotlib and Seaborn, "
         "and machine learning prediction using Linear and Logistic Regression.")
    viva("Why did you use Streamlit?",
         "Streamlit is a Python library that converts Python scripts into interactive web applications. "
         "It allowed me to build a user-friendly interface where users can upload files, select options, "
         "and see results — all without any web development knowledge.")
    viva("What makes your project universal?",
         "The code uses dynamic column detection — it automatically identifies numeric and categorical "
         "columns from any dataset. No column names are hardcoded. It works for student, sales, "
         "medical, or any other CSV dataset.")