import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ==========================================
# PAGE CONFIGURATION & MODERN STYLING
# ==========================================
st.set_page_config(
    page_title="Bank Term Deposit Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Remove default Streamlit header/footer padding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
    }
    
    /* Custom Info & Callout Cards */
    .feature-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    .leakage-banner {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 1px solid #F59E0B;
        border-left: 5px solid #D97706;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1.2rem 0;
    }
    
    .score-badge-blue {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .score-badge-green {
        background-color: #ECFDF5;
        color: #047857;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Form & Action Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 500;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# FILE PATH RESOLUTION & CACHED LOADERS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(relative_paths):
    for rel_path in relative_paths:
        full_path = os.path.abspath(os.path.join(BASE_DIR, rel_path))
        if os.path.exists(full_path):
            return full_path
    return None

DATA_PATH = find_file([
    os.path.join('..', 'dataSet', 'bank-full.csv'),
    os.path.join('dataSet', 'bank-full.csv'),
    os.path.join('..', '..', 'dataSet', 'bank-full.csv')
])

MODEL_PATH = find_file([
    os.path.join('model', 'final_bank_model_smotenc.pkl'),
    os.path.join('..', 'model', 'final_bank_model_smotenc.pkl')
])

COLS_PATH = find_file([
    os.path.join('model', 'final_feature_columns.pkl'),
    os.path.join('..', 'model', 'final_feature_columns.pkl')
])

@st.cache_data
def load_and_preprocess_data():
    if not DATA_PATH or not os.path.exists(DATA_PATH):
        st.error("Error: Could not locate `bank-full.csv` dataset.")
        return None
    df = pd.read_csv(DATA_PATH, sep=';')
    
    # Feature Engineering matching Notebook EDA
    age_bins = [18, 25, 35, 45, 55, 65, 100]
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=['18-25', '25-35', '35-45', '45-55', '55-65', '65+'], right=False)
    df['y_numeric'] = df['y'].map({'no': 0, 'yes': 1})
    return df

@st.cache_resource
def load_production_model():
    if MODEL_PATH and COLS_PATH and os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
        try:
            pipe = joblib.load(MODEL_PATH)
            cols = joblib.load(COLS_PATH)
            return pipe, cols
        except Exception as e:
            return None, None
    return None, None

df_raw = load_and_preprocess_data()
model_pipeline, feature_cols = load_production_model()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("## 🏦 **Term Deposit AI**")
    st.markdown("<p style='font-size:0.9rem; color:#64748B;'>Advanced Bank Marketing Analytics & Targeted Machine Learning Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "📊 Visual EDA & Market Insights",
            "📈 Model Scores & Performance Analytics",
            "🎯 Customer Prediction Engine",
            "💼 Campaign ROI Simulator"
        ]
    )
    
    st.markdown("---")
    if model_pipeline is not None:
        st.success("✅ Production Model Active: `GradientBoosting + SMOTENC` (Realistic, No Duration)")
    else:
        st.warning("⚠️ Using fallback predictions.")

# ==========================================
# MODULE 1: COMPREHENSIVE VISUAL EDA
# ==========================================
if page == "📊 Visual EDA & Market Insights":
    st.title("Comprehensive Exploratory Data Analysis (EDA)")
    st.markdown("Deep dive into customer demographics, behavioral dynamics, and marketing campaign performance derived directly from historical data.")
    
    if df_raw is not None:
        # High Level KPIs
        total_customers = len(df_raw)
        total_subscribers = (df_raw['y'] == 'yes').sum()
        conv_rate = (total_subscribers / total_customers) * 100
        avg_balance = df_raw['balance'].mean()
        avg_duration = df_raw['duration'].mean()
        
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.metric("Total Prospects", f"{total_customers:,}")
        with kpi2:
            st.metric("Total Subscribed", f"{total_subscribers:,}")
        with kpi3:
            st.metric("Baseline Conversion", f"{conv_rate:.2f}%")
        with kpi4:
            st.metric("Average Balance", f"€{avg_balance:,.0f}")
        with kpi5:
            st.metric("Avg Call Duration", f"{avg_duration:.0f} sec")
            
        st.markdown("---")
        
        # Tabs for full Visual EDA breakdown
        eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
            "🎯 1. Target & Demographics",
            "💰 2. Financial Profiles",
            "📞 3. Campaign & Contact Dynamics",
            "🔬 4. Correlation & Unknowns Audit"
        ])
        
        # TAB 1: Target & Demographics
        with eda_tab1:
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Class Imbalance: Target Distribution")
                target_counts = df_raw['y'].value_counts().reset_index()
                target_counts.columns = ['Subscribed', 'Count']
                target_counts['Percentage'] = (target_counts['Count'] / total_customers) * 100
                
                fig_target = px.pie(
                    target_counts, values='Count', names='Subscribed',
                    color='Subscribed', color_discrete_map={'no': '#EF4444', 'yes': '#10B981'},
                    hole=0.45,
                    hover_data=['Percentage']
                )
                fig_target.update_traces(textposition='inside', textinfo='percent+label')
                fig_target.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
                st.plotly_chart(fig_target, use_container_width=True)
                st.caption("Heavily imbalanced dataset: 88.3% 'No' vs 11.7% 'Yes', requiring SMOTENC synthesis.")
                
            with c2:
                st.subheader("Conversion Rate by Age Group")
                age_conv = (pd.crosstab(df_raw['age_group'], df_raw['y'], normalize='index')['yes'] * 100).reset_index()
                age_conv.columns = ['Age Group', 'Conversion Rate (%)']
                
                fig_age = px.bar(
                    age_conv, x='Age Group', y='Conversion Rate (%)',
                    color='Conversion Rate (%)', color_continuous_scale='Teal',
                    text_auto='.1f'
                )
                fig_age.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
                st.plotly_chart(fig_age, use_container_width=True)
                st.caption("Key Insight: Senior citizens (65+) and young adults (18-25) exhibit the highest conversion propensity.")
                
            st.markdown("---")
            
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Subscription Rate by Job Sector")
                job_conv = (pd.crosstab(df_raw['job'], df_raw['y'], normalize='index')['yes'] * 100).reset_index()
                job_conv.columns = ['Job Sector', 'Conversion Rate (%)']
                job_conv = job_conv.sort_values(by='Conversion Rate (%)', ascending=True)
                
                fig_job = px.bar(
                    job_conv, x='Conversion Rate (%)', y='Job Sector', orientation='h',
                    color='Conversion Rate (%)', color_continuous_scale='Viridis',
                    text_auto='.1f'
                )
                fig_job.update_layout(height=420, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
                st.plotly_chart(fig_job, use_container_width=True)
                st.caption("Students (28.7%) and Retirees (22.8%) convert at more than double the national baseline.")
                
            with c4:
                st.subheader("Marital & Education Cross-Analysis")
                edu_conv = (pd.crosstab(df_raw['education'], df_raw['y'], normalize='index')['yes'] * 100).reset_index()
                edu_conv.columns = ['Education', 'Conversion Rate (%)']
                
                mar_conv = (pd.crosstab(df_raw['marital'], df_raw['y'], normalize='index')['yes'] * 100).reset_index()
                mar_conv.columns = ['Marital Status', 'Conversion Rate (%)']
                
                fig_edu_mar = make_subplots(rows=2, cols=1, subplot_titles=("Conversion by Education Level", "Conversion by Marital Status"))
                fig_edu_mar.add_trace(go.Bar(x=edu_conv['Education'], y=edu_conv['Conversion Rate (%)'], marker_color='#3B82F6', name='Education'), row=1, col=1)
                fig_edu_mar.add_trace(go.Bar(x=mar_conv['Marital Status'], y=mar_conv['Conversion Rate (%)'], marker_color='#8B5CF6', name='Marital'), row=2, col=1)
                fig_edu_mar.update_layout(height=420, margin=dict(t=30, b=20, l=20, r=20), showlegend=False)
                st.plotly_chart(fig_edu_mar, use_container_width=True)
                st.caption("Tertiary educated and single individuals convert at noticeably higher rates.")

        # TAB 2: Financial Profiles
        with eda_tab2:
            st.subheader("Financial Assets & Debt Indicators")
            
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.markdown("#### Account Balance vs Subscription")
                # Sample 5000 rows for smooth boxplot visualization
                sample_df = df_raw.sample(min(5000, len(df_raw)), random_state=67)
                fig_bal = px.box(
                    sample_df, x='y', y='balance', color='y',
                    color_discrete_map={'no': '#EF4444', 'yes': '#10B981'},
                    labels={'y': 'Subscribed', 'balance': 'Yearly Balance (€)'},
                    points=False
                )
                fig_bal.update_yaxes(range=[-1000, 10000])
                fig_bal.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
                st.plotly_chart(fig_bal, use_container_width=True)
                st.caption("Customers who subscribe possess higher average and median balances across all age segments.")
                
            with f_col2:
                st.markdown("#### The Debt Burden Effect")
                loan_metrics = []
                for feat, name in [('housing', 'Housing Loan'), ('loan', 'Personal Loan'), ('default', 'Credit in Default')]:
                    ct = pd.crosstab(df_raw[feat], df_raw['y'], normalize='index')['yes'] * 100
                    loan_metrics.append({'Category': f"{name}: No", 'Conversion Rate (%)': ct.get('no', 0)})
                    loan_metrics.append({'Category': f"{name}: Yes", 'Conversion Rate (%)': ct.get('yes', 0)})
                    
                loan_df = pd.DataFrame(loan_metrics)
                fig_loan = px.bar(
                    loan_df, x='Category', y='Conversion Rate (%)',
                    color='Category', color_discrete_sequence=['#10B981', '#EF4444', '#10B981', '#EF4444', '#10B981', '#EF4444'],
                    text_auto='.1f'
                )
                fig_loan.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
                st.plotly_chart(fig_loan, use_container_width=True)
                st.caption("Critical Finding: Having a housing or personal loan cuts conversion probability by more than 50%!")

        # TAB 3: Campaign & Contact Dynamics
        with eda_tab3:
            st.markdown("""
            <div class="leakage-banner">
                <h4 style="margin:0 0 6px 0; color:#92400E;">⚠️ Critical Discovery: Data Leakage via 'Contact Duration'</h4>
                <p style="margin:0; font-size:0.92rem; color:#78350F;">
                    In initial EDA, <b>call duration</b> correlates extremely strongly with subscriptions (+0.39). However, a call's duration is only known <i>after</i> placing the call! 
                    Using duration in a real-world predictive model is <b>Data Leakage</b> and cannot be used before dialing. In our final production model, <b>duration is removed</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            c_row1, c_row2 = st.columns(2)
            with c_row1:
                st.subheader("Call Duration Disparity (Data Leakage)")
                sample_dur = df_raw.sample(min(4000, len(df_raw)), random_state=67)
                fig_dur = px.box(
                    sample_dur, x='y', y='duration', color='y',
                    color_discrete_map={'no': '#EF4444', 'yes': '#10B981'},
                    labels={'y': 'Subscribed', 'duration': 'Call Duration (seconds)'},
                    points=False
                )
                fig_dur.update_yaxes(range=[0, 1500])
                fig_dur.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
                st.plotly_chart(fig_dur, use_container_width=True)
                st.caption("Subscribers average 537 seconds on the phone vs 221 seconds for non-subscribers.")
                
            with c_row2:
                st.subheader("Diminishing Returns on Campaign Contacts")
                camp_conv = df_raw[df_raw['campaign'] <= 10].groupby('campaign')['y_numeric'].agg(['count', 'mean']).reset_index()
                camp_conv['Conversion Rate (%)'] = camp_conv['mean'] * 100
                
                fig_camp = px.line(
                    camp_conv, x='campaign', y='Conversion Rate (%)',
                    markers=True, line_shape='spline',
                    labels={'campaign': 'Number of Contacts During Campaign'}
                )
                fig_camp.update_traces(line_color='#2563EB', line_width=3, marker=dict(size=8, color='#1E40AF'))
                fig_camp.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_camp, use_container_width=True)
                st.caption("Conversion peaks on contacts 1-2 (14.6% to 11.6%) and plummets rapidly with further dialing.")
                
            st.markdown("---")
            c_row3, c_row4 = st.columns(2)
            with c_row3:
                st.subheader("Seasonal Targeting: Conversion by Month")
                month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
                month_conv = (pd.crosstab(df_raw['month'], df_raw['y'], normalize='index')['yes'] * 100).reindex(month_order).reset_index()
                month_conv.columns = ['Month', 'Conversion Rate (%)']
                
                fig_month = px.bar(
                    month_conv, x='Month', y='Conversion Rate (%)',
                    color='Conversion Rate (%)', color_continuous_scale='Blues',
                    text_auto='.1f'
                )
                fig_month.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
                st.plotly_chart(fig_month, use_container_width=True)
                st.caption("Off-peak campaign months (March: 52%, Sept: 46%, Oct: 44%, Dec: 47%) yield massive success compared to high-volume May (6.7%).")
                
            with c_row4:
                st.subheader("Impact of Previous Campaign Outcome")
                poutcome_conv = (pd.crosstab(df_raw['poutcome'], df_raw['y'], normalize='index')['yes'] * 100).reset_index()
                poutcome_conv.columns = ['Previous Outcome', 'Conversion Rate (%)']
                poutcome_conv = poutcome_conv.sort_values(by='Conversion Rate (%)', ascending=False)
                
                fig_pout = px.bar(
                    poutcome_conv, x='Previous Outcome', y='Conversion Rate (%)',
                    color='Conversion Rate (%)', color_continuous_scale='Greens',
                    text_auto='.1f'
                )
                fig_pout.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
                st.plotly_chart(fig_pout, use_container_width=True)
                st.caption("Customers who agreed in previous campaigns convert at a staggering 64.7%!")

        # TAB 4: Correlation Matrix & Unknowns Audit
        with eda_tab4:
            corr_c1, corr_c2 = st.columns([3, 2])
            
            with corr_c1:
                st.subheader("Numerical Features Correlation Matrix")
                num_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y_numeric']
                corr_matrix = df_raw[num_features].corr()
                
                fig_corr = px.imshow(
                    corr_matrix, text_auto='.2f', aspect='auto',
                    color_continuous_scale='RdBu_r', zmin=-0.5, zmax=0.5
                )
                fig_corr.update_layout(height=420, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_corr, use_container_width=True)
                
            with corr_c2:
                st.subheader("Missing & 'Unknown' Value Audit")
                unknown_cols = ["contact", "education", "job", "poutcome"]
                unknown_data = []
                for col in unknown_cols:
                    cnt = (df_raw[col] == "unknown").sum()
                    pct = (cnt / total_customers) * 100
                    unknown_data.append({'Feature': col, 'Unknown Count': cnt, 'Unknown (%)': pct})
                
                unknown_df = pd.DataFrame(unknown_data).sort_values(by='Unknown Count', ascending=False)
                st.dataframe(
                    unknown_df.style.format({'Unknown Count': '{:,}', 'Unknown (%)': '{:.1f}%'}),
                    use_container_width=True
                )
                st.markdown("""
                * **No NaNs Found:** The dataset has 0 standard missing values.
                * **Literal 'unknown's Retained:** Instead of imputing arbitrary modes, 'unknown' is retained as an informative category in OneHotEncoding.
                """)

# ==========================================
# MODULE 2: MODEL SCORE & PERFORMANCE ANALYTICS
# ==========================================
elif page == "📈 Model Scores & Performance Analytics":
    st.title("Model Performance & Evaluation Analytics")
    st.markdown("Detailed breakdown of machine learning algorithms evaluated across Baseline, SMOTENC Balancing, and Realistic (No-Duration) Cross-Validation.")
    
    # ----------------------------------------------------
    # HARD DATA RECORDED DIRECTLY FROM THE EXPERIMENTAL PIPELINE
    # ----------------------------------------------------
    baseline_scores = pd.DataFrame([
        {"Model": "Random Forest", "Accuracy": 0.9084, "Precision": 0.6634, "Recall": 0.4414, "F1 Score": 0.5301},
        {"Model": "Gradient Boosting", "Accuracy": 0.9082, "Precision": 0.6633, "Recall": 0.4376, "F1 Score": 0.5273},
        {"Model": "Logistic Regression", "Accuracy": 0.9058, "Precision": 0.6807, "Recall": 0.3667, "F1 Score": 0.4767},
        {"Model": "KNN", "Accuracy": 0.8959, "Precision": 0.5913, "Recall": 0.3582, "F1 Score": 0.4461},
        {"Model": "Decision Tree", "Accuracy": 0.8737, "Precision": 0.4625, "Recall": 0.4896, "F1 Score": 0.4757}
    ])
    
    smote_scores = pd.DataFrame([
        {"Model": "Random Forest", "CV Accuracy": 0.8854, "CV Precision": 0.5078, "CV Recall": 0.6852, "CV F1 Score": 0.5831},
        {"Model": "Gradient Boosting", "CV Accuracy": 0.8568, "CV Precision": 0.4378, "CV Recall": 0.7866, "CV F1 Score": 0.5625},
        {"Model": "Logistic Regression", "CV Accuracy": 0.8538, "CV Precision": 0.4278, "CV Recall": 0.7388, "CV F1 Score": 0.5419},
        {"Model": "KNN", "CV Accuracy": 0.8585, "CV Precision": 0.4329, "CV Recall": 0.6736, "CV F1 Score": 0.5270},
        {"Model": "Decision Tree", "CV Accuracy": 0.8550, "CV Precision": 0.4179, "CV Recall": 0.6103, "CV F1 Score": 0.4960}
    ])
    
    realistic_scores = pd.DataFrame([
        {"Model": "Random Forest", "CV Accuracy": 0.8370, "CV Precision": 0.3539, "CV Recall": 0.4762, "CV F1 Score": 0.4059},
        {"Model": "Gradient Boosting", "CV Accuracy": 0.7580, "CV Precision": 0.2658, "CV Recall": 0.6065, "CV F1 Score": 0.3695},
        {"Model": "KNN", "CV Accuracy": 0.7759, "CV Precision": 0.2697, "CV Recall": 0.5356, "CV F1 Score": 0.3587},
        {"Model": "Decision Tree", "CV Accuracy": 0.7806, "CV Precision": 0.2564, "CV Recall": 0.4602, "CV F1 Score": 0.3292},
        {"Model": "Logistic Regression", "CV Accuracy": 0.7246, "CV Precision": 0.2319, "CV Recall": 0.5859, "CV F1 Score": 0.3323}
    ])
    
    st.markdown("### 📊 Multi-Phase Model Score Comparison")
    st.markdown("Compare the evolution of model metrics across the three architectural phases:")
    
    # 3-COLUMN ANALYTICS LAYOUT REQUESTED BY USER
    col_phase1, col_phase2, col_phase3 = st.columns(3)
    
    with col_phase1:
        st.markdown("""
        <div style='background:#F1F5F9; padding:12px; border-radius:8px; margin-bottom:12px;'>
            <h4 style='margin:0; color:#334155;'>1. Baseline (Unbalanced)</h4>
            <span style='font-size:0.8rem; color:#64748B;'>Includes Duration • Single Split</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            baseline_scores.style.format({
                "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption("⚠️ **Deceptive Accuracy**: 90% accuracy is an illusion because guessing 'no' on 100% of rows gives 89% accuracy. Recall was under 45%.")
        
    with col_phase2:
        st.markdown("""
        <div style='background:#EFF6FF; padding:12px; border-radius:8px; margin-bottom:12px;'>
            <h4 style='margin:0; color:#1E40AF;'>2. SMOTENC (5-Fold CV)</h4>
            <span style='font-size:0.8rem; color:#3B82F6;'>Includes Duration • Balanced Folds</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            smote_scores.style.format({
                "CV Accuracy": "{:.2%}", "CV Precision": "{:.2%}", "CV Recall": "{:.2%}", "CV F1 Score": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption("✅ **Recall Skyrockets**: Synthetic sampling boosts Recall from ~40% to **78.7%** (Gradient Boosting), catching almost all depositors.")
        
    with col_phase3:
        st.markdown("""
        <div style='background:#ECFDF5; padding:12px; border-radius:8px; margin-bottom:12px;'>
            <h4 style='margin:0; color:#065F46;'>3. Realistic (No Duration)</h4>
            <span style='font-size:0.8rem; color:#059669;'>Production Target • 5-Fold CV</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            realistic_scores.style.format({
                "CV Accuracy": "{:.2%}", "CV Precision": "{:.2%}", "CV Recall": "{:.2%}", "CV F1 Score": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption("🚀 **Production Selection**: No Data Leakage. Gradient Boosting achieves a **60.6% Recall** without knowing call length in advance!")
        
    st.markdown("---")
    
    # Visual Analytics Chart Section
    st.subheader("Interactive Metric Comparison (Production Candidates)")
    metric_choice = st.selectbox(
        "Select Performance Metric to Compare:",
        ["CV Recall", "CV Precision", "CV F1 Score", "CV Accuracy"]
    )
    
    fig_metric = px.bar(
        realistic_scores.sort_values(by=metric_choice, ascending=False),
        x="Model", y=metric_choice,
        color=metric_choice,
        color_continuous_scale="Teal",
        text_auto=".2%",
        title=f"Realistic 5-Fold Stratified Cross-Validation: {metric_choice}"
    )
    fig_metric.update_layout(height=380, margin=dict(t=40, b=20, l=20, r=20), coloraxis_showscale=False)
    st.plotly_chart(fig_metric, use_container_width=True)
    
    st.markdown("---")
    
    # Test Set Confusion Matrix & Cost Breakdown
    cm_col1, cm_col2 = st.columns([1, 1])
    
    with cm_col1:
        st.subheader("Final Production Confusion Matrix (Test Set)")
        st.markdown("Evaluation on **9,043 untouched test prospects** (Without Duration):")
        
        # Test Set Confusion Matrix values from notebook Cell 67
        cm_matrix = [[6142, 1843], [410, 648]]
        fig_cm = px.imshow(
            cm_matrix,
            text_auto=True,
            labels=dict(x="Predicted Label", y="True Label", color="Customers"),
            x=['Predicted NO', 'Predicted YES'],
            y=['Actual NO', 'Actual YES'],
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(height=360, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with cm_col2:
        st.subheader("Operational Efficiency & Call Savings")
        st.markdown("""
        * **Total Untouched Test Prospects:** `9,043`
        * **Targeted Calls Recommended by Model:** `2,491 (27.5%)`
        * **Calls Safely Avoided:** `6,552 (72.5%)`
        """)
        
        eff_m1, eff_m2 = st.columns(2)
        with eff_m1:
            st.metric("Calls Eliminated", "72.5%", "Budget Preserved")
        with eff_m2:
            st.metric("Telemarketer Hours Saved", "546.0 hrs", "Assuming 5 min/call")
            
        st.markdown("""
        <div class="feature-card" style="margin-top:1rem;">
            <b>Why Gradient Boosting Won:</b><br>
            While Random Forest had higher precision, Gradient Boosting captured significantly more actual subscribers (60.6% Recall vs 47.6%), providing optimal total ROI for campaign budgets.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Feature Importance of Deployed Model
    st.subheader("Top Feature Importances (Production Model)")
    st.markdown("What attributes drive customer deposit decisions when call duration is properly excluded?")
    
    try:
        # Extract dynamic feature importances if model is available
        if model_pipeline is not None and feature_cols is not None:
            pre = model_pipeline.named_steps['preprocessor']
            model_step = model_pipeline.named_steps['model']
            importances = model_step.feature_importances_
            
            num_cols_real = [c for c in ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']]
            cat_cols_real = [c for c in feature_cols if c not in num_cols_real]
            
            cat_encoder = pre.named_transformers_['categorical']
            try:
                cat_names = cat_encoder.get_feature_names_out(cat_cols_real)
            except:
                cat_names = cat_encoder.get_feature_names(cat_cols_real)
                
            all_feature_names = list(num_cols_real) + list(cat_names)
            feat_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=True).tail(10)
        else:
            feat_df = pd.DataFrame([
                {'Feature': 'housing_no', 'Importance': 0.0249},
                {'Feature': 'education_primary', 'Importance': 0.0250},
                {'Feature': 'balance', 'Importance': 0.0321},
                {'Feature': 'loan_yes', 'Importance': 0.0469},
                {'Feature': 'housing_yes', 'Importance': 0.0515},
                {'Feature': 'day', 'Importance': 0.0544},
                {'Feature': 'loan_no', 'Importance': 0.0605},
                {'Feature': 'campaign', 'Importance': 0.0715},
                {'Feature': 'poutcome_success', 'Importance': 0.1526},
                {'Feature': 'contact_cellular', 'Importance': 0.2942}
            ])
            
        fig_feat = px.bar(
            feat_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Teal',
            labels={'Importance': 'Relative Importance Score', 'Feature': ''}
        )
        fig_feat.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
        st.plotly_chart(fig_feat, use_container_width=True)
        st.caption("Cellular contact channel, past campaign success, and lack of debt (housing/personal loans) are the top predictive features.")
    except Exception as e:
        st.info("Feature importance display fallback.")

# ==========================================
# MODULE 3: PREDICTION ENGINE
# ==========================================
elif page == "🎯 Customer Prediction Engine":
    st.title("Prospect Targeting Engine")
    st.markdown("Input client attributes below to generate an instantaneous, data-driven targeting recommendation. *Note: Contact duration is excluded to prevent data leakage.*")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demographics")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox("Job Type", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
            marital = st.selectbox("Marital Status", ['divorced', 'married', 'single'])
            education = st.selectbox("Education Level", ['primary', 'secondary', 'tertiary', 'unknown'])
        
        with col2:
            st.markdown("#### Financial Profile")
            balance = st.number_input("Yearly Balance (€)", min_value=-10000, max_value=200000, value=1500)
            default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
            housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
            loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
        
        with col3:
            st.markdown("#### Campaign History")
            contact = st.selectbox("Communication Channel", ['cellular', 'telephone', 'unknown'])
            month = st.selectbox("Target Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
            day = st.number_input("Target Day of Month", min_value=1, max_value=31, value=15)
            campaign = st.number_input("Contacts During This Campaign", min_value=1, max_value=60, value=1)
            pdays = st.number_input("Days Since Last Campaign (-1 if never)", min_value=-1, max_value=900, value=-1)
            previous = st.number_input("Prior Campaign Contacts", min_value=0, max_value=300, value=0)
            poutcome = st.selectbox("Previous Campaign Outcome", ['failure', 'other', 'success', 'unknown'])

        submit = st.form_submit_button("Run Targeting Assessment")

    if submit:
        if model_pipeline is not None and feature_cols is not None:
            input_df = pd.DataFrame([{
                'age': int(age), 'job': job, 'marital': marital, 'education': education,
                'default': default, 'balance': float(balance), 'housing': housing, 'loan': loan,
                'contact': contact, 'day': int(day), 'month': month, 'campaign': int(campaign),
                'pdays': int(pdays), 'previous': int(previous), 'poutcome': poutcome
            }], columns=feature_cols)
            
            prob = model_pipeline.predict_proba(input_df)[0][1]
        else:
            # Intuitive heuristic probability for demonstration if model pkl unlinked
            prob = 0.65 if (poutcome == 'success' or balance > 3000) else 0.22
            
        st.markdown("---")
        st.subheader("Targeting Assessment Result")
        
        res_c1, res_c2 = st.columns([1, 2])
        with res_c1:
            if prob >= 0.5:
                st.success("🎯 **HIGH PRIORITY TARGET**")
                st.markdown("The model identifies this prospect as highly likely to convert. **Recommend scheduling call.**")
            else:
                st.warning("⛔ **LOW PRIORITY / DO NOT TARGET**")
                st.markdown("Low conversion probability. Reallocate telemarketing bandwidth elsewhere.")
                
        with res_c2:
            st.markdown(f"**Calculated Subscription Probability: {prob * 100:.1f}%**")
            st.progress(float(prob))
            
            if prob >= 0.5:
                st.markdown("🟢 *Action: Place at the top of the telemarketing priority queue.*")
            else:
                st.markdown("🔴 *Action: Exclude from cold call list to conserve budget.*")

# ==========================================
# MODULE 4: BUSINESS ROI SIMULATOR
# ==========================================
elif page == "💼 Campaign ROI Simulator":
    st.title("Campaign ROI & Financial Simulator")
    st.markdown("Estimate the financial impact of utilizing the Gradient Boosting (SMOTENC) targeting model compared to an indiscriminate 'call-everyone' approach.")
    
    st.markdown("---")
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        call_cost = st.slider("Cost per Telemarketing Call (€)", min_value=1.0, max_value=25.0, value=5.0, step=0.5)
    with c_col2:
        conversion_value = st.slider("Revenue per Successful Deposit (€)", min_value=50, max_value=1500, value=350, step=25)
        
    total_prospects = st.slider("Campaign Batch Size (Prospects)", min_value=1000, max_value=50000, value=10000, step=1000)
    
    actual_subscribers = int(total_prospects * 0.117)
    
    # Scenario A: Call Everyone
    cost_a = total_prospects * call_cost
    revenue_a = actual_subscribers * conversion_value
    profit_a = revenue_a - cost_a
    
    # Scenario B: Machine Learning Targeted (Based on Test-Set Validation)
    # 60.6% Recall, 26.6% Precision (No duration)
    subscribers_caught = int(actual_subscribers * 0.606)
    total_ml_calls = int(subscribers_caught / 0.266) if subscribers_caught > 0 else 0
    
    cost_b = total_ml_calls * call_cost
    revenue_b = subscribers_caught * conversion_value
    profit_b = revenue_b - cost_b
    
    st.subheader(f"Financial Impact on {total_prospects:,} Prospects")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Traditional Profit (Call All)", f"€{profit_a:,.2f}")
    with m2:
        st.metric("AI-Targeted Profit", f"€{profit_b:,.2f}", f"+€{profit_b - profit_a:,.2f}")
    with m3:
        avoided_calls = total_prospects - total_ml_calls
        st.metric("Unnecessary Calls Avoided", f"{avoided_calls:,}", f"Saved €{avoided_calls * call_cost:,.2f}")
        
    st.markdown("---")
    
    # Comparison Bar Chart
    comp_df = pd.DataFrame({
        'Strategy': ['Traditional (Call Everyone)', 'AI-Targeted Model'],
        'Cost (€)': [cost_a, cost_b],
        'Revenue (€)': [revenue_a, revenue_b],
        'Net Profit (€)': [profit_a, profit_b]
    })
    
    fig_roi = px.bar(
        comp_df, x='Strategy', y=['Cost (€)', 'Revenue (€)', 'Net Profit (€)'],
        barmode='group',
        color_discrete_map={'Cost (€)': '#EF4444', 'Revenue (€)': '#3B82F6', 'Net Profit (€)': '#10B981'},
        title="Strategy Comparison: Costs, Revenue, and Net Profit"
    )
    fig_roi.update_layout(height=400, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_roi, use_container_width=True)
    
    st.info("💡 **Bottom Line:** Machine learning prevents hundreds of hours of fruitless cold calling. By strategically bypassing the ~73% of non-converting prospects, campaign profitability increases while operational burnout decreases.")
