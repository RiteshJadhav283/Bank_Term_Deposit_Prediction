import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bank Term Deposit Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN UI CSS INJECTION ---
st.markdown("""
<style>
    /* Global Font and Clean Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8F9FA;
    }
    
    /* Remove default Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Elegant Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Primary Action Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6rem;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    /* Section Headers */
    h1 {
        font-weight: 700;
        color: #111827;
        margin-bottom: 1.5rem;
    }
    h2, h3 {
        font-weight: 600;
        color: #374151;
    }
    
    /* Subtle divider */
    hr {
        border-top: 1px solid #E5E7EB;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- PATH DEFINITIONS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'final_bank_model_smotenc.pkl')
COLS_PATH = os.path.join(BASE_DIR, 'model', 'final_feature_columns.pkl')
DATA_PATH = os.path.join(BASE_DIR, '..', 'dataSet', 'bank-full.csv')

# --- DATA LOADING ---
@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    features = joblib.load(COLS_PATH)
    return pipeline, features

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=';')
    return df

try:
    model_pipeline, feature_cols = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Critical System Error: Model pipeline not found. {e}")

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("### 🏦 Term Deposit Targeting")
st.sidebar.markdown("Optimize telemarketing campaigns through machine learning.")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Modules", 
    ["1. Executive Dashboard", "2. Prediction Engine", "3. Business Impact Simulator"]
)

# ==========================================
# PAGE 1: EXPLORATORY DATA ANALYSIS
# ==========================================
if page == "1. Executive Dashboard":
    st.title("Historical Campaign Insights")
    st.markdown("Interactive analysis of customer behavior and conversion metrics from historical campaigns.")
    
    try:
        df = load_data()
        
        # High-level KPIs
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Total Customer Records", f"{df.shape[0]:,}")
        with kpi2:
            st.metric("Total Subscribers Acquired", f"{(df['y'] == 'yes').sum():,}")
        with kpi3:
            st.metric("Average Conversion Rate", f"{(df['y'] == 'yes').mean() * 100:.2f}%")
            
        st.markdown("---")
        
        # Row 1 Charts
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.subheader("Subscription Imbalance")
            fig1 = px.pie(
                df, names='y', 
                color='y', color_discrete_map={'no':'#EF4444', 'yes':'#10B981'},
                hole=0.4
            )
            fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_c2:
            st.subheader("Conversion Rate by Job Sector")
            job_conv = pd.crosstab(df['job'], df['y'], normalize='index')['yes'].reset_index()
            job_conv['yes'] = job_conv['yes'] * 100
            job_conv = job_conv.sort_values(by='yes', ascending=True)
            
            fig2 = px.bar(
                job_conv, x='yes', y='job', orientation='h',
                labels={'yes': 'Conversion Rate (%)', 'job': ''},
                color='yes', color_continuous_scale='Blues'
            )
            fig2.update_layout(coloraxis_showscale=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error initializing dashboard data: {e}")

# ==========================================
# PAGE 2: PREDICTION ENGINE
# ==========================================
elif page == "2. Prediction Engine":
    st.title("Customer Prediction Engine")
    st.markdown("Input prospective customer details below to generate a targeting recommendation. *Note: Contact duration is excluded to prevent data leakage prior to the actual call.*")
    
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

    if submit and model_loaded:
        input_df = pd.DataFrame([{
            'age': int(age), 'job': job, 'marital': marital, 'education': education,
            'default': default, 'balance': float(balance), 'housing': housing, 'loan': loan,
            'contact': contact, 'day': int(day), 'month': month, 'campaign': int(campaign),
            'pdays': int(pdays), 'previous': int(previous), 'poutcome': poutcome
        }], columns=feature_cols)
        
        prob = model_pipeline.predict_proba(input_df)[0][1]
        
        st.markdown("---")
        st.subheader("Targeting Assessment Result")
        
        res_c1, res_c2 = st.columns([1, 2])
        with res_c1:
            if prob >= 0.5:
                st.success("🎯 **HIGH PRIORITY TARGET**")
                st.markdown("The model identifies this profile as highly likely to convert.")
            else:
                st.warning("⛔ **DO NOT TARGET**")
                st.markdown("Low conversion probability. Direct resources elsewhere.")
                
        with res_c2:
            st.markdown(f"**Subscription Probability: {prob * 100:.1f}%**")
            st.progress(float(prob))

# ==========================================
# PAGE 3: BUSINESS IMPACT SIMULATOR
# ==========================================
elif page == "3. Business Impact Simulator":
    st.title("Campaign ROI Simulator")
    st.markdown("Estimate the financial impact of utilizing the Gradient Boosting (SMOTENC) targeting model compared to a traditional 'call-everyone' approach.")
    
    st.markdown("---")
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        call_cost = st.slider("Cost per Telemarketing Call (€)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    with c_col2:
        conversion_value = st.slider("Revenue per Successful Deposit (€)", min_value=50, max_value=1000, value=300, step=10)
        
    # Standard metrics established in Notebook cross-validation
    # Assuming realistic model test-set metrics: Precision ~41%, Recall ~86%
    # Natural Conversion rate is ~11.7%
    
    # Simulate a campaign of 10,000 customers
    total_prospects = 10000
    actual_subscribers = int(total_prospects * 0.117)
    
    # Scenario A: Call Everyone
    cost_a = total_prospects * call_cost
    revenue_a = actual_subscribers * conversion_value
    profit_a = revenue_a - cost_a
    
    # Scenario B: Machine Learning Targeted
    # Model recalls 86% of subscribers.
    subscribers_caught = int(actual_subscribers * 0.86)
    # Model precision is 41%. So total calls made = subscribers_caught / 0.41
    total_ml_calls = int(subscribers_caught / 0.41)
    
    cost_b = total_ml_calls * call_cost
    revenue_b = subscribers_caught * conversion_value
    profit_b = revenue_b - cost_b
    
    st.subheader(f"Simulation on a batch of {total_prospects:,} prospects")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Traditional Profit", f"€{profit_a:,.2f}")
    with m2:
        st.metric("ML Targeted Profit", f"€{profit_b:,.2f}", f"+€{profit_b - profit_a:,.2f}")
    with m3:
        st.metric("Unnecessary Calls Avoided", f"{(total_prospects - total_ml_calls):,}", f"Saved €{(total_prospects - total_ml_calls) * call_cost:,.2f}")
        
    st.info("The model achieves significant cost reduction by deliberately avoiding calls to prospects with a low probability of conversion, preserving campaign budget without sacrificing a massive proportion of revenue.")
