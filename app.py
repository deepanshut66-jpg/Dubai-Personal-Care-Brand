import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Dubai Personal Care — Consumer Intelligence",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
    .stMetric { background: #f0f7f7; padding: 12px; border-radius: 8px; }
    div[data-testid="stMetric"] { background: #f0f7f7; padding: 12px; border-radius: 8px; }
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; }
    h1 { color: #1B2A4A; }
    h2 { color: #0D7377; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/moisturizing-cream.png", width=60)
    st.title("Dubai Personal Care")
    st.caption("Consumer Intelligence Dashboard")
    st.divider()
    st.markdown("**Navigation**")
    page = st.radio("Select Analysis:", [
        "📊 Descriptive",
        "🔍 Diagnostic",
        "🤖 Predictive",
        "🔬 Clustering Deep-Dive",
        "💡 Prescriptive",
        "🆕 New Customer Predictor"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Dataset**")
    data_path = os.path.join(os.path.dirname(__file__), "dubai_personal_care_survey.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.success(f"✅ {len(df):,} records loaded")
    else:
        st.error("Dataset not found!")
        st.stop()

    st.caption(f"Columns: {len(df.columns)}")
    st.caption("Source: Synthetic survey data simulating Dubai personal care consumers")
    st.divider()
    st.markdown("**About**")
    st.caption(
        "Built for data-driven go-to-market strategy. "
        "Applies Classification, Clustering (K-Means), "
        "Latent Class Analysis, Association Rule Mining, "
        "and Regression to design segment-specific "
        "marketing playbooks for a Dubai personal care brand launch."
    )

# Header
st.markdown("""
# 🧴 Dubai Personal Care — Consumer Intelligence Dashboard
**Data-driven go-to-market strategy for a personal care brand launch in Dubai**
""")
st.divider()

# Page Routing
if page == "📊 Descriptive":
    import tab_descriptive
    tab_descriptive.render(df)
elif page == "🔍 Diagnostic":
    import tab_diagnostic
    tab_diagnostic.render(df)
elif page == "🤖 Predictive":
    import tab_predictive
    tab_predictive.render(df)
elif page == "🔬 Clustering Deep-Dive":
    import tab_clustering_deepdive
    tab_clustering_deepdive.render(df)
elif page == "💡 Prescriptive":
    import tab_prescriptive
    tab_prescriptive.render(df)
elif page == "🆕 New Customer Predictor":
    import tab_predict_new
    tab_predict_new.render(df)
