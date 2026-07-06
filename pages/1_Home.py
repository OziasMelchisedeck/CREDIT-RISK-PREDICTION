import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.title("🏦 AI Credit Risk Assessment System")

st.markdown("""
### Welcome 👋

This application predicts the probability of loan repayment using a **LightGBM model** trained on over **500,000 records**.

---

### 🔥 Features

- AI-powered credit scoring
- Real-time prediction
- Explainable AI (SHAP)
- Model performance dashboard
""")

# KPI cards
col1, col2, col3 = st.columns(3)

col1.metric("Dataset Size", "500K+")
col2.metric("Model", "LightGBM")
col3.metric("ROC AUC", "0.92")

st.divider()

st.info("Use the sidebar to navigate through the application.")