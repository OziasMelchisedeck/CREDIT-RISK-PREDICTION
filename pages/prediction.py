import streamlit as st

st.title("💳 Loan Prediction")

st.markdown("""
<style>
.big-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income", 0, 1_000_000, 50000)
    credit_score = st.slider("Credit Score", 300, 850, 700)
    debt_ratio = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.3)

with col2:
    loan_amount = st.number_input("Loan Amount", 0, 500000, 10000)
    interest_rate = st.slider("Interest Rate", 0.0, 30.0, 10.0)

    education = st.selectbox(
        "Education Level",
        ["High School", "Bachelor's", "Master's", "PhD"]
    )

st.divider()

if st.button("🔮 Predict"):
    st.success("Prediction will appear here after model integration")
    st.info("Next step: connect your LightGBM pipeline")