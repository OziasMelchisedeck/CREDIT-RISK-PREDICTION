import streamlit as st
import pandas as pd
from utils.prediction import pipeline_load_predict, make_prediction
import sklearn
import sys

st.title("💳 Loan Prediction")
st.layout = "wide"

# Sidebar design
st.sidebar.title("🏦 Credit Risk AI")
st.sidebar.image("https://img.icons8.com/fluency/96/bank.png")

st.markdown("""
<style>
.big-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

pipeline = pipeline_load_predict() #Chargement du pipeline

st.write(sklearn.__version__)
st.write(sys.executable)
st.markdown("#### Entrer les informations du client dont vous souhaitez avoir une prédiction")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    annual_income = st.number_input("Annual Income", 0, 1_000_000, 50000, key='annual_income')
    credit_score = st.slider("Credit Score", 300, 850, 700, key='credit_score')
    debt_to_income_ratio = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.3, key='debt_to_income_ratio')

with col2:
    loan_amount = st.number_input("Loan Amount", 0, 500000, 10000, key='loan_amount')
    interest_rate = st.slider("Interest Rate", 0.0, 30.0, 10.0, key='interest_rate')

    education = st.selectbox(
        "Education Level",
        ["High School", "Bachelor's", "Master's", "PhD", "Other"],
        key='education_level'
    )

with col3:
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], key='gender')
    marital_status = st.selectbox("Marital status", ["Married", "Single", "Divorced", "Widowed"], key='marital_status')
    loan_purpose = st.selectbox("Loan purpose", ["Other", "Debt Consolidation", "Home", "Education", "Vacation", "Car", "Medical", "Business"], key='loan_purpose')

with col4:
    grade_subgrade = st.selectbox("Grade Subgrade", ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5", "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5"], key='grade_subgrade')
    employment_status = st.selectbox("Employment status", ["Employed", "Self-employed", "Retired", "Unemployed", "Student"], key='employment_status')

st.divider()

#Fonction d'enregistrement des données de prediction
def save_prediction_data():
    data = pd.DataFrame(
        [{
            "annual_income" : st.session_state.annual_income,
            "debt_to_income_ratio" : st.session_state.debt_to_income_ratio,
            "credit_score" : st.session_state.credit_score,
            "loan_amount" : st.session_state.loan_amount,
            "interest_rate" : st.session_state.interest_rate,
            "gender" : st.session_state.gender,
            "marital_status" : st.session_state.marital_status,
            "education_level" : st.session_state.education_level,
            "employment_status" : st.session_state.employment_status,
            "loan_purpose" : st.session_state.loan_purpose,
            "grade_subgrade" : st.session_state.grade_subgrade
        }]
    )
    return data



if st.button("🔮 Predict"):
    st.success("Prediction will appear here after model integration")
    prediction_data = save_prediction_data()
    pred, proba = make_prediction(pipeline, prediction_data)
    st.write(f"{pred}, {proba} ")
    st.info("Next step: connect your LightGBM pipeline")