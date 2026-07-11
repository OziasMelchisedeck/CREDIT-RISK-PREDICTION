import streamlit as st
import pandas as pd
from utils.prediction import pipeline_load_predict, make_prediction
from utils.shap import load_shape
import shap
from matplotlib import pyplot as plt


#VARIABLES
pipeline = pipeline_load_predict() #Chargement du pipeline
explainer, shap_values, expected_value, dataset = load_shape() #Chargement des infos d'explicabilité Shap


resultsOk = False
pred = 0.0 #resultat(classe) de la prediction
proba = 0.0 #probabilite de la classe


#FONCTIONS
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

#fonction d'affichage du waterfall plot pour lexplicabilité d'un client
def print_waterfall(client):
    encoded_client = pipeline['preprocessing'].transform(client)
    shap_values_client = explainer.shap_values(encoded_client)
    if isinstance(shap_values_client, list):
        shap_values_client = shap_values_client[1]
    
    shap.waterfall_plot(shap.Explanation(
        values=shap_values_client[0],
        base_values=expected_value,
        data=encoded_client[0],
        feature_names=dataset.columns
    ), max_display = 10, show = False)
    fig = plt.gcf()
    st.pyplot(fig, bbox_inches = 'tight')
    plt.close(fig)


#INTERFACE

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

if st.button("🔮 Predict"):
    data_client = save_prediction_data()
    pred, proba = make_prediction(pipeline, data_client)
    print(pred, proba)
    if pred or proba:
        st.success("Prédiction effectuée")
        resultsOk = True
    else :
        st.error("Une erreur est survenue")
    
    
    #resultats
    if resultsOk:
        st.markdown("#### Décision + Score")
        proba_defaut = 1 - proba
        if proba >= 0.7 :
            st.markdown("##### ✅ CREDIT ACCORDE")
            st.metric(label="Probabilité de remboursement", value =f"{proba * 100:.2f}%")
            st.metric(label="Probalite de defaut", value=f"{proba_defaut * 100:.2f}%")
        elif proba >= 0.5 and proba < 0.7 :
            st.markdown("##### ⚠️ A ETUDIER")
            st.metric(label="Probabilité de remboursement", value =f"{proba * 100:.2f}%")
            st.metric(label="Probalite de defaut", value=f"{proba_defaut * 100:.2f}%")
        elif proba < 0.5 :
            st.markdown("##### ❌ CREDIT REFUSE")
            st.metric(label="Probabilité de remboursement", value =f"{proba * 100:.2f}%")
            st.metric(label="Probalite de defaut", value=f"{proba_defaut * 100:.2f}%")

    # ==========================================================
    # EXPLICATION LOCALE DE LA PRÉDICTION
    # ==========================================================

    st.subheader("🧠 Explication de la décision du modèle")

    st.markdown("""
    Le graphique ci-dessous présente une **explication locale** de la prédiction réalisée pour cet emprunteur à l'aide des valeurs **SHAP (SHapley Additive exPlanations)**.

    Il montre comment chaque caractéristique du client influence progressivement la décision du modèle jusqu'à obtenir la probabilité finale de remboursement.
    """)

    st.caption(
        "💡 Chaque variable contribue positivement ou négativement à la prédiction finale du modèle."
    )

    print_waterfall(data_client)

    st.info("""
    ### 📖 Comment interpréter ce graphique ?

    Le graphique Waterfall décompose la prédiction du modèle en montrant la contribution de chaque variable.

    - 🔴 **Barres rouges** : représentent les variables qui **augmentent le risque de défaut de remboursement**. Elles orientent donc la prédiction vers un **non-remboursement** du prêt.

    - 🔵 **Barres bleues** : représentent les variables qui **réduisent le risque de défaut** et favorisent ainsi la prédiction d'un **remboursement effectif**.

    - 📏 **La longueur de chaque barre** traduit l'importance de l'influence de la variable sur la décision du modèle : plus la barre est longue, plus son impact est significatif.

    - 🎯 **La somme de toutes les contributions** permet de passer de la prédiction moyenne du modèle (valeur de base) à la probabilité finale affichée pour cet emprunteur.

    Cette visualisation permet ainsi de comprendre précisément **pour quelles raisons le modèle considère qu'un client présente un risque élevé ou faible de défaut de remboursement**.
    """)
