import streamlit as st
from matplotlib import pyplot as plt
from utils.shap import load_shape
import shap


#VARIABLES
explainer, shap_values, expected_value, dataset = load_shape() #Chargement des infos d'explicabilité Shap


#FONCTIONS


#INTERFACE
#==========================================================
# TITRE
#==========================================================
st.title("🔍 Explicabilité du modèle")
st.layout = "wide"

# Sidebar design
st.sidebar.title("🏦 Credit Risk AI")
st.sidebar.image("https://img.icons8.com/fluency/96/bank.png")
 
st.markdown("""
Cette section présente l'interprétation globale des décisions du modèle **LightGBM Classifier** grâce à la méthode **SHAP (SHapley Additive exPlanations)**.

L'objectif est d'identifier les variables qui influencent le plus les prédictions et de comprendre leur impact sur l'évaluation du risque de crédit.
""")

st.divider()


# ==========================================================
# IMPACT DES VARIABLES
# ==========================================================

st.subheader("📊 Impact des variables sur les prédictions")

st.markdown("""
Le graphique ci-dessous présente la contribution de chaque variable aux prédictions du modèle.

- Chaque point représente une observation du jeu de données.
- La couleur indique la valeur de la variable :
    - 🔴 Rouge : valeur élevée de la variable.
    - 🔵 Bleu : valeur faible de la variable.
- L'axe horizontal représente l'influence de la variable sur la prédiction :
    - ➡️ Une contribution positive augmente le score associé au risque de défaut.
    - ⬅️ Une contribution négative réduit ce risque.

Les variables sont classées de haut en bas selon leur importance dans les décisions du modèle.
""")

fig, ax = plt.subplots(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    dataset,
    show=False
)

fig = plt.gcf()

st.pyplot(fig)

plt.close(fig)


st.info("""
### Interprétation métier

Les variables situées en haut du graphique sont celles qui ont le plus d'influence sur la décision du modèle.

Une contribution SHAP positive indique que la variable pousse la prédiction vers un risque plus élevé de défaut de remboursement.

À l'inverse, une contribution négative indique que la variable réduit le risque estimé par le modèle.
""")


st.divider()


# ==========================================================
# IMPORTANCE GLOBALE
# ==========================================================

st.subheader("🏆 Importance globale des variables")

st.markdown("""
Ce graphique présente le classement des variables selon leur contribution moyenne aux décisions du modèle.

Il permet d'identifier rapidement les facteurs les plus déterminants dans l'évaluation du risque de crédit.
""")

fig2, ax2 = plt.subplots(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    dataset,
    plot_type="bar",
    show=False
)

fig2 = plt.gcf()

st.pyplot(fig2)

plt.close(fig2)


st.divider()


# ==========================================================
# CONCLUSION
# ==========================================================

st.success("""
Grâce à SHAP, le modèle LightGBM devient plus transparent en permettant d'identifier les facteurs influençant ses décisions.

Cette interprétabilité est essentielle dans les domaines financiers, où les modèles doivent combiner performance prédictive, fiabilité et capacité d'explication.
""")