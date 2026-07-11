import streamlit as st

st.set_page_config(
    page_title="AI Credit Risk App",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar design
st.sidebar.title("🏦 Credit Risk AI")
st.sidebar.image("https://img.icons8.com/fluency/96/bank.png")

st.title("🏦 Système d'Évaluation du Risque de Crédit par Intelligence Artificielle")

st.markdown("""
### Bienvenue 👋

Cette application permet d'estimer la **probabilité de remboursement d'un prêt** à l'aide d'un modèle **LightGBM** entraîné sur un jeu de données de plus de **500 000 observations**.

---

### 🚀 Fonctionnalités

- Évaluation automatisée du risque de crédit par Intelligence Artificielle
- Prédiction de la probabilité de remboursement en temps réel
- Explication des prédictions grâce à SHAP (IA explicable)
- Tableau de bord des performances du modèle
""")

# Cartes d'information
col1, col2, col3 = st.columns(3)

col1.metric("Taille du jeu de données", "500K+")
col2.metric("Modèle utilisé", "LightGBM")
col3.metric("ROC AUC", "0.92")

st.divider()

st.info("Utilisez le menu de navigation situé à gauche pour accéder aux différentes fonctionnalités de l'application.")