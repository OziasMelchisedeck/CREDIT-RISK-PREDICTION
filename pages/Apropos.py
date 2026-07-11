import streamlit as st

st.title("📖 À propos de l'Application")
st.layout = "wide"

# Sidebar design
st.sidebar.title("🏦 Credit Risk AI")
st.sidebar.image("https://img.icons8.com/fluency/96/bank.png")

# ==========================================================
# OBJECTIF
# ==========================================================

st.header("🎯 Objectif du projet")

st.markdown("""
Cette application a été développée dans le cadre d'un projet complet de **Machine Learning** appliqué à l'évaluation du **risque de crédit**.

Son objectif est d'estimer la **probabilité qu'un emprunteur rembourse son prêt** à partir de ses caractéristiques personnelles, professionnelles et financières.

Au-delà de la prédiction, cette application met l'accent sur **l'interprétabilité des décisions du modèle** grâce à des techniques d'**Intelligence Artificielle Explicable (Explainable AI)**.

L'ensemble du projet couvre toutes les étapes d'un pipeline de Data Science, depuis l'analyse des données jusqu'au déploiement d'une application web interactive avec Streamlit.
""")

st.divider()

# ==========================================================
# DATASET
# ==========================================================

st.header("📊 Jeu de données")

st.markdown("""
Le modèle a été entraîné sur le jeu de données **Loan Approval Prediction**, proposé dans le cadre de la compétition **Kaggle Playground Series - Season 5 Episode 11**.

Ce jeu de données comporte **plus de 500 000 observations** représentant des demandes de prêt.

Chaque observation contient plusieurs informations relatives au profil de l'emprunteur, notamment :

- Revenus annuels
- Score de crédit
- Ratio d'endettement
- Montant du prêt
- Taux d'intérêt
- Niveau d'étude
- Situation professionnelle
- Situation matrimoniale
- Objet du prêt
- etc.

L'objectif de la compétition consiste à prédire la probabilité qu'un emprunteur rembourse intégralement son prêt.

Les soumissions sont évaluées selon la métrique **ROC AUC**, particulièrement adaptée aux problèmes de classification binaire déséquilibrés.
""")

st.link_button(
    "🔗 Consulter la compétition Kaggle",
    "https://www.kaggle.com/competitions/playground-series-s5e11/data"
)

st.divider()

# ==========================================================
# MODEL
# ==========================================================

st.header("🤖 Modèle de Machine Learning")

st.markdown("""
Le modèle retenu est un **LightGBM Classifier**, choisi après comparaison avec plusieurs algorithmes de classification supervisée.

Le pipeline d'entraînement comprend notamment :

- Encodage des variables catégorielles (Ordinal Encoding et One-Hot Encoding)
- Création de nouvelles variables (Feature Engineering)
- Gestion du déséquilibre des classes
- Optimisation des hyperparamètres
- Validation croisée
- Évaluation du modèle selon la métrique ROC AUC

LightGBM a été retenu pour ses excellentes performances, sa rapidité d'entraînement et sa capacité à gérer efficacement des jeux de données volumineux.
""")

st.divider()

# ==========================================================
# DEVELOPMENT PROCESS
# ==========================================================

st.header("🔬 Démarche de développement")

st.markdown("""
La réalisation de ce projet suit les principales étapes d'un projet professionnel de Data Science :

- Compréhension du problème métier
- Analyse exploratoire des données (EDA)
- Nettoyage et préparation des données
- Encodage des variables
- Création de nouvelles variables (Feature Engineering)
- Gestion du déséquilibre entre les classes
- Comparaison de plusieurs modèles de Machine Learning
- Optimisation du modèle LightGBM
- Validation croisée
- Évaluation des performances
- Déploiement sous forme d'application Streamlit
""")

st.divider()

# ==========================================================
# APPLICATION ARCHITECTURE
# ==========================================================

st.header("🏗️ Architecture générale de l'application")

st.markdown("""
L'application suit un pipeline complet de Machine Learning :

1. 📥 Saisie des informations du demandeur de prêt.

2. ⚙️ Prétraitement automatique des données.

3. 🧠 Prédiction à l'aide du modèle LightGBM.

4. 📊 Calcul de la probabilité de remboursement.

5. 🔍 Explication des prédictions grâce aux valeurs SHAP.

6. 📈 Consultation des performances globales du modèle.

7. 💻 Restitution des résultats dans une interface intuitive développée avec Streamlit.
""")

st.divider()

# ==========================================================
# EXPLAINABILITY
# ==========================================================

st.header("🧠 Intelligence Artificielle Explicable (Explainable AI)")

st.markdown("""
L'application ne se limite pas à fournir une prédiction.

Elle permet également de comprendre **pourquoi** le modèle prend une décision donnée grâce à la bibliothèque **SHAP (SHapley Additive exPlanations)**.

Les graphiques proposés permettent de :

- identifier les variables les plus influentes ;
- expliquer chaque prédiction individuellement ;
- mesurer l'impact positif ou négatif de chaque variable ;
- renforcer la transparence des décisions du modèle.

Cette approche favorise une meilleure confiance dans les résultats, notamment dans les domaines bancaire et financier où l'explicabilité des modèles est essentielle.
""")

st.divider()

# ==========================================================
# FEATURES
# ==========================================================

st.header("🚀 Fonctionnalités de l'application")

st.markdown("""
L'application permet notamment :

✅ d'estimer la probabilité de remboursement d'un prêt ;

✅ de calculer le niveau de risque associé à un emprunteur ;

✅ de visualiser les performances globales du modèle ;

✅ d'explorer les graphiques d'explicabilité SHAP ;

✅ d'analyser l'importance des variables ;

✅ d'offrir une interface utilisateur intuitive développée avec Streamlit.
""")

st.divider()

# ==========================================================
# TECHNOLOGIES
# ==========================================================

st.header("🛠️ Technologies utilisées")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
### Data Science

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
""")

with col2:

    st.markdown("""
### Développement

- Streamlit
- SHAP
- Matplotlib
- Joblib
- Git & GitHub
""")

st.divider()

# ==========================================================
# SKILLS
# ==========================================================

st.header("🎯 Compétences mobilisées")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
### Data Science

- Analyse exploratoire (EDA)
- Prétraitement des données
- Encodage des variables
- Feature Engineering
- Validation croisée
- Optimisation des hyperparamètres
- Évaluation des modèles
""")

with col2:

    st.markdown("""
### Machine Learning & Déploiement

- Classification supervisée
- LightGBM
- Explainable AI (SHAP)
- Développement Streamlit
- Git & GitHub
- Déploiement d'applications ML
""")

st.divider()

# ==========================================================
# PROJECT VALUE
# ==========================================================

st.header("🎓 Ce que démontre ce projet")

st.markdown("""
Ce projet illustre ma capacité à concevoir une solution complète de Machine Learning répondant à une problématique réelle.

À travers cette application, j'ai mobilisé des compétences en :

- Data Science ;
- Machine Learning ;
- Intelligence Artificielle Explicable (XAI) ;
- Développement d'applications interactives avec Streamlit ;
- Génie logiciel avec Git et GitHub ;
- Déploiement de modèles de Machine Learning.

L'objectif est de proposer une solution robuste, interprétable et facilement exploitable dans un contexte réel d'aide à la décision.
""")

st.divider()

# ==========================================================
# RESOURCES
# ==========================================================

st.header("🔗 Ressources")

st.markdown("""
Vous pouvez consulter les ressources associées à ce projet :
""")

col1, col2= st.columns(2)

with col1:

    st.link_button(
        "📂 GitHub",
        "https://github.com/OziasMelchisedeck/CREDIT-RISK-PREDICTION"
    )

with col2:

    st.link_button(
        "🏆 Kaggle",
        "https://www.kaggle.com/competitions/playground-series-s5e11/data"
    )


st.divider()

# ==========================================================
# CONTACT
# ==========================================================

st.header("📬 Contact")

st.markdown("""
**Développeur :** Ozias Melchisedeck Effa'a

📧 **Email :** effaozias3@gmail.com

📱 **Téléphone :** +237 696 85 16 32

🏆 **Profil Kaggle :**
""")

st.link_button(
    "Consulter mon profil Kaggle",
    "https://www.kaggle.com/oziasmelchi"
)

st.divider()

st.success(
    "💡 « Une bonne prédiction ne suffit pas ; il est tout aussi important de pouvoir expliquer les raisons qui la motivent. »"
)