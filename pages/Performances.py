import streamlit as st
from matplotlib import pyplot as plt


#VARIABLES



#FONCTIONS


#INTERFACE
import streamlit as st

# ==========================================================
# TITRE
# ==========================================================

st.title("🧠 Performances du modèle")

st.markdown("""
Cette page présente les principales métriques d'évaluation du modèle **LightGBM Classifier** utilisé pour prédire la probabilité de remboursement d'un prêt.

Les performances ont été obtenues sur le jeu de validation à l'aide d'une validation croisée.
""")

st.divider()

# ==========================================================
# METRIQUES
# ==========================================================

st.subheader("📊 Indicateurs de performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="ROC AUC",
        value="0.921"
    )

with col2:
    st.metric(
        label="Accuracy",
        value="86.8 %"
    )

with col3:
    st.metric(
        label="Précision",
        value="94.3 %"
    )

col4, col5 = st.columns(2)

with col4:
    st.metric(
        label="Rappel",
        value="88.7 %"
    )

with col5:
    st.metric(
        label="F1-score",
        value="91.5 %"
    )

st.divider()

# ==========================================================
# INTERPRETATION
# ==========================================================

st.subheader("📖 Interprétation des résultats")

st.info("""
- **ROC AUC :** le modèle distingue efficacement les emprunteurs solvables de ceux présentant un risque de défaut.

- **Accuracy :** près de **87 %** des prédictions sont correctes.

- **Précision :** lorsque le modèle prédit un remboursement, cette prédiction est correcte dans **94 %** des cas.

- **Rappel :** le modèle identifie correctement près de **89 %** des emprunteurs qui rembourseront effectivement leur prêt.

- **F1-score :** l'équilibre entre la précision et le rappel confirme la robustesse globale du modèle.
""")

st.divider()

# ==========================================================
# INFORMATIONS
# ==========================================================

col1, col2 = st.columns([2,1])

with col1:

    st.subheader("📌 Résumé")

    st.markdown("""
Le modèle **LightGBM Classifier** a été sélectionné après comparaison avec plusieurs algorithmes de classification supervisée.

Ses performances résultent notamment de :

- l'ingénierie de nouvelles variables (**Feature Engineering**) ;
- l'encodage des variables catégorielles ;
- la gestion du déséquilibre des classes ;
- l'optimisation des hyperparamètres ;
- une validation croisée afin d'améliorer la capacité de généralisation.
""")

with col2:

    st.subheader("ℹ️ Informations")

    st.markdown("""
**Modèle**

LightGBM Classifier

---

**Problème**

Classification binaire

---

**Jeu de données**

≈ 500 000 observations

---

**Variable cible**

Loan Paid Back

---

**Métrique principale**

ROC AUC
""")

st.divider()

# ==========================================================
# CONCLUSION
# ==========================================================

st.success("""
Le modèle présente un excellent compromis entre capacité de discrimination, précision et robustesse.

Associé aux visualisations d'explicabilité SHAP, il fournit des prédictions performantes tout en restant interprétable, un aspect essentiel dans le domaine bancaire et financier.
""")