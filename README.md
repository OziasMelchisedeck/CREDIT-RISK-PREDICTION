# 🏦 AI Credit Risk Assessment System

## 📌 Présentation du projet

**AI Credit Risk Assessment System** est une application web développée avec **Streamlit** permettant d'évaluer le risque de crédit d'un emprunteur grâce à un modèle de Machine Learning.

L'objectif principal est de prédire la probabilité qu'un demandeur de prêt rembourse effectivement son crédit, tout en fournissant une explication claire des facteurs ayant influencé la décision du modèle.

Ce projet met en œuvre un pipeline complet de Data Science allant de la préparation des données jusqu'au déploiement d'une application interactive intégrant des techniques d'**Explainable AI (XAI)**.

---

# 🎯 Objectifs du projet

Dans un contexte bancaire, l'évaluation du risque de crédit constitue un enjeu majeur permettant :

- d'améliorer la prise de décision lors de l'octroi des prêts ;
- de réduire le risque de défaut ;
- d'automatiser l'analyse des dossiers clients ;
- de fournir des décisions plus rapides et cohérentes.

Cette application vise à proposer une solution combinant :

✅ performance prédictive  
✅ interprétabilité des résultats  
✅ simplicité d'utilisation  

---

# 📊 Jeu de données

Le projet utilise le jeu de données issu de la compétition Kaggle :

**Playground Series - Season 5 Episode 11 : Loan Approval Prediction**

🔗 Dataset :
https://www.kaggle.com/competitions/playground-series-s5e11/data

Le dataset contient plus de :

> **500 000 observations**

Chaque observation représente une demande de prêt avec différentes caractéristiques financières et personnelles :

- revenu annuel ;
- score de crédit ;
- ratio d'endettement ;
- montant du prêt ;
- taux d'intérêt ;
- niveau d'éducation ;
- situation professionnelle ;
- situation matrimoniale ;
- objectif du prêt ;
- etc.

L'objectif est une classification binaire :

| Classe | Signification |
|---|---|
| 0 | Défaut de remboursement |
| 1 | Remboursement du prêt |

La métrique principale utilisée dans la compétition est :

> **ROC AUC**

car elle permet d'évaluer efficacement la capacité du modèle à distinguer les différentes classes.

---

# 🧠 Approche Machine Learning

## 1. Analyse et préparation des données

Les principales étapes réalisées :

- analyse exploratoire des données ;
- étude des distributions ;
- analyse des variables importantes ;
- gestion des valeurs manquantes ;
- préparation des variables catégorielles.

---

## 2. Feature Engineering

Une attention particulière a été portée à la création et la sélection des variables afin d'améliorer la capacité du modèle à détecter les profils présentant un risque.

Les transformations réalisées permettent notamment de mieux représenter :

- la capacité financière de l'emprunteur ;
- son niveau d'endettement ;
- son profil de risque ;
- les relations entre les différentes caractéristiques financières.

---

## 3. Encodage des variables catégorielles

Les variables qualitatives ont été adaptées au modèle grâce à :

- **Ordinal Encoding** pour les variables possédant une notion d'ordre ;
- **One Hot Encoding** pour les variables nominales.

Cette étape est intégrée directement dans un pipeline afin d'assurer la reproductibilité du processus.

---

## 4. Gestion du déséquilibre des classes

Le problème de classification présente un déséquilibre entre les différentes catégories.

Différentes stratégies ont été étudiées afin d'améliorer la capacité du modèle à apprendre correctement les deux classes :

- analyse de la distribution de la variable cible ;
- ajustement du poids des classes ;
- optimisation selon la métrique ROC AUC.

---

# 🤖 Modèle utilisé

Le modèle final utilisé est :

## LightGBM Classifier

LightGBM a été choisi pour :

- ses excellentes performances sur les données tabulaires ;
- sa rapidité d'entraînement ;
- sa capacité à gérer des volumes importants de données ;
- son efficacité sur les problèmes de classification.

Le modèle a été entraîné avec :

- optimisation des hyperparamètres ;
- validation croisée ;
- recherche du meilleur compromis entre précision et généralisation.

---

# 📈 Performances du modèle

Les performances obtenues sur le jeu de validation :

| Métrique | Score |
|---|---:|
| ROC AUC | 0.921 |
| Accuracy | 86.8 % |
| Precision | 94.3 % |
| Recall | 88.7 % |
| F1-score | 91.5 % |

Ces résultats montrent une bonne capacité du modèle à identifier les profils présentant un risque tout en limitant les erreurs de classification.

---

# 🔍 Explainable AI (SHAP)

Une attention particulière a été portée à l'interprétabilité du modèle.

L'application intègre la bibliothèque :

## SHAP (SHapley Additive exPlanations)

Elle permet de :

- comprendre l'importance globale des variables ;
- identifier les facteurs influençant les prédictions ;
- expliquer individuellement une décision du modèle ;
- améliorer la transparence des résultats.

Deux types d'explications sont disponibles :

### 🌍 Explication globale

Analyse des variables les plus influentes sur l'ensemble du dataset :

- SHAP Summary Plot ;
- Importance globale des variables.

### 👤 Explication locale

Analyse d'une prédiction individuelle :

- Waterfall Plot ;
- contribution de chaque variable dans la décision finale.

Cette approche est particulièrement importante dans les domaines financiers où les modèles doivent être explicables et auditables.

---

# 🖥️ Application Streamlit

L'application propose plusieurs interfaces :

## 🏠 Accueil

Présentation générale du système d'évaluation du risque de crédit.

---

## 🔮 Prediction

Permet à un utilisateur :

- de renseigner les informations d'un emprunteur ;
- d'obtenir une prédiction ;
- d'obtenir la probabilité de remboursement.

---

## 📊 Performances

Présente :

- les métriques du modèle ;
- les résultats d'évaluation ;
- les analyses de performance.

---

## 🔍 Explicabilité

Permet d'explorer :

- l'importance des variables ;
- les contributions SHAP ;
- les facteurs influençant une décision.

---

# 🏗️ Architecture du projet
Credit-Risk-Prediction
│
├── Accueil.py
│
├── pages/
│ ├── Apropos.py
│ ├── Explicabilite.py
│ ├── Performances.py
│ └── Prediction.py
│
├── utils/
│ └── prediction.py
│ └── shap.py
├── models/
│ └── loan_pipeline_lgbmClassifier.pkl
¦ └── shape_cache_loan.pkl
│
├── notebooks/
│ └── predicted_loan.ipynb
│
├── requirements.txt
│
└── README.md


---

# 🛠️ Technologies utilisées

## Langage

- Python

## Machine Learning

- LightGBM
- Scikit-learn
- SHAP

## Data Processing

- Pandas
- NumPy

## Visualisation

- Matplotlib

## Déploiement

- Streamlit

## Versioning

- Git
- GitHub

---

# 🚀 Installation et utilisation

## Cloner le projet

```bash
git clone https://github.com/OziasMelchisedeck/CREDIT-RISK-PREDICTION.git