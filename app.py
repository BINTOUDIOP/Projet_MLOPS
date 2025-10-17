import streamlit as st
import joblib
import pandas as pd

# Chargement modèle et scaler
rf = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Titre de l'app
st.title("Prédicteur de défaut de crédit bancaire")

# Formulaire de saisie utilisateur
credit_lines_outstanding = st.number_input("Nombre de lignes de crédit en cours", min_value=0, max_value=20, value=2)
loan_amt_outstanding = st.number_input("Montant du prêt en cours", min_value=0.0, value=5000.0)
total_debt_outstanding = st.number_input("Dette totale en cours", min_value=0.0, value=7000.0)
income = st.number_input("Revenu annuel", min_value=0.0, value=45000.0)
years_employed = st.number_input("Nombre d'années d'emploi", min_value=0, max_value=50, value=5)
fico_score = st.number_input("Score de crédit FICO", min_value=300, max_value=850, value=680)

# Fonction prédictive
def predict_default(inputs):
    df = pd.DataFrame([inputs])
    X_scaled = scaler.transform(df)
    proba = rf.predict_proba(X_scaled)[0, 1]
    label = rf.predict(X_scaled)[0]
    return {"proba_defaut": proba, "defaut": label}

# Bouton de prédiction
if st.button("Estimer le risque de défaut"):
    data = {
        "credit_lines_outstanding": credit_lines_outstanding,
        "loan_amt_outstanding": loan_amt_outstanding,
        "total_debt_outstanding": total_debt_outstanding,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico_score
    }
    result = predict_default(data)
    st.write(f"Probabilité estimée de défaut : {result['proba_defaut']:.2%}")
    st.write(f"Prédiction finale : {'Défaillant' if result['defaut'] == 1 else 'Non défaillant'}")
