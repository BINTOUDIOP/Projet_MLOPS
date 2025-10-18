# pipeline/test_model.py

import pickle
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "BestModel_LogisticRegression_0.9920.pkl")

# Chargement du modèle
with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

# Données factices pour le test
test_data = pd.DataFrame([{
    "credit_lines_outstanding": 3,
    "loan_amt_outstanding": 5000,
    "total_debt_outstanding": 1500,
    "income": 450,
    "years_employed": 5,
    "fico_score": 700
}])


# --- Fonction de test ---
def test_prediction():
    prediction = pipeline.predict(test_data)[0]
    proba = pipeline.predict_proba(test_data)[:, 1][0]

    print("Prediction:", prediction)
    print("Probability of default:", proba)

    # Assertion simple pour pytest
    assert prediction in [0, 1], "La prédiction doit être 0 ou 1"
    assert 0.0 <= proba <= 1.0, "La probabilité doit être comprise entre 0 et 1"


# --- Exécution si on lance directement le script ---
if __name__ == "__main__":
    test_prediction()