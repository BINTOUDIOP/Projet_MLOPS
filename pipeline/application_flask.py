import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Définir le chemin absolu du modèle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "BestModel_LogisticRegression_0.9920.pkl")

# Chargement du pipeline
try:
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    print(" Modèle chargé avec succès.")
except Exception as e:
    pipeline = None
    print(f" Erreur lors du chargement du modèle : {e}")

# Liste des features utilisées pour la prédiction
FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score"
]

def model_pred(features):
    """Fait une prédiction avec le pipeline ML."""
    df = pd.DataFrame([features])
    prediction = int(pipeline.predict(df)[0])
    probability = float(pipeline.predict_proba(df)[:, 1][0])
    return prediction, probability


@app.route("/", methods=["GET"])
def home():
    """Page d’accueil simple (HTML si présent ou JSON par défaut)."""
    try:
        return render_template("index.html", features=FEATURES)
    except Exception:
        return jsonify({
            "message": "Bienvenue sur l’API de scoring MLOps déployée sur Azure ",
            "features": FEATURES
        })


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint principal de prédiction."""
    if pipeline is None:
        return jsonify({"error": "Le modèle n’a pas été chargé."}), 500

    try:
        data = request.get_json()

        # Vérification des clés attendues
        for feature in FEATURES:
            if feature not in data:
                return jsonify({"error": f"Valeur manquante pour '{feature}'"}), 400

        prediction, probability = model_pred(data)

        if prediction == 1:
            interpretation = (
                "Risque ÉLEVÉ : le client présente une probabilité élevée de défaut."
            )
        else:
            interpretation = (
                "  Le client semble fiable : le prêt peut être accordé."
            )

        return jsonify({
            "predicted_class": prediction,
            "default_probability": round(probability, 4),
            "interpretation": interpretation
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Azure définit automatiquement la variable d’environnement PORT
    port = int(os.environ.get("PORT", 5051))
    app.run(host="0.0.0.0", port=port)