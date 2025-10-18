import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "BestModel_LogisticRegression_0.9920.pkl")

# Chargement du pipeline complet
with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score"
]

def model_pred(features):

    df = pd.DataFrame([features])
    prediction = int(pipeline.predict(df)[0])
    probability = float(pipeline.predict_proba(df)[:, 1][0])
    return prediction, probability



@app.route("/", methods=["GET"])
def home():
    # On passe la liste des features à l’interface HTML
    return render_template("index.html", features=FEATURES)



@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()


        for feature in FEATURES:
            if feature not in data:
                return jsonify({"error": f"Valeur manquante pour {feature}"}), 400


        prediction, probability = model_pred(data)


        if prediction == 1:
            interpretation = (
                f"⚠ Le modèle estime que ce client présente un RISQUE ÉLEVÉ "
                f"de faire défaut sur le prêt (probabilité {probability*100:.1f}%)."
            )
        else:
            interpretation = (
                f" Le client semble fiable : faible probabilité de défaut "
                f"({probability*100:.1f}%). Le prêt peut être accordé."
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
    app.run(host="0.0.0.0", port=5051, debug=True)