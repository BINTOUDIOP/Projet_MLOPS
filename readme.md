# Prédiction de défaut de prêt

## 1. Contexte
Dans le secteur bancaire, les prêts personnels représentent une source de revenus mais comportent un risque de défaut. L’objectif de ce projet est de prédire la probabilité de défaut de chaque client afin d’aider la banque à anticiper les pertes potentielles et à allouer le capital nécessaire.

## 2. Objectifs
- Prétraiter les données et effectuer une analyse exploratoire.
- Construire et entraîner des modèles de classification : Logistic Regression, Decision Tree, Random Forest.
- Suivre et comparer les modèles avec MLflow.
- Déployer une API Flask pour les prédictions.
- Mettre en place un pipeline CI/CD avec Docker et Azure.

## 3. Installation
1. Cloner le repository :

       git clone https://github.com/BINTOUDIOP/Projet_MLOPS
       cd Projet_MLOPS

2. Créer un environnement virtuel et installer les dépendances :


    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows
    pip install --upgrade pip
    pip install -r requirements.txt

## 4. Prétraitement et pipeline
- Le notebook `Exploration_Pretraitement_Donnees.ipynb` contient l’analyse exploratoire et le prétraitement.
- Les données sont standardisées et sauvegardées dans `loan_data_preprocessed.csv`.
- Le pipeline inclut la sélection des features et le scaling des données avant l’entraînement.

## 5. Entraînement et suivi des modèles
- Trois modèles testés : Logistic Regression, Decision Tree, Random Forest.
- Les données sont équilibrées via RandomOverSampler pour gérer le déséquilibre.
- Meilleur modèle sauvegardé automatiquement dans le dossier `Models`.
- Métriques calculées : Accuracy, F1, Precision, Recall, ROC AUC, matrice de confusion.
- Figures et importances des features sauvegardées pour présentation.

## 6. MLflow Tracking
- Chaque modèle correspond à un experiment dans MLflow.
- Chaque itération correspond à un run.
- Tous les paramètres, métriques et artifacts (matrices, figures) sont loggés.

## 7. API Flask
- L’API est dans `pipeline/application_flask.py`.
- Endpoint `/predict` : reçoit JSON avec les features et retourne la classe prédite, la probabilité et une phrase d’interprétation.
- Interface web simple accessible via le port 5051 en local.

## 8. CI/CD avec Docker
- Dockerfile créé dans le dossier `pipeline`.
- Image Docker buildée et poussée sur Docker Hub.
- Azure Web App configurée pour récupérer l’image Docker (déploiement testé et reussi sur azure).

## 9. Déploiement
- L’API Flask est accessible publiquement sur : https://mlops-flask-app.azurewebsites.net/


## 10. Conclusion et perspectives
- Projet MLOps end-to-end : exploration, prétraitement, modèles, tracking MLflow, API Flask.
