# ==== Étape 1 : Image de base ====
FROM python:3.11-slim

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers de projet
COPY . /app

# Mise à jour des paquets et installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Installation des bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposition du port pour Streamlit et MLflow
EXPOSE 8501
EXPOSE 5000

# Variable d'environnement pour MLflow
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Démarrage par défaut de Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
