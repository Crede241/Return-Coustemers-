🧠 Iris — Multilingual Sentiment Analysis App

Iris est une application web développée avec Streamlit permettant d’analyser automatiquement des retours clients en français et en anglais grâce à un modèle NLP basé sur BERT.

L’objectif est de transformer des commentaires textuels en indicateurs décisionnels exploitables.

🚀 Features

✅ Analyse automatique du sentiment (1–5 ⭐)

✅ Conversion en catégories métier : Positif / Neutre / Négatif

✅ Visualisations interactives (répartition, évolution temporelle)

✅ Extraction des mots-clés récurrents

✅ Filtrage par date et canal

✅ Export des résultats en CSV

✅ Support multilingue (FR/EN)

🧠 Model

Modèle utilisé :
nlptown/bert-base-multilingual-uncased-sentiment

Basé sur BERT

Classification en 5 niveaux (1 à 5 étoiles)

Adapté aux textes multilingues

🛠️ Tech Stack

Python

Streamlit

HuggingFace Transformers

Pandas & NumPy

Plotly

Regex preprocessing

📊 Pipeline

Chargement des données (CSV)

Nettoyage du texte (lowercase, suppression URLs, regex)

Prédiction par batch

Transformation en catégories métier

Visualisation & KPI

Export des résultats

⚙️ Installation
git clone https://github.com/ton-username/iris-sentiment-app.git
cd iris-sentiment-app
pip install -r requirements.txt
streamlit run app.py
📁 Expected CSV Format

Minimum required column:

comment

Optional columns:

date
channel
rating
💡 Use Cases

Analyse d’avis clients (e-commerce)

Enquêtes de satisfaction

Customer Experience analytics

Monitoring de réputation

📌 What This Project Demonstrates

End-to-end NLP pipeline

Integration of pre-trained transformer models

Interactive data visualization

Business-oriented data interpretation

Clean and reusable Streamlit architecture
