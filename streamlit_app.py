
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from collections import Counter
from datetime import datetime

# Lazy import for transformers to speed up app start
@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=-1)
    return pipe

def rate_to_label(stars: int) -> str:
    if stars <= 2:
        return "Négatif"
    elif stars == 3:
        return "Neutre"
    else:
        return "Positif"

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-ZàâäéèêëîïôöùûüçœÀÂÄÉÈÊËÎÏÔÖÙÛÜÇŒ' -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Simple FR/EN stopwords to keep the app lightweight (you can extend this list)
STOPWORDS = set("""
    a aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me même 
    mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous 
    c d j l à m n s t y était êtes être été avoir ai as a avons avez ont suis es est sommes êtes sont 
    the and is are was were be been being i you he she it we they me him her us them my your yours his hers its our their 
    this that these those of to for from in on at by with about into over after before under above again further then once
""".split())

def top_keywords(texts, k=20):
    tokens = []
    for t in texts:
        t = clean_text(str(t))
        tokens.extend([w for w in t.split() if len(w) > 2 and w not in STOPWORDS])
    return Counter(tokens).most_common(k)

def predict_dataframe(df, text_col, pipe):
    # Run predictions in batches for speed
    texts = df[text_col].fillna("").astype(str).tolist()
    preds = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        outputs = pipe(batch)
        # nlptown model returns labels like '1 star' .. '5 stars'
        for out in outputs:
            label = out['label']
            stars = int(label.split()[0])
            preds.append(stars)
    df = df.copy()
    df["rating_pred"] = preds
    df["sentiment"] = df["rating_pred"].apply(rate_to_label)
    return df

st.set_page_config(page_title="Iris - Analyse de Sentiments", layout="wide")
import os

if os.path.exists("iris_logo.png"):
    st.image("iris_logo.png", width=120)
st.title("Analyse de Sentiments des Retours Clients")
st.caption("Multilingue (FR/EN) • Modèle : nlptown/bert-base-multilingual-uncased-sentiment (1–5 ⭐)")

with st.sidebar:
    st.header("⚙️ Paramètres")
    st.write("1) Importez un fichier CSV des commentaires clients.")
    demo = st.toggle("Utiliser le jeu de démo", value=True)
    uploaded = st.file_uploader("Ou chargez votre CSV", type=["csv"])
    text_col = st.text_input("Nom de la colonne Texte", value="comment")
    date_col = st.text_input("Nom de la colonne Date (optionnel)", value="date")
    channel_col = st.text_input("Nom de la colonne Canal (optionnel)", value="channel")
    rating_col = st.text_input("Nom de la colonne Note existante 1–5 (optionnel)", value="rating")
    st.divider()
    st.write("2) Modèle de classification")
    model_name = st.text_input("Modèle HF", value="nlptown/bert-base-multilingual-uncased-sentiment")
    run_btn = st.button("🔮 Lancer l'analyse", use_container_width=True)

if demo and uploaded is None:
    df = pd.read_csv("sample_data.csv", sep=';')
elif uploaded is not None:
    df = pd.read_csv(uploaded, sep=';')
else:
    st.info("Chargez un CSV ou activez le jeu de démo.", icon="ℹ️")
    st.stop()

# Basic validation
if text_col not in df.columns:
    st.error(f"Colonne texte '{text_col}' introuvable dans le CSV. Colonnes dispo : {list(df.columns)}")
    st.stop()

# Parse date if present
if date_col in df.columns:
    with st.spinner("Conversion des dates…"):
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            pass

# Filters
with st.expander(" Filtres", expanded=True):
    if channel_col in df.columns:
        channels = ["(Tous)"] + sorted([c for c in df[channel_col].dropna().unique()])
        ch = st.selectbox("Canal", channels, index=0)
        if ch != "(Tous)":
            df = df[df[channel_col] == ch]
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        dmin = df[date_col].min()
        dmax = df[date_col].max()
        if pd.notnull(dmin) and pd.notnull(dmax):
            start, end = st.slider("Période", min_value=dmin.to_pydatetime(), max_value=dmax.to_pydatetime(),
                                   value=(dmin.to_pydatetime(), dmax.to_pydatetime()))
            df = df[(df[date_col] >= start) & (df[date_col] <= end)]

if run_btn:
    with st.spinner("Chargement du modèle et inférence… (1ère fois un peu plus longue)"):
        pipe = load_model(model_name)
        df_pred = predict_dataframe(df, text_col, pipe)
else:
    # Si l'utilisateur n'a pas encore cliqué, proposer un aperçu sans prédictions
    df_pred = df.copy()
    if "rating" in df_pred.columns:
        # Use pandas nullable Int64 type to allow NaN values
        df_pred["rating_pred"] = df_pred["rating"].clip(1,5).astype('Int64')
        df_pred["sentiment"] = df_pred["rating_pred"].apply(lambda x: rate_to_label(x) if pd.notnull(x) else np.nan)
    else:
        df_pred["rating_pred"] = np.nan
        df_pred["sentiment"] = np.nan

# KPIs
total = len(df_pred)
pos = int((df_pred["sentiment"] == "Positif").sum())
neu = int((df_pred["sentiment"] == "Neutre").sum())
neg = int((df_pred["sentiment"] == "Négatif").sum())
pos_pct = (pos / total * 100) if total else 0
neg_pct = (neg / total * 100) if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avis total", f"{total}")
c2.metric("😊 % Positifs", f"{pos_pct:.1f}%")
c3.metric("😐 Neutres", f"{neu}")
c4.metric("☹️ Négatifs", f"{neg} ({neg_pct:.1f}%)")

st.divider()
colA, colB = st.columns([3,2])

# Pie sentiment
with colA:
    if df_pred["sentiment"].notna().any():
        fig = px.pie(df_pred, names="sentiment", title="Répartition des sentiments")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune prédiction encore. Cliquez sur **Lancer l'analyse** pour générer les sentiments.", icon="🤖")

# Top keywords
with colB:
    st.subheader(" Mots-clés récurrents (Top 20)")
    topk = top_keywords(df_pred[text_col].dropna().tolist(), k=20)
    if topk:
        kw_df = pd.DataFrame(topk, columns=["mot", "fréquence"])
        st.dataframe(kw_df, use_container_width=True, hide_index=True)
    else:
        st.write("—")

# Timeline
if date_col in df_pred.columns and pd.api.types.is_datetime64_any_dtype(df_pred[date_col]):
    st.subheader("⏳ Sentiment dans le temps")
    tmp = df_pred.dropna(subset=[date_col]).copy()
    if len(tmp):
        tmp["date_only"] = tmp[date_col].dt.date
        daily = tmp.groupby(["date_only", "sentiment"]).size().reset_index(name="count")
        fig2 = px.line(daily, x="date_only", y="count", color="sentiment", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

# Negative themes
st.subheader("Thèmes négatifs récurrents")
neg_rows = df_pred[df_pred["sentiment"] == "Négatif"]
if len(neg_rows):
    neg_top = top_keywords(neg_rows[text_col].tolist(), k=20)
    neg_df = pd.DataFrame(neg_top, columns=["mot", "fréquence"])
    st.dataframe(neg_df, use_container_width=True, hide_index=True)
else:
    st.write("Aucun commentaire négatif sur l'échantillon.")

st.divider()
st.subheader("Aperçu des avis")
show_cols = [c for c in [date_col if date_col in df_pred.columns else None,
                         channel_col if channel_col in df_pred.columns else None,
                         text_col, "rating_pred", "sentiment"] if c]
st.dataframe(df_pred[show_cols].head(200), use_container_width=True, hide_index=True)

# Download
st.download_button("Télécharger les résultats (CSV)",
                   data=df_pred.to_csv(index=False).encode("utf-8"),
                   file_name="sentiment_results.csv",
                   mime="text/csv")

st.caption("Astuce : ajoutez des colonnes (p.ex. 'product', 'region') et utilisez le filtre pour explorer vos segments.")
