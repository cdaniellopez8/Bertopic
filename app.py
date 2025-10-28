# app.py
# Full app: Shakira + BERTopic — limpieza interactiva, embeddings, UMAP, TF-IDF interactivo, renombrado de tópicos, evolución y listas.

import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
import json
import os

# NLP / embeddings / topics
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

# visualization & reduction
from umap import UMAP
import plotly.express as px

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Shakira · BERTopic (Interactivo)", layout="wide", page_icon="🎤")
st.title("🎤 Análisis de letras de Shakira — BERTopic")

# -------------------------
# Helpers
# -------------------------
def find_shak_file():
    candidates = [Path.cwd() / "shak.xlsx", Path.cwd().parent / "shak.xlsx", Path(__file__).resolve().parent.parent / "shak.xlsx"]
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None

def normalize_text(text, lower=True, remove_urls=True, remove_punct=True, remove_digits=True):
    t = str(text) if pd.notna(text) else ""
    if lower:
        t = t.lower()
    if remove_urls:
        t = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join([c for c in t if not unicodedata.combining(c)])
    if remove_digits:
        t = re.sub(r"\d+", " ", t)
    if remove_punct:
        t = re.sub(r"[^a-zA-ZñÑáéíóúÁÉÍÓÚüÜ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def basic_text_stats(series):
    s = series.dropna().astype(str)
    counts = s.apply(lambda x: len(x.split()))
    return {"n_docs": int(s.shape[0]), "avg_words": float(counts.mean()) if counts.shape[0]>0 else 0, "min_words": int(counts.min()) if counts.shape[0]>0 else 0, "max_words": int(counts.max()) if counts.shape[0]>0 else 0}

def load_stopwords_musical():
    # Descargar el recurso solo si no está disponible
    nltk.download("stopwords", quiet=True)
    
    # Stopwords base de NLTK
    en = set(stopwords.words("english"))
    es = set(stopwords.words("spanish"))
    
    # Stopwords adicionales musicales y coloquiales
    extra = {
        # sonidos comunes en canciones
        "oh","ohh","ooh","oohh","uh","uhh","ah","ahh","eh","ehh","yeah","ya",
        "yea","na","naah","la","woo","hey","ha","hmm","mmm","yah","uoh",
        # variantes coloquiales o informales
        "pa","ay","toy","ta","e","na","ma","toa","to","vamo","vamonos",
        # palabras comunes en letras
        "amor","vida","corazon","corazón","quiero","quieres","amo","amas","amar",
        "aqui","aquí","ahi","ahí","bien","mal","bebé","nena","neni","mi","tu",
        "tus","mis","sueño","sueños","beso","besos","manos","cuerpo","noche",
        "día","dias","mañana","hoy","ayer","si","sí","mas","más","solo","sola",
        "baby","lady","mami","papi","ohhh","yeahhh","dale","anda","ven","voy",
        "estoy","estas","estás","dime","decir","sabés","sabes","dijo","ver",
        "mirar","mira","mirame","miráme","aqui","ahi","ahi","voy","dia","ena","asi",
        "pasa", 'alo','creo', 'tan', 
    }
    
    # Unión de todas
    all_stopwords = en.union(es).union(extra)
    return all_stopwords

STOPWORDS = load_stopwords_musical()

def remove_stopwords(text, stopwords_set=STOPWORDS):
    if not text or pd.isna(text):
        return ""
    tokens = [tok for tok in text.split() if tok not in stopwords_set and len(tok)>1]
    return " ".join(tokens)

# -------------------------
# 1) Carga del dataset (archivo local)
# -------------------------
st.header("1 — Carga del dataset (archivo local)")
shak_path = find_shak_file()
if shak_path is None:
    st.error("No se encontró 'shak.xlsx' en la carpeta del proyecto ni en la carpeta padre. Coloca el archivo y recarga.")
    st.stop()

st.write(f"Archivo detectado: `{shak_path}`")
try:
    df_raw = pd.read_excel(shak_path, engine="openpyxl")
except Exception as e:
    st.error(f"Error leyendo shak.xlsx: {e}")
    st.stop()

df_raw.columns = [c.lower() for c in df_raw.columns]
required = {"song","year","lyrics"}
if not required.issubset(set(df_raw.columns)):
    st.error(f"El archivo debe tener columnas: {required}. Columnas encontradas: {df_raw.columns.tolist()}")
    st.stop()

st.subheader("Preview (primeras filas)")
st.dataframe(df_raw.head(6))

stats = basic_text_stats(df_raw["lyrics"])
c1, c2, c3 = st.columns(3)
c1.metric("Canciones (no nulas)", stats["n_docs"])
c2.metric("Promedio palabras (raw)", f"{stats['avg_words']:.1f}")
c3.metric("Rango palabras (raw)", f"{stats['min_words']} - {stats['max_words']}")

st.markdown("---")

# -------------------------
# 2) Preprocesamiento interactivo
# -------------------------
st.header("2 — Preprocesamiento (elige qué aplicar)")

with st.expander("Opciones de limpieza (selecciona lo que quieras aplicar)", expanded=True):
    do_lower = st.checkbox("Convertir a minúsculas", value=True)
    do_remove_urls = st.checkbox("Eliminar URLs", value=True)
    do_remove_punct = st.checkbox("Eliminar puntuación (simbolos)", value=True)
    do_remove_digits = st.checkbox("Eliminar números", value=True)
    do_remove_stopwords = st.checkbox("Eliminar stopwords (es + en + interjecciones musicales)", value=True)
    preview_n = st.number_input("Mostrar primeras N filas en preview", min_value=3, max_value=50, value=8)

# If cleaned df exists in session, show it
if "df_clean" in st.session_state:
    st.success("Dataset limpio cargado desde sesión.")
    st.dataframe(st.session_state["df_clean"][["song","year","lyrics_clean"]].head(preview_n))
else:
    st.info("Aún no has aplicado la limpieza. Usa el botón 'Aplicar limpieza' para generar 'lyrics_clean'.")

if st.button("🧹 Aplicar limpieza"):
    df = df_raw.copy()
    df["lyrics_original"] = df["lyrics"].astype(str)
    df["lyrics_norm"] = df["lyrics_original"].apply(lambda t: normalize_text(t, lower=do_lower, remove_urls=do_remove_urls, remove_punct=do_remove_punct, remove_digits=do_remove_digits))
    if do_remove_stopwords:
        df["lyrics_clean"] = df["lyrics_norm"].apply(lambda t: remove_stopwords(t))
    else:
        df["lyrics_clean"] = df["lyrics_norm"]
    st.session_state["df_clean"] = df
    st.success("Limpieza aplicada y guardada en sesión.")
    st.dataframe(df[["song","year","lyrics_original","lyrics_clean"]].head(preview_n))
    post_stats = basic_text_stats(df["lyrics_clean"])
    p1,p2,p3 = st.columns(3)
    p1.metric("Canciones (no nulas)", post_stats["n_docs"])
    p2.metric("Promedio palabras (limpio)", f"{post_stats['avg_words']:.1f}")
    p3.metric("Rango palabras (limpio)", f"{post_stats['min_words']} - {post_stats['max_words']}")
    st.markdown("---")

# Ensure df variable refers to cleaned if exists
if "df_clean" in st.session_state:
    df = st.session_state["df_clean"]
else:
    df = df_raw.copy()

# -------------------------
# 3) Embeddings (igual que antes) + primer embedding ejemplo
# -------------------------
st.header("3 — Generación de embeddings")

# Button to (re)generate embeddings
if "embeddings" not in st.session_state:
    if st.button("⚙️ Generar embeddings (all-MiniLM-L6-v2)"):
        if "lyrics_clean" not in df.columns or df["lyrics_clean"].isna().all():
            st.error("No hay 'lyrics_clean'. Aplica la limpieza en la Sección 2 antes de generar embeddings.")
        else:
            with st.spinner("Cargando modelo y generando embeddings... esto puede tardar"):
                model_name = "all-MiniLM-L6-v2"
                sentence_model = SentenceTransformer(model_name)
                embeddings = sentence_model.encode(df["lyrics_clean"].tolist(), show_progress_bar=True)
                st.session_state["embeddings"] = embeddings
                st.success(f"Embeddings generados ({embeddings.shape}) y guardados en sesión.")
                st.subheader("Ejemplo: primer embedding (primeros 15 valores)")
                st.code(np.round(embeddings[0][:15], 6).tolist())
else:
    st.success("Embeddings ya presentes en sesión.")
    emb = st.session_state["embeddings"]
    st.subheader("Ejemplo: primer embedding (primeros 15 valores)")
    st.code(np.round(emb[0][:15], 6).tolist())
    st.write(f"Dimensiones: {emb.shape}")

st.markdown("---")

# -------------------------
# 4) UMAP 2D visualization (Plotly) with song titles visible
# -------------------------
st.header("4 — Visualización UMAP 2D")

if "embeddings" not in st.session_state:
    st.info("Genera los embeddings en la sección 3 antes de visualizar.")
else:
    emb = st.session_state["embeddings"]
    if st.button("🔍 Reducir a 2D (UMAP) y mostrar"):
        with st.spinner("Reduciendo embeddings a 2D con UMAP..."):
            umap_vis = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=42)
            emb_2d = umap_vis.fit_transform(emb)
            df_vis = pd.DataFrame({"x": emb_2d[:,0], "y": emb_2d[:,1], "song": df["song"].astype(str), "year": pd.to_numeric(df["year"], errors="coerce")})
            st.session_state["emb_2d"] = df_vis
            st.success("Embeddings reducidos a 2D.")
            # Plotly: use hover_name and show song labels option
            fig = px.scatter(df_vis, x="x", y="y", color="year", hover_name="song",
                             color_continuous_scale="Viridis", title="UMAP 2D — canciones (color continuo = year)")
            fig.update_layout(coloraxis_colorbar_title="year")
            st.plotly_chart(fig, use_container_width=True)
            # optional: show all labels (can overlap)
            if st.checkbox("Mostrar etiquetas de canción (puede solaparse)", value=False):
                fig2 = px.scatter(df_vis, x="x", y="y", text="song", color="year", color_continuous_scale="Viridis")
                fig2.update_traces(textposition="top center")
                fig2.update_layout(coloraxis_colorbar_title="year")
                st.plotly_chart(fig2, use_container_width=True)
    elif "emb_2d" in st.session_state:
        st.info("Visualización UMAP disponible en sesión.")
        df_vis = st.session_state["emb_2d"]
        fig = px.scatter(df_vis, x="x", y="y", color="year", hover_name="song", color_continuous_scale="Viridis", title="UMAP 2D — canciones (color continuo = year)")
        fig.update_layout(coloraxis_colorbar_title="year")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -------------------------
# 5) BERTopic
# -------------------------
st.header("5 — BERTopic")
st.markdown("""
**Breve explicación (paso a paso)**:
- **Embeddings**: representaciones numéricas de cada letra (capturan semántica).
- **Reducción UMAP**: conserva estructura semántica para clustering.
- **Clustering**: HDBSCAN encuentra grupos densos (topics) y ruido (topic -1).
- **TF-IDF / c-TF-IDF**: extrae términos representativos por cluster (BERTopic lo combina internamente).
""")

use_umap_for_bt = st.checkbox("Reducir embeddings con UMAP para BERTopic (recomendado, 5D)", value=True)
if use_umap_for_bt:
    if "embeddings_umap" not in st.session_state:
        if st.button("🔁 Reducir embeddings a 5D para BERTopic"):
            with st.spinner("Reduciendo embeddings a 5D con UMAP..."):
                umap_bt = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
                st.session_state["embeddings_umap"] = umap_bt.fit_transform(st.session_state["embeddings"])
                st.success("Embeddings reducidos y guardados en sesión.")
    else:
        st.info("Embeddings para BERTopic ya reducidos en sesión.")
else:
    st.info("No usar reducción previa: se enviarán embeddings originales a BERTopic.")

if st.button("🚀 Entrenar BERTopic"):
    if "embeddings" not in st.session_state:
        st.error("Genera embeddings primero (Sección 3).")
    else:
        emb_to_use = st.session_state.get("embeddings_umap", st.session_state["embeddings"]) if use_umap_for_bt else st.session_state["embeddings"]
        with st.spinner("Entrenando BERTopic (UMAP + HDBSCAN + extracción TF-IDF)... Esto puede tardar."):
            topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=False)
            topics, probs = topic_model.fit_transform(df["lyrics_clean"].tolist(), emb_to_use)
            df["topic"] = topics
            st.session_state["topic_model"] = topic_model
            st.session_state["df_topics"] = df
            st.success("BERTopic entrenado y resultados guardados en sesión.")
            st.dataframe(df[["song","year","topic"]].head(12))

st.markdown("---")

# -------------------------
# 5.1) Visualización Intertopic Distance Map
# -------------------------
st.header("📊 Mapa de intertópicos (BERTopic)")

if "topic_model" not in st.session_state:
    st.info("Entrena BERTopic en la sección 5 para visualizar el mapa de intertópicos.")
else:
    topic_model = st.session_state["topic_model"]
    try:
        topic_info = topic_model.get_topic_info()
        n_topics_total = len(topic_info)

        if n_topics_total <= 2:
            st.info("Hay muy pocos tópicos para generar el mapa de intertópicos (se necesitan al menos 3).")
        else:
            with st.spinner("Generando visualización de intertópicos..."):
                # Las versiones nuevas no usan n_components ni n_clusters
                n_show = min(20, max(3, n_topics_total - 1))
                fig_inter = topic_model.visualize_topics(top_n_topics=n_show)
                st.plotly_chart(fig_inter, use_container_width=True)
                st.caption(f"Mostrando {n_show} tópicos (de {n_topics_total} totales)")
    except Exception as e:
        st.error(f"Error al generar el gráfico de intertópicos: {e}")

st.markdown("---")

# -------------------------
# 6) TF-IDF
# -------------------------
st.header("6 — Explorar términos por tópico (Top-N) usando TF-IDF")

if "topic_model" not in st.session_state:
    st.info("Entrena BERTopic en la sección 5 para explorar términos y renombrar tópicos.")
else:
    topic_model = st.session_state["topic_model"]
    df_topics = st.session_state["df_topics"]

    # Slider para número de términos
    top_n = st.number_input(
        "Selecciona el número de términos por tópico (Top N):",
        min_value=5,
        max_value=100,
        value=10,
        step=5
    )

    # Cargar stopwords
    STOPWORDS = load_stopwords_musical()

    # Obtener IDs de tópicos válidos
    topic_info = topic_model.get_topic_info()
    topic_ids = [int(x) for x in topic_info["Topic"].tolist() if x != -1]

    st.write("Términos por tópico (recalculados desde las letras clasificadas):")
    cols = st.columns(2)

    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    for tid in topic_ids:
        docs_topic = df_topics[df_topics["topic"] == tid]["lyrics_clean"].dropna().tolist()
        docs_topic = [str(doc) for doc in docs_topic if isinstance(doc, str) and doc.strip() != ""]

        if not docs_topic:
            continue

        # Usar lista en lugar de set para evitar error de validación
        vectorizer = CountVectorizer(stop_words=list(STOPWORDS))
        X = vectorizer.fit_transform(docs_topic)
        freqs = X.toarray().sum(axis=0)
        terms = vectorizer.get_feature_names_out()

        freq_df = pd.DataFrame({"word": terms, "freq": freqs})
        freq_df = freq_df.sort_values("freq", ascending=False).head(top_n)

        with cols[0 if tid % 2 == 0 else 1]:
            st.markdown(f"**Tópico {tid}** — {len(docs_topic)} canciones")
            if not freq_df.empty:
                st.write(", ".join(freq_df["word"].tolist()))
            else:
                st.write("_(todas las top palabras eran stopwords o interjecciones musicales)_")

            current_label = st.session_state.get("topic_labels", {}).get(tid, f"Tópico {tid}")
            new_label = st.text_input(f"Nombre para tópico {tid}", value=current_label, key=f"label_{tid}")

            if "topic_labels_temp" not in st.session_state:
                st.session_state["topic_labels_temp"] = {}
            st.session_state["topic_labels_temp"][tid] = new_label

    # Botón para guardar nombres personalizados
    if st.button("💾 Guardar nombres personalizados de tópicos"):
        labels = st.session_state.get("topic_labels", {})
        temp = st.session_state.get("topic_labels_temp", {})
        labels.update(temp)
        st.session_state["topic_labels"] = labels
        st.success("Nombres de tópicos guardados en sesión.")

st.markdown("---")

# -------------------------
# 7) Visualizaciones de topicos y su evolución por año
# -------------------------
st.header("7 — Visualizaciones de tópicos y su evolución por año")

if "topic_model" not in st.session_state:
    st.info("Entrena BERTopic y guarda nombres de tópicos para ver visualizaciones.")
else:
    topic_model = st.session_state["topic_model"]
    df_topics = st.session_state["df_topics"].copy()
    labels = st.session_state.get("topic_labels", {})
    # map labels
    df_topics["topic_label"] = df_topics["topic"].apply(lambda t: labels.get(int(t), f"Topic {int(t)}"))

    # Bar chart — distribution by label
    st.subheader("Distribución de canciones por tópico (labels personalizados)")
    counts = df_topics["topic_label"].value_counts().reset_index()
    counts.columns = ["topic_label","count"]
    fig_bar = px.bar(counts, x="topic_label", y="count", title="Songs per topic (custom labels)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Evolution per year (continuous)
    st.subheader("Evolución de tópicos por año (continuo)")
    df_topics["year_num"] = pd.to_numeric(df_topics["year"], errors="coerce")
    trends = df_topics.dropna(subset=["year_num"]).groupby(["year_num","topic_label"]).size().reset_index(name="count")
    if trends.empty:
        st.info("No hay datos numéricos de 'year' suficientes para trazar evolución.")
    else:
        # limit to top K labels for clarity
        top_labels = df_topics["topic_label"].value_counts().index.tolist()[:8]
        trends_plot = trends[trends["topic_label"].isin(top_labels)]
        fig_trend = px.line(trends_plot, x="year_num", y="count", color="topic_label", markers=True,
                            title="Evolución (count) de topics por year (top labels)")
        fig_trend.update_xaxes(title="Year", type="linear")
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    # Songs and lyrics by topic (expanders)
    st.subheader("Lista de canciones y letras por tópico")
    for label in sorted(df_topics["topic_label"].unique(), key=lambda x: str(x)):
        subset = df_topics[df_topics["topic_label"] == label][["song","year","lyrics_original","lyrics_clean"]].copy()
        with st.expander(f"{label} — {len(subset)} canciones", expanded=False):
            nshow = st.number_input(f"Mostrar cuántas canciones de {label}?", min_value=1, max_value=100, value=5, key=f"nshow_{label}")
            st.dataframe(subset.head(nshow).reset_index(drop=True))

    # export csv
    if st.button("💾 Exportar asignaciones (shak_topics.csv)"):
        out = df_topics[["song","year","topic","topic_label","lyrics_clean"]].copy()
        out.to_csv("shak_topics.csv", index=False, encoding="utf-8")
        st.success("shak_topics.csv guardado en la carpeta del proyecto.")

st.markdown("---")
st.caption("Flujo: 1) carga → 2) limpieza interactiva → 3) embeddings (igual que antes) → 4) UMAP (plotly) → 5) BERTopic → 6) TF-IDF top-N + renombrado → 7) visualizaciones y listas.")






