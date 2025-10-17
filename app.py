import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import openai

st.set_page_config(layout="wide")
st.title("🎵 Demo pedagógica de BERTopic paso a paso")

# -------------------------------
# API Key de OpenAI desde st.secrets
# -------------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY", "TU_API_KEY_AQUI")

# -------------------------------
# Función para generar nombres de temas con GPT
# -------------------------------
def generate_topic_name(keywords):
    prompt = f"Estas son las palabras clave de un tema: {keywords}. " \
             "Sugiere un nombre corto y representativo para este tema."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        return response.choices[0].message['content'].strip()
    except:
        return "Tema automático"

# -------------------------------
# Paso 0: Subir archivo
# -------------------------------
uploaded_file = st.file_uploader("Sube tu archivo shak.xlsx", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # -------------------------------
    # Paso 1: Crear embeddings
    # -------------------------------
    if st.button("1️⃣ Crear embeddings"):
        with st.spinner("Generando embeddings..."):
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embedder.encode(df['lyrics'].astype(str), show_progress_bar=True)
            st.session_state['embeddings'] = embeddings
        st.success("✅ Embeddings generados")
        st.write("Ejemplos de embeddings (primeros 3):")
        st.write(embeddings[:3])

        # Visualización 3D
        st.subheader("Embeddings en 3D (primeras 3 dimensiones)")
        fig = px.scatter_3d(
            x=embeddings[:,0], y=embeddings[:,1], z=embeddings[:,2],
            hover_data={'Song': df['song'], 'Year': df['year']},
            title="Embeddings 3D"
        )
        st.plotly_chart(fig)

    # -------------------------------
    # Paso 2: Clustering con KMeans
    # -------------------------------
    if 'embeddings' in st.session_state and st.button("2️⃣ Clustering con KMeans"):
        num_topics = st.slider("Número de tópicos (clusters)", min_value=3, max_value=15, value=5)
        kmeans_model = KMeans(n_clusters=num_topics, random_state=42)
        labels = kmeans_model.fit_predict(st.session_state['embeddings'])
        df['topic'] = labels
        st.session_state['kmeans_labels'] = labels
        st.session_state['num_topics'] = num_topics
        st.success(f"✅ Clustering completado: {num_topics} tópicos")
        st.dataframe(df[['song','year','topic']])

    # -------------------------------
    # Paso 3: BERTopic + TF-IDF palabras clave
    # -------------------------------
    if 'kmeans_labels' in st.session_state and st.button("3️⃣ Extraer palabras clave con TF-IDF (BERTopic)"):
        topic_model = BERTopic(embedding_model=None, verbose=True)
        topics, probs = topic_model.fit_transform(df['lyrics'].astype(str),
                                                 embeddings=st.session_state['embeddings'])
        # Reemplazar clusters con KMeans
        topic_model.update_topics(df['lyrics'].astype(str), topics=st.session_state['kmeans_labels'],
                                  embeddings=st.session_state['embeddings'])
        df['topic'] = st.session_state['kmeans_labels']
        st.session_state['topic_model'] = topic_model

        # Mostrar palabras clave TF-IDF
        st.subheader("Palabras clave de cada tópico")
        for topic_id in range(st.session_state['num_topics']):
            words = [word for word, _ in topic_model.get_topic(topic_id)]
            st.write(f"Tópico {topic_id}: {', '.join(words)}")

    # -------------------------------
    # Paso 4: Nombrar tópicos con GPT
    # -------------------------------
    if 'topic_model' in st.session_state and st.button("4️⃣ Nombrar tópicos con GPT"):
        topic_names = {}
        for topic_id in range(st.session_state['num_topics']):
            words = [word for word, _ in st.session_state['topic_model'].get_topic(topic_id)]
            topic_names[topic_id] = generate_topic_name(words)

        df['topic_name'] = df['topic'].map(topic_names)
        st.session_state['df'] = df
        st.success("✅ Nombres generados por GPT")
        st.dataframe(df[['song','year','topic','topic_name']])

    # -------------------------------
    # Paso 5: Visualización interactiva
    # -------------------------------
    if 'topic_model' in st.session_state and st.button("5️⃣ Visualizar tópicos"):
        st.subheader("Visualización de tópicos")
        try:
            fig = st.session_state['topic_model'].visualize_topics()
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"No se pudo generar la visualización: {e}")
