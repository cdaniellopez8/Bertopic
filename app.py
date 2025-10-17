import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import openai
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("🎵 BERTopic paso a paso con nombres de temas usando GPT")

# -------------------------------
# API Key de OpenAI
# -------------------------------
# Guarda tu API key en Streamlit Secrets: st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets.get("OPENAI_API_KEY", "TU_API_KEY_AQUI")

# -------------------------------
# Función para generar nombres de temas
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
        name = response.choices[0].message['content'].strip()
        return name
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

        # -------------------------------
        # Mostrar embeddings en 3D
        # -------------------------------
        st.subheader("Visualización 3D de embeddings")
        # Reducimos dimensionalidad a 3D con KMeans (opcional)
        fig = px.scatter_3d(
            x=embeddings[:,0], y=embeddings[:,1], z=embeddings[:,2],
            hover_data={'Song': df['song'], 'Year': df['year']},
            title="Embeddings 3D (primeras 3 dimensiones)"
        )
        st.plotly_chart(fig)

    # -------------------------------
    # Paso 2: Generar tópicos con BERTopic
    # -------------------------------
    if 'embeddings' in st.session_state and st.button("2️⃣ Generar temas (BERTopic)"):
        with st.spinner("Generando temas..."):
            topic_model = BERTopic(
                embedding_model=None,
                hdbscan_model=None,  # desactiva HDBSCAN
                verbose=True
            )
            topics, probs = topic_model.fit_transform(df['lyrics'].astype(str), embeddings=st.session_state['embeddings'])
            df['topic'] = topics
            st.session_state['topic_model'] = topic_model
            st.session_state['df'] = df
        st.success("✅ Temas generados")

        st.subheader("Tabla de temas asignados")
        st.dataframe(df[['song','year','topic']])

    # -------------------------------
    # Paso 3: Nombrar los temas con GPT
    # -------------------------------
    if 'topic_model' in st.session_state and st.button("3️⃣ Nombrar temas con GPT"):
        topic_info = st.session_state['topic_model'].get_topic_info()
        topic_names = {}
        for topic_id in topic_info['Topic']:
            if topic_id != -1:  # ignorar outliers
                keywords = [word for word, _ in st.session_state['topic_model'].get_topic(topic_id)]
                topic_names[topic_id] = generate_topic_name(keywords)
        df['topic_name'] = df['topic'].map(topic_names)
        st.session_state['df'] = df
        st.session_state['topic_names'] = topic_names
        st.success("✅ Nombres de temas generados")
        st.dataframe(df[['song','year','topic_name']])

    # -------------------------------
    # Paso 4: Visualización de temas interactiva
    # -------------------------------
    if 'topic_model' in st.session_state and st.button("4️⃣ Visualización interactiva"):
        st.subheader("Visualización de temas")
        fig = st.session_state['topic_model'].visualize_topics()
        st.plotly_chart(fig)