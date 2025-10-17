import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import openai

st.set_page_config(layout="wide")
st.title("🎵 Demo interactiva de BERTopic paso a paso con GPT")

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

        # -------------------------------
        # Visualización 3D
        # -------------------------------
        st.subheader("Embeddings en 3D (primeras 3 dimensiones)")
        fig = px.scatter_3d(
            x=embeddings[:,0], y=embeddings[:,1], z=embeddings[:,2],
            hover_data={'Song': df['song'], 'Year': df['year']},
            title="Embeddings 3D"
        )
        st.plotly_chart(fig)

    # -------------------------------
    # Paso 2: Generar temas con BERTopic + KMeans
    # -------------------------------
    if 'embeddings' in st.session_state and st.button("2️⃣ Generar temas (BERTopic)"):
        with st.spinner("Generando temas..."):
            num_topics = st.slider("Número de tópicos (clusters)", min_value=3, max_value=15, value=5)
            kmeans_model = KMeans(n_clusters=num_topics, random_state=42)

            topic_model = BERTopic(
                embedding_model=None,
                verbose=True,
                calculate_probabilities=True
            )
            topics, probs = topic_model.fit_transform(df['lyrics'].astype(str),
                                                     embeddings=st.session_state['embeddings'],
                                                     clustering_model=kmeans_model)
            df['topic'] = topics
            st.session_state['topic_model'] = topic_model
            st.session_state['df'] = df
        st.success("✅ Temas generados")
        st.subheader("Tabla de temas asignados")
        st.dataframe(df[['song','year','topic']])

    # -------------------------------
    # Paso 3: Nombrar temas con GPT
    # -------------------------------
    if 'topic_model' in st.session_state and 'df' in st.session_state and st.button("3️⃣ Nombrar temas con GPT"):
        topic_info = st.session_state['topic_model'].get_topic_info()
        topic_names = {}
        for topic_id in topic_info['Topic']:
            if topic_id != -1:  # ignorar outliers
                keywords = [word for word, _ in st.session_state['topic_model'].get_topic(topic_id)]
                topic_names[topic_id] = generate_topic_name(keywords)

        # Asignación segura de nombres
        df = st.session_state['df']
        df['topic_name'] = None
        for topic_id, name in topic_names.items():
            df.loc[df['topic'] == topic_id, 'topic_name'] = name
        df['topic_name'].fillna("Outlier / Sin tema", inplace=True)

        st.session_state['df'] = df
        st.session_state['topic_names'] = topic_names
        st.success("✅ Nombres de temas generados")
        st.dataframe(df[['song','year','topic_name']])

    # -------------------------------
    # Paso 4: Visualización interactiva de temas
    # -------------------------------
    if 'topic_model' in st.session_state and st.button("4️⃣ Visualización interactiva de temas"):
        try:
            st.subheader("Visualización de temas")
            fig = st.session_state['topic_model'].visualize_topics()
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"No se pudo generar la visualización: {e}")

    # -------------------------------
    # Paso 5: Filtrar por año
    # -------------------------------
    if 'df' in st.session_state:
        st.subheader("Filtrar por año")
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        year_selected = st.slider(
            "Selecciona rango de años",
            min_year, max_year,
            (min_year, max_year)
        )
        filtered_df = df[(df['year'] >= year_selected[0]) & (df['year'] <= year_selected[1])]
        st.dataframe(filtered_df)
