import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

# -------------------------------
# Configuración
# -------------------------------
st.set_page_config(layout="wide")
st.title("🎵 Demo pedagógica BERTopic + GPT para nombres de tópicos")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------
# Paso 0: Subir archivo
# -------------------------------
uploaded_file = st.file_uploader("Sube tu archivo shak.xlsx", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())
    st.session_state['df'] = df

    # -------------------------------
    # Paso 1: Crear embeddings
    # -------------------------------
    if st.button("1️⃣ Crear embeddings"):
        with st.spinner("Generando embeddings..."):
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embedder.encode(df['lyrics'].astype(str), show_progress_bar=True)
            st.session_state['embeddings'] = embeddings
        st.success("✅ Embeddings generados")

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
        st.session_state['df'] = df
        st.success(f"✅ Clustering completado: {num_topics} tópicos")
        st.dataframe(df[['song','year','topic']])

    # -------------------------------
    # Paso 3: Extraer palabras clave TF-IDF por cluster
    # -------------------------------
    if 'kmeans_labels' in st.session_state and st.button("3️⃣ Extraer palabras clave TF-IDF"):
        df = st.session_state['df']
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(df['lyrics'].astype(str))
        topic_keywords = {}

        st.subheader("Palabras clave por tópico")
        for t in range(st.session_state['num_topics']):
            idx = np.where(st.session_state['kmeans_labels'] == t)[0]
            if len(idx) == 0:
                topic_keywords[t] = ["N/A"]
                continue
            tfidf_avg = X[idx].mean(axis=0)
            words = [(vectorizer.get_feature_names_out()[i], tfidf_avg[0, i]) 
                     for i in range(tfidf_avg.shape[1])]
            words = sorted(words, key=lambda x: x[1], reverse=True)[:5]  # top 5 palabras
            topic_keywords[t] = [w for w,_ in words]
            st.write(f"Tópico {t}: {', '.join(topic_keywords[t])}")

        st.session_state['topic_keywords'] = topic_keywords

    # -------------------------------
    # Paso 4: Nombrar tópicos con GPT
    # -------------------------------
    def generar_prompt(words):
        return f"""
        Tengo un conjunto de palabras clave extraídas de un grupo de canciones: {', '.join(words)}.
        Estas palabras representan un tema central en las letras. 
        Por favor, dame un nombre corto y descriptivo para este tópico,
        que refleje de manera clara y concisa el tema de las canciones.
        Solo devuelve el nombre, sin explicaciones adicionales.
        """

    if 'topic_keywords' in st.session_state and st.button("4️⃣ Nombrar tópicos automáticamente con GPT"):
        topic_names = {}
        for topic_id, words in st.session_state['topic_keywords'].items():
            prompt = generar_prompt(words)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            topic_name = response.choices[0].message.content.strip()
            topic_names[topic_id] = topic_name

        df['topic_name'] = df['topic'].map(topic_names)
        st.session_state['df'] = df
        st.session_state['topic_names'] = topic_names
        st.success("✅ Nombres generados por GPT")
        st.dataframe(df[['song','year','topic','topic_name']])

    # -------------------------------
    # Paso 5: Visualización 3D coloreada por nombre de tópico
    # -------------------------------
    if 'df' in st.session_state and 'embeddings' in st.session_state and st.button("5️⃣ Visualizar tópicos en 3D"):
        df_vis = st.session_state['df']
        embeddings = st.session_state['embeddings']

        if 'topic_name' not in df_vis.columns:
            st.warning("Primero debes nombrar los tópicos para colorearlos.")
        else:
            fig = px.scatter_3d(
                x=embeddings[:,0], y=embeddings[:,1], z=embeddings[:,2],
                color=df_vis['topic_name'],
                hover_data={'Song': df_vis['song'], 'Year': df_vis['year'], 'Topic': df_vis['topic_name']},
                title="Embeddings 3D coloreados por tópico"
            )
            st.plotly_chart(fig)
