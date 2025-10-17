import streamlit as st
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import OpenAI, KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import openai
import os
import plotly.express as px
import nltk
import numpy as np
from nltk.corpus import stopwords

# --- CONFIGURACIÓN INICIAL ---
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# Título y Configuración Inicial
st.set_page_config(layout="wide", page_title="ShakiraGPT: Análisis de Tópicos")
st.title("🎤 ShakiraGPT: La Evolución Temática de una Loba 🐺")
st.markdown("---")
st.info("⚠️ Ejecutando en **Modo Demostración** (usando la **mitad del corpus**) para cumplir con los límites de memoria de la plataforma.")

# 🔑 Carga de la Clave API de OpenAI (para el Paso 4)
openai_api_key = None
try:
    openai_api_key = st.secrets["openai"]["api_key"]
except (KeyError, AttributeError):
    openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.sidebar.error("⚠️ Clave OpenAI no configurada. El Paso 4 (Mejora con LLM) está deshabilitado.")

# 1. Cargar datos (letras) y embeddings (vectores) - SIN CACHÉ PESADA
def load_data(file_path_shakira, file_path_embeddings):
    """Carga los dos archivos Excel y los sincroniza, usando solo la mitad de las filas."""
    try:
        df_shakira_full = pd.read_excel(file_path_shakira)
        df_embeddings_full = pd.read_excel(file_path_embeddings, header=None)
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró uno de los archivos requeridos: {e}.")
        st.stop()
        
    df_shakira_full = df_shakira_full.dropna(subset=['lyrics', 'song', 'year']).sort_values(by='year').reset_index(drop=True)
    df_shakira_full['lyrics'] = df_shakira_full['lyrics'].astype(str)
    df_shakira_full['year'] = pd.to_numeric(df_shakira_full['year'], errors='coerce').fillna(0).astype(int)
    
    # 🚨 ESTRATEGIA DE MEMORIA: USAR SOLO LA MITAD DEL CORPUS
    full_size = len(df_shakira_full)
    sample_size = full_size // 2  # Usar la mitad entera
    
    df_shakira = df_shakira_full.head(sample_size)
    df_embeddings = df_embeddings_full.head(sample_size)
    
    # Validación CRUCIAL de orden
    if len(df_shakira) != len(df_embeddings):
        st.error(f"Error: La muestra de canciones ({len(df_shakira)}) no coincide con los embeddings ({len(df_embeddings)}).")
        st.stop()
        
    embeddings = df_embeddings.values
    
    st.sidebar.info(f"Usando **{sample_size}** de {full_size} canciones originales.")

    return df_shakira, embeddings


# 2. Entrenar el modelo BERTopic - SIN CACHÉ PESADA
def train_bertopic(docs, embeddings, use_llm_representation=False):
    """Inicializa y entrena el modelo BERTopic con embeddings precalculados en un subconjunto."""
    
    # --- Definición de Modelos BERTopic ---
    umap_model = UMAP(n_neighbors=5, n_components=3, min_dist=0.0, metric='cosine', random_state=42)
    spanish_stopwords = stopwords.words('spanish')
    
    # CountVectorizer: Sin min_df para evitar el ValueError
    vectorizer_model = CountVectorizer(stop_words=spanish_stopwords) 
    
    representation_model = KeyBERTInspired()
    
    if use_llm_representation and openai_api_key:
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            prompt = "Genera un título corto y conciso (máximo 6 palabras) para este tópico de canciones. El título debe ser profesional y capturar la esencia del tema."
            representation_model = OpenAI(client, model="gpt-4o-mini", chat=True, prompt=prompt, delay_in_seconds=5)
        except:
             pass # Si falla el LLM, usa KeyBERT

    topic_model = BERTopic(
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model, 
        language="multilingual", 
        calculate_probabilities=True,
        verbose=False,
    )
    
    with st.spinner("✨ Entrenando modelo en subconjunto de datos (Modo Demostración)... ⏳"):
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings) 
    
    return topic_model, topics, probs

# --- EJECUCIÓN DEL FLUJO PRINCIPAL ---

FILE_PATH_SHAKIRA = 'shak.xlsx'
FILE_PATH_EMBEDDINGS = 'embeddings.xlsx'
df_shakira, embeddings = load_data(FILE_PATH_SHAKIRA, FILE_PATH_EMBEDDINGS)
docs = df_shakira['lyrics'].tolist()

st.sidebar.header("Opciones de Modelado")
use_llm = st.sidebar.toggle(
    "Usar GPT-4o-mini para nombres de Tópicos (Paso 4)", 
    value=False,
    disabled=not openai_api_key
)

# EJECUCIÓN CON MUESTRA REDUCIDA
topic_model, topics, probs = train_bertopic(docs, embeddings, use_llm_representation=use_llm)
df_shakira['topic'] = topics

# Preparación de datos para la visualización
df_topics_info = topic_model.get_topic_info()
df_topics = df_topics_info[df_topics_info['Topic'] != -1]
df_topics = df_topics.rename(columns={
    'Name': 'Nombre del Tópico (Final)', 
    'Count': 'Canciones', 
    'Representation': 'Palabras Clave (c-TF-IDF)'
})


# --------------------------------------------------------------------------------------
# ➡️ PASO 1: EXPLORACIÓN DE DATOS
# --------------------------------------------------------------------------------------

st.header("1️⃣ Paso Inicial: Carga y Limpieza de Datos")
st.markdown(f"**Materia Prima:** Solo se usa una muestra de **{len(df_shakira)}** canciones para demostración.")
st.dataframe(df_shakira[['song', 'year', 'lyrics']].head(), use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------------------
# ➡️ PASO 2: CARGA DE EMBEDDINGS Y REDUCCIÓN DE DIMENSIONALIDAD (UMAP)
# --------------------------------------------------------------------------------------

st.header("2️⃣ Carga de Embeddings y Proyección (UMAP)")
st.markdown(f"""
**Embeddings:** Cargamos los vectores precalculados ({embeddings.shape[1]} dimensiones) para saltar el costoso paso de BERT.
**UMAP:** Proyecta esos vectores a 3D para la visualización.
""")

try:
    fig_docs = topic_model.visualize_documents(docs, custom_labels=True, title="Mapa de Tópicos (UMAP)")
    st.plotly_chart(fig_docs, use_container_width=True)
except Exception as e:
    st.error(f"Error al generar la visualización UMAP: {e}.")
    
st.markdown("---")

# --------------------------------------------------------------------------------------
# ➡️ PASO 3: AGRUPACIÓN (HDBSCAN) Y TÓPICOS BASE (c-TF-IDF)
# --------------------------------------------------------------------------------------

st.header("3️⃣ Agrupación (HDBSCAN) y Tópicos Base (c-TF-IDF)")
st.markdown("HDBSCAN encuentra *clusters* en el espacio UMAP. c-TF-IDF extrae las palabras clave que definen esos *clusters*.")

st.subheader("Palabras Clave por Tópico (Representación Estadística)")
st.dataframe(
    df_topics[['Topic', 'Canciones', 'Palabras Clave (c-TF-IDF)']], 
    use_container_width=True
)

st.subheader("Visualización de las Palabras Clave")
fig_bar = topic_model.visualize_barchart(top_n_topics=10, n_words=8, custom_labels=True)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------------------
# ➡️ PASO 4: MEJORA DE LA REPRESENTACIÓN CON LLMS
# --------------------------------------------------------------------------------------

st.header("4️⃣ Mejora de la Representación con LLMs")
st.markdown("Este paso demuestra cómo una IA (GPT-4o-mini) puede etiquetar los tópicos de forma más clara que los métodos estadísticos.")

if use_llm:
    st.success("✅ ¡GPT-4o-mini ha transformado las palabras estadísticas en nombres legibles!")
    st.dataframe(
        df_topics[['Topic', 'Nombre del Tópico (Final)', 'Canciones', 'Palabras Clave (c-TF-IDF)']], 
        use_container_width=True
    )
else:
    st.info("💡 Activa el interruptor en la barra lateral para ver la mejora de GPT-4o-mini.")

st.markdown("---")

# --------------------------------------------------------------------------------------
# ➡️ PASO 5: ANÁLISIS FINAL (Visualizaciones)
# --------------------------------------------------------------------------------------

st.header("5️⃣ Análisis Final: Tópicos y Tendencias Temporales")

st.subheader("Evolución de la Prominencia Temática")
st.markdown("El gráfico muestra cómo la importancia de los temas ha cambiado con el tiempo. (La precisión es limitada debido a la muestra).")

try:
    topics_over_time = topic_model.topics_over_time(docs, df_shakira['year'])
    fig_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, custom_labels=True)
    st.plotly_chart(fig_time, use_container_width=True)
except Exception as e:
    st.warning(f"Error al generar el gráfico temporal: {e}. Puede ser por la pequeña muestra de datos.")

st.markdown("---")
st.caption("Solución final implementada para priorizar la ejecución pedagógica sobre la precisión en entornos de recursos limitados.")
