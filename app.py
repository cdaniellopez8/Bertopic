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

# --- CONFIGURACIÓN INICIAL Y DESCARGA DE RECURSOS ---

# 1. Configuración de NLTK
try:
    nltk.data.find('corpora/stopwords')
except:
    st.info("Descargando el recurso 'stopwords' de NLTK (solo la primera vez).")
    nltk.download('stopwords')
from nltk.corpus import stopwords
# ---------------------------------------------------

# Título y Configuración Inicial
st.set_page_config(layout="wide", page_title="ShakiraGPT: Análisis de Tópicos")
st.title("🎤 ShakiraGPT: La Evolución Temática de una Loba 🐺")
st.markdown("---")
st.header("Modelo BERTopic con Embeddings Precalculados (Optimización de Memoria)")

# 🔑 Carga de la Clave API de OpenAI (para el Paso 4)
openai_api_key = None
try:
    openai_api_key = st.secrets["openai"]["api_key"]
except (KeyError, AttributeError):
    openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.sidebar.error("⚠️ Clave OpenAI no configurada. El Paso 4 (Mejora con LLM) está deshabilitado.")

# 1. Cargar datos (letras) y embeddings (vectores)
@st.cache_data
def load_data(file_path_shakira, file_path_embeddings):
    """Carga los dos archivos Excel y los sincroniza por orden de fila."""
    try:
        df_shakira = pd.read_excel(file_path_shakira)
        # Cargamos embeddings sin encabezado para tratar cada columna como una dimensión
        df_embeddings = pd.read_excel(file_path_embeddings, header=None) 
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró uno de los archivos requeridos: {e}. Asegúrate de tener '{file_path_shakira}' y '{file_path_embeddings}'.")
        st.stop()
        
    df_shakira = df_shakira.dropna(subset=['lyrics', 'song', 'year'])
    df_shakira['lyrics'] = df_shakira['lyrics'].astype(str)
    df_shakira['year'] = pd.to_numeric(df_shakira['year'], errors='coerce').fillna(0).astype(int)
    
    # Validación CRUCIAL de orden
    if len(df_shakira) != len(df_embeddings):
        st.error(f"Error de sincronización: El número de canciones ({len(df_shakira)}) no coincide con el número de embeddings ({len(df_embeddings)}).")
        st.stop()
        
    # Convertir el DataFrame de embeddings a un array de numpy (formato requerido por BERTopic)
    embeddings = df_embeddings.values
    
    return df_shakira.sort_values(by='year').reset_index(drop=True), embeddings


# 2. Entrenar o cargar el modelo BERTopic (Usando caché)
@st.cache_resource
def train_bertopic(docs, embeddings, use_llm_representation=False):
    """Inicializa y entrena el modelo BERTopic con embeddings precalculados."""
    
    # --- Definición de Modelos BERTopic ---
    
    # UMAP: Modelo de reducción optimizado para ahorrar memoria
    umap_model = UMAP(n_neighbors=15, 
                      n_components=3, 
                      min_dist=0.0, 
                      metric='cosine', 
                      random_state=42)

    # CountVectorizer: Stopwords en español y min_df=3 para filtrar palabras muy infrecuentes
    spanish_stopwords = stopwords.words('spanish')
    vectorizer_model = CountVectorizer(stop_words=spanish_stopwords, min_df=3) 

    representation_model = KeyBERTInspired() # Representación base
    
    # 4. Configurar el Modelo de Representación (LLM)
    if use_llm_representation and openai_api_key:
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            prompt = "Genera un título corto y conciso (máximo 6 palabras) para este tópico de canciones. El título debe ser profesional y capturar la esencia del tema."
            
            representation_model = OpenAI(client, 
                                          model="gpt-4o-mini", 
                                          chat=True,
                                          prompt=prompt,
                                          delay_in_seconds=5)
        except Exception as e:
            st.warning(f"Error al inicializar OpenAI: {e}. Se usará KeyBERT.")
            
    # Inicialización de BERTopic (sin el embedding_model, ya que lo pasaremos precalculado)
    topic_model = BERTopic(
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model, 
        language="multilingual", 
        calculate_probabilities=True,
        verbose=False,
    )
    
    # Entrenamiento (EL PASO LENTO: Aquí se usa 'embeddings=embeddings' para saltar BERT)
    with st.spinner("✨ Descubriendo Tópicos con Embeddings Precalculados... ⏳"):
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
# ➡️ PASO 1: EXPLORACIÓN DE DATOS (Materia Prima)
# --------------------------------------------------------------------------------------

st.header("1️⃣ Paso Inicial: Carga y Limpieza de Datos")
st.markdown("Preparamos las letras y aplicamos un filtro básico, eliminando *stopwords* comunes en español para el análisis.")
st.write(f"Corpus cargado: **{len(df_shakira)}** canciones de Shakira.")
st.dataframe(df_shakira[['song', 'year', 'lyrics']].head(), use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------------------
# ➡️ PASO 2: CARGA DE EMBEDDINGS Y REDUCCIÓN DE DIMENSIONALIDAD (UMAP)
# --------------------------------------------------------------------------------------

st.header("2️⃣ Carga de Embeddings Precalculados y Proyección (UMAP)")
st.markdown("""
- **Embeddings:** En lugar de entrenar BERT (que agota la memoria), cargamos los **vectores numéricos** de las canciones que ya tenías listos.
- **UMAP:** Utiliza esos vectores para proyectar el significado de las canciones en solo 3 dimensiones.
""")

st.markdown("Cada punto en el gráfico es una canción. La **proximidad** indica similitud semántica de las letras.")

try:
    fig_docs = topic_model.visualize_documents(docs, custom_labels=True, title="Mapa de Tópicos (UMAP)")
    st.plotly_chart(fig_docs, use_container_width=True)
except Exception as e:
    st.error(f"Error al generar la visualización UMAP: {e}.")
    
st.markdown("---")

# --------------------------------------------------------------------------------------
# ➡️ PASO 3: AGRUPACIÓN (HDBSCAN) Y GENERACIÓN INICIAL DE TÓPICOS (c-TF-IDF)
# --------------------------------------------------------------------------------------

st.header("3️⃣ Agrupación (HDBSCAN) y Tópicos Base (c-TF-IDF)")
st.markdown("""
- **HDBSCAN:** Agrupa los puntos cercanos en el espacio UMAP.
- **c-TF-IDF:** Genera las **palabras clave** estadísticas para cada *cluster*.
""")

st.subheader("Palabras Clave por Tópico (Representación Estadística)")
st.info("La columna 'Palabras Clave' muestra los términos más importantes, sin incluir *stopwords* comunes.")
st.dataframe(
    df_topics[['Topic', 'Canciones', 'Palabras Clave (c-TF-IDF)']], 
    use_container_width=True
)

st.subheader("Visualización de las Palabras Clave")
fig_bar = topic_model.visualize_barchart(top_n_topics=10, n_words=8, custom_labels=True)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------------------
# ➡️ PASO 4: MEJORA DE LA REPRESENTACIÓN CON LLMS (La Pedagogía)
# --------------------------------------------------------------------------------------

st.header("4️⃣ Mejora de la Representación con LLMs (Paso Opcional)")

if use_llm:
    st.success("✅ ¡GPT-4o-mini ha transformado las palabras estadísticas en nombres legibles!")
    
    st.subheader("Etiquetas de Tópicos Mejoradas (GPT-4o-mini)")
    st.dataframe(
        df_topics[['Topic', 'Nombre del Tópico (Final)', 'Canciones', 'Palabras Clave (c-TF-IDF)']], 
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("Ejemplo Pedagógico: Comparación de Etiquetas")
    
    first_topic_id = df_topics['Topic'].iloc[0]
    c_tfidf_words = ", ".join([word[0] for word in topic_model.get_topic(first_topic_id)])
    llm_name = df_topics[df_topics['Topic'] == first_topic_id]['Nombre del Tópico (Final)'].iloc[0]

    st.code(f"Palabras Clave (c-TF-IDF): {c_tfidf_words}", language='text')
    st.code(f"Nombre Generado por LLM: {llm_name}", language='text')

    
else:
    st.info("💡 Activa el interruptor en la barra lateral para ver la mejora de GPT-4o-mini.")

st.markdown("---")

# --------------------------------------------------------------------------------------
# ➡️ PASO 5: ANÁLISIS FINAL (Visualizaciones)
# --------------------------------------------------------------------------------------

st.header("5️⃣ Análisis Final: Tópicos y Tendencias Temporales")

st.subheader("Evolución de la Prominencia Temática")

try:
    topics_over_time = topic_model.topics_over_time(docs, df_shakira['year'])
    fig_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, custom_labels=True)
    st.plotly_chart(fig_time, use_container_width=True)
except Exception as e:
    st.warning(f"Error al generar el gráfico temporal: {e}.")


st.markdown("---")

st.header("🔍 Exploración Detallada de Canciones")

selected_topic_id = st.selectbox(
    "Selecciona un Tópico para ver sus Canciones:",
    options=df_topics['Topic'].tolist(),
    format_func=lambda x: f"Tópico {x}: {df_topics[df_topics['Topic'] == x]['Nombre del Tópico (Final)'].iloc[0]}"
)

songs_in_topic = df_shakira[df_shakira['topic'] == selected_topic_id]

if not songs_in_topic.empty:
    st.subheader(f"Canciones para el Tópico {selected_topic_id}")
    st.dataframe(songs_in_topic[['year', 'song', 'lyrics']], use_container_width=True)
else:
    st.info("No hay canciones para este tópico.")

st.caption("¡Proyecto pedagógico finalizado! El uso de embeddings precalculados optimiza el rendimiento y mantiene el enfoque educativo.")
