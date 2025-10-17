import streamlit as st
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import OpenAI, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import openai
import os
import plotly.express as px
import nltk
import numpy as np
from nltk.corpus import stopwords

# --- CONFIGURACIÓN INICIAL Y DESCARGA DE RECURSOS ---
try:
    nltk.data.find('corpora/stopwords')
except:
    st.info("Descargando el recurso 'stopwords' de NLTK (solo la primera vez).")
    nltk.download('stopwords')

# Título y Configuración Inicial
st.set_page_config(layout="wide", page_title="ShakiraGPT: Análisis de Tópicos")
st.title("🎤 ShakiraGPT: La Evolución Temática de una Loba 🐺")
st.markdown("Un tutorial interactivo sobre **Modelado de Tópicos** con BERTopic.")
st.markdown("---")
st.warning("⚠️ **ATENCIÓN:** Ejecutando con el **cálculo interno de embeddings (41 canciones)**. Esto consumirá más recursos y tiempo, pero asegura la integridad del modelo para las visualizaciones.")


# 🔑 Carga de la Clave API de OpenAI (para el Paso 4)
openai_api_key = None
try:
    openai_api_key = st.secrets["openai"]["api_key"]
except (KeyError, AttributeError):
    openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.sidebar.error("⚠️ Clave OpenAI no configurada. El Paso 4 (Mejora con LLM) está deshabilitado.")

# 1. Cargar datos (letras) - SÓLO EL ARCHIVO DE SHAKIRA
def load_data(file_path_shakira):
    """Carga los datos de Shakira, usando el corpus completo."""
    try:
        df_shakira = pd.read_excel(file_path_shakira)
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo requerido: {e}.")
        st.stop()
        
    df_shakira = df_shakira.dropna(subset=['lyrics', 'song', 'year']).sort_values(by='year').reset_index(drop=True)
    df_shakira['lyrics'] = df_shakira['lyrics'].astype(str)
    df_shakira['year'] = pd.to_numeric(df_shakira['year'], errors='coerce').fillna(0).astype(int)
    
    # ESTRATEGIA: USAR EL CORPUS COMPLETO
    full_size = len(df_shakira)
    
    st.sidebar.info(f"Usando **{full_size}** canciones completas. El modelo calculará los embeddings.")

    return df_shakira


# 2. Entrenar o cargar el modelo BERTopic (Usando caché)
# 🚨 Los embeddings se calculan internamente por BERTopic (Sentence Transformer)
@st.cache_resource
def train_bertopic(docs, use_llm_representation=False):
    """Inicializa y entrena el modelo BERTopic, incluyendo el cálculo de embeddings."""
    
    # --- Definición de Modelos BERTopic ---
    umap_model = UMAP(n_neighbors=5, n_components=3, min_dist=0.0, metric='cosine', random_state=42)
    spanish_stopwords = stopwords.words('spanish')
    vectorizer_model = CountVectorizer(stop_words=spanish_stopwords) 

    representation_model = MaximalMarginalRelevance(diversity=0.3) 
    
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
        except Exception:
            st.sidebar.warning("Fallo al conectar con GPT-4o-mini. Usando MMR.")
            
    # Inicialización de BERTopic 
    topic_model = BERTopic(
        # ⚠️ Dejamos el embedding_model por defecto (Sentence Transformer) para que calcule
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model, 
        language="multilingual", 
        calculate_probabilities=True,
        verbose=False,
    )
    
    # ⚠️ NO se usa 'embeddings=embeddings' en fit_transform
    with st.spinner("✨ Calculando Embeddings y Descubriendo Tópicos... ⏳ (Esto puede tardar varios minutos)"):
        topics, probs = topic_model.fit_transform(docs) 
    
    return topic_model, topics, probs

# --- EJECUCIÓN DEL FLUJO PRINCIPAL ---

FILE_PATH_SHAKIRA = 'shak.xlsx'
df_shakira = load_data(FILE_PATH_SHAKIRA)
docs = df_shakira['lyrics'].tolist()

st.sidebar.header("Opciones de Modelado")
use_llm = st.sidebar.toggle(
    "Usar GPT-4o-mini para nombres de Tópicos (Paso 4)", 
    value=False,
    disabled=not openai_api_key
)

# ⚠️ LLAMADA SIN EMBEDDINGS
topic_model, topics, probs = train_bertopic(docs, use_llm_representation=use_llm)
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
## ➡️ PASO 1: EXPLORACIÓN DE DATOS
# --------------------------------------------------------------------------------------

st.header("1️⃣ Paso Inicial: Carga y Limpieza de Datos")
st.markdown("Se ha cargado el **corpus completo** de Shakira.")
st.dataframe(df_shakira[['song', 'year', 'lyrics']].head(), use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------------------
## ➡️ PASO 2: CÁLCULO DE EMBEDDINGS Y REDUCCIÓN DE DIMENSIONALIDAD (UMAP)
# --------------------------------------------------------------------------------------

st.header("2️⃣ Cálculo de Embeddings y Proyección (UMAP)")
st.markdown("""
- **Embeddings:** Se ha utilizado el modelo BERT predeterminado para calcular los vectores de cada canción.
- **UMAP:** Proyecta esos vectores a 3D. El modelo ahora es completo y estable.
""")

try:
    # 🚨 Visualización ESTÁNDAR, sin pasar embeddings=...
    fig_docs = topic_model.visualize_documents(docs, custom_labels=True, title="Mapa de Tópicos (UMAP)")
    st.plotly_chart(fig_docs, use_container_width=True)
except Exception as e:
    st.error(f"Error al generar la visualización UMAP: {e}.")
    
st.markdown("---")

# --------------------------------------------------------------------------------------
## ➡️ PASO 3: AGRUPACIÓN (HDBSCAN) Y TÓPICOS BASE (c-TF-IDF)
# --------------------------------------------------------------------------------------

st.header("3️⃣ Agrupación (HDBSCAN) y Tópicos Base (c-TF-IDF)")
st.markdown("HDBSCAN encuentra *clusters* en el espacio UMAP. c-TF-IDF extrae las palabras clave que definen esos *clusters*.")

st.subheader("Palabras Clave por Tópico (Representación Estadística)")
st.dataframe(
    df_topics[['Topic', 'Canciones', 'Palabras Clave (c-TF-IDF)']], 
    use_container_width=True
)

# Generar gráfico solo si hay tópicos válidos
if not df_topics.empty:
    st.subheader("Visualización de las Palabras Clave")
    fig_bar = topic_model.visualize_barchart(top_n_topics=min(10, len(df_topics)), n_words=8, custom_labels=True)
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.warning("El modelo no pudo generar tópicos válidos con la muestra actual (solo ruido).")

st.markdown("---")

# --------------------------------------------------------------------------------------
## ➡️ PASO 4: MEJORA DE LA REPRESENTACIÓN CON LLMS
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
## ➡️ PASO 5: ANÁLISIS FINAL (Visualizaciones)
# --------------------------------------------------------------------------------------

st.header("5️⃣ Análisis Final: Tópicos y Tendencias Temporales")

st.subheader("Evolución de la Prominencia Temática")
st.markdown("El gráfico muestra cómo la importancia de los temas ha cambiado con el tiempo.")

try:
    topics_over_time = topic_model.topics_over_time(docs, df_shakira['year'])
    fig_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, custom_labels=True)
    st.plotly_chart(fig_time, use_container_width=True)
except Exception as e:
    st.warning(f"Error al generar el gráfico temporal: {e}.")

st.markdown("---")
st.caption("Esta implementación utiliza el flujo estándar de BERTopic, lo que garantiza la integridad de los componentes del modelo.")
