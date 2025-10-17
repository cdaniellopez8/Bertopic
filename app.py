import streamlit as st
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import OpenAI, KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
import openai
import os
import plotly.express as px
import nltk

# --- CONFIGURACIÓN DE NLTK ---
try:
    nltk.data.find('corpora/stopwords')
except:
    st.info("Descargando el recurso 'stopwords' de NLTK (solo la primera vez).")
    nltk.download('stopwords')
from nltk.corpus import stopwords
# -----------------------------

# --- CONFIGURACIÓN Y CACHÉ DE RECURSOS ---

st.set_page_config(layout="wide", page_title="ShakiraGPT: Análisis de Tópicos")
st.title("🎤 ShakiraGPT: La Evolución Temática de una Loba 🐺")
st.markdown("Un tutorial interactivo sobre **Modelado de Tópicos** con BERTopic, desde la incrustación hasta la mejora con LLMs.")
st.markdown("---")

# 🔑 Carga de la Clave API de OpenAI (para el Paso 4)
openai_api_key = None
try:
    # Usar st.secrets (forma recomendada para Streamlit Cloud)
    openai_api_key = st.secrets["openai"]["api_key"]
except (KeyError, AttributeError):
    # Si no está en secrets, usar variable de entorno
    openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.sidebar.error("⚠️ Clave OpenAI no configurada. El Paso 4 (Mejora con LLM) está deshabilitado.")

# 1. Cargar y Preprocesar los datos
@st.cache_data
def load_data(file_path):
    """Carga el Excel y hace una limpieza básica."""
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{file_path}'. Asegúrate de que esté en la carpeta del proyecto.")
        st.stop()
        
    df = df.dropna(subset=['lyrics', 'song', 'year'])
    df['lyrics'] = df['lyrics'].astype(str)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    return df.sort_values(by='year')


# 2. Entrenar o cargar el modelo BERTopic (Optimización de Memoria y Caché)
@st.cache_resource
def train_bertopic(docs, use_llm_representation=False):
    """Inicializa y entrena el modelo BERTopic con optimización de memoria."""
    
    # ⚙️ OPTIMIZACIÓN DE MEMORIA: Modelos Ligeros
    
    # Embedding: MiniLM-L4-v2 usa menos memoria que el L6 o L12
    embedding_model = SentenceTransformer("all-MiniLM-L4-v2")
    
    # UMAP: Reducir la dimensionalidad a 3D (para visualización 3D o 2D)
    umap_model = UMAP(n_neighbors=15, 
                      n_components=3, 
                      min_dist=0.0, 
                      metric='cosine', 
                      random_state=42)

    # CountVectorizer: Incluye stopwords en español y relaja el umbral de frecuencia (min_df=3)
    spanish_stopwords = stopwords.words('spanish')
    vectorizer_model = CountVectorizer(stop_words=spanish_stopwords, min_df=3) 

    representation_model = KeyBERTInspired() # Representación base si no se usa LLM
    
    # 4. Configurar el Modelo de Representación (Paso Opcional)
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
            
    # Inicialización de BERTopic con los modelos optimizados
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model, 
        language="multilingual", 
        calculate_probabilities=True,
        verbose=False,
    )
    
    # Entrenamiento (EL PASO LENTO)
    with st.spinner("✨ Creando Embeddings y Descubriendo Tópicos... ¡Esto puede tardar unos minutos! ⏳"):
        topics, probs = topic_model.fit_transform(docs)
    
    return topic_model, topics, probs

# --- INTERFAZ DE USUARIO Y EJECUCIÓN ---

FILE_PATH = 'shak.xlsx'
df_shakira = load_data(FILE_PATH)
docs = df_shakira['lyrics'].tolist()

st.sidebar.header("Opciones de Modelado")
use_llm = st.sidebar.toggle(
    "Usar GPT-4o-mini para nombres de Tópicos (Paso 4)", 
    value=False,
    disabled=not openai_api_key
)

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
# ➡️ PASO 1: EXPLORACIÓN DE DATOS (Materia Prima)
# --------------------------------------------------------------------------------------

st.header("1️⃣ Paso Inicial: Carga y Limpieza de Datos")
st.markdown("Antes de modelar, preparamos las letras y aplicamos un filtro básico (eliminar *stopwords* en español).")
st.write(f"Corpus cargado: **{len(df_shakira)}** canciones de Shakira.")
st.dataframe(df_shakira[['song', 'year', 'lyrics']].head(), use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------------------
# ➡️ PASO 2: INCUSTACIÓN (BERT) Y REDUCCIÓN DE DIMENSIONALIDAD (UMAP)
# --------------------------------------------------------------------------------------

st.header("2️⃣ Incrustación (BERT) y Proyección (UMAP)")
st.markdown("""
- **BERT:** Cada letra se convierte en un **vector numérico** (embedding) que captura su significado.
- **UMAP:** Reduce esos vectores a solo 3 dimensiones para que podamos verlos en un gráfico 2D/3D.
""")

st.markdown("Cada punto en el gráfico es una canción. La **proximidad** indica similitud semántica de las letras.")

# Generar y mostrar el gráfico de documentos UMAP (¡funciona con el modelo optimizado!)
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
- **HDBSCAN:** Agrupa los puntos cercanos en el espacio UMAP, formando *clusters* (tópicos).
- **c-TF-IDF:** Genera las **palabras clave** para cada *cluster* (tópico), ignorando las *stopwords* que filtramos inicialmente.
""")

st.subheader("Palabras Clave por Tópico (Representación Estadística)")
st.info("La columna 'Palabras Clave' contiene la lista de términos más importantes generados por c-TF-IDF.")
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
    st.success("✅ GPT-4o-mini ha generado nombres coherentes y fáciles de entender para los tópicos.")
    
    st.subheader("Etiquetas de Tópicos Mejoradas (GPT-4o-mini)")
    st.dataframe(
        df_topics[['Topic', 'Nombre del Tópico (Final)', 'Canciones', 'Palabras Clave (c-TF-IDF)']], 
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("Ejemplo Pedagógico: Comparación de Etiquetas")
    
    first_topic_id = df_topics['Topic'].iloc[0]
    
    # Recuperamos las palabras clave del tópico principal
    c_tfidf_words = ", ".join([word[0] for word in topic_model.get_topic(first_topic_id)])
    llm_name = df_topics[df_topics['Topic'] == first_topic_id]['Nombre del Tópico (Final)'].iloc[0]

    st.markdown(f"**Tópico más frecuente (Tópico {first_topic_id}):**")
    st.code(f"Palabras Clave (c-TF-IDF): {c_tfidf_words}", language='text')
    st.code(f"Nombre Generado por LLM: {llm_name}", language='text')
    st.markdown("Esto demuestra cómo un LLM transforma una lista de palabras estadísticas en un concepto legible, ideal para informes.")

    
else:
    st.info("💡 Activa el interruptor en la barra lateral. Usamos KeyBERT para una representación base, pero GPT-4o-mini es superior.")

st.markdown("---")

# --------------------------------------------------------------------------------------
# ➡️ PASO 5: ANÁLISIS FINAL (Visualizaciones)
# --------------------------------------------------------------------------------------

st.header("5️⃣ Análisis Final: Tópicos y Tendencias Temporales")

st.subheader("Evolución de la Prominencia Temática")
st.markdown("Este gráfico muestra cómo la importancia de cada tópico ha cambiado a lo largo de la carrera de Shakira (eje X: Año).")

try:
    df_shakira['year'] = df_shakira['year'].astype(int)
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

st.caption("¡Proyecto pedagógico finalizado! Esperamos que esta demostración te haya ayudado a comprender el flujo de trabajo de BERTopic.")
