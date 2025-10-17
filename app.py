import streamlit as st
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import OpenAI, KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import openai
import os
import plotly.express as px
import nltk

# Descargar stopwords de NLTK si no están presentes
try:
    nltk.data.find('corpora/stopwords')
except: # Capturamos cualquier excepción, incluyendo la de recurso no encontrado
    st.info("Descargando el recurso 'stopwords' de NLTK por primera vez. Esto solo sucede una vez.")
    nltk.download('stopwords')

# --- CONFIGURACIÓN Y CACHÉ DE RECURSOS ---

# Título y Configuración Inicial
st.set_page_config(layout="wide", page_title="ShakiraGPT: Análisis de Tópicos")
st.title("🎤 ShakiraGPT: La Evolución Temática de una Loba 🐺")
st.markdown("---")

# 🔑 Carga de la Clave API de OpenAI
openai_api_key = None
try:
    openai_api_key = st.secrets["openai"]["api_key"]
except (KeyError, AttributeError):
    openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.sidebar.error("⚠️ La clave API de OpenAI no está configurada. El Paso 4 estará deshabilitado.")

# 1. Cargar y Preprocesar los datos (Usando caché de datos)
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


# 2. Entrenar o cargar el modelo BERTopic (Usando caché de recursos)
@st.cache_resource
def train_bertopic(docs, use_llm_representation=False):
    """Inicializa y entrena el modelo BERTopic."""
    
    # --- Definición de Stopwords para el preprocesamiento (¡Mejora 3!)
    spanish_stopwords = stopwords.words('spanish')
    vectorizer_model = CountVectorizer(stop_words=spanish_stopwords, min_df=2)

    representation_model = None
    
    # 4. Configurar el Modelo de Representación (Paso Clave)
    if use_llm_representation and openai_api_key:
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            
            prompt = """
            Genera un título corto y conciso (máximo 6 palabras) para este tópico. 
            El tópico contiene letras de canciones. El título debe ser profesional y capturar la esencia del tema.
            """
            
            representation_model = OpenAI(client, 
                                          model="gpt-4o-mini", 
                                          chat=True,
                                          prompt=prompt,
                                          delay_in_seconds=5)
        except Exception as e:
            st.warning(f"Error al inicializar OpenAI: {e}. Se usará KeyBERT.")
            representation_model = None
    
    # Si no se usa LLM, usamos KeyBERT como representación secundaria más legible
    if not representation_model:
        representation_model = KeyBERTInspired()
            
    # Inicialización de BERTopic
    topic_model = BERTopic(
        language="multilingual", 
        calculate_probabilities=True,
        verbose=False,
        representation_model=representation_model,
        # Se añade el vectorizador con las stopwords aquí
        vectorizer_model=vectorizer_model 
    )
    
    # Entrenamiento
    with st.spinner("✨ Creando Embeddings y Descubriendo Tópicos... ¡Esto puede tardar unos minutos! ⏳"):
        topics, probs = topic_model.fit_transform(docs)
    
    return topic_model, topics, probs

# --- INTERFAZ DE USUARIO Y EJECUCIÓN ---

# Cargar los datos
FILE_PATH = 'shak.xlsx'
df_shakira = load_data(FILE_PATH)
docs = df_shakira['lyrics'].tolist()

# Sidebar para la configuración pedagógica (Paso 4)
st.sidebar.header("Opciones de Modelado")
use_llm = st.sidebar.toggle(
    "Usar GPT-4o-mini para nombres de Tópicos (Paso 4)", 
    value=False,
    disabled=not openai_api_key
)

# Entrenar el modelo
topic_model, topics, probs = train_bertopic(docs, use_llm_representation=use_llm)
df_shakira['topic'] = topics

# Filtrar outliers (-1) para las visualizaciones de tópicos principales
df_topics_info = topic_model.get_topic_info()
df_topics = df_topics_info[df_topics_info['Topic'] != -1]
df_topics = df_topics.rename(columns={
    'Name': 'Nombre del Tópico (Final)', 
    'Count': 'Canciones', 
    'Representation': 'Palabras Clave (c-TF-IDF)'
})

# --------------------------------------------------------------------------------------
# ➡️ ORDEN PEDAGÓGICO CORREGIDO
# --------------------------------------------------------------------------------------

## 1️⃣ Exploración de Datos
st.header("1️⃣ Exploración de Datos: La Materia Prima")
st.write(f"Corpus cargado: **{len(df_shakira)}** canciones de Shakira.")
st.dataframe(df_shakira[['song', 'year', 'lyrics']].head(), use_container_width=True)

st.markdown("---")

## 2️⃣ Incrustación y Reducción de Dimensionalidad (BERT + UMAP)
st.header("2️⃣ Incrustación (BERT) y Reducción (UMAP)")
st.markdown("""
El primer paso de BERTopic es convertir las letras en **vectores** (Embeddings) usando BERT para capturar su significado. 
Luego, **UMAP** reduce esos vectores de alta dimensión a 2D para que podamos visualizarlos.
""")
st.markdown("Cada punto en el gráfico es una canción. La **proximidad** indica similitud semántica de las letras.")

# Generar y mostrar el gráfico de documentos (¡Mejora 2 - UMAP!)
# Se necesita obtener los embeddings y el UMAP_model para visualizar
if hasattr(topic_model, 'umap_model') and topic_model.umap_model is not None:
    try:
        # Forzar la generación si no existe (aunque fit_transform debería haberlo hecho)
        embeddings = topic_model._extract_embeddings(docs)
        reduced_embeddings = topic_model.umap_model.transform(embeddings)
        
        # Usar la función visualize_documents de BERTopic
        fig_docs = topic_model.visualize_documents(docs, custom_labels=True, title="Mapa de Tópicos (UMAP)")
        st.plotly_chart(fig_docs, use_container_width=True)
    except Exception as e:
        st.error(f"Error al generar la visualización UMAP: {e}. Puede deberse a pocos datos o problemas de clustering.")
else:
     st.warning("El modelo UMAP no está disponible, el entrenamiento pudo fallar o se requieren más datos.")
    
st.markdown("---")

## 3️⃣ Agrupación (HDBSCAN) y Generación Inicial de Tópicos (c-TF-IDF)
st.header("3️⃣ Agrupación (HDBSCAN) y Tópicos Base (c-TF-IDF)")
st.markdown("""
**HDBSCAN** agrupa los puntos cercanos en el espacio UMAP para formar *clusters* de canciones.
**c-TF-IDF** genera las palabras clave para cada *cluster* (tópico), ignorando las *stopwords* comunes en español.
""")

st.subheader("Palabras Clave por Tópico (¡Mejora 3 - Keywords visibles!)")
st.info("La columna 'Palabras Clave' muestra la representación inicial, **sin incluir palabras comunes como 'el', 'la', 'que'**.")
st.dataframe(
    df_topics[['Topic', 'Canciones', 'Palabras Clave (c-TF-IDF)']], 
    use_container_width=True
)

st.subheader("Visualización de Palabras Clave")
fig_bar = topic_model.visualize_barchart(top_n_topics=10, n_words=8, custom_labels=True)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

## 4️⃣ Mejora de la Representación con LLMs (Paso Clave)
st.header("4️⃣ Mejora de la Representación con LLMs (Paso Opcional de Calidad)")

if use_llm:
    st.success("✅ ¡GPT-4o-mini ha generado nombres coherentes para los tópicos!")
    
    st.subheader("Etiquetas de Tópicos Mejoradas (GPT-4o-mini)")
    st.dataframe(
        df_topics[['Topic', 'Nombre del Tópico (Final)', 'Canciones', 'Palabras Clave (c-TF-IDF)']], 
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("Ejemplo Pedagógico: Comparación de Etiquetas")
    
    # Mostrar la diferencia en el primer tópico
    first_topic_id = df_topics['Topic'].iloc[0]
    
    # Nota: BERTopic sobrescribe 'Name'. Para comparación, se usa la representación c-TF-IDF
    c_tfidf_words = ", ".join([word[0] for word in topic_model.get_topic(first_topic_id)])
    llm_name = df_topics[df_topics['Topic'] == first_topic_id]['Nombre del Tópico (Final)'].iloc[0]

    st.markdown(f"**Tópico más frecuente (Tópico {first_topic_id}):**")
    st.code(f"Palabras Clave (c-TF-IDF): {c_tfidf_words}", language='text')
    st.code(f"Nombre Generado por LLM: {llm_name}", language='text')
    st.markdown("Esto demuestra cómo un LLM transforma una lista de palabras en un concepto legible.")

    
else:
    st.info("💡 Activa el interruptor en la barra lateral para ver cómo GPT-4o-mini mejora las etiquetas de tópicos (se requiere API Key).")
    st.subheader("Etiquetas por Defecto (KeyBERT)")
    st.dataframe(
        df_topics[['Topic', 'Nombre del Tópico (Final)', 'Canciones', 'Palabras Clave (c-TF-IDF)']], 
        use_container_width=True
    )

st.markdown("---")

## 5️⃣ Conclusiones y Análisis a Través del Tiempo
st.header("5️⃣ Análisis Final: Tópicos a Través del Tiempo")

st.subheader("Evolución de la Prominencia Temática")
st.markdown("Este gráfico muestra cómo la importancia de cada tópico ha evolucionado a lo largo de la carrera de Shakira.")

try:
    df_shakira['year'] = df_shakira['year'].astype(int)
    topics_over_time = topic_model.topics_over_time(docs, df_shakira['year'])
    fig_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, custom_labels=True)
    st.plotly_chart(fig_time, use_container_width=True)
except Exception as e:
    st.warning(f"Error al generar el gráfico temporal: {e}. Asegúrate de que tienes datos de años variados.")


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

st.caption("¡Gracias por explorar la evolución de los tópicos en la discografía de Shakira!")


