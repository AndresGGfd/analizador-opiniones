"""
Dashboard Streamlit - Analizador de Opiniones de Docentes
=========================================================
Interfaz web interactiva para visualizar resultados del análisis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# Configuración de la página
st.set_page_config(
    page_title="Analizador de Opiniones Docentes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y presentación
st.title("📊 Analizador de Opiniones Docentes")
st.markdown("""
Este dashboard permite explorar el análisis de opiniones estudiantiles sobre docentes
utilizando modelado de tópicos (LDA).
""")

# Barra lateral
st.sidebar.title("⚙️ Configuración")
seccion = st.sidebar.radio("Selecciona una sección:", [
    "Inicio",
    "Datos",
    "Tópicos",
    "Análisis",
    "Sobre el Proyecto"
])

# SECCIÓN: INICIO
if seccion == "Inicio":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Bienvenido al Análisis de Opiniones")
        st.write("""
        Este proyecto utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) 
        para analizar automáticamente las opiniones estudiantiles sobre docentes.
        
        **Características principales:**
        - 📊 Análisis automático de encuestas
        - 🏷️ Extracción de tópicos principales
        - 📈 Visualización interactiva
        - 🔍 Exploración de patrones
        """)
    
    with col2:
        st.info("""
        **Stack Tecnológico:**
        - Python 3.10+
        - NLTK
        - scikit-learn
        - gensim
        - Streamlit
        """)

# SECCIÓN: DATOS
elif seccion == "Datos":
    st.header("📁 Datos del Proyecto")
    
    tab1, tab2, tab3 = st.tabs(["Cargar Datos", "Estadísticas", "Exploración"])
    
    with tab1:
        st.subheader("Cargar archivo de encuesta")
        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Archivo cargado correctamente")
            st.write(f"Forma del dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
            
            # Guardar datos
            if st.button("💾 Guardar en data/raw/"):
                ruta = Path("data/raw/encuesta.csv")
                ruta.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(ruta, index=False)
                st.success(f"Archivo guardado en {ruta}")
    
    with tab2:
        st.subheader("Estadísticas del Dataset")
        st.info("Carga un archivo CSV primero para ver estadísticas")
    
    with tab3:
        st.subheader("Exploración de Datos")
        st.info("Carga un archivo CSV primero para explorar los datos")

# SECCIÓN: TÓPICOS
elif seccion == "Tópicos":
    st.header("🏷️ Tópicos Identificados")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Distribución de Tópicos")
        st.info("Los resultados del modelo LDA aparecerán aquí una vez entrenado el modelo")
    
    with col2:
        st.subheader("Parámetros del Modelo")
        num_topics = st.slider("Número de tópicos:", 2, 20, 5)
        passes = st.slider("Pasadas del algoritmo:", 1, 50, 10)
        
        if st.button("🚀 Entrenar Modelo"):
            st.info("Funcionalidad en desarrollo...")

# SECCIÓN: ANÁLISIS
elif seccion == "Análisis":
    st.header("📈 Análisis Detallado")
    
    tab1, tab2, tab3 = st.tabs(["Coherencia", "Palabras Clave", "Documentos"])
    
    with tab1:
        st.subheader("Métricas de Coherencia")
        st.info("Ejecuta el modelo para ver métricas de coherencia")
    
    with tab2:
        st.subheader("Palabras Clave por Tópico")
        st.info("Las palabras clave más representativas aparecerán aquí")
    
    with tab3:
        st.subheader("Tópicos por Documento")
        st.info("Asignación de tópicos a cada documento")

# SECCIÓN: SOBRE EL PROYECTO
elif seccion == "Sobre el Proyecto":
    st.header("ℹ️ Sobre el Proyecto")
    
    st.markdown("""
    ## Analizador de Opiniones de Docentes
    
    ### Objetivo
    Analizar automáticamente encuestas de opinión estudiantil sobre docentes 
    para identificar tópicos principales y patrones de satisfacción.
    
    ### Metodología
    1. **Preprocesamiento**: Limpieza, tokenización y stemming
    2. **Vectorización**: Transformación a TF-IDF y Bag of Words
    3. **Modelado**: Entrenamiento de LDA (Latent Dirichlet Allocation)
    4. **Evaluación**: Métricas de coherencia
    5. **Visualización**: Gráficos e insights
    
    ### Stack Tecnológico
    | Componente | Tecnología |
    |---|---|
    | Lenguaje | Python 3.10+ |
    | Preprocesamiento | NLTK |
    | Vectorización | scikit-learn |
    | Modelado | gensim |
    | Visualización | pyLDAvis, matplotlib, wordcloud |
    | Dashboard | Streamlit |
    
    ### Equipo
    Proyecto colaborativo de 8 integrantes
    
    ### Referencias
    - [Gensim LDA](https://radimrehurek.com/gensim/)
    - [pyLDAvis](https://pyldavis.readthedocs.io/)
    - [NLTK](https://www.nltk.org/)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Analizador de Opiniones Docentes 📊 | Proyecto Educativo 2026</p>
</div>
""", unsafe_allow_html=True)
