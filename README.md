# Analizador de Opiniones de Docentes

Proyecto colaborativo para análisis de opiniones estudiantiles sobre docentes usando modelado de tópicos con LDA (Latent Dirichlet Allocation).

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis completo de encuestas de opinión estudiantil sobre la calidad docente mediante:
- Preprocesamiento de texto (tokenización, limpieza, stemming)
- Vectorización con TF-IDF y Bag of Words
- Modelado de tópicos con LDA (Latent Dirichlet Allocation)
- Visualización interactiva de resultados
- Dashboard interactivo con Streamlit

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Editor | Visual Studio Code |
| Control de versiones | Git + GitHub |
| Tokenización | NLTK |
| Stemming | NLTK SnowballStemmer |
| Vectorización | scikit-learn |
| Modelo de tópicos | gensim (LDA) |
| Evaluación | gensim (CoherenceModel) |
| Visualización | pyLDAvis, matplotlib, wordcloud |
| Dashboard | Streamlit |
| Datos | pandas |

## 📁 Estructura del Proyecto

```
analizador-opiniones/
├── README.md                      # Este archivo
├── requirements.txt               # Dependencias Python
├── .gitignore                     # Archivos ignorados por Git
│
├── data/
│   ├── raw/                       # Datos originales (CSV de encuesta)
│   └── processed/                 # Datos preprocesados
│
├── notebooks/
│   ├── 01_exploracion.ipynb       # Análisis exploratorio
│   ├── 02_preprocesamiento.ipynb  # Limpieza y tokenización
│   ├── 03_vectorizacion.ipynb     # TF-IDF y Bag of Words
│   ├── 04_modelo_lda.ipynb        # Entrenamiento LDA
│   ├── 05_evaluacion.ipynb        # Métricas de coherencia
│   └── 06_cruce_likert.ipynb      # Análisis con escala Likert
│
├── src/
│   ├── preprocesamiento.py        # Funciones de limpieza reutilizables
│   ├── vectorizacion.py           # Funciones de vectorización
│   ├── modelo.py                  # Funciones del modelo LDA
│   └── visualizacion.py           # Funciones de gráficos
│
├── app/
│   └── app.py                     # Dashboard Streamlit
│
└── docs/
    └── informe.pdf                # Documento final del proyecto
```

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/usuario/analizador-opiniones-docentes.git
cd analizador-opiniones-docentes
```

### 2. Crear entorno virtual (opcional pero recomendado)
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Descargar recursos NLTK
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📊 Flujo de Trabajo

### Equipo de Datos (2 personas)
- Diseñan la encuesta
- Recolectan respuestas
- Generan CSV limpio en `data/raw/`

### Equipo de Preprocesamiento (2 personas)
- Limpieza de texto
- Tokenización y eliminación de stopwords
- Stemming con Snowball
- Guardan corpus procesado en `data/processed/`

### Equipo de Modelado LDA (2 personas)
- Vectorización (TF-IDF/BoW)
- Entrenamiento del modelo LDA
- Ajuste de hiperparámetros
- Evaluación con métricas de coherencia

### Equipo de Visualización (2 personas)
- Gráficos con matplotlib
- Visualización interactiva con pyLDAvis
- Análisis cruzado con escala Likert
- Dashboard con Streamlit

## 🔧 Cómo Usar

### Ejecutar un notebook
```bash
# Los notebooks se abren directamente en VS Code
# Con la extensión Jupyter puedes correr celdas individuales
```

### Ejecutar el dashboard
```bash
streamlit run app/app.py
```

## 📝 Flujo de Git

**Rama principal:** `main` (nunca editar directamente)

**Para cada tarea:**
```bash
git checkout main
git pull origin main
git checkout -b feature/descripcion-tarea

# Hacer cambios...

git add .
git commit -m "Descripción clara del cambio"
git push origin feature/descripcion-tarea
```

Luego abrir Pull Request en GitHub para revisión.

## 👥 Equipo

andress mi señor

## 📚 Referencias

- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Gensim LDA](https://radimrehurek.com/gensim/)
- [pyLDAvis](https://pyldavis.readthedocs.io/)
- [Streamlit](https://streamlit.io/)

## 📄 Licencia

Proyecto educativo del grupo (2026)
