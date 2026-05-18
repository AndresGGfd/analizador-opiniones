"""
Módulo de Visualización
=======================
Funciones para crear gráficos e visualizaciones.
"""

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import seaborn as sns


def configurar_estilo():
    """Configura el estilo de matplotlib."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def grafico_distribucion_topicos(topicos_documentos, num_docs=None):
    """
    Crea un gráfico de barras con distribución de tópicos.
    
    Args:
        topicos_documentos (list): Lista de tópicos dominantes (ID de tópico)
        num_docs (int): Número de documentos a mostrar (default: todos)
        
    Returns:
        fig, ax: Matplotlib figure y axis
    """
    configurar_estilo()
    
    data = pd.Series(topicos_documentos)
    counts = data.value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind='bar', ax=ax, color='steelblue')
    
    ax.set_title('Distribución de Tópicos en Documentos', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tópico ID', fontsize=12)
    ax.set_ylabel('Cantidad de Documentos', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def grafico_probabilidades_topicos(probabilidades_doc, doc_idx=0):
    """
    Crea un gráfico de probabilidades de tópicos para un documento.
    
    Args:
        probabilidades_doc (list): Tuplas (topic_id, probabilidad) para un documento
        doc_idx (int): Índice del documento
        
    Returns:
        fig, ax: Matplotlib figure y axis
    """
    configurar_estilo()
    
    if not probabilidades_doc:
        print("No hay datos para graficar")
        return None, None
    
    topicos = [f'T{int(item[0])}' for item in probabilidades_doc]
    probs = [item[1] for item in probabilidades_doc]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(topicos, probs, color='coral')
    
    ax.set_title(f'Probabilidades de Tópicos - Documento {doc_idx}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Tópico', fontsize=12)
    ax.set_ylabel('Probabilidad', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def crear_wordcloud(palabras_pesos, titulo='Word Cloud', ancho=12, alto=6):
    """
    Crea una nube de palabras a partir de palabras y pesos.
    
    Args:
        palabras_pesos (dict): Diccionario {palabra: peso}
        titulo (str): Título de la nube
        ancho (int): Ancho en pulgadas
        alto (int): Alto en pulgadas
        
    Returns:
        fig, ax: Matplotlib figure y axis
    """
    wordcloud = WordCloud(
        width=ancho*100,
        height=alto*100,
        background_color='white',
        colormap='viridis',
        prefer_horizontal=0.7,
        max_words=100
    ).generate_from_frequencies(palabras_pesos)
    
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax


def grafico_coherencia_vs_topicos(results_coherencia):
    """
    Gráfico de coherencia vs número de tópicos.
    
    Args:
        results_coherencia (dict): {num_topics: coherence_score}
        
    Returns:
        fig, ax: Matplotlib figure y axis
    """
    configurar_estilo()
    
    num_topics = sorted(results_coherencia.keys())
    coherencias = [results_coherencia[n] for n in num_topics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(num_topics, coherencias, marker='o', linewidth=2, markersize=8, color='steelblue')
    
    # Marcar el óptimo
    optimal_topics = num_topics[coherencias.index(max(coherencias))]
    ax.scatter([optimal_topics], [max(coherencias)], 
              color='red', s=200, marker='*', label=f'Óptimo: {optimal_topics} tópicos')
    
    ax.set_title('Coherencia vs Número de Tópicos', fontsize=14, fontweight='bold')
    ax.set_xlabel('Número de Tópicos', fontsize=12)
    ax.set_ylabel('Score de Coherencia', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig, ax


def grafico_heatmap_documentos_topicos(matriz_doc_topic):
    """
    Crea un heatmap de documentos vs tópicos.
    
    Args:
        matriz_doc_topic (pd.DataFrame): DataFrame con documentos x tópicos
        
    Returns:
        fig, ax: Matplotlib figure y axis
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(matriz_doc_topic, cmap='YlOrRd', cbar_kws={'label': 'Probabilidad'}, ax=ax)
    
    ax.set_title('Matriz Documentos vs Tópicos', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tópicos', fontsize=12)
    ax.set_ylabel('Documentos', fontsize=12)
    
    plt.tight_layout()
    return fig, ax


def guardar_figura(fig, ruta):
    """
    Guarda una figura en archivo.
    
    Args:
        fig: Matplotlib figure object
        ruta (str): Ruta de salida
    """
    fig.savefig(ruta, dpi=300, bbox_inches='tight')
    print(f"Figura guardada en: {ruta}")
