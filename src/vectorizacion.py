"""
Módulo de Vectorización
=======================
Funciones para convertir texto a representaciones numéricas (TF-IDF, Bag of Words).
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd


def crear_tfidf_vectorizer(documentos, max_features=None, ngram_range=(1, 1)):
    """
    Crea una matriz TF-IDF a partir de documentos.
    
    Args:
        documentos (list): Lista de textos o tokens (debe ser textos unidos por espacios)
        max_features (int): Número máximo de features (default: None)
        ngram_range (tuple): Rango de n-gramas (default: (1,1) para unigramas)
        
    Returns:
        tuple: (matriz_tfidf, vectorizer, feature_names)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,  # Mínimo 2 documentos
        max_df=0.8  # Máximo 80% de documentos
    )
    
    matriz = vectorizer.fit_transform(documentos)
    feature_names = vectorizer.get_feature_names_out()
    
    return matriz, vectorizer, feature_names


def crear_bow_vectorizer(documentos, max_features=None, ngram_range=(1, 1)):
    """
    Crea una matriz Bag of Words a partir de documentos.
    
    Args:
        documentos (list): Lista de textos
        max_features (int): Número máximo de features (default: None)
        ngram_range (tuple): Rango de n-gramas (default: (1,1) para unigramas)
        
    Returns:
        tuple: (matriz_bow, vectorizer, feature_names)
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.8
    )
    
    matriz = vectorizer.fit_transform(documentos)
    feature_names = vectorizer.get_feature_names_out()
    
    return matriz, vectorizer, feature_names


def matriz_a_dataframe(matriz, feature_names, indices_doc=None):
    """
    Convierte una matriz sparse a DataFrame para visualización.
    
    Args:
        matriz: Matriz sparse (TF-IDF o BoW)
        feature_names: Nombres de las características
        indices_doc: Índices de los documentos (default: numeración)
        
    Returns:
        pd.DataFrame: Matriz en formato DataFrame
    """
    if indices_doc is None:
        indices_doc = [f'Doc_{i}' for i in range(matriz.shape[0])]
    
    df = pd.DataFrame(
        matriz.toarray(),
        columns=feature_names,
        index=indices_doc
    )
    
    return df


def get_top_features(matriz, feature_names, documento_idx, top_n=10):
    """
    Obtiene las características más importantes de un documento.
    
    Args:
        matriz: Matriz sparse
        feature_names: Nombres de características
        documento_idx: Índice del documento
        top_n: Número de top features (default: 10)
        
    Returns:
        pd.DataFrame: Features ordenadas por importancia
    """
    doc_vector = matriz[documento_idx].toarray().ravel()
    top_indices = doc_vector.argsort()[-top_n:][::-1]
    
    resultado = pd.DataFrame({
        'feature': [feature_names[i] for i in top_indices],
        'valor': doc_vector[top_indices]
    })
    
    return resultado.sort_values('valor', ascending=False)


def filtrar_vocabulario(documentos, min_freq=5):
    """
    Filtra vocabulario por frecuencia mínima.
    
    Args:
        documentos (list): Lista de textos
        min_freq (int): Frecuencia mínima (default: 5)
        
    Returns:
        set: Vocabulario filtrado
    """
    from collections import Counter
    
    vocab_freq = Counter()
    for doc in documentos:
        vocab_freq.update(doc.split())
    
    vocab_filtrado = {word for word, freq in vocab_freq.items() if freq >= min_freq}
    
    return vocab_filtrado
