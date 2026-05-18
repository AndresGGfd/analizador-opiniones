"""
Módulo del Modelo LDA
====================
Funciones para entrenar y evaluar modelos de tópicos LDA.
"""

import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


def crear_diccionario_y_corpus(documentos_tokens):
    """
    Crea un diccionario y corpus en formato gensim a partir de tokens.
    
    Args:
        documentos_tokens (list): Lista de listas de tokens
        
    Returns:
        tuple: (dictionary, corpus)
    """
    # Crear diccionario del corpus
    dictionary = Dictionary(documentos_tokens)
    
    # Filtrar tokens extremos
    dictionary.filter_extremes(no_below=2, no_above=0.8, keep_n=100000)
    
    # Crear corpus (bag of words)
    corpus = [dictionary.doc2bow(doc) for doc in documentos_tokens]
    
    return dictionary, corpus


def entrenar_lda(corpus, dictionary, num_topics=5, passes=10, workers=4):
    """
    Entrena un modelo LDA.
    
    Args:
        corpus: Corpus en formato gensim (lista de bag-of-words)
        dictionary: Diccionario gensim
        num_topics (int): Número de tópicos (default: 5)
        passes (int): Número de pasadas (default: 10)
        workers (int): Número de workers (default: 4)
        
    Returns:
        LdaModel: Modelo LDA entrenado
    """
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        workers=workers,
        alpha='auto',  # Optimizar alpha automáticamente
        eta='auto',    # Optimizar eta automáticamente
        minimum_probability=0.0,
        random_state=42,
        per_word_topics=True
    )
    
    return lda_model


def calcular_coherencia(lda_model, corpus, diccionario, textos_tokens):
    """
    Calcula la métrica de coherencia del modelo.
    
    Args:
        lda_model: Modelo LDA entrenado
        corpus: Corpus en formato gensim
        diccionario: Diccionario gensim
        textos_tokens (list): Documentos originales tokenizados
        
    Returns:
        float: Score de coherencia
    """
    coherence_model = CoherenceModel(
        model=lda_model,
        corpus=corpus,
        dictionary=diccionario,
        texts=textos_tokens,
        coherence='c_v'
    )
    
    return coherence_model.get_coherence()


def optimizar_num_topics(corpus, dictionary, textos_tokens, 
                         min_topics=2, max_topics=10, step=1):
    """
    Busca el número óptimo de tópicos según coherencia.
    
    Args:
        corpus: Corpus gensim
        dictionary: Diccionario gensim
        textos_tokens: Documentos tokenizados
        min_topics (int): Número mínimo de tópicos
        max_topics (int): Número máximo de tópicos
        step (int): Incremento en cada iteración
        
    Returns:
        dict: Diccionario con num_topics como clave y coherencia como valor
    """
    resultados = {}
    
    for num_topics in range(min_topics, max_topics + 1, step):
        print(f"Entrenando modelo con {num_topics} tópicos...")
        
        lda_model = entrenar_lda(corpus, dictionary, num_topics=num_topics)
        coherencia = calcular_coherencia(lda_model, corpus, dictionary, textos_tokens)
        
        resultados[num_topics] = coherencia
        print(f"  Coherencia: {coherencia:.4f}")
    
    return resultados


def obtener_topicos(lda_model, num_words=10):
    """
    Obtiene los Top-N palabras de cada tópico.
    
    Args:
        lda_model: Modelo LDA entrenado
        num_words (int): Número de palabras por tópico (default: 10)
        
    Returns:
        dict: Diccionario con tópicos y sus palabras principales
    """
    topicos = {}
    
    for idx, topic in lda_model.print_topics(num_topics=-1, num_words=num_words):
        topicos[f'Topico_{idx}'] = topic
    
    return topicos


def asignar_topicos(lda_model, corpus):
    """
    Asigna los tópicos dominantes a cada documento.
    
    Args:
        lda_model: Modelo LDA entrenado
        corpus: Corpus gensim
        
    Returns:
        list: Lista con tópicos dominantes por documento
    """
    topicos_doc = []
    
    for doc_topics in lda_model.get_document_topics(corpus):
        if doc_topics:
            # Obtener tópico con mayor probabilidad
            topico_dominante = max(doc_topics, key=lambda x: x[1])
            topicos_doc.append(topico_dominante)
        else:
            topicos_doc.append((None, 0.0))
    
    return topicos_doc


def visualizar_lda(lda_model, corpus, dictionary):
    """
    Crea una visualización interactiva del modelo LDA.
    
    Args:
        lda_model: Modelo LDA entrenado
        corpus: Corpus gensim
        dictionary: Diccionario gensim
        
    Returns:
        pyLDAvis visualization object
    """
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    return vis
