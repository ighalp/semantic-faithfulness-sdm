"""
Text Utilities for Semantic Faithfulness Analysis

This module provides shared utility functions for text processing,
embedding generation, and distribution computation used across
the SDM package and GUI pipeline.
"""

import re
import numpy as np
from nltk.tokenize import sent_tokenize


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using NLTK's sent_tokenize.
    Normalizes whitespace and filters empty sentences.

    Args:
        text: Text to split into sentences

    Returns:
        List of sentence strings
    """
    # Ensure NLTK punkt tokenizer is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize into sentences
    sentences = sent_tokenize(text)

    # Filter out empty sentences
    return [s for s in sentences if s]


def embed_texts(model, texts: list[str], show_progress_bar: bool = False, batch_size: int = 8) -> np.ndarray:
    """
    Generate embeddings for texts using a SentenceTransformer model.

    Uses reduced batch_size=8 by default for MPS memory optimization.

    Args:
        model: SentenceTransformer model instance
        texts: List of text strings to embed
        show_progress_bar: Whether to show progress bar during encoding
        batch_size: Batch size for encoding (default: 8 for MPS memory)

    Returns:
        Numpy array of embeddings with shape (len(texts), embedding_dim)
    """
    return model.encode(texts, show_progress_bar=show_progress_bar, batch_size=batch_size)


def compute_triplet_distributions(
    question: str,
    context: str,
    answer: str,
    embedding_model,
    clustering_method: str = 'dib',
    tau_values: np.ndarray = None,
    max_n_clusters: int = 15,
    min_clusters: int = 3,
    seed: int = 42
) -> dict:
    """
    Compute probability distributions for a question-context-answer triplet.

    This function performs the complete pipeline:
    1. Split Q, C, A into sentences
    2. Generate embeddings for all sentences
    3. Cluster sentences using DIB
    4. Compute probability distributions p_q, p_c, p_a

    Args:
        question: Question text
        context: Context text
        answer: Answer text
        embedding_model: SentenceTransformer model instance (already loaded)
        clustering_method: Clustering method ('dib' or other)
        tau_values: DIB tau values (default: np.logspace(-2, 2, 30))
        max_n_clusters: Maximum number of clusters
        min_clusters: Minimum number of clusters for DIB
        seed: Random seed for reproducibility

    Returns:
        Dictionary with keys:
            - 'p_q': List of probabilities for question distribution
            - 'p_c': List of probabilities for context distribution
            - 'p_a': List of probabilities for answer distribution
            - 'n_topics': Number of discovered topics/clusters
    """
    from sdm_package.DIB_with_KL_upper_bound import DIBAnalyzer

    # Default tau values for DIB
    if tau_values is None:
        tau_values = np.logspace(-2, 2, 30)

    # Step 1: Split into sentences
    question_sentences = split_into_sentences(question)
    context_sentences = split_into_sentences(context)
    answer_sentences = split_into_sentences(answer)
    all_sentences = question_sentences + context_sentences + answer_sentences

    # Step 2: Generate embeddings (model already loaded and passed in)
    embeddings = embed_texts(embedding_model, all_sentences, show_progress_bar=False)

    # Step 3: Cluster using DIB
    dib_analyzer = DIBAnalyzer(embeddings, all_sentences)
    dib_analyzer.run(tau_values, max_n_clusters=max_n_clusters, seed=seed)
    recommendation, _ = dib_analyzer.get_recommendation(min_clusters=min_clusters, metric='kink_angle')

    if recommendation is None:
        raise ValueError(
            f"DIB clustering failed to find stable solutions. "
            f"This can happen when there are too few sentences ({len(all_sentences)} total). "
            f"Try providing longer context or answer text."
        )

    assignments = recommendation['assignments']
    n_topics = recommendation['n_clusters']

    # Step 4: Compute probability distributions
    n_q = len(question_sentences)
    n_c = len(context_sentences)
    n_a = len(answer_sentences)

    assignments_q = assignments[:n_q]
    assignments_c = assignments[n_q:n_q+n_c]
    assignments_a = assignments[n_q+n_c:]

    def to_distribution(assigns, n_topics):
        """Convert cluster assignments to probability distribution"""
        counts = np.bincount(assigns, minlength=n_topics)
        return (counts / counts.sum()).tolist()

    return {
        'p_q': to_distribution(assignments_q, n_topics),
        'p_c': to_distribution(assignments_c, n_topics),
        'p_a': to_distribution(assignments_a, n_topics),
        'n_topics': int(n_topics)
    }
