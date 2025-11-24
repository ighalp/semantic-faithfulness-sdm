"""
Analysis Service - Wraps the SDM pipeline for GUI execution

This service handles the complete workflow from raw QCA triplet text
to computed semantic faithfulness metrics.
"""

import sys
from pathlib import Path
from typing import Dict, Callable, Optional
from dataclasses import dataclass
import asyncio

# Add parent directory to path for sdm_package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize
from sdm_package import DIBAnalyzer, compute_semantic_faithfulness


@dataclass
class AnalysisConfig:
    """Configuration for semantic faithfulness analysis"""
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    clustering_method: str = 'udib'  # 'udib', 'kmeans', or 'agglomerative'
    n_clusters: Optional[int] = None  # Auto-determine if None
    tolerance: float = 1e-7
    max_iterations: int = 100


@dataclass
class AnalysisProgress:
    """Progress information for analysis"""
    stage: str  # 'tokenization', 'embedding', 'clustering', 'optimization'
    progress: float  # 0.0 to 1.0
    message: str


@dataclass
class AnalysisResults:
    """Complete results from semantic faithfulness analysis"""
    # Core metrics
    F_S: float
    SEP_total: float  # D_min
    SEP_system: float  # Will be computed from matrices

    # Entropy metrics
    H_Q: float
    H_C: float
    H_A: float

    # Probability distributions
    p_q: np.ndarray
    p_c: np.ndarray
    p_a: np.ndarray

    # Transition matrices
    Q_star: np.ndarray
    A_star: np.ndarray

    # Metadata
    n_clusters: int
    iterations: int
    converged: bool
    convergence_history: Optional[list] = None

    # Raw data for visualization
    question_sentences: list
    context_sentences: list
    answer_sentences: list


class AnalysisService:
    """Service for executing semantic faithfulness analysis"""

    def __init__(self):
        self._embedding_model = None
        self._current_model_name = None

    def _load_embedding_model(self, model_name: str):
        """Load embedding model (cached)"""
        if self._embedding_model is None or self._current_model_name != model_name:
            self._embedding_model = SentenceTransformer(model_name)
            self._current_model_name = model_name
        return self._embedding_model

    async def analyze(
        self,
        question: str,
        context: str,
        answer: str,
        config: AnalysisConfig,
        progress_callback: Optional[Callable[[AnalysisProgress], None]] = None
    ) -> AnalysisResults:
        """
        Perform complete semantic faithfulness analysis

        Args:
            question: Question text
            context: Context text
            answer: Answer text
            config: Analysis configuration
            progress_callback: Optional callback for progress updates

        Returns:
            AnalysisResults object with all computed metrics
        """

        def report_progress(stage: str, progress: float, message: str):
            if progress_callback:
                progress_callback(AnalysisProgress(stage, progress, message))

        # Stage 1: Tokenization
        report_progress('tokenization', 0.0, 'Tokenizing text into sentences...')
        await asyncio.sleep(0)  # Yield control

        question_sentences = sent_tokenize(question)
        context_sentences = sent_tokenize(context)
        answer_sentences = sent_tokenize(answer)

        all_sentences = question_sentences + context_sentences + answer_sentences
        n_q = len(question_sentences)
        n_c = len(context_sentences)
        n_a = len(answer_sentences)

        report_progress('tokenization', 1.0,
                       f'Tokenized: {n_q} Q, {n_c} C, {n_a} A sentences')

        # Stage 2: Embedding
        report_progress('embedding', 0.0, f'Loading model {config.embedding_model}...')
        await asyncio.sleep(0)

        embedding_model = await asyncio.to_thread(
            self._load_embedding_model, config.embedding_model
        )

        report_progress('embedding', 0.3, 'Computing sentence embeddings...')
        await asyncio.sleep(0)

        embeddings = await asyncio.to_thread(
            embedding_model.encode, all_sentences, show_progress_bar=False
        )

        report_progress('embedding', 1.0, f'Computed {len(embeddings)} embeddings')

        # Stage 3: Clustering
        report_progress('clustering', 0.0, f'Clustering with {config.clustering_method}...')
        await asyncio.sleep(0)

        # Determine number of clusters if not specified
        n_clusters = config.n_clusters
        if n_clusters is None:
            # Heuristic: sqrt of number of sentences, bounded
            n_clusters = max(3, min(20, int(np.sqrt(len(all_sentences)))))

        # Perform clustering
        if config.clustering_method == 'udib':
            dib = DIBAnalyzer(verbose=False)
            cluster_labels = await asyncio.to_thread(
                dib.fit_predict,
                embeddings,
                n_clusters=n_clusters,
                init_method='kmeans++',
                max_iter=100,
                tol=1e-6
            )
        elif config.clustering_method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = await asyncio.to_thread(kmeans.fit_predict, embeddings)
        elif config.clustering_method == 'agglomerative':
            agg = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = await asyncio.to_thread(agg.fit_predict, embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {config.clustering_method}")

        report_progress('clustering', 1.0, f'Clustered into {n_clusters} topics')

        # Compute probability distributions from clusters
        labels_q = cluster_labels[:n_q]
        labels_c = cluster_labels[n_q:n_q + n_c]
        labels_a = cluster_labels[n_q + n_c:]

        # Create probability distributions
        p_q = np.bincount(labels_q, minlength=n_clusters).astype(float)
        p_c = np.bincount(labels_c, minlength=n_clusters).astype(float)
        p_a = np.bincount(labels_a, minlength=n_clusters).astype(float)

        # Normalize
        p_q = p_q / (p_q.sum() + 1e-12)
        p_c = p_c / (p_c.sum() + 1e-12)
        p_a = p_a / (p_a.sum() + 1e-12)

        # Stage 4: Optimization
        report_progress('optimization', 0.0, 'Computing semantic faithfulness...')
        await asyncio.sleep(0)

        sf_result = await asyncio.to_thread(
            compute_semantic_faithfulness,
            p_c=p_c,
            p_q=p_q,
            p_a=p_a,
            tol_outer=config.tolerance,
            max_outer_iter=config.max_iterations,
            debug=False
        )

        report_progress('optimization', 1.0, 'Optimization complete')

        # Compute entropy metrics
        H_Q = entropy(p_q, base=2)
        H_C = entropy(p_c, base=2)
        H_A = entropy(p_a, base=2)

        # Compute SEP_system (from paper: SEP_system = H(A) - H(C))
        SEP_system = H_A - H_C

        # Package results
        return AnalysisResults(
            F_S=sf_result['F_S'],
            SEP_total=sf_result['D_min'],
            SEP_system=SEP_system,
            H_Q=H_Q,
            H_C=H_C,
            H_A=H_A,
            p_q=p_q,
            p_c=p_c,
            p_a=p_a,
            Q_star=sf_result['Q_star'],
            A_star=sf_result['A_star'],
            n_clusters=n_clusters,
            iterations=sf_result['iterations'],
            converged=sf_result['converged'],
            convergence_history=sf_result.get('convergence_history'),
            question_sentences=question_sentences,
            context_sentences=context_sentences,
            answer_sentences=answer_sentences
        )

    def validate_input(self, question: str, context: str, answer: str) -> tuple[bool, str]:
        """
        Validate QCA triplet input

        Returns:
            (is_valid, error_message)
        """
        if not question or not question.strip():
            return False, "Question cannot be empty"

        if not context or not context.strip():
            return False, "Context cannot be empty"

        if not answer or not answer.strip():
            return False, "Answer cannot be empty"

        # Check minimum sentence counts
        try:
            q_sents = sent_tokenize(question.strip())
            c_sents = sent_tokenize(context.strip())
            a_sents = sent_tokenize(answer.strip())

            if len(q_sents) < 1:
                return False, "Question must contain at least 1 sentence"

            if len(c_sents) < 2:
                return False, "Context must contain at least 2 sentences"

            if len(a_sents) < 1:
                return False, "Answer must contain at least 1 sentence"

        except Exception as e:
            return False, f"Error tokenizing text: {str(e)}"

        return True, ""


# Singleton instance
_service = None

def get_analysis_service() -> AnalysisService:
    """Get singleton analysis service instance"""
    global _service
    if _service is None:
        _service = AnalysisService()
    return _service
