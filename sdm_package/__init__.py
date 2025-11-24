"""
Semantic Divergence Metrics (SDM) Package

This package implements information-theoretically principled metrics for evaluating
LLM faithfulness and semantic alignment with source contexts.

Main Components:
- SemanticFaithfulnessAnalyzer: Core SDM framework for computing Semantic Faithfulness (F_S)
                                 and Semantic Entropy Production (SEP) metrics
- DIBAnalyzer: Upper-Bounded Deterministic Information Bottleneck clustering
- compute_semantic_faithfulness: Compute Semantic Faithfulness (F_S) and Entropy Production (SEP) metrics

Citation:
    If you use this package in your research, please cite:

    Halperin, I. (2025). Information-Theoretic Faithfulness Metrics for Large Language Models.
    arXiv preprint arXiv:XXXX.XXXXX
"""

__version__ = "1.0.0"
__author__ = "Igor Halperin"

from .SDM import SemanticFaithfulnessAnalyzer
from .DIB_with_KL_upper_bound import DIBAnalyzer
from .compute_semantic_faithfulness import compute_semantic_faithfulness

__all__ = [
    'SemanticFaithfulnessAnalyzer',
    'DIBAnalyzer',
    'compute_semantic_faithfulness',
]
