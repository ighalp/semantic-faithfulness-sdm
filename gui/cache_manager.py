"""
Cache Manager for Semantic Faithfulness Pipeline
Provides intelligent caching to avoid regenerating expensive computations
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any


class CacheManager:
    """Manages caching for pipeline components"""

    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager

        Args:
            cache_dir: Root directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.paraphrases_dir = self.cache_dir / 'paraphrases'
        self.answers_dir = self.cache_dir / 'answers'
        self.distributions_dir = self.cache_dir / 'distributions'
        self.fs_scores_dir = self.cache_dir / 'fs_scores'

        # Create directories
        for dir_path in [self.paraphrases_dir, self.answers_dir,
                         self.distributions_dir, self.fs_scores_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_string(text: str) -> str:
        """Generate hash for a string"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    @staticmethod
    def _hash_dict(data: Dict) -> str:
        """Generate hash for a dictionary"""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]

    def get_paraphrases(
        self,
        question: str,
        model: str,
        num_paraphrases: int
    ) -> Optional[List[str]]:
        """
        Get cached paraphrases if available

        Args:
            question: Original question
            model: LLM model name
            num_paraphrases: Number of paraphrases requested

        Returns:
            List of paraphrases if cached, None otherwise
        """
        question_hash = self._hash_string(question)
        cache_file = self.paraphrases_dir / f'{question_hash}_{model}_{num_paraphrases}.json'

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data['paraphrases']
            except Exception:
                return None
        return None

    def save_paraphrases(
        self,
        question: str,
        model: str,
        num_paraphrases: int,
        paraphrases: List[str]
    ):
        """
        Save paraphrases to cache

        Args:
            question: Original question
            model: LLM model name
            num_paraphrases: Number of paraphrases
            paraphrases: List of paraphrases to cache
        """
        question_hash = self._hash_string(question)
        cache_file = self.paraphrases_dir / f'{question_hash}_{model}_{num_paraphrases}.json'

        data = {
            'question': question,
            'model': model,
            'num_paraphrases': num_paraphrases,
            'paraphrases': paraphrases
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_answer(
        self,
        question: str,
        context: str,
        model: str
    ) -> Optional[str]:
        """
        Get cached answer if available

        Args:
            question: Question text
            context: Context text
            model: LLM model name

        Returns:
            Answer if cached, None otherwise
        """
        qc_hash = self._hash_dict({'question': question, 'context': context})
        cache_file = self.answers_dir / f'{qc_hash}_{model}.json'

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data['answer']
            except Exception:
                return None
        return None

    def save_answer(
        self,
        question: str,
        context: str,
        model: str,
        answer: str
    ):
        """
        Save answer to cache

        Args:
            question: Question text
            context: Context text
            model: LLM model name
            answer: Answer to cache
        """
        qc_hash = self._hash_dict({'question': question, 'context': context})
        cache_file = self.answers_dir / f'{qc_hash}_{model}.json'

        data = {
            'question': question,
            'context': context,
            'model': model,
            'answer': answer
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_distributions(
        self,
        question: str,
        context: str,
        answer: str,
        embedding_model: str,
        clustering_method: str
    ) -> Optional[Dict]:
        """
        Get cached distributions if available

        Args:
            question: Question text
            context: Context text
            answer: Answer text
            embedding_model: Embedding model name
            clustering_method: Clustering method

        Returns:
            Distribution data if cached, None otherwise
        """
        qca_hash = self._hash_dict({
            'question': question,
            'context': context,
            'answer': answer
        })
        cache_file = self.distributions_dir / f'{qca_hash}_{embedding_model}_{clustering_method}.json'

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_distributions(
        self,
        question: str,
        context: str,
        answer: str,
        embedding_model: str,
        clustering_method: str,
        distributions: Dict
    ):
        """
        Save distributions to cache

        Args:
            question: Question text
            context: Context text
            answer: Answer text
            embedding_model: Embedding model name
            clustering_method: Clustering method
            distributions: Distribution data to cache
        """
        qca_hash = self._hash_dict({
            'question': question,
            'context': context,
            'answer': answer
        })
        cache_file = self.distributions_dir / f'{qca_hash}_{embedding_model}_{clustering_method}.json'

        data = {
            'question': question,
            'context': context,
            'answer': answer,
            'embedding_model': embedding_model,
            'clustering_method': clustering_method,
            'distributions': distributions
        }

        # Ensure parent directory exists (for model names with slashes like "Qwen/Qwen3-Embedding-0.6B")
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_fs_score(
        self,
        p_q: List[float],
        p_c: List[float],
        p_a: List[float]
    ) -> Optional[Dict]:
        """
        Get cached F_S score if available

        Args:
            p_q: Question distribution
            p_c: Context distribution
            p_a: Answer distribution

        Returns:
            F_S score data if cached, None otherwise
        """
        dist_hash = self._hash_dict({
            'p_q': p_q,
            'p_c': p_c,
            'p_a': p_a
        })
        cache_file = self.fs_scores_dir / f'{dist_hash}.json'

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_fs_score(
        self,
        p_q: List[float],
        p_c: List[float],
        p_a: List[float],
        fs_score: Dict
    ):
        """
        Save F_S score to cache

        Args:
            p_q: Question distribution
            p_c: Context distribution
            p_a: Answer distribution
            fs_score: F_S score data to cache
        """
        dist_hash = self._hash_dict({
            'p_q': p_q,
            'p_c': p_c,
            'p_a': p_a
        })
        cache_file = self.fs_scores_dir / f'{dist_hash}.json'

        data = {
            'distributions': {
                'p_q': p_q,
                'p_c': p_c,
                'p_a': p_a
            },
            'fs_score': fs_score
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache files

        Args:
            cache_type: Type of cache to clear ('paraphrases', 'answers',
                       'distributions', 'fs_scores', or None for all)
        """
        if cache_type is None:
            dirs = [self.paraphrases_dir, self.answers_dir,
                   self.distributions_dir, self.fs_scores_dir]
        elif cache_type == 'paraphrases':
            dirs = [self.paraphrases_dir]
        elif cache_type == 'answers':
            dirs = [self.answers_dir]
        elif cache_type == 'distributions':
            dirs = [self.distributions_dir]
        elif cache_type == 'fs_scores':
            dirs = [self.fs_scores_dir]
        else:
            raise ValueError(f"Invalid cache_type: {cache_type}")

        for dir_path in dirs:
            for file_path in dir_path.glob('*.json'):
                file_path.unlink()

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics

        Returns:
            Dictionary with count of cached items per type
        """
        return {
            'paraphrases': len(list(self.paraphrases_dir.glob('*.json'))),
            'answers': len(list(self.answers_dir.glob('*.json'))),
            'distributions': len(list(self.distributions_dir.glob('*.json'))),
            'fs_scores': len(list(self.fs_scores_dir.glob('*.json')))
        }
