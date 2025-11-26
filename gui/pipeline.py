"""
Pipeline Orchestrator
Chains together LLM generation, embedding, clustering, and distribution computation
"""

from pathlib import Path
from typing import Dict, List, Optional, Callable
import json
import asyncio
from datetime import datetime

from llm_client import LLMClient, LLMProvider, LLMModel
from cache_manager import CacheManager


# Global variable for worker process (each worker gets its own copy with spawn)
_worker_model = None


def _init_embedding_worker(model_name: str):
    """
    Initialize worker process with embedding model.

    This function is called once per worker process when the ProcessPoolExecutor
    starts up. With 'spawn' multiprocessing, each worker gets a fresh Python
    interpreter, avoiding PyTorch's threading conflicts with NiceGUI.

    Args:
        model_name: Name of the SentenceTransformer model to load
    """
    global _worker_model
    from sentence_transformers import SentenceTransformer
    import os
    import torch

    # Set single-threaded mode for PyTorch in worker process
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Clear any existing GPU memory before loading model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load model - use CPU if MPS has memory issues
    # You can set FORCE_CPU=1 environment variable to force CPU usage
    device = 'cpu' if os.environ.get('FORCE_CPU') else None
    _worker_model = SentenceTransformer(model_name, device=device)


def _compute_embeddings_in_worker(triplet_data: Dict, clustering_method: str) -> Dict:
    """
    Compute embeddings and distributions for a single triplet in worker process.

    This function runs in an isolated worker process, using the pre-loaded
    embedding model from _init_embedding_worker. It calls the existing
    compute_triplet_distributions function from sdm_package.

    Args:
        triplet_data: Dictionary with 'question', 'context', 'answer', 'prompt_id'
        clustering_method: Clustering method to use

    Returns:
        Dictionary with distributions and prompt_id
    """
    global _worker_model
    import torch
    from sdm_package.text_utils import compute_triplet_distributions

    # Compute distributions using existing SDM infrastructure
    dist_result = compute_triplet_distributions(
        question=triplet_data['question'],
        context=triplet_data['context'],
        answer=triplet_data['answer'],
        embedding_model=_worker_model,
        clustering_method=clustering_method
    )

    # Add prompt_id to result
    dist_result['prompt_id'] = triplet_data.get('prompt_id', 'unknown')

    # Clear GPU memory after computation to prevent OOM errors
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dist_result


class SemanticFaithfulnessPipeline:
    """End-to-end pipeline for semantic faithfulness analysis"""

    def __init__(
        self,
        llm_client: LLMClient,
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize pipeline

        Args:
            llm_client: LLM client for generation
            output_dir: Directory for output files
            progress_callback: Optional callback for progress updates
            cache_dir: Directory for cache files (default: output_dir.parent / 'cache')
        """
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback

        # Initialize cache manager
        if cache_dir is None:
            cache_dir = self.output_dir.parent / 'cache'
        self.cache = CacheManager(cache_dir)

        # Initialize ProcessPoolExecutor for embedding computation (created on-demand)
        # Using spawn method (set in app.py) to avoid PyTorch threading conflicts
        self.embedding_executor = None

    async def _update_progress(self, step: str, current: int, total: int, message: str = ""):
        """Update progress if callback is set"""
        if self.progress_callback:
            await self.progress_callback(step, current, total, message)

    async def run_full_pipeline(
        self,
        original_question: str,
        context: str,
        num_paraphrases: int,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        clustering_method: str = "spectral",
        run_id: Optional[str] = None,
        force_regenerate: bool = False
    ) -> Dict:
        """
        Run the complete pipeline from LLM generation to F_S computation

        Args:
            original_question: Original question/prompt
            context: Context information
            num_paraphrases: Number of paraphrases to generate
            embedding_model: Model for embeddings
            clustering_method: Clustering method
            run_id: Optional run identifier
            force_regenerate: If True, bypass cache and regenerate all data

        Returns:
            Dictionary with results and file paths
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'original_question': original_question,
                'context': context,
                'num_paraphrases': num_paraphrases,
                'embedding_model': embedding_model,
                'clustering_method': clustering_method,
                'llm_provider': self.llm_client.provider.value,
                'llm_model': self.llm_client.model.value
            }
        }

        # Initialize cache statistics
        cache_stats = {
            'paraphrases_cached': False,
            'answers_cached': 0,
            'answers_generated': 0,
            'distributions_cached': 0,
            'distributions_computed': 0,
            'fs_scores_cached': 0,
            'fs_scores_computed': 0
        }

        try:
            # Step 1: Generate paraphrases (with caching)
            await self._update_progress("Generating paraphrases", 0, 5, "Checking cache...")

            cached_paraphrases = None
            if not force_regenerate:
                cached_paraphrases = self.cache.get_paraphrases(
                    original_question,
                    self.llm_client.model.value,
                    num_paraphrases
                )

            if cached_paraphrases is not None:
                paraphrases = cached_paraphrases
                cache_stats['paraphrases_cached'] = True
                await self._update_progress("Generating paraphrases", 1, 5,
                                           f"Using cached paraphrases ({len(paraphrases)} prompts)")
            else:
                await self._update_progress("Generating paraphrases", 0, 5, "Calling LLM...")
                paraphrases = await self.llm_client.generate_paraphrases(
                    original_question, context, num_paraphrases
                )
                # Save to cache
                self.cache.save_paraphrases(
                    original_question,
                    self.llm_client.model.value,
                    num_paraphrases,
                    paraphrases
                )
                await self._update_progress("Generating paraphrases", 1, 5,
                                           f"Generated {len(paraphrases)} prompts")

            # Save prompts
            prompts_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'run_id': run_id,
                    'llm_provider': self.llm_client.provider.value,
                    'llm_model': self.llm_client.model.value
                },
                'prompts': [
                    {
                        'prompt_id': f'PROMPT_{i}',
                        'text': paraphrase,
                        'type': 'original' if i == 0 else 'paraphrase',
                        'context': context
                    }
                    for i, paraphrase in enumerate(paraphrases)
                ]
            }

            prompts_file = self.output_dir / f'prompts_{run_id}.json'
            with open(prompts_file, 'w') as f:
                json.dump(prompts_data, f, indent=2)

            results['prompts_file'] = str(prompts_file)
            results['prompts'] = paraphrases

            # Step 2: Generate answers (with caching)
            await self._update_progress("Generating answers", 1, 5, "Checking cache...")

            answers = []
            answers_to_generate = []
            answers_to_generate_indices = []

            for i, paraphrase in enumerate(paraphrases):
                cached_answer = None
                if not force_regenerate:
                    cached_answer = self.cache.get_answer(
                        paraphrase,
                        context,
                        self.llm_client.model.value
                    )

                if cached_answer is not None:
                    answers.append(cached_answer)
                    cache_stats['answers_cached'] += 1
                else:
                    answers.append(None)  # Placeholder
                    answers_to_generate.append(paraphrase)
                    answers_to_generate_indices.append(i)

            if answers_to_generate:
                await self._update_progress("Generating answers", 1, 5,
                                           f"Generating {len(answers_to_generate)} answers ({cache_stats['answers_cached']} cached)...")

                async def answer_progress(current, total):
                    await self._update_progress("Generating answers", 1, 5,
                                               f"Generated {current}/{total} answers ({cache_stats['answers_cached']} cached)")

                generated_answers = await self.llm_client.generate_all_answers(
                    answers_to_generate, context, progress_callback=answer_progress
                )

                # Fill in generated answers and save to cache
                for idx, answer in zip(answers_to_generate_indices, generated_answers):
                    answers[idx] = answer
                    self.cache.save_answer(
                        paraphrases[idx],
                        context,
                        self.llm_client.model.value,
                        answer
                    )
                cache_stats['answers_generated'] = len(generated_answers)
            else:
                await self._update_progress("Generating answers", 1, 5,
                                           f"Using cached answers ({cache_stats['answers_cached']} answers)")

            # Save answers
            answers_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'run_id': run_id,
                    'llm_provider': self.llm_client.provider.value,
                    'llm_model': self.llm_client.model.value
                },
                'answers': [
                    {
                        'prompt_id': f'PROMPT_{i}',
                        'answer': answer
                    }
                    for i, answer in enumerate(answers)
                ]
            }

            answers_file = self.output_dir / f'answers_{run_id}.json'
            with open(answers_file, 'w') as f:
                json.dump(answers_data, f, indent=2)

            results['answers_file'] = str(answers_file)
            results['answers'] = answers

            await self._update_progress("Generating answers", 2, 5,
                                       f"Complete: {cache_stats['answers_cached']} cached, {cache_stats['answers_generated']} generated")

            # Step 3: Create QCA triplets
            await self._update_progress("Creating triplets", 2, 5, "Building QCA structure...")

            triplets = []
            for i in range(len(paraphrases)):
                triplets.append({
                    'prompt_id': f'PROMPT_{i}',
                    'question': paraphrases[i],
                    'context': context,
                    'answer': answers[i]
                })

            results['triplets'] = triplets

            await self._update_progress("Creating triplets", 3, 5,
                                       f"Created {len(triplets)} QCA triplets")

            # Step 4: Generate embeddings and cluster (with caching)
            # This follows the pattern from Semantic_Faithfulness_SDM_demo.ipynb cells 7-21
            await self._update_progress("Computing embeddings", 3, 5,
                                       "Checking cache...")

            # First pass: check cache and collect uncached triplets
            distributions_list = [None] * len(triplets)  # Placeholder list
            uncached_triplets = []
            uncached_indices = []

            for i, triplet in enumerate(triplets):
                # Check cache first
                cached_dist = None
                if not force_regenerate:
                    cached_dist = self.cache.get_distributions(
                        triplet['question'],
                        triplet['context'],
                        triplet['answer'],
                        embedding_model,
                        clustering_method
                    )

                if cached_dist is not None:
                    # Use cached distributions
                    distributions_list[i] = {
                        'prompt_id': triplet['prompt_id'],
                        'p_q': cached_dist['distributions']['p_q'],
                        'p_c': cached_dist['distributions']['p_c'],
                        'p_a': cached_dist['distributions']['p_a']
                    }
                    cache_stats['distributions_cached'] += 1
                else:
                    # Collect for batch processing
                    uncached_triplets.append({
                        'question': triplet['question'],
                        'context': triplet['context'],
                        'answer': triplet['answer'],
                        'prompt_id': triplet.get('prompt_id', 'unknown')
                    })
                    uncached_indices.append(i)

            await self._update_progress("Computing embeddings", 3, 5,
                                       f"Found {cache_stats['distributions_cached']} cached, {len(uncached_triplets)} to compute")

            # Process uncached triplets using ProcessPoolExecutor
            # This avoids PyTorch threading conflicts by using spawn-based multiprocessing
            if uncached_triplets:
                from concurrent.futures import ProcessPoolExecutor

                # Create executor if not already initialized
                if self.embedding_executor is None:
                    await self._update_progress("Loading model", 3, 5,
                                               f"Initializing worker process with {embedding_model} model...")

                    # Create executor with spawn method (set in app.py)
                    # Single worker is sufficient and avoids resource contention
                    self.embedding_executor = ProcessPoolExecutor(
                        max_workers=1,
                        initializer=_init_embedding_worker,
                        initargs=(embedding_model,)
                    )

                await self._update_progress("Computing embeddings", 3, 5,
                                           f"Model loaded in worker. Processing {len(uncached_triplets)} triplets...")

                # Get event loop for executor calls
                loop = asyncio.get_event_loop()

                # Process each uncached triplet in isolated worker process
                computed_distributions = []
                for i, triplet_data in enumerate(uncached_triplets):
                    await self._update_progress("Computing embeddings", 3, 5,
                                               f"Processing triplet {i+1}/{len(uncached_triplets)}...")

                    # Run computation in worker process (isolated from asyncio event loop)
                    dist_result = await loop.run_in_executor(
                        self.embedding_executor,
                        _compute_embeddings_in_worker,
                        triplet_data,
                        clustering_method
                    )

                    computed_distributions.append(dist_result)

                # Insert computed distributions back into the list and save to cache
                for idx, dist_data in zip(uncached_indices, computed_distributions):
                    distributions_list[idx] = dist_data
                    triplet = triplets[idx]

                    # Save to cache
                    self.cache.save_distributions(
                        triplet['question'],
                        triplet['context'],
                        triplet['answer'],
                        embedding_model,
                        clustering_method,
                        {
                            'p_q': dist_data['p_q'],
                            'p_c': dist_data['p_c'],
                            'p_a': dist_data['p_a'],
                            'n_topics': dist_data['n_topics']
                        }
                    )
                cache_stats['distributions_computed'] = len(uncached_triplets)

            # Save distributions
            distributions_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'run_id': run_id,
                    'embedding_model': embedding_model,
                    'clustering_method': clustering_method
                },
                'triplets': distributions_list
            }

            distributions_file = self.output_dir / f'distributions_{run_id}.json'
            with open(distributions_file, 'w') as f:
                json.dump(distributions_data, f, indent=2)

            results['distributions_file'] = str(distributions_file)
            results['distributions'] = distributions_list

            await self._update_progress("Computing embeddings", 4, 5,
                                       f"Complete: {cache_stats['distributions_cached']} cached, {cache_stats['distributions_computed']} computed")

            # Step 5: Compute F_S scores (with caching)
            await self._update_progress("Computing F_S", 4, 5,
                                       "Checking cache...")

            import numpy as np
            import importlib.util
            # Path is already imported at module level

            # Import compute_semantic_faithfulness
            csf_path = Path(__file__).parent.parent / "sdm_package" / "compute_semantic_faithfulness.py"
            spec = importlib.util.spec_from_file_location("compute_semantic_faithfulness_module", str(csf_path))
            csf_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(csf_module)
            compute_semantic_faithfulness = csf_module.compute_semantic_faithfulness

            fs_scores = {}
            for i, dist in enumerate(distributions_list):
                p_q = dist['p_q']
                p_c = dist['p_c']
                p_a = dist['p_a']

                cached_fs = None
                if not force_regenerate:
                    cached_fs = self.cache.get_fs_score(p_q, p_c, p_a)

                if cached_fs is not None:
                    fs_scores[dist['prompt_id']] = cached_fs['fs_score']
                    cache_stats['fs_scores_cached'] += 1
                    await self._update_progress("Computing F_S", 4, 5,
                                               f"Computing F_S for triplet {i+1}/{len(distributions_list)} (cached)")
                else:
                    await self._update_progress("Computing F_S", 4, 5,
                                               f"Computing F_S for triplet {i+1}/{len(distributions_list)} (computing)")

                    p_q_array = np.array(p_q)
                    p_c_array = np.array(p_c)
                    p_a_array = np.array(p_a)

                    result = await asyncio.to_thread(
                        compute_semantic_faithfulness,
                        p_c=p_c_array,
                        p_q=p_q_array,
                        p_a=p_a_array,
                        tol_outer=1e-7,
                        max_outer_iter=100,
                        debug=False
                    )

                    fs_result = {
                        'F_S': float(result['F_S']),
                        'D_min': float(result['D_min']),
                        'iterations': int(result['iterations'])
                    }
                    fs_scores[dist['prompt_id']] = fs_result

                    # Save to cache
                    self.cache.save_fs_score(p_q, p_c, p_a, fs_result)
                    cache_stats['fs_scores_computed'] += 1

            results['fs_scores'] = fs_scores

            await self._update_progress("Computing F_S", 5, 5,
                                       f"Complete: {cache_stats['fs_scores_cached']} cached, {cache_stats['fs_scores_computed']} computed")

            # Add cache stats to results
            results['cache_stats'] = cache_stats

            # Save complete results
            results_file = self.output_dir / f'results_{run_id}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            results['results_file'] = str(results_file)
            results['status'] = 'success'

        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()
            raise

        return results
