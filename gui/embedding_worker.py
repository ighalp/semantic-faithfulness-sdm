"""
Embedding Worker - Runs PyTorch/transformers in isolated subprocess
This avoids PyTorch mutex blocking issues in async contexts

This module uses the shared text_utils module from sdm_package to avoid code duplication.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_distributions_subprocess(triplet_data: dict, embedding_model: str, clustering_method: str) -> dict:
    """
    Compute distributions for a single triplet in isolated process.
    Uses shared text_utils.compute_triplet_distributions function.

    Args:
        triplet_data: Dict with 'question', 'context', 'answer', 'prompt_id'
        embedding_model: Name of the sentence transformer model
        clustering_method: Clustering method ('dib' or 'kmeans')

    Returns:
        Dict with 'p_q', 'p_c', 'p_a', 'n_topics', 'prompt_id'
    """
    # Import heavy modules ONLY in subprocess
    from sentence_transformers import SentenceTransformer
    from sdm_package.text_utils import compute_triplet_distributions

    # Load embedding model
    model = SentenceTransformer(embedding_model)

    # Use shared utility function to compute distributions
    result = compute_triplet_distributions(
        question=triplet_data['question'],
        context=triplet_data['context'],
        answer=triplet_data['answer'],
        embedding_model=model,  # Pass loaded model instance
        clustering_method=clustering_method
    )

    # Add prompt_id to result
    result['prompt_id'] = triplet_data.get('prompt_id', 'unknown')

    return result


def compute_distributions_batch(triplets_data: list, embedding_model: str, clustering_method: str) -> list:
    """
    Compute distributions for multiple triplets in a single subprocess.
    Loads the embedding model ONCE and reuses it for all triplets (optimization).
    Uses shared text_utils.compute_triplet_distributions function.

    Args:
        triplets_data: List of dicts with 'question', 'context', 'answer', 'prompt_id'
        embedding_model: Name of the sentence transformer model
        clustering_method: Clustering method ('udib' or 'kmeans')

    Returns:
        List of dicts with 'p_q', 'p_c', 'p_a', 'n_topics', 'prompt_id'
    """
    # Import heavy modules ONLY in subprocess
    from sentence_transformers import SentenceTransformer
    from sdm_package.text_utils import compute_triplet_distributions

    # Load model ONCE for all triplets (optimization for batch processing)
    model = SentenceTransformer(embedding_model)

    results = []
    for triplet_data in triplets_data:
        # Use shared utility function to compute distributions
        result = compute_triplet_distributions(
            question=triplet_data['question'],
            context=triplet_data['context'],
            answer=triplet_data['answer'],
            embedding_model=model,  # Pass loaded model instance (reused)
            clustering_method=clustering_method
        )

        # Add prompt_id to result
        result['prompt_id'] = triplet_data.get('prompt_id', 'unknown')
        results.append(result)

    return results


if __name__ == '__main__':
    # Read input from stdin
    input_data = json.load(sys.stdin)

    try:
        # Check if we're processing a batch or single triplet
        if 'triplets' in input_data:
            # Batch processing (loads model once)
            results = compute_distributions_batch(
                triplets_data=input_data['triplets'],
                embedding_model=input_data['embedding_model'],
                clustering_method=input_data['clustering_method']
            )
            json.dump({'success': True, 'results': results}, sys.stdout)
            sys.stdout.flush()  # Ensure output is sent immediately
        else:
            # Single triplet (backward compatibility)
            result = compute_distributions_subprocess(
                triplet_data=input_data['triplet'],
                embedding_model=input_data['embedding_model'],
                clustering_method=input_data['clustering_method']
            )
            json.dump({'success': True, 'result': result}, sys.stdout)
            sys.stdout.flush()  # Ensure output is sent immediately

        sys.exit(0)

    except Exception as e:
        # Write error to stdout with explicit flush
        json.dump({'success': False, 'error': str(e), 'type': type(e).__name__}, sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
