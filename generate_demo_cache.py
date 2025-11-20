#!/usr/bin/env python3
"""
Generate cached data for the demo notebook for full reproducibility.

This script generates and saves:
1. Sentence embeddings (from nvidia_demo.json)
2. UDIB clustering results
3. Marginal probability distributions

These cached files enable users to run the notebook immediately without
waiting for embedding/clustering computation.
"""

import json
import pickle
import hashlib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk

from sdm_package.DIB_with_KL_upper_bound import DIBAnalyzer

# Ensure punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download('punkt')

# Set up paths
DATA_DIR = Path("data")
EXAMPLES_DIR = DATA_DIR / "examples"
EMBEDDINGS_CACHE_DIR = DATA_DIR / "cache" / "embeddings"
DISTRIBUTIONS_CACHE_DIR = DATA_DIR / "cache" / "distributions"

# Create directories
EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DISTRIBUTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("GENERATING DEMO CACHE DATA")
print("="*80)

# Load QCA data
example_file = EXAMPLES_DIR / "nvidia_demo.json"
print(f"\n1. Loading QCA triplet from: {example_file.name}")

with open(example_file, 'r') as f:
    data = json.load(f)

triplet = data['triplets'][0]
question = triplet['question']
context = triplet['context']
answer = triplet['answer']

print(f"   ✓ Loaded triplet ID: {triplet['id']}")

# Tokenize sentences
print("\n2. Tokenizing sentences...")
question_sentences = sent_tokenize(question.strip())
context_sentences = sent_tokenize(context.strip())
answer_sentences = sent_tokenize(answer.strip())
all_sentences = question_sentences + context_sentences + answer_sentences

print(f"   Question: {len(question_sentences)} sentences")
print(f"   Context: {len(context_sentences)} sentences")
print(f"   Answer: {len(answer_sentences)} sentences")
print(f"   Total: {len(all_sentences)} sentences")

# Generate cache key
def get_cache_key(data, prefix=""):
    """Generate a cache key from data using hash."""
    if isinstance(data, str):
        content = data
    elif isinstance(data, list):
        content = "|".join(str(item) for item in data)
    else:
        content = str(data)

    hash_obj = hashlib.md5(content.encode())
    return f"{prefix}_{hash_obj.hexdigest()[:12]}"

embedding_cache_key = get_cache_key(all_sentences, "embeddings")
print(f"\n   Embedding cache key: {embedding_cache_key}")

# Generate embeddings
print("\n3. Generating sentence embeddings...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"   Model: all-MiniLM-L6-v2")
print(f"   Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")

embeddings = embedding_model.encode(all_sentences, show_progress_bar=True)
print(f"   ✓ Generated embeddings: shape {embeddings.shape}")

# Save embeddings
embeddings_path = EMBEDDINGS_CACHE_DIR / f"{embedding_cache_key}.pkl"
with open(embeddings_path, 'wb') as f:
    pickle.dump(embeddings, f)
print(f"   ✓ Saved to: {embeddings_path.relative_to(Path.cwd())}")

# Generate clustering
print("\n4. Running UDIB clustering...")
tau_values = np.logspace(-2, 2, 30)
max_n_clusters = 15
seed = 42

clustering_params = f"{embedding_cache_key}_tau{len(tau_values)}_maxc{max_n_clusters}_seed{seed}"
clustering_cache_key = get_cache_key(clustering_params, "clustering")
print(f"   Clustering cache key: {clustering_cache_key}")
print(f"   Tau range: [{tau_values.min():.4f}, {tau_values.max():.4f}]")
print(f"   Max clusters: {max_n_clusters}")

dib_analyzer = DIBAnalyzer(embeddings, all_sentences)
dib_analyzer.run(tau_values, max_n_clusters=max_n_clusters, seed=seed)

recommendation, _ = dib_analyzer.get_recommendation(min_clusters=3, metric='kink_angle')

print(f"   ✓ UDIB clustering complete")
print(f"     Recommended clusters: {recommendation['n_clusters']}")
print(f"     Kink angle: {recommendation['kink_angle']:.2f}°")
print(f"     Cluster entropy H(c): {recommendation['H(c)']:.3f} bits")

# Save clustering results
cached_clustering = {
    'recommendation': recommendation,
    'dib_analyzer': dib_analyzer
}

clustering_path = DISTRIBUTIONS_CACHE_DIR / f"{clustering_cache_key}.pkl"
with open(clustering_path, 'wb') as f:
    pickle.dump(cached_clustering, f)
print(f"   ✓ Saved to: {clustering_path.relative_to(Path.cwd())}")

# Compute and display marginal probabilities
print("\n5. Computing marginal probability distributions...")
assignments = recommendation['assignments']
n_topics = recommendation['n_clusters']

n_q = len(question_sentences)
n_c = len(context_sentences)
n_a = len(answer_sentences)

assignments_q = assignments[:n_q]
assignments_c = assignments[n_q:n_q+n_c]
assignments_a = assignments[n_q+n_c:]

def compute_distribution(assignments, n_topics):
    counts = np.bincount(assignments, minlength=n_topics)
    return counts / counts.sum()

p_question = compute_distribution(assignments_q, n_topics)
p_context = compute_distribution(assignments_c, n_topics)
p_answer = compute_distribution(assignments_a, n_topics)

print(f"   ✓ Computed distributions over {n_topics} topics")
print(f"     p_question.sum() = {p_question.sum():.6f}")
print(f"     p_context.sum() = {p_context.sum():.6f}")
print(f"     p_answer.sum() = {p_answer.sum():.6f}")

# Summary
print("\n" + "="*80)
print("CACHE GENERATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print(f"  1. {embeddings_path.relative_to(Path.cwd())}")
print(f"     - Sentence embeddings ({embeddings.shape[0]} sentences × {embeddings.shape[1]} dims)")
print(f"     - Size: {embeddings_path.stat().st_size / 1024:.1f} KB")
print(f"\n  2. {clustering_path.relative_to(Path.cwd())}")
print(f"     - UDIB clustering results ({n_topics} topics)")
print(f"     - Size: {clustering_path.stat().st_size / 1024:.1f} KB")

total_size = embeddings_path.stat().st_size + clustering_path.stat().st_size
print(f"\nTotal cache size: {total_size / 1024:.1f} KB")

print("\n✓ These files are now committed to the repository for reproducibility.")
print("  Users can run the notebook immediately without regenerating embeddings/clustering.")
print("\n" + "="*80)
