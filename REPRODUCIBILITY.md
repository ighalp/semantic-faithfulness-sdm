# Reproducibility Guide

This repository includes **pre-computed cached data** to enable immediate reproducibility of all results without requiring expensive recomputation of embeddings and clustering.

## What's Included

### 1. Example Data
- **`data/examples/nvidia_demo.json`**
  - Single QCA triplet demonstrating the pipeline
  - Based on NVIDIA business risks
  - Ideal for quick testing and tutorials

### 2. Pre-computed Cache (5.5 MB total)
All cached files are **committed to the repository** for full reproducibility:

#### Embeddings Cache (`data/cache/embeddings/`)
- **`embeddings_v2.npz`** (5.5 MB)
  - Sentence embeddings for 10 QCA triplets
  - Model: Qwen/Qwen3-Embedding-0.6B
  - Embedding dimension: 768

- **`cluster_labels_v2.npz`** (5.8 KB)
  - UDIB cluster assignments for all sentences
  - 23 semantic topics discovered

#### Distributions Cache (`data/cache/distributions/`)
- **`distributions_v2.json`** (17 KB)
  - Marginal probability distributions: p(Q), p(C), p(A)
  - For all 10 triplets across 23 topics

- **`metadata_v2.json`** (438 bytes)
  - Dataset metadata and configuration

## Quick Start

### Option 1: Use Pre-computed Cache (Instant)
```bash
# Clone repository (includes cached data)
git clone https://github.com/ighalp/semantic-faithfulness-sdm.git
cd semantic-faithfulness-sdm

# Install dependencies
pip install -e .

# Run notebook - uses cached data immediately
jupyter notebook Semantic_Faithfulness_SDM_demo.ipynb
```

**Result:** Notebook runs in seconds, using pre-computed embeddings and clustering.

### Option 2: Regenerate Cache from Scratch
```bash
# Delete existing cache
rm -rf data/cache/embeddings/* data/cache/distributions/*

# Regenerate using script
python generate_demo_cache.py

# Or just run the notebook - it will regenerate automatically
jupyter notebook Semantic_Faithfulness_SDM_demo.ipynb
```

**Note:** First-time generation takes ~2-5 minutes depending on hardware.

## Benefits of Committed Cache

✅ **Instant Results** - Users can run notebooks immediately without waiting for embeddings

✅ **Exact Reproducibility** - Everyone gets identical results matching the paper

✅ **Educational** - Newcomers can see results instantly, understand the method, then regenerate if desired

✅ **Comparison Baseline** - Provides reference outputs for validating custom implementations

✅ **Bandwidth Efficient** - 5.5 MB is small enough to include in repository

## File Size Breakdown

```
data/
├── examples/
│   └── nvidia_demo.json           3 KB
├── cache/
│   ├── embeddings/
│   │   ├── embeddings_v2.npz      5.5 MB  ← Pre-computed sentence embeddings
│   │   └── cluster_labels_v2.npz  5.8 KB  ← UDIB cluster assignments
│   └── distributions/
│       ├── distributions_v2.json  17 KB   ← Probability distributions
│       └── metadata_v2.json       438 B   ← Dataset metadata
└── results/                       (gitignored - generated locally)
```

**Total committed cache:** 5.5 MB

## Cache Key Format

The notebook uses content-based hashing for cache keys:
- Embedding cache: `embeddings_{md5_hash[:12]}.pkl`
- Clustering cache: `clustering_{md5_hash[:12]}.pkl`

The committed `*_v2.npz` files are from the paper's experimental dataset and are loaded by filename when available.

## Regeneration Script

The `generate_demo_cache.py` script:
1. Loads QCA triplets from `data/examples/`
2. Generates sentence embeddings
3. Runs UDIB clustering (τ values, kink detection)
4. Saves cache to `data/cache/`

## When to Regenerate

Regenerate cache when:
- ✏️ Modifying QCA triplets in `data/examples/`
- 🔧 Changing embedding models
- ⚙️ Adjusting UDIB parameters (τ range, max clusters)
- 🧪 Testing custom clustering algorithms

Otherwise, use the pre-computed cache for consistent, fast results.

## Citation

If you use this pre-computed data, please cite:

```bibtex
@article{halperin2025faithfulness,
  title={Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations},
  author={Halperin, Igor},
  year={2025}
}
```

## Questions?

See the main [README.md](README.md) for full documentation.
