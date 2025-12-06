# Semantic Faithfulness SDM Repository Navigation

## Project Overview
This is the **Semantic Divergence Metrics (SDM)** package implementing information-theoretic faithfulness metrics for LLMs. It includes a web-based GUI application called "Paraphrase Me".

## Python Environment
- **Python version**: 3.13.0
- **Virtual environment**: `.venv/` (activate with `source .venv/bin/activate`)
- **Core requirements**: `requirements.txt`
- **GUI requirements**: `gui/requirements-gui.txt`

## Key Directories

### Core Package: `sdm_package/`
| File | Purpose |
|------|---------|
| `SDM.py` | Main SemanticFaithfulnessAnalyzer class - handles embeddings, clustering, distributions |
| `compute_semantic_faithfulness.py` | Computes F_S (Semantic Faithfulness) and SEP (Semantic Entropy Production) metrics |
| `DIB_with_KL_upper_bound.py` | UDIB (Upper-bounded Deterministic Information Bottleneck) clustering algorithm |
| `text_utils.py` | Text preprocessing utilities |
| `create_SF_SEP_plots.py` | Script to generate visualization plots |

### GUI Application: `gui/`
| File | Purpose |
|------|---------|
| `app.py` | Main NiceGUI application entry point (run with `python app.py`) |
| `llm_client.py` | LLM API client for OpenAI, Anthropic, Google Gemini |
| `pipeline.py` | Orchestrates the full analysis pipeline |
| `cache_manager.py` | Manages caching of expensive computations |
| `embedding_worker.py` | Handles embedding generation in background |

### GUI Pages: `gui/pages/`
| File | Purpose |
|------|---------|
| `home.py` | Landing page with overview |
| `input_page.py` | Input form for Question-Context-Answer triplets |
| `analyze.py` | Analysis execution page |
| `results.py` | Results visualization with Plotly charts |
| `compare.py` | Side-by-side answer comparison with diff highlighting |
| `judge.py` | LLM-as-a-Judge evaluation and verdict export |
| `markdown_utils.py` | Markdown to HTML conversion, highlighting utilities |

## Cache Structure: `data/cache/`

All cache files use SHA256 hashes (first 16 chars) for keys.

### `data/cache/paraphrases/`
- **Pattern**: `{question_hash}_{model}_{num_paraphrases}.json`
- **Example**: `2a9165440fa430e9_claude-sonnet-4-5-20250929_3.json`
- **Contents**: `{question, model, num_paraphrases, paraphrases[]}`

### `data/cache/answers/`
- **Pattern**: `{question_hash}_{model}.json`
- **Example**: `00a12dcaf71a9fb8_gemini-2.5-pro.json`
- **Contents**: `{question, context, model, answer}`

### `data/cache/distributions/`
- **Pattern**: `{triplet_hash}_{embedding_model_short}/`
- **Example**: `042890b446ecbd07_Qwen/Qwen3-Embedding-0.6B_udib.json`
- **Contents**: Full SDM analysis results including:
  - `p_question`, `p_context`, `p_answer` - marginal probability distributions
  - `embeddings` - sentence embeddings for Q, C, A
  - `cluster_assignments` - UDIB cluster assignments for each sentence
  - `n_clusters` - number of discovered semantic topics
  - Metadata about the analysis parameters

### `data/cache/fs_scores/`
- **Pattern**: `{triplet_hash}.json`
- **Contents**: Pre-computed F_S and SEP metrics for triplets

### `data/cache/embeddings/`
- Legacy embedding cache (less commonly used)

## Data Files: `data/`
- `data/examples/` - Example input files (e.g., NVIDIA 10-K PDF)
- `data/qca_triplets.json` - Sample Question-Context-Answer triplets

## Documentation: `docs/`
- `docs/methodology.md` - Theoretical foundations
- `docs/architecture.md` - System architecture
- `docs/examples/` - Documented examples including hallucination detection case

## Key Metrics

### Semantic Faithfulness (F_S)
- Range: 0 to 1 (higher = more faithful)
- Measures alignment between answer channel and optimal goal channel
- Computed via Csiszár-Tusnády/Blahut-Arimoto alternating minimization

### Semantic Entropy Production (SEP)
- System entropy change: Ṡ = H(A) - H(C) (semantic expansion/compression)
- Dissipated heat: Ṡ_m (see paper for formula)

## Running the Application

```bash
# Activate environment
source .venv/bin/activate

# Run GUI (opens at http://localhost:8080)
cd gui && python app.py

# Run Jupyter demo
jupyter notebook Semantic_Faithfulness_SDM_demo.ipynb
```

## LLM Provider Configuration

Environment variables for API keys:
- `OPENAI_API_KEY` - OpenAI
- `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` - Anthropic
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` - Google Gemini

## Common Tasks

### Clear all caches
```bash
rm -rf data/cache/paraphrases/* data/cache/answers/* data/cache/distributions/* data/cache/fs_scores/*
```

### Generate paper figures
```bash
python sdm_package/create_SF_SEP_plots.py --output ./plots
```

### Run the demo notebook
```bash
jupyter notebook Semantic_Faithfulness_SDM_demo.ipynb
```

## Static Assets
- `gui/LLM_Maxwell_demon_logo.png` - App logo (served at `/static/`)
- `gui/static/` - Other static files

## Research Paper Location
The LaTeX source for the Semantic Faithfulness paper:
- **Path**: `~/Library/Mobile Documents/com~apple~CloudDocs/Documents/Docs/My Papers/LLM_Faithfulness_metrics/Semantic_Faithfulness_for_LLM.tex`
- **Short**: iCloud Docs → Documents → Docs → My Papers → LLM_Faithfulness_metrics
