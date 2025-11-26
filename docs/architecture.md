# System Architecture

## Overview

The Semantic Faithfulness SDM system consists of two main components:
1. **Core SDM Package** (`sdm_package/`) - Python library for computing semantic metrics
2. **GUI Application** (`gui/`) - Web-based interface for interactive analysis

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SEMANTIC FAITHFULNESS SDM                              │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         USER INTERFACES                                     │ │
│  │                                                                             │ │
│  │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │ │
│  │   │  Jupyter        │    │  Python API     │    │  Web GUI        │       │ │
│  │   │  Notebook       │    │  (Library)      │    │  (NiceGUI)      │       │ │
│  │   │                 │    │                 │    │                 │       │ │
│  │   │  Interactive    │    │  from sdm_pkg   │    │  localhost:8080 │       │ │
│  │   │  exploration    │    │  import SDM     │    │  Browser-based  │       │ │
│  │   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘       │ │
│  │            │                      │                      │                 │ │
│  └────────────┼──────────────────────┼──────────────────────┼─────────────────┘ │
│               │                      │                      │                   │
│               ▼                      ▼                      ▼                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                        SDM CORE PACKAGE                                     │ │
│  │                                                                             │ │
│  │   ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │   │                  SemanticFaithfulnessAnalyzer                        │  │ │
│  │   │                                                                      │  │ │
│  │   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │ │
│  │   │  │   Tokenizer  │  │   Embedder   │  │   Clusterer  │               │  │ │
│  │   │  │   (NLTK)     │  │  (Sentence-  │  │   (UDIB)     │               │  │ │
│  │   │  │              │  │  Transformers│  │              │               │  │ │
│  │   │  │  text → sents│  │  sents → vecs│  │  vecs → topics               │  │ │
│  │   │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │  │ │
│  │   │         │                 │                 │                        │  │ │
│  │   │         ▼                 ▼                 ▼                        │  │ │
│  │   │  ┌──────────────────────────────────────────────────────────────┐   │  │ │
│  │   │  │              Probability Distributions                        │   │  │ │
│  │   │  │                                                               │   │  │ │
│  │   │  │   p(Q) = [p₁, p₂, ..., pₙ]  Question distribution            │   │  │ │
│  │   │  │   p(C) = [p₁, p₂, ..., pₙ]  Context distribution             │   │  │ │
│  │   │  │   p(A) = [p₁, p₂, ..., pₙ]  Answer distribution              │   │  │ │
│  │   │  │                                                               │   │  │ │
│  │   │  └────────────────────────────┬─────────────────────────────────┘   │  │ │
│  │   │                               │                                      │  │ │
│  │   └───────────────────────────────┼──────────────────────────────────────┘  │ │
│  │                                   │                                         │ │
│  │                                   ▼                                         │ │
│  │   ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │   │              compute_semantic_faithfulness()                         │  │ │
│  │   │                                                                      │  │ │
│  │   │   ┌────────────────────────────────────────────────────────────┐    │  │ │
│  │   │   │           Csiszár-Tusnády Optimization                      │    │  │ │
│  │   │   │                                                             │    │  │ │
│  │   │   │   Compute optimal transition matrices Q* and A*             │    │  │ │
│  │   │   │   via alternating minimization of KL divergence            │    │  │ │
│  │   │   │                                                             │    │  │ │
│  │   │   │   Q* = argmin_Q  KL(A || Q)                                 │    │  │ │
│  │   │   │       s.t.  p(Q) = p(C)ᵀ · Q                                │    │  │ │
│  │   │   │                                                             │    │  │ │
│  │   │   └─────────────────────────┬──────────────────────────────────┘    │  │ │
│  │   │                             │                                        │  │ │
│  │   │                             ▼                                        │  │ │
│  │   │   ┌────────────────────────────────────────────────────────────┐    │  │ │
│  │   │   │                    OUTPUT METRICS                           │    │  │ │
│  │   │   │                                                             │    │  │ │
│  │   │   │   F_S = 1 / (1 + D_min)         Semantic Faithfulness       │    │  │ │
│  │   │   │   SEP_system = H(A) - H(C)      System Entropy Production   │    │  │ │
│  │   │   │   SEP_total ≈ 1/F_S - 1         Total Entropy Production    │    │  │ │
│  │   │   │                                                             │    │  │ │
│  │   │   └────────────────────────────────────────────────────────────┘    │  │ │
│  │   │                                                                      │  │ │
│  │   └──────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## GUI Application Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PARAPHRASE ME - GUI WORKFLOW                             │
│                                                                                  │
│   ┌─────────┐                                                                    │
│   │  START  │                                                                    │
│   └────┬────┘                                                                    │
│        │                                                                         │
│        ▼                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                           HOME PAGE                                      │   │
│   │                                                                          │   │
│   │   "Welcome to Paraphrase Me - LLM Semantic Assistant"                   │   │
│   │                                                                          │   │
│   │   Navigation: [Home] [Input] [Analyze] [Results] [Compare] [LLM Judge]  │   │
│   │                                                                          │   │
│   └─────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
│                                 ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                          INPUT PAGE                                      │   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │  Question-Context-Answer Input                                   │   │   │
│   │   │                                                                  │   │   │
│   │   │  [Question]: Enter your question here...                         │   │   │
│   │   │  [Context]:  Enter the source context...                         │   │   │
│   │   │  [Answer]:   Enter the LLM answer (optional)...                  │   │   │
│   │   │                                                                  │   │   │
│   │   │  [Load Example]  [Import JSON]  [Clear]                          │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │  LLM Pipeline Configuration                                      │   │   │
│   │   │                                                                  │   │   │
│   │   │  Provider: [OpenAI ▼]  Model: [gpt-4o ▼]                         │   │   │
│   │   │  Paraphrases: [5]  Temperature: [0.7]                            │   │   │
│   │   │  [x] Force regeneration                                          │   │   │
│   │   │                                                                  │   │   │
│   │   │  [Run LLM Pipeline]                                              │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │                            OR                                            │   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │  Quick Analysis (without LLM)                                    │   │   │
│   │   │                                                                  │   │   │
│   │   │  [Analyze Single Triplet]                                        │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   └─────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
│         ┌───────────────────────┴───────────────────────┐                       │
│         │                                               │                       │
│         ▼                                               ▼                       │
│   ┌─────────────────────┐                   ┌─────────────────────────────┐     │
│   │  LLM PIPELINE       │                   │  DIRECT ANALYSIS            │     │
│   │                     │                   │                             │     │
│   │  1. Generate        │                   │  Skip to Analyze Page       │     │
│   │     paraphrases     │                   │  with single triplet        │     │
│   │                     │                   │                             │     │
│   │  2. Generate        │                   └──────────────┬──────────────┘     │
│   │     answers         │                                  │                    │
│   │                     │                                  │                    │
│   │  3. Compute F_S     │                                  │                    │
│   │     for each        │                                  │                    │
│   │                     │                                  │                    │
│   └──────────┬──────────┘                                  │                    │
│              │                                             │                    │
│              └──────────────────┬──────────────────────────┘                    │
│                                 │                                                │
│                                 ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         ANALYZE PAGE                                     │   │
│   │                                                                          │   │
│   │   Progress: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 35%          │   │
│   │                                                                          │   │
│   │   [✓] Tokenization    - Split into sentences                            │   │
│   │   [●] Embedding       - Converting to vectors...                         │   │
│   │   [ ] Clustering      - Group into topics                                │   │
│   │   [ ] Optimization    - Compute F_S                                      │   │
│   │                                                                          │   │
│   │   Log Output:                                                            │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │ [12:34:56] Loading Qwen3-Embedding model...                      │  │   │
│   │   │ [12:34:58] Tokenized 45 sentences from context                   │  │   │
│   │   │ [12:34:59] Embedding sentences...                                │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                          │   │
│   └─────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
│                                 ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         RESULTS PAGE                                     │   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │  F_S Scores by Paraphrase                                        │   │   │
│   │   │                                                                  │   │   │
│   │   │  prompt_0: ████████████████████████░░  0.8234  ← Initial        │   │   │
│   │   │  prompt_1: ██████████████████████████  0.8891  ← Highest        │   │   │
│   │   │  prompt_2: ███████████████████████░░░  0.8567                   │   │   │
│   │   │  prompt_3: █████████████████░░░░░░░░░  0.7123  ← Lowest         │   │   │
│   │   │  prompt_4: ██████████████████████░░░░  0.8012                   │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │   [Distributions Tab]  [Matrices Tab]  [Statistics Tab]                 │   │
│   │                                                                          │   │
│   │   [Export JSON]  [Go to Compare]                                         │   │
│   │                                                                          │   │
│   └─────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
│                                 ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         COMPARE PAGE                                     │   │
│   │                                                                          │   │
│   │   Quick Select: [Initial vs Highest] [Initial vs Lowest] [High vs Low]  │   │
│   │                                                                          │   │
│   │   ┌───────────────────────────┐   ┌───────────────────────────┐         │   │
│   │   │  Answer A (prompt_0)      │   │  Answer B (prompt_3)      │         │   │
│   │   │  F_S: 0.8234              │   │  F_S: 0.7123              │         │   │
│   │   │                           │   │                           │         │   │
│   │   │  The company reported     │   │  The company reported     │         │   │
│   │   │  strong Q3 results...     │   │  quarterly results that   │         │   │
│   │   │                           │   │  [highlighted diff]       │         │   │
│   │   │  Revenue increased by     │   │  showed mixed signals...  │         │   │
│   │   │  15% year-over-year...    │   │                           │         │   │
│   │   └───────────────────────────┘   └───────────────────────────┘         │   │
│   │                                                                          │   │
│   │   [Analyze with LLM Judge →]                                             │   │
│   │                                                                          │   │
│   └─────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
│                                 ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        LLM JUDGE PAGE                                    │   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │  🏆 Winner: Answer A (prompt_0)                                  │   │   │
│   │   │                                                                  │   │   │
│   │   │  ┌─────────────────┐    ┌─────────────────┐                     │   │   │
│   │   │  │  Answer A        │    │  Answer B        │                     │   │   │
│   │   │  │  LLM Score: 8/10 │    │  LLM Score: 6/10 │                     │   │   │
│   │   │  │  F_S: 0.8234     │    │  F_S: 0.7123     │                     │   │   │
│   │   │  └─────────────────┘    └─────────────────┘                     │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │   Criteria Breakdown:                                                    │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │ Criterion      │  A  │  B  │ Diff │ Better │                    │   │   │
│   │   │ Faithfulness   │  8  │  6  │  +2  │   A    │                    │   │   │
│   │   │ Completeness   │  7  │  7  │   0  │   =    │                    │   │   │
│   │   │ Coherence      │  8  │  6  │  +2  │   A    │                    │   │   │
│   │   │ Relevance      │  9  │  7  │  +2  │   A    │                    │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   │   [Export as Markdown]  [Export as PDF]                                  │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                                  │
│                                                                                 │
│   USER INPUT                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │  Question: "What are the main financial highlights?"                     │  │
│   │  Context:  "Apple Inc. reported Q3 2023 revenue of $81.8B..."           │  │
│   │  Answer:   (optional - can be generated by LLM)                          │  │
│   └───────────────────────────────────┬─────────────────────────────────────┘  │
│                                       │                                         │
│                                       ▼                                         │
│   LLM PIPELINE (Optional)                                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                          │  │
│   │  Question + Context ──────────────────────────────────────┐              │  │
│   │                                                           │              │  │
│   │                                                           ▼              │  │
│   │                                                  ┌─────────────────┐     │  │
│   │                                                  │  LLM API        │     │  │
│   │                                                  │  (OpenAI/       │     │  │
│   │                                                  │   Anthropic/    │     │  │
│   │                                                  │   Gemini)       │     │  │
│   │                                                  └────────┬────────┘     │  │
│   │                                                           │              │  │
│   │                            ┌──────────────────────────────┼────────────┐ │  │
│   │                            │                              │            │ │  │
│   │                            ▼                              ▼            │ │  │
│   │                   ┌───────────────────┐        ┌──────────────────┐   │ │  │
│   │                   │  Paraphrases      │        │  Answers         │   │ │  │
│   │                   │                   │        │                  │   │ │  │
│   │                   │  Q₀ (original)    │───────▶│  A₀              │   │ │  │
│   │                   │  Q₁ (paraphrase)  │───────▶│  A₁              │   │ │  │
│   │                   │  Q₂ (paraphrase)  │───────▶│  A₂              │   │ │  │
│   │                   │  ...              │        │  ...             │   │ │  │
│   │                   │  Qₙ (paraphrase)  │───────▶│  Aₙ              │   │ │  │
│   │                   │                   │        │                  │   │ │  │
│   │                   └───────────────────┘        └────────┬─────────┘   │ │  │
│   │                                                         │             │ │  │
│   └─────────────────────────────────────────────────────────┼─────────────┘ │  │
│                                                             │                │  │
│                          Cached in data/cache/              │                │  │
│                                                             │                │  │
│                                                             ▼                │  │
│   SDM ANALYSIS ENGINE                                                        │  │
│   ┌─────────────────────────────────────────────────────────────────────────┐│  │
│   │                                                                          ││  │
│   │  For each triplet (Qᵢ, C, Aᵢ):                                          ││  │
│   │                                                                          ││  │
│   │  STEP 1: TOKENIZATION                                                    ││  │
│   │  ┌──────────────────────────────────────────────────────────────────┐   ││  │
│   │  │  Text ──▶ NLTK sent_tokenize() ──▶ List of sentences             │   ││  │
│   │  │                                                                   │   ││  │
│   │  │  Q: ["What are the highlights?"]                                  │   ││  │
│   │  │  C: ["Apple reported...", "Revenue was...", "iPhone sales..."]   │   ││  │
│   │  │  A: ["The main highlights...", "First, revenue...", ...]         │   ││  │
│   │  └──────────────────────────────────────────────────────────────────┘   ││  │
│   │                              │                                           ││  │
│   │                              ▼                                           ││  │
│   │  STEP 2: EMBEDDING                                                       ││  │
│   │  ┌──────────────────────────────────────────────────────────────────┐   ││  │
│   │  │  Sentences ──▶ SentenceTransformer ──▶ Dense vectors (768-dim)   │   ││  │
│   │  │                                                                   │   ││  │
│   │  │  "Apple reported..." ──▶ [0.12, -0.45, 0.78, ...]                │   ││  │
│   │  │  "Revenue was..."    ──▶ [0.34, 0.22, -0.11, ...]                │   ││  │
│   │  └──────────────────────────────────────────────────────────────────┘   ││  │
│   │                              │                                           ││  │
│   │                              ▼                                           ││  │
│   │  STEP 3: CLUSTERING (UDIB)                                               ││  │
│   │  ┌──────────────────────────────────────────────────────────────────┐   ││  │
│   │  │  Vectors ──▶ UDIB Algorithm ──▶ Topic assignments                │   ││  │
│   │  │                                                                   │   ││  │
│   │  │  Automatic cluster selection via Information Bottleneck          │   ││  │
│   │  │  Minimizes: I(X;T) - β·I(T;Y)                                    │   ││  │
│   │  │                                                                   │   ││  │
│   │  │  Output: N topics (typically 3-8)                                │   ││  │
│   │  └──────────────────────────────────────────────────────────────────┘   ││  │
│   │                              │                                           ││  │
│   │                              ▼                                           ││  │
│   │  STEP 4: DISTRIBUTION COMPUTATION                                        ││  │
│   │  ┌──────────────────────────────────────────────────────────────────┐   ││  │
│   │  │  Topic assignments ──▶ Probability distributions                 │   ││  │
│   │  │                                                                   │   ││  │
│   │  │  p(Q) = [0.5, 0.3, 0.2, 0.0]     (4 topics)                      │   ││  │
│   │  │  p(C) = [0.2, 0.4, 0.2, 0.2]                                     │   ││  │
│   │  │  p(A) = [0.4, 0.3, 0.2, 0.1]                                     │   ││  │
│   │  └──────────────────────────────────────────────────────────────────┘   ││  │
│   │                              │                                           ││  │
│   │                              ▼                                           ││  │
│   │  STEP 5: OPTIMIZATION (Csiszár-Tusnády)                                  ││  │
│   │  ┌──────────────────────────────────────────────────────────────────┐   ││  │
│   │  │  Distributions ──▶ Alternating Minimization ──▶ Optimal matrices │   ││  │
│   │  │                                                                   │   ││  │
│   │  │  Iterate until convergence (tol=1e-7):                           │   ││  │
│   │  │    A-step: Update A given Q                                      │   ││  │
│   │  │    Q-step: Update Q given A                                      │   ││  │
│   │  │                                                                   │   ││  │
│   │  │  Output: Q* (optimal goal), A* (actual)                          │   ││  │
│   │  └──────────────────────────────────────────────────────────────────┘   ││  │
│   │                              │                                           ││  │
│   │                              ▼                                           ││  │
│   │  STEP 6: METRIC COMPUTATION                                              ││  │
│   │  ┌──────────────────────────────────────────────────────────────────┐   ││  │
│   │  │  D_min = KL(A* || Q*)                                            │   ││  │
│   │  │  F_S = 1 / (1 + D_min)                                           │   ││  │
│   │  │  SEP_system = H(A) - H(C)                                        │   ││  │
│   │  │  SEP_total ≈ D_min                                               │   ││  │
│   │  └──────────────────────────────────────────────────────────────────┘   ││  │
│   │                                                                          ││  │
│   └──────────────────────────────────────────────────────────────────────────┘│  │
│                                                                                │  │
│   OUTPUT                                                                       │  │
│   ┌────────────────────────────────────────────────────────────────────────┐  │  │
│   │                                                                         │  │  │
│   │  {                                                                      │  │  │
│   │    "prompt_0": { "F_S": 0.8234, "SEP_system": 0.12, ... },             │  │  │
│   │    "prompt_1": { "F_S": 0.8891, "SEP_system": 0.08, ... },             │  │  │
│   │    "prompt_2": { "F_S": 0.8567, "SEP_system": 0.10, ... },             │  │  │
│   │    ...                                                                  │  │  │
│   │  }                                                                      │  │  │
│   │                                                                         │  │  │
│   └────────────────────────────────────────────────────────────────────────┘  │  │
│                                                                                │  │
└────────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
semantic-faithfulness-sdm/
│
├── sdm_package/                        # Core SDM Library
│   ├── __init__.py                     # Package exports
│   ├── SDM.py                          # SemanticFaithfulnessAnalyzer class
│   ├── DIB_with_KL_upper_bound.py      # UDIB clustering algorithm
│   └── compute_semantic_faithfulness.py # F_S and SEP computation
│
├── gui/                                # Web Application
│   ├── app.py                          # Main entry point (NiceGUI)
│   ├── llm_client.py                   # LLM API wrapper
│   ├── requirements-gui.txt            # GUI dependencies
│   ├── README.md                       # GUI documentation
│   │
│   ├── pages/                          # Page modules
│   │   ├── __init__.py
│   │   ├── home.py                     # Landing page
│   │   ├── input_page.py               # Input form + LLM pipeline
│   │   ├── analyze.py                  # Analysis execution
│   │   ├── results.py                  # Results visualization
│   │   ├── compare.py                  # Answer comparison
│   │   ├── judge.py                    # LLM-as-a-Judge
│   │   └── markdown_utils.py           # Markdown → HTML conversion
│   │
│   └── services/                       # Business logic
│       └── analysis_service.py         # SDM pipeline wrapper
│
├── docs/                               # Documentation
│   ├── methodology.md                  # Theoretical foundations
│   └── architecture.md                 # This file
│
├── data/                               # Data and cache
│   ├── README.md
│   └── cache/                          # Cached results
│       ├── paraphrases/                # Cached LLM paraphrases
│       ├── answers/                    # Cached LLM answers
│       ├── embeddings/                 # Cached sentence embeddings
│       └── distributions/              # Cached probability distributions
│
├── examples/                           # Usage examples
├── tests/                              # Unit tests
├── Semantic_Faithfulness_SDM_demo.ipynb # Demo notebook
├── requirements.txt                    # Core dependencies
├── setup.py                            # Package installation
└── README.md                           # Main documentation
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Core Library | Python 3.8+ | SDM algorithms |
| Embedding | SentenceTransformers, Qwen3 | Text → vectors |
| Clustering | UDIB, scikit-learn | Topic discovery |
| Optimization | NumPy, SciPy | Csiszár-Tusnády |
| Web Framework | NiceGUI 3.x | Browser interface |
| Visualization | Plotly | Interactive charts |
| LLM APIs | OpenAI, Anthropic, Google | Paraphrase/answer generation |
| PDF Export | ReportLab | Document generation |
