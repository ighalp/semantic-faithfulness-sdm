# Methodology: Semantic Faithfulness and Entropy Production

## Overview

This document explains the theoretical foundations and computational methods behind the Semantic Divergence Metrics (SDM) framework.

## 1. Topical Representation

### 1.1 Sentence Embedding

Each text (question Q, context C, answer A) is decomposed into sentences:
- Tokenization using NLTK's `sent_tokenize`
- Embedding using pre-trained sentence transformers (e.g., `all-MiniLM-L6-v2`)
- Each sentence → dense vector in ℝ^d

### 1.2 Topic Discovery via UDIB Clustering

The **Upper-Bounded Deterministic Information Bottleneck (UDIB)** algorithm clusters sentence embeddings into N semantic topics. UDIB is based on the Deterministic Information Bottleneck (DIB) framework for geometric clustering, adapted for practical computation on high-dimensional embeddings.

**Background: DIB for Geometric Clustering**

The DIB method assigns each data point to a cluster by minimizing a per-point Lagrangian:

```
L[q(c|i)] = -log q(c) + β · D_KL(p(x|i) || q(x|c))
```

Where:
- q(c): marginal probability of cluster c (its relative size)
- p(x|i): smoothed distribution centered at point i (Gaussian with variance s²)
- q(x|c): cluster conditional distribution (a Gaussian mixture)
- β: trade-off parameter between compression and information preservation

**The UDIB Innovation**

The KL divergence between a Gaussian and a Gaussian mixture is analytically intractable. UDIB replaces this with a computationally efficient upper bound based on Jensen's inequality (Hershey & Olsen, 2007):

```
D_KL(p(x|i) || q(x|c)) ≤ (1 / 2s²n_c) Σ_{j∈S_c} ||x_i - x_j||²
```

This yields the practical UDIB assignment rule with a single effective hyperparameter τ = 2s²/β:

```
c*(i) = argmin_c [ (1/n_c) Σ_{j∈S_c} ||x_i - x_j||² - τ·log q(c) ]
```

**Interpretation**

UDIB can be viewed as an **entropy-regularized, robust version of K-means**:
- The first term measures average squared distance to all cluster members (an upper bound on K-means distance to centroid)
- The second term (-τ·log q(c)) penalizes cluster entropy, encouraging fewer, more balanced clusters

**Model Selection**

Unlike K-means which requires pre-specifying the number of clusters, UDIB provides built-in model selection. By sweeping the parameter τ and analyzing where the number of clusters remains stable (via "kink angle" analysis of the information profile), the algorithm automatically determines the optimal number of topics N.

For full algorithmic details, see [Halperin (2025), "Topic Identification in LLM Input-Output Pairs through the Lens of Information Bottleneck"](https://arxiv.org/abs/2509.03533).

### 1.3 Probability Distributions

For each text, compute the probability distribution over N topics:

```
p^(c) = (p_1^(c), ..., p_N^(c))  # Context distribution
p^(q) = (p_1^(q), ..., p_N^(q))  # Question distribution
p^(a) = (p_1^(a), ..., p_N^(a))  # Answer distribution
```

where p_j^(·) = (count of sentences in topic j) / (total sentences).

These marginal distributions are computed using the SDM method by counting frequencies of cluster assignments of sentences in each text.

## 2. Semantic Faithfulness (SF) Metric

### 2.1 Information Flow as Transition Matrices

We model topic transformations from context C to question Q and answer A as two N×N transition matrices:

- **Q**: Q_ij = P(topic j in Q | topic i in C) — encodes the "goal" of the query
- **A**: A_ij = P(topic j in A | topic i in C) — encodes the "result" of the LLM

Both matrices must be row-stochastic and satisfy marginal constraints:

```
Σ_j Q_ij = 1,  Σ_j A_ij = 1  ∀i
p^(q) = p^(c)ᵀ · Q
p^(a) = p^(c)ᵀ · A
```

### 2.2 Definition

The **Semantic Faithfulness** score is defined as:

```
F_S = 1 / (1 + D_min)
```

where D_min is the minimal KL divergence between matrices A and Q:

```
D_min = min_{A,Q} D(A ‖ Q) = min_{A,Q} Σ_{i,j} p_i^(c) A_ij log(A_ij / Q_ij)
```

subject to the row-stochastic and marginal constraints above.

**Properties:**
- Range: F_S ∈ (0, 1]
- F_S = 1: Perfect faithfulness (A = Q)
- F_S → 0: Low faithfulness (high divergence)

### 2.3 Algorithm: Computing Semantic Faithfulness

The optimization is jointly convex and solved via **Csiszár-Tusnády alternating minimization**:

```
Algorithm 1: Semantic Faithfulness

Input: Marginal distributions p^(c), p^(a), p^(q), initial matrix Q^(0),
       tolerances ε_outer, ε_inner

Repeat until convergence:

    // A-Step: Update A given Q^(k)
    1. Find scaling factors u_j by solving the fixed-point equation:
       u_j = p_j^(a) / Σ_i [p_i^(c) Q_ij / Σ_j' Q_ij' u_j']

    2. Compute A^(k+1):
       A_ij = Q_ij u_j / Σ_j' Q_ij' u_j'

    // Q-Step: Update Q given A^(k+1)
    3. Find Lagrange multipliers ξ, ν by alternating maximization of:
       L_2(ξ,ν) = Σ_{i,j} p_i^(c) A_ij log(ν_i + ξ_j) - Σ_i p_i^(c) ν_i - Σ_j p_j^(q) ξ_j + 1

    4. Compute Q^(k+1):
       Q_ij = A_ij / (ν_i + ξ_j)

Until: change in D(A^(k) ‖ Q^(k)) < ε_outer

Output: Optimal matrices A*, Q*, minimal divergence D_min,
        SF score F_S = 1/(1 + D_min)
```

## 3. Semantic Entropy Production (SEP)

### 3.1 Thermodynamic Framework

We model the LLM as a **bipartite information engine**:
- **Sub-system X**: Observable channel (context C → answer A)
- **Sub-system Y**: Hidden controller (the "Maxwell's demon" — LLM's internal computation)

Total entropy production decomposes as:

```
SEP = Ṡ + Ṡ_m
```

where:
- **Ṡ = H(A) - H(C)**: System entropy change (semantic expansion/compression)
- **Ṡ_m**: Dissipated heat (entropy flow to the environment)

### 3.2 System Entropy Change

```
Ṡ = H(A) - H(C) = -Σ_j p_j^(a) log p_j^(a) + Σ_i p_i^(c) log p_i^(c)
```

Interpretation:
- Ṡ > 0: Semantic expansion (LLM elaborates beyond context)
- Ṡ < 0: Semantic compression (LLM summarizes)
- Ṡ = 0: Semantic conservation

### 3.3 Algorithm: Computing SEP

SEP is computed as D(A* ‖ A^R), where A^R is the optimal reverse (time-reversed) transition matrix:

```
Algorithm 2: Semantic Entropy Production (SEP)

Input: Optimal transition matrix A* (from Algorithm 1), p^(c), p^(a)

1. Initialize ξ, ν ∈ ℝ^N_{>0}

Repeat until convergence:

    // ξ-step: Maximize L_S w.r.t. ξ (fixing ν)
    ξ_i* = argmax_ξ [Σ_j p_i^(c) A*_ij log(ξ_i + ν_j) - p_i^(c) ξ_i]

    // ν-step: Maximize L_S w.r.t. ν (fixing ξ)
    ν_j* = argmax_ν [Σ_i p_i^(c) A*_ij log(ξ_i + ν_j) - p_j^(a) ν_j]

2. Recover reverse matrix:
   A^R_ji = A*_ij / (ξ_i* + ν_j*)

Output: SEP = D(A* ‖ A^R) = Σ_{i,j} p_i^(c) A*_ij log(A*_ij / A^R_ji)
```

### 3.4 Relationship Between SF and SEP

The SF and SEP metrics are related but capture distinct aspects:
- **SF (F_S)**: Information-theoretic alignment between question intent and answer content
- **SEP**: Thermodynamic efficiency of the information transformation

Empirically, they show moderate negative correlation (r ≈ -0.6): higher faithfulness generally implies lower entropy production, but the relationship is not deterministic.

For full theoretical derivations, see [Halperin (2025), "Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations"](https://arxiv.org/abs/2512.05156).

## 4. Practical Considerations

### 4.1 Embedding Model Selection

Recommended models:
- **Lightweight**: `all-MiniLM-L6-v2` (80M params, fast)
- **Balanced**: `all-mpnet-base-v2` (110M params)
- **High-quality**: `sentence-t5-base` (220M params)
- **Domain-specific**: Fine-tuned models for specialized domains

### 4.2 Clustering Method Selection

**UDIB (Recommended):**
- Pros: Automatic cluster selection, information-theoretic grounding
- Cons: Slower than k-means
- Use when: Want optimal topic granularity

**K-means with elbow method:**
- Pros: Fast, simple
- Cons: Requires manual cluster selection
- Use when: Speed is critical

### 4.3 Computational Complexity

For N topics, S sentences, K iterations:
- Embedding: O(S · d) where d = embedding dimension
- Clustering: O(S · N · K_cluster)
- F_S computation: O(N^2 · K_opt)

Typical runtime: ~10-30 seconds per triplet on CPU

## 5. Validation and Interpretation

### 5.1 Faithfulness Thresholds

**Note:** The following thresholds are *indicative* only. The actual range of F_S values depends on your specific prompt structure, context complexity, and experimental setup. Calibrate thresholds based on your domain and use case.

Indicative ranges based on empirical analysis:
- F_S > 0.85: Typically indicates high faithfulness
- 0.65 < F_S < 0.85: Typically indicates moderate faithfulness
- F_S < 0.65: May indicate lower faithfulness (potential hallucination)

### 5.2 Entropy Production Interpretation

**Note:** The following thresholds are *indicative* only. The actual range of SEP values will vary depending on your specific prompt, context variability, and experimental conditions. Use these as starting points and calibrate based on your particular experiments.

Indicative SEP ranges:
- < 0.1 bits: Typically very low (near-optimal)
- 0.1-0.3 bits: Typically low (good faithfulness)
- 0.3-0.5 bits: Typically moderate
- > 0.5 bits: Typically high (potential hallucination)

### 5.3 Validation Approaches

1. **Correlation with human judgment**: LLM-as-a-Judge evaluation
2. **Test-retest reliability**: Consistent scores across paraphrases
3. **Known-good/known-bad examples**: Sanity checks
4. **Theoretical consistency**: Verify inverse F_S-SEP relationship

## References

1. Halperin, I. (2025). "Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations." [GitHub](https://github.com/ighalp/semantic-faithfulness-sdm)
2. Halperin, I. (2025). "Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination Detection in Large Language Models." [arXiv:2508.10192](https://arxiv.org/abs/2508.10192)
3. Halperin, I. (2025). "Topic Identification in LLM Input-Output Pairs through the Lens of Information Bottleneck." [arXiv:2509.03533](https://arxiv.org/abs/2509.03533)
4. Csiszár, I., & Tusnády, G. (1984). Information geometry and alternating minimization procedures. Statistics & Decisions, Supplement Issue, 1, 205-237.
5. Parrondo, J. M. R., Horowitz, J. M., & Sagawa, T. (2015). Thermodynamics of information. Nature Physics, 11, 131-139.
6. Seifert, U. (2012). Stochastic thermodynamics, fluctuation theorems and molecular machines. Rep. Prog. Phys., 75, 126001.
7. Hershey, J. R., & Olsen, P. A. (2007). Approximating the Kullback Leibler divergence between Gaussian mixture models. ICASSP.
