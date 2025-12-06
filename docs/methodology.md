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

The **Upper-Bounded Deterministic Information Bottleneck (UDIB)** algorithm clusters sentence embeddings into N semantic topics:

```
minimize: I(X; T) - β·I(T; Y)
subject to: I(X; T) ≤ I_max
```

Where:
- X: sentence embeddings
- T: topic assignments (clusters)
- Y: text identity (Q/C/A)
- β: trade-off parameter
- I_max: upper bound on compression

The UDIB algorithm automatically determines the optimal number of topics N by maximizing the information bottleneck objective subject to the compression bound.

### 1.3 Probability Distributions

For each text, compute the probability distribution over N topics:

p^(q) = (p_1^(q), ..., p_N^(q))  # Question distribution
p^(c) = (p_1^(c), ..., p_N^(c))  # Context distribution
p^(a) = (p_1^(a), ..., p_N^(a))  # Answer distribution

where p_j^(·) = (count of sentences in topic j) / (total sentences).

## 2. Information Flow as Transition Matrices

### 2.1 Transition Matrix Framework

Model information flow from context to question/answer as transition matrices:

**Q-matrix** (Goal Channel):
```
Q_ij = P(topic j in Q | topic i in C)
```

**A-matrix** (Actual Channel):
```
A_ij = P(topic j in A | topic i in C)
```

Both matrices must:
1. Be row-stochastic: Σ_j Q_ij = 1, Σ_j A_ij = 1
2. Satisfy marginal constraints:
   - p^(q) = p^(c)^T · Q
   - p^(a) = p^(c)^T · A

### 2.2 Optimal Goal Channel Q*

The optimal goal channel Q* minimizes KL divergence from the actual answer channel:

```
Q* = argmin_Q  KL(A || Q) = Σ_{i,j} p_i^(c) A_ij log(A_ij / Q_ij)

subject to:
  Σ_j Q_ij = 1  ∀i
  p^(q) = p^(c)^T · Q
```

This is computed using **Csiszár-Tusnády alternating minimization**.

## 3. Semantic Faithfulness Metric

### 3.1 Definition

**Semantic Faithfulness** quantifies how closely A aligns with Q*:

```
F_S = 1 / (1 + D_min)

where D_min = KL(A || Q*) = Σ_{i,j} p_i^(c) A_ij log(A_ij / Q*_ij)
```

Properties:
- Range: F_S ∈ (0, 1]
- F_S = 1: Perfect faithfulness (A = Q*)
- F_S → 0: Low faithfulness (high divergence)

### 3.2 Computation Algorithm

**Iterative A-Q Alternating Minimization:**

Initialize: Q^(0) = uniform, A^(0) = uniform

Repeat until convergence:
  # A-step: Update A given Q^(k)
  A_ij^(k+1) = p_i^(c) A_ij^(k) / Z_i^A
              · exp(-ξ_i - ν_j)
              · Q_ij^(k)

  # Q-step: Update Q given A^(k+1)
  Q_ij^(k+1) = p_i^(c) Q_ij^(k) / Z_i^Q
              · exp(-ξ_i - ν_j)
              · A_ij^(k+1)

where ξ, ν are Lagrange multipliers enforcing constraints.

Convergence criterion: |F_S^(k+1) - F_S^(k)| < ε (typically ε = 10^-7)

## 4. Semantic Entropy Production

### 4.1 Thermodynamic Interpretation

View LLM as a **bipartite information engine**:
- Sub-system X: Observable (context C → answer A)
- Sub-system Y: Hidden controller (Maxwell's demon)

### 4.2 System Entropy Change

**Semantic expansion/compression:**

```
Ṡ = H(A) - H(C) = Σ_j p_j^(a) log(1/p_j^(a)) - Σ_i p_i^(c) log(1/p_i^(c))
```

Interpretation:
- Ṡ > 0: Semantic expansion (LLM elaborates)
- Ṡ < 0: Semantic compression (LLM summarizes)
- Ṡ = 0: Semantic conservation

### 4.3 Dissipated Heat

**Irreversibility of information flow:**

The dissipated heat Ṡ_m quantifies the irreversibility of the LLM's answer generation process. See the paper for the full derivation and formula.

This establishes the **inverse relationship**:
- High F_S → Low Ṡ_m (faithful answers have low dissipated heat)
- Low F_S → High Ṡ_m (unfaithful answers have high dissipated heat)

## 5. Practical Considerations

### 5.1 Embedding Model Selection

Recommended models:
- **Lightweight**: `all-MiniLM-L6-v2` (80M params, fast)
- **Balanced**: `all-mpnet-base-v2` (110M params)
- **High-quality**: `sentence-t5-base` (220M params)
- **Domain-specific**: Fine-tuned models for specialized domains

### 5.2 Clustering Method Selection

**UDIB (Recommended):**
- Pros: Automatic cluster selection, information-theoretic grounding
- Cons: Slower than k-means
- Use when: Want optimal topic granularity

**K-means with elbow method:**
- Pros: Fast, simple
- Cons: Requires manual cluster selection
- Use when: Speed is critical

### 5.3 Computational Complexity

For N topics, S sentences, K iterations:
- Embedding: O(S · d) where d = embedding dimension
- Clustering: O(S · N · K_cluster)
- F_S computation: O(N^2 · K_opt)

Typical runtime: ~10-30 seconds per triplet on CPU

## 6. Validation and Interpretation

### 6.1 Faithfulness Thresholds

Based on empirical analysis:
- F_S > 0.85: High faithfulness
- 0.65 < F_S < 0.85: Moderate faithfulness
- F_S < 0.65: Low faithfulness

### 6.2 Entropy Production Interpretation

Typical SEP_total values:
- < 0.1 bits: Very low (near-optimal)
- 0.1-0.3 bits: Low (good faithfulness)
- 0.3-0.5 bits: Moderate
- > 0.5 bits: High (potential hallucination)

### 6.3 Validation Approaches

1. **Correlation with human judgment**: LLM-as-a-Judge evaluation
2. **Test-retest reliability**: Consistent scores across paraphrases
3. **Known-good/known-bad examples**: Sanity checks
4. **Theoretical consistency**: Verify inverse F_S-SEP relationship

## References

1. Csiszár, I., & Tusnády, G. (1984). Information geometry and alternating minimization procedures.
2. Amari, S. (2016). Information Geometry and Its Applications.
3. Seifert, U. (2012). Stochastic thermodynamics, fluctuation theorems and molecular machines.
4. Parrondo, J. M. R., et al. (2015). Thermodynamics of information.
