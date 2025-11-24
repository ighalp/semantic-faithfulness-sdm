"""
Basic Usage Example: Computing Semantic Faithfulness for a Single QCA Triplet

This example demonstrates how to use the SDM package to compute
Semantic Faithfulness (F_S) and Entropy Production (SEP) metrics
for a question-context-answer triplet.
"""

import sys
sys.path.append('..')

from sdm_package import SemanticFaithfulnessAnalyzer, compute_semantic_faithfulness

def main():
    # Example QCA triplet
    question = """What are the primary competitive threats facing NVIDIA in the AI
    accelerator market over the next 3-5 years?"""

    context = """NVIDIA Corporation faces competitive pressures from multiple sources.
    First, hyperscale cloud providers including Google, Amazon, Microsoft, and Meta
    are developing custom AI accelerators (TPUs, Trainium, Inferentia, Maia) to reduce
    dependency on NVIDIA's GPUs. Second, traditional semiconductor competitors like AMD
    and Intel are launching competing AI acceleration products (MI300, Gaudi). Third,
    NVIDIA's CUDA software ecosystem, historically a strong competitive moat, faces
    erosion from cross-platform frameworks like PyTorch and JAX."""

    answer = """NVIDIA faces three main competitive threats: 1) Hyperscaler custom
    silicon from Google (TPU), AWS (Trainium/Inferentia), Microsoft (Maia), and Meta
    (MTIA), which could reduce demand from these major customers. 2) Traditional
    competitors AMD (MI300 with unified memory) and Intel (Gaudi accelerators) offering
    viable alternatives. 3) Erosion of CUDA lock-in as cross-platform frameworks enable
    easier hardware switching."""

    print("="*80)
    print("Semantic Faithfulness Analysis")
    print("="*80)

    # Initialize the analyzer
    print("\nInitializing semantic analyzer...")
    analyzer = SemanticFaithfulnessAnalyzer(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        clustering_method="udib",  # Upper-Bounded DIB
        verbose=True
    )

    # Compute semantic distributions over topics
    print("\nComputing semantic distributions...")
    analyzer.fit_transform(
        questions=[question],
        contexts=[context],
        answers=[answer]
    )

    # Get distributions
    p_context = analyzer.get_distribution('context', 0)
    p_question = analyzer.get_distribution('question', 0)
    p_answer = analyzer.get_distribution('answer', 0)

    print(f"\nNumber of semantic topics: {len(p_context)}")
    print(f"Context entropy: H(C) = {analyzer.compute_entropy(p_context):.3f} bits")
    print(f"Question entropy: H(Q) = {analyzer.compute_entropy(p_question):.3f} bits")
    print(f"Answer entropy: H(A) = {analyzer.compute_entropy(p_answer):.3f} bits")

    # Compute Semantic Faithfulness and Entropy Production
    print("\nComputing Semantic Faithfulness metrics...")
    results = compute_semantic_faithfulness(
        p_context=p_context,
        p_question=p_question,
        p_answer=p_answer,
        return_all=True
    )

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nSemantic Faithfulness (F_S): {results['F_S']:.3f}")
    print(f"  Interpretation: {'High faithfulness' if results['F_S'] > 0.8 else 'Moderate faithfulness' if results['F_S'] > 0.6 else 'Low faithfulness'}")

    print(f"\nMinimal KL Divergence (D_min): {results['D_min']:.3f} bits")
    print(f"  (Distance from optimal channel)")

    print(f"\nTotal Entropy Production (SEP_total): {results['SEP_total']:.3f} bits")
    print(f"  (Irreversibility in information flow)")

    print(f"\nSystem Entropy Production (SEP_system): {results['SEP_system']:.3f} bits")
    print(f"  Interpretation: {'Semantic expansion' if results['SEP_system'] > 0 else 'Semantic compression'}")

    print(f"\nOptimization converged: {results['converged']}")
    print(f"Number of iterations: {results['iterations']}")

    # Inverse relationship verification
    approx_sep = 1/results['F_S'] - 1
    print(f"\nTheoretical approximation check:")
    print(f"  SEP_total ≈ 1/F_S - 1 = {approx_sep:.3f} bits")
    print(f"  Actual SEP_total = {results['SEP_total']:.3f} bits")
    print(f"  Approximation error: {abs(approx_sep - results['SEP_total'])/results['SEP_total']*100:.1f}%")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
