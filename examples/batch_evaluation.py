"""
Batch Evaluation Example: Comparing Multiple Answers

This example demonstrates how to evaluate multiple QCA triplets
and compare them based on Semantic Faithfulness scores.
"""

import sys
sys.path.append('..')

import json
import numpy as np
from sdm_package import SemanticFaithfulnessAnalyzer, compute_semantic_faithfulness

def main():
    print("="*80)
    print("Batch Evaluation: Comparing Multiple LLM Answers")
    print("="*80)

    # Sample dataset: same question and context, different answer variations
    question = "What are NVIDIA's main supply chain vulnerabilities?"

    context = """NVIDIA's fabless manufacturing model creates dependency on third-party
    foundries, primarily TSMC, which produces 90-95% of NVIDIA's advanced semiconductors.
    Geographic concentration in Taiwan presents geopolitical risks. Manufacturing lead
    times exceed 12 months during capacity constraints. Demand forecasting challenges
    arise from unpredictable generative AI adoption and cryptocurrency mining volatility."""

    answers = {
        "answer_1": """NVIDIA relies heavily on TSMC for manufacturing (90-95% of production),
        creating single-point-of-failure risk. Taiwan geographic concentration adds
        geopolitical vulnerability. Long lead times (12+ months) during shortages combined
        with poor demand forecasting from AI/crypto volatility compounds supply risks.""",

        "answer_2": """Supply chain vulnerabilities include dependency on external foundries,
        concentration in geopolitically sensitive regions, extended procurement cycles,
        and demand forecasting difficulties.""",

        "answer_3": """NVIDIA faces challenges from relying on TSMC and other factors
        affecting semiconductor supply chains globally. The company must manage these
        risks carefully to maintain competitiveness.""",
    }

    # Initialize analyzer once (shared clustering across all triplets)
    print("\nInitializing semantic analyzer...")
    analyzer = SemanticFaithfulnessAnalyzer(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        clustering_method="udib",
        verbose=False
    )

    # Prepare for joint analysis
    questions_list = [question] * len(answers)
    contexts_list = [context] * len(answers)
    answers_list = list(answers.values())

    print(f"Analyzing {len(answers)} answer variations...")
    analyzer.fit_transform(
        questions=questions_list,
        contexts=contexts_list,
        answers=answers_list
    )

    # Compute metrics for each answer
    results = []
    for i, (answer_id, answer_text) in enumerate(answers.items()):
        p_c = analyzer.get_distribution('context', i)
        p_q = analyzer.get_distribution('question', i)
        p_a = analyzer.get_distribution('answer', i)

        metrics = compute_semantic_faithfulness(
            p_context=p_c,
            p_question=p_q,
            p_answer=p_a,
            return_all=True
        )

        results.append({
            'id': answer_id,
            'text': answer_text[:100] + "...",
            'length': len(answer_text),
            'F_S': metrics['F_S'],
            'SEP_total': metrics['SEP_total'],
            'SEP_system': metrics['SEP_system'],
            'H_A': analyzer.compute_entropy(p_a)
        })

    # Sort by faithfulness (highest first)
    results.sort(key=lambda x: x['F_S'], reverse=True)

    # Display results
    print("\n" + "="*80)
    print("RANKING BY SEMANTIC FAITHFULNESS")
    print("="*80)

    for rank, r in enumerate(results, 1):
        print(f"\n#{rank} - {r['id']}")
        print(f"  F_S: {r['F_S']:.3f}  |  SEP_total: {r['SEP_total']:.3f} bits  |  Length: {r['length']} chars")
        print(f"  Text: {r['text']}")

    # Statistical summary
    fs_scores = [r['F_S'] for r in results]
    sep_scores = [r['SEP_total'] for r in results]

    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(f"\nSemantic Faithfulness (F_S):")
    print(f"  Range: [{min(fs_scores):.3f}, {max(fs_scores):.3f}]")
    print(f"  Mean: {np.mean(fs_scores):.3f} ± {np.std(fs_scores):.3f}")
    print(f"  Spread: {(max(fs_scores) - min(fs_scores))/np.mean(fs_scores)*100:.1f}%")

    print(f"\nTotal Entropy Production (SEP_total):")
    print(f"  Range: [{min(sep_scores):.3f}, {max(sep_scores):.3f}] bits")
    print(f"  Mean: {np.mean(sep_scores):.3f} ± {np.std(sep_scores):.3f} bits")

    # Inverse correlation check
    correlation = np.corrcoef(fs_scores, sep_scores)[0, 1]
    print(f"\nCorrelation F_S vs SEP_total: r = {correlation:.3f}")
    print(f"  Expected: negative (high F_S → low SEP)")

    print("\n" + "="*80)
    print("\nConclusion:")
    best = results[0]
    print(f"  Best answer: {best['id']} with F_S = {best['F_S']:.3f}")
    print(f"  This answer shows {'high' if best['F_S'] > 0.8 else 'moderate'} semantic faithfulness")
    print(f"  and {'low' if best['SEP_total'] < 0.2 else 'moderate'} entropy production.")

if __name__ == "__main__":
    main()
