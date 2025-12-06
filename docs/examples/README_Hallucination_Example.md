# LLM Judge Hallucination Example: SF Method Detects What LLM Judge Misses

## Overview

This folder contains an example demonstrating a case where the **Semantic Faithfulness (F_S) metric correctly identifies a lower-quality answer** that the LLM-as-a-Judge evaluation failed to penalize.

## The Hallucination

In this comparison between two LLM-generated summaries of NVIDIA's 10-K Risk Factors:

- **Answer A (PROMPT_0)**: F_S = 0.3236
- **Answer B (PROMPT_2)**: F_S = 0.2496

The LLM Judge (Claude Sonnet 4.5) declared a **TIE** with both answers receiving 9/10 scores across all criteria.

However, **Answer B contains a significant hallucination** that the LLM Judge overlooked:

> "In fiscal year 2025, direct sales to **Customers A, B, and C represented 12%, 11%, and 11% of total revenue**, respectively."

The original context document only states that **"three direct customers"** represented these revenue percentages. It does **NOT** name them as "Customers A, B, and C."

## Why This Matters

This hallucination is potentially dangerous for downstream systems:

1. **Fabricated Entity Names**: The LLM invented placeholder names ("Customers A, B, C") that could be mistaken for actual company identifiers.

2. **Downstream Propagation Risk**: An automated system processing this summary might:
   - Attempt to look up information about "Customer A," "Customer B," or "Customer C"
   - Make incorrect inferences about NVIDIA's customer relationships
   - Generate further hallucinated content based on these fabricated names

3. **Subtle Nature**: The hallucination is subtle enough that both human reviewers and LLM judges might miss it, as the numerical facts (12%, 11%, 11%) are accurate.

## F_S Correctly Identifies the Issue

The Semantic Faithfulness metric assigned a **lower score (0.2496) to Answer B** compared to Answer A (0.3236), correctly reflecting that Answer B is less faithful to the source material despite appearing equally comprehensive to the LLM Judge.

This demonstrates the value of information-theoretic metrics like F_S as a complement to LLM-as-a-Judge evaluations:

- **LLM Judge**: Evaluates coherence, completeness, and apparent quality
- **F_S Metric**: Measures information-theoretic alignment with the source, catching subtle deviations

## Files

- `LLM_Judge_Hallucination_Example_SF_Detects_Fabricated_Customer_Names.md` - Full verdict in Markdown format
- `LLM_Judge_Hallucination_Example_SF_Detects_Fabricated_Customer_Names.pdf` - Full verdict in PDF format

## Conclusion

This example illustrates why combining multiple evaluation methods (F_S metric + LLM Judge + human review) provides more robust assessment of LLM outputs than any single method alone.
