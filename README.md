# Semantic Divergence Metrics (SDM): Information-Theoretic Faithfulness for LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package implementing **novel metrics grounded in information theory and stochastic thermodynamics** for evaluating Large Language Model (LLM) faithfulness and semantic alignment with source contexts. These metrics are computable using lightweight algorithms requiring only sentence embeddings and standard convex optimization. The package provides the Semantic Faithfulness ($\mathcal{F}_S$) and Semantic Entropy Production (SEP) metrics introduced in our paper.

**Now with a beautiful web-based GUI application!** See the [GUI Documentation](gui/README.md) for details.

## 📖 Overview

When LLMs generate answers based on provided context, how do we measure whether they faithfully represent the information in that context? Traditional metrics like BLEU or ROUGE measure surface-level similarity, while this package provides **information-theoretic measures** that capture semantic alignment at a deeper level.

### Key Features

- **Semantic Faithfulness ($\mathcal{F}_S$)**: Quantifies how well an LLM's answer aligns with the optimal information channel from context to question
- **Semantic Entropy Production (SEP)**: Computed as the sum of system entropy change and dissipated heat, SEP quantifies the degree of irreversibility in the LLM's answer generation process, grounded in stochastic thermodynamics
- **Upper-Bounded DIB (UDIB) Clustering**: Automated semantic topic discovery from text
- **Black-box Evaluation**: Works with any LLM without requiring access to internal activations or logits
- **Convex Optimization**: Guaranteed convergence to global optimum using Csiszár-Tusnády/Blahut-Arimoto Alternating Minimization algorithm
- **Web-Based GUI**: Interactive browser application with LLM integration, paraphrase generation, answer comparison, and LLM-as-a-Judge evaluation

### Theoretical Foundation

The SDM framework models LLM question-answering as information flow through transition matrices:
- Context $C$ → Question $Q$ (goal channel)
- Context $C$ → Answer $A$ (actual channel)

**Semantic Faithfulness** measures how closely the answer channel approximates the optimal goal channel by minimizing KL divergence.

**Semantic Entropy Production** decomposes into:
- System entropy change: $\overset{.}{S} = H(A) - H(C)$ (semantic expansion/compression)
- Dissipated heat: $\overset{.}{S}_m$ (see paper for formula)

#### Core Algorithms and Principles

- **Information theory and information geometry principles** for characterizing optimal divergence measures
- **Csiszár-Tusnády/Blahut-Arimoto alternating minimization** for computing optimal information channels
- **Stochastic thermodynamics framework** for quantifying entropy production and system irreversibility

## 🚀 Quick Start

### Demo Notebook (Recommended)

The fastest way to get started is with our **comprehensive Jupyter notebook** that includes:
- ✅ Step-by-step pipeline walkthrough
- ✅ Multi-triplet batch analysis (10 QCA triplets from the paper)
- ✅ Publication-quality visualizations (5 figures)
- ✅ Pre-computed cache for instant reproducibility

```bash
# Clone and setup
git clone https://github.com/ighalp/semantic-faithfulness-sdm.git
cd semantic-faithfulness-sdm
pip install -e .

# Run the demo notebook
jupyter notebook Semantic_Faithfulness_SDM_demo.ipynb
```

The notebook is divided into two parts:
1. **Part I**: Single-triplet walkthrough (educational)
2. **Part II**: Multi-triplet analysis with visualization suite (research)

**Performance Note**: The notebook uses pre-computed cached distributions and does NOT regenerate embeddings. Full execution time for all 10 triplets is approximately 2-3 minutes, with most time spent on the Csiszár-Tusnády optimization algorithm (not embedding generation).

### Web-Based GUI Application

For an interactive experience, use our **Paraphrase Me** web application:

```bash
# Navigate to the GUI directory
cd gui

# Install GUI dependencies
pip install -r requirements-gui.txt

# Run the application
python app.py
```

Open your browser to **http://localhost:8080** and enjoy:

- **Input Page**: Enter your Question-Context-Answer triplet or use built-in examples
- **LLM Pipeline**: Generate paraphrases and answers using OpenAI, Anthropic, or Google Gemini
- **Analysis**: Compute F_S scores for all answer variants automatically
- **Results**: Interactive visualizations with Plotly charts
- **Compare**: Side-by-side answer comparison with diff highlighting
- **LLM Judge**: Use LLM-as-a-Judge to evaluate which answer is best

See the [GUI Documentation](gui/README.md) for complete details.

### Installation

```bash
# Clone the repository
git clone https://github.com/ighalp/semantic-faithfulness-sdm.git
cd semantic-faithfulness-sdm

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from sdm_package import SemanticFaithfulnessAnalyzer, compute_semantic_faithfulness

# Initialize analyzer
analyzer = SemanticFaithfulnessAnalyzer(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    clustering_method="udib"  # Upper-Bounded DIB
)

# Prepare your QCA triplet
question = "What are the main risks to NVIDIA's business?"
context = """NVIDIA faces risks from supply chain dependencies..."""
answer = """The main risks include: 1) Supply chain vulnerabilities..."""

# Compute semantic distributions
analyzer.fit_transform([question], [context], [answer])

# Compute Semantic Faithfulness and Entropy Production
results = compute_semantic_faithfulness(
    p_context=analyzer.get_distribution('context', 0),
    p_question=analyzer.get_distribution('question', 0),
    p_answer=analyzer.get_distribution('answer', 0),
    return_all=True
)

print(f"Semantic Faithfulness (F_S): {results['F_S']:.3f}")
print(f"Semantic Entropy Production (SEP): {results['SEP_total']:.3f} bits")
print(f"System Entropy Change (Ṡ): {results['SEP_system']:.3f} bits")
```

### Example: Comparing Multiple Answers

```python
import json
from sdm_package import SemanticFaithfulnessAnalyzer, compute_semantic_faithfulness

# Load your QCA triplets
with open('data/qca_triplets.json', 'r') as f:
    triplets = json.load(f)

# Analyze all triplets
analyzer = SemanticFaithfulnessAnalyzer()
results = []

for triplet in triplets:
    analyzer.fit_transform(
        [triplet['question']],
        [triplet['context']],
        [triplet['answer']]
    )

    sf_result = compute_semantic_faithfulness(
        p_context=analyzer.get_distribution('context', 0),
        p_question=analyzer.get_distribution('question', 0),
        p_answer=analyzer.get_distribution('answer', 0)
    )

    results.append({
        'id': triplet['id'],
        'F_S': sf_result['F_S'],
        'SEP_total': sf_result['SEP_total']
    })

# Sort by faithfulness (higher is better)
results.sort(key=lambda x: x['F_S'], reverse=True)
print(f"Best answer: {results[0]['id']} (F_S={results[0]['F_S']:.3f})")
```

## 📊 Experimental Results

Our experiments on NVIDIA 10-K financial disclosures show:

| Metric | Group A (Comprehensive) | Group B (Focused) |
|--------|------------------------|-------------------|
| Mean $\mathcal{F}_S$ | 0.906 | 0.780 |
| Mean SEP<sub>total</sub> | 0.102 bits | 0.294 bits |
| Question Structure | Multi-topic (4+ categories) | Single-topic (competitive) |

**Key Finding**: Question semantic structure—not just entropy—drives faithfulness. Comprehensive questions with explicit structure achieve ~16% higher faithfulness.

### Detecting Hallucinations Missed by LLM Judges

The $\mathcal{F}_S$ metric can detect subtle hallucinations that LLM-as-a-Judge evaluations miss. See [`docs/examples/`](docs/examples/) for a documented case where:

- **LLM Judge** (Claude Sonnet 4.5) rated two answers as a **tie** (both 9/10)
- **$\mathcal{F}_S$ metric** correctly scored the hallucinating answer lower (0.25 vs 0.32)

The hallucination: One answer fabricated customer names ("Customers A, B, C") when the source only mentioned "three direct customers" without naming them. This subtle fabrication could mislead downstream LLM systems into researching non-existent entities.

## 📂 Repository Structure

```
semantic-faithfulness-sdm/
├── sdm_package/                    # Core SDM package
│   ├── __init__.py
│   ├── SDM.py                      # Main SDM analyzer class
│   ├── DIB_with_KL_upper_bound.py  # UDIB clustering algorithm
│   └── compute_semantic_faithfulness.py  # F_S and SEP computation
├── gui/                            # Web-based GUI application
│   ├── app.py                      # Main application entry point
│   ├── requirements-gui.txt        # GUI-specific dependencies
│   ├── README.md                   # GUI documentation
│   ├── llm_client.py               # LLM API client (OpenAI, Anthropic, Gemini)
│   ├── pages/                      # NiceGUI page modules
│   │   ├── home.py                 # Landing page
│   │   ├── input_page.py           # Input form with LLM pipeline
│   │   ├── analyze.py              # Analysis execution
│   │   ├── results.py              # Results visualization
│   │   ├── compare.py              # Answer comparison
│   │   ├── judge.py                # LLM-as-a-Judge evaluation
│   │   └── markdown_utils.py       # Markdown to HTML conversion
│   └── services/                   # Business logic
│       └── analysis_service.py     # SDM pipeline wrapper
├── examples/                       # Usage examples
│   ├── basic_usage.py
│   └── batch_evaluation.py
├── tests/                          # Unit tests
│   ├── test_sdm.py
│   └── test_semantic_faithfulness.py
├── docs/                           # Documentation
│   └── methodology.md              # Theoretical foundations
├── data/                           # Example data and cache
│   ├── README.md
│   └── cache/                      # Cached embeddings and distributions
├── Semantic_Faithfulness_SDM_demo.ipynb  # Interactive demo notebook
├── requirements.txt                # Core dependencies
├── setup.py                        # Package installation
├── README.md                       # This file
└── LICENSE                         # MIT License
```

## 🔧 Requirements

- Python 3.8+
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- sentence-transformers >= 2.2.0
- nltk >= 3.6
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- POT (Python Optimal Transport) >= 0.8.0
- kneed >= 0.7.0

## 📖 Citation

If you use this package in your research, please cite:

```bibtex
@article{halperin2025faithfulness,
  title={Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations},
  author={Halperin, Igor},
  note={To be published},
  year={2025}
}
```

**Note**: Full citation details will be available upon publication. Please also consider citing the foundational SDM papers listed in the References section below.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References and Related Work

### This Work

**Halperin, I. (2025).** "Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations."
*To be published soon.*

This paper introduces the Semantic Faithfulness ($\mathcal{F}_S$) and Semantic Entropy Production (SEP) metrics implemented in this package.

### Foundational SDM Papers

1. **Halperin, I. (2025).** "Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination Detection in Large Language Models."
   - arXiv: [https://arxiv.org/abs/2508.10192](https://arxiv.org/abs/2508.10192)
   - SSRN: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5390586](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5390586)

   *The original SDM framework introducing semantic uncertainty and divergence metrics.*

2. **Halperin, I. (2025).** "Topic Identification in LLM Input-Output Pairs through the Lens of Information Bottleneck."
   - arXiv: [https://arxiv.org/abs/2509.03533](https://arxiv.org/abs/2509.03533)
   - SSRN: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5403971](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5403971)

   *Introduces the Upper-Bounded Deterministic Information Bottleneck (UDIB) clustering algorithm used in this package.*

### Related Research

- **Farquhar, S. et al. (2024).** "Detecting Hallucinations in Large Language Models Using Semantic Entropy."
  NeurIPS 2024. [Paper](https://arxiv.org/abs/2406.15012)

- **Zheng, L. et al. (2023).** "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."
  NeurIPS 2023. [Paper](https://arxiv.org/abs/2306.05685)

## 📧 Contact

Igor Halperin

Project Link: [https://github.com/ighalp/semantic-faithfulness-sdm](https://github.com/ighalp/semantic-faithfulness-sdm)

## 🙏 Acknowledgments

This section is reserved for acknowledging future contributors and collaborators.
