# Paraphrase Me - LLM Semantic Assistant

A simple, Apple-inspired web application for analyzing semantic faithfulness of Large Language Model (LLM) responses using information-theoretic methods.

## Overview

**Paraphrase Me** is a complete pipeline for:
1. Generating question paraphrases using LLMs (OpenAI, Anthropic, Google Gemini)
2. Generating answers for each paraphrase
3. Compute the Semantic Faithfulness (SF) and Semantic Entropy Production (SEP) scores
4. Comparing answers side-by-side with diff highlighting
5. Using LLM-as-a-Judge to evaluate answer quality
6. Exporting results in Markdown or PDF format

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PARAPHRASE ME - GUI APPLICATION                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │   Home   │───▶│  Input   │───▶│ Analyze  │───▶│ Results  │───▶│Compare │ │
│  │   Page   │    │   Page   │    │   Page   │    │   Page   │    │  Page  │ │
│  └──────────┘    └────┬─────┘    └──────────┘    └──────────┘    └───┬────┘ │
│                       │                                               │      │
│                       ▼                                               ▼      │
│              ┌────────────────┐                              ┌──────────────┐│
│              │  LLM Pipeline  │                              │  LLM Judge   ││
│              │                │                              │              ││
│              │ • Paraphrases  │                              │ • Compare A  ││
│              │ • Answers      │                              │   vs B       ││
│              │ • F_S Scores   │                              │ • Scores     ││
│              └───────┬────────┘                              │ • Export     ││
│                      │                                       └──────────────┘│
│                      ▼                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        SDM ANALYSIS ENGINE                             │  │
│  │                                                                        │  │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌───────────┐  │  │
│  │  │ Tokenize    │──▶│  Embed      │──▶│  Cluster    │──▶│ Optimize  │  │  │
│  │  │ (NLTK)      │   │ (Qwen3)     │   │ (UDIB)      │   │ (SF Algo) │  │  │
│  │  └─────────────┘   └─────────────┘   └─────────────┘   └───────────┘  │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           LLM API PROVIDERS                                  │
│     ┌──────────┐         ┌──────────┐         ┌──────────┐                  │
│     │  OpenAI  │         │Anthropic │         │  Gemini  │                  │
│     │  GPT-4o  │         │ Claude   │         │  2.5 Pro │                  │
│     └──────────┘         └──────────┘         └──────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Input Page
- **Manual Input**: Enter Question, Context, and Answer directly
- **Example Data**: Built-in Apple Q3 2023 financial analysis example
- **JSON Import**: Upload QCA triplets from file
- **LLM Configuration**: Choose provider (OpenAI/Anthropic/Gemini) and model
- **Pipeline Options**:
  - Number of paraphrases to generate (1-10)
  - Answer temperature control (0.0-1.0)
  - Force regeneration option to bypass cache

### Analysis Page
- **Real-time Progress**: 4-stage visualization (Tokenize → Embed → Cluster → Optimize)
- **Detailed Logging**: View processing steps and timings
- **Auto-navigation**: Proceeds to Results when complete

### Results Page
- **Key Metrics Display**:
  - Semantic Faithfulness (F_S): 0-1 scale
  - System Entropy Production (SEP)
  - Entropy values H(Q), H(C), H(A)
- **Interactive Visualizations**:
  - Probability distributions over semantic topics
  - Transition matrix heatmaps (Q* and A*)
  - F_S score comparisons across paraphrases
- **Export Options**: JSON download

### Compare Page
- **Side-by-Side View**: Compare any two answers
- **Diff Highlighting**: Yellow highlights show substantially different sentences
- **Quick Selection Buttons**:
  - Initial vs Highest F_S
  - Initial vs Lowest F_S
  - Highest vs Lowest F_S
- **Tabs**: Switch between Answers and Prompts comparison

### LLM Judge Page
- **LLM-as-a-Judge Evaluation**: Use AI to compare answer quality
- **Criteria Breakdown**:
  - Faithfulness to Context
  - Completeness
  - Coherence
  - Relevance
- **Score Comparison**: LLM scores vs F_S metric agreement
- **Export Options**:
  - Markdown with full evaluation report
  - PDF with proper formatting (headers, bold, lists)

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Navigate to the GUI directory:**
   ```bash
   cd gui
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements-gui.txt
   ```

3. **Set up API keys** (optional, for LLM features):

   Create a `.env` file in the `gui/` directory:
   ```bash
   OPENAI_API_KEY=your-openai-key
   ANTHROPIC_API_KEY=your-anthropic-key
   GOOGLE_API_KEY=your-google-key
   ```

## Usage

### Starting the Application

```bash
python app.py
```

The application will:
- Start a local web server on **http://localhost:8080**
- Display startup information in the terminal
- Be accessible from your browser

### Workflow

1. **Home** → Overview and navigation
2. **Input** → Enter your QCA triplet and configure LLM pipeline
3. **Analyze** → Watch real-time analysis progress
4. **Results** → View metrics and visualizations
5. **Compare** → Side-by-side answer comparison
6. **LLM Judge** → AI-powered answer evaluation

## Configuration

### Embedding Models
- `Qwen/Qwen3-Embedding-0.6B` (Default, high quality)
- `sentence-transformers/all-MiniLM-L6-v2` (Lightweight, fast)
- `sentence-transformers/all-mpnet-base-v2` (Balanced)

### LLM Providers
- **OpenAI**: GPT-4o, GPT-4o-mini, o1-preview
- **Anthropic**: Claude Sonnet 4.5, Claude Opus 4.1, Claude Haiku 4.5
- **Google**: Gemini 2.5 Pro, Gemini 2.5 Flash

### Clustering Methods
- **UDIB** (Default): Upper-Bounded Deterministic Information Bottleneck
- **K-means**: Standard clustering with elbow method
- **Agglomerative**: Hierarchical clustering

## Output Metrics

### Semantic Faithfulness (F_S)
Value between 0 and 1 indicating how well the answer aligns with the optimal information channel.
- **F_S > 0.85**: High faithfulness
- **0.65 < F_S < 0.85**: Moderate faithfulness
- **F_S < 0.65**: Low faithfulness (potential hallucination)

### Semantic Entropy Production (SEP)
Measures irreversibility in the answer generation process.
- **SEP_system**: H(A) - H(C), semantic expansion/compression
- **SEP_total**: Divergence from optimal channel

## Technical Details

### Architecture
- **Frontend**: NiceGUI (Vue.js + Quasar)
- **Backend**: FastAPI (built into NiceGUI)
- **Visualization**: Plotly
- **Styling**: Apple-inspired design with CSS custom properties

### Processing Pipeline
1. **Tokenization**: Split text into sentences using NLTK
2. **Embedding**: Convert sentences to vectors using Qwen3-Embedding
3. **Clustering**: Group sentences into semantic topics using UDIB
4. **Optimization**: Compute optimal transition matrices via Csiszár-Tusnády algorithm

### Caching
- Paraphrases cached by question+context hash
- Answers cached by prompt+model hash
- Embeddings and distributions cached for fast re-analysis

## Troubleshooting

### Port Already in Use
Edit `app.py` and change `APP_PORT = 8080` to another port.

### PyTorch Threading Issues (macOS)

**Symptom**: Application hangs or crashes during embedding generation on macOS.

**Cause**: PyTorch's default `fork()` multiprocessing conflicts with its internal thread pools.

**Solution**: The app automatically sets `multiprocessing.set_start_method('spawn')` at startup. If you still see issues:
1. Ensure you're running `python app.py` directly (not through an IDE)
2. The following environment variables are set automatically in `app.py`:
   ```python
   os.environ['OMP_NUM_THREADS'] = '1'
   os.environ['MKL_NUM_THREADS'] = '1'
   os.environ['TOKENIZERS_PARALLELISM'] = 'false'
   ```

### Sentence Transformers Model Loading

**Symptom**: "RuntimeError: Cannot re-initialize CUDA in forked subprocess" or similar errors.

**Solution**:
- Models are loaded lazily on first use
- Embedding happens in the main process, not in subprocesses
- If using multiprocessing, ensure `spawn` start method is used

### Slow First Run
First analysis with Qwen3 model takes ~30 seconds to download and load. Subsequent runs use the cached model from `~/.cache/huggingface/`.

### Memory Issues with Large Documents

**Symptom**: Out of memory errors or very slow processing.

**Solutions**:
1. Use a smaller embedding model: `sentence-transformers/all-MiniLM-L6-v2`
2. Break very long contexts into smaller chunks
3. Reduce the number of paraphrases generated
4. Close other memory-intensive applications

### API Key Not Found

**Symptom**: "No API key found" error when running LLM pipeline.

**Solutions**:
1. Create a `.env` file in the `gui/` directory:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   OPENAI_API_KEY=sk-...
   GOOGLE_API_KEY=AIza...
   ```
2. Or set environment variables directly:
   ```bash
   export ANTHROPIC_API_KEY=your-key
   ```
3. The app checks multiple `.env` locations: `gui/`, project root, and parent directory

### NiceGUI Version Compatibility

**Symptom**: `TypeError: Html.__init__() missing required argument` or similar errors.

**Cause**: NiceGUI API changes between versions.

**Solution**: Ensure you have `nicegui>=3.0.0`:
```bash
pip install --upgrade nicegui
```

### Plotly Charts Not Rendering

**Symptom**: Empty chart containers or JavaScript errors.

**Solutions**:
1. Clear browser cache and refresh
2. Try a different browser (Chrome recommended)
3. Ensure Plotly is installed: `pip install plotly>=5.0.0`

## Cross-Platform Compatibility

Tested on:
- macOS (Apple Silicon & Intel)
- Linux
- Windows 10/11

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{halperin2025faithfulness,
  title={Semantic Faithfulness and Entropy Production Measures for LLM Evaluation},
  author={Halperin, Igor},
  year={2025}
}
```

## License

MIT License - See parent directory for details.

## Support

- **GitHub Issues**: https://github.com/ighalp/semantic-faithfulness-sdm/issues
- **Documentation**: https://github.com/ighalp/semantic-faithfulness-sdm/blob/main/docs/methodology.md

## Version

Current version: 2.0.0

### Changelog

**v2.0.0** (Current)
- Added LLM pipeline for paraphrase and answer generation
- Added Compare page with diff highlighting
- Added LLM-as-a-Judge evaluation
- Added PDF export with proper markdown formatting
- Apple-inspired UI design
- Support for OpenAI, Anthropic, and Google Gemini APIs

**v1.0.0**
- Initial release with basic analysis pipeline
- Single QCA triplet analysis
- JSON export
