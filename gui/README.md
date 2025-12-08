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
  - Semantic Entropy Production (SEP)
  - System Entropy Change (Ṡ)
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

### Authentication (Enterprise) - EXPERIMENTAL

> **⚠️ Development Status**: This authentication module is currently in development and has not been fully tested in production environments. It may require debugging and tuning when deployed. This feature is **only needed for enterprise deployments** that require corporate SSO integration. For personal or development use, leave authentication disabled (the default).

For corporate environments requiring authentication, the application supports OAuth2/OIDC with multiple identity providers.

#### Enable Authentication

Set the following environment variables in your `.env` file:

```bash
# Enable authentication (options: disabled, okta, azure, pyauth)
AUTH_PROVIDER=okta

# Session configuration
AUTH_SESSION_STORE=memory    # Options: memory, redis, oracle
AUTH_SESSION_TIMEOUT_HOURS=9  # Default: 9 hours
```

#### Provider Configuration

**Okta:**
```bash
AUTH_PROVIDER=okta
OKTA_ISSUER_URL=https://your-org.okta.com/oauth2/default
OKTA_CLIENT_ID=your-client-id
OKTA_CLIENT_SECRET=your-client-secret
OKTA_REDIRECT_URI=http://localhost:8080/auth/callback
```

**Azure AD:**
```bash
AUTH_PROVIDER=azure
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_REDIRECT_URI=http://localhost:8080/auth/callback
```

**PyAuth (Internal Corporate SSO):**
```bash
AUTH_PROVIDER=pyauth
PYAUTH_CLIENT_ID=your-client-id
PYAUTH_CLIENT_SECRET=your-client-secret
PYAUTH_AUTHORIZATION_ENDPOINT=https://auth.corp.com/authorize
PYAUTH_TOKEN_ENDPOINT=https://auth.corp.com/token
PYAUTH_USERINFO_ENDPOINT=https://auth.corp.com/userinfo
PYAUTH_REDIRECT_URI=http://localhost:8080/auth/callback
# Optional: API keys endpoint for centralized key management
PYAUTH_API_KEYS_ENDPOINT=https://api.corp.com/keys
```

#### Session Storage (Production)

For production deployments with multiple instances:

**Redis:**
```bash
AUTH_SESSION_STORE=redis
AUTH_REDIS_URL=redis://localhost:6379/0
```

**Oracle:**
```bash
AUTH_SESSION_STORE=oracle
AUTH_ORACLE_DSN=user/password@host:1521/service
AUTH_ORACLE_TABLE=auth_sessions
```

See `gui/auth/session_store.py` for Oracle table schema.

#### Protected Pages

When authentication is enabled:
- **Home page** (`/`): Always public
- **Input, Analyze, Results, Compare, LLM Judge**: Require login

API keys can be provided via:
1. Environment variables (traditional)
2. OAuth session (via API keys endpoint) - keys are injected automatically

### Embedding Models
- `Qwen/Qwen3-Embedding-0.6B` (Default, high quality)
- `sentence-transformers/all-MiniLM-L6-v2` (Lightweight, fast)
- `sentence-transformers/all-mpnet-base-v2` (Balanced)

### LLM Providers
- **OpenAI**: GPT-4o, GPT-4o-mini, o1-preview, o1-mini
- **Anthropic**: Claude Sonnet 4.5, Claude Opus 4.1, Claude Sonnet 4, Claude Haiku 4.5, Claude Sonnet 3.5
- **Google**: Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash Exp

### Clustering Methods
- **UDIB** (Default): Upper-Bounded Deterministic Information Bottleneck
- **K-means**: Standard clustering with elbow method
- **Agglomerative**: Hierarchical clustering

## Output Metrics

### Semantic Faithfulness (SF)
Value between 0 and 1 indicating how well the answer aligns with the optimal information channel.

The following are *indicative* ranges for interpretation (the method does not specify hard thresholds):
- **SF ~ 0.85+**: Typically indicates high faithfulness
- **SF ~ 0.65-0.85**: Typically indicates moderate faithfulness
- **SF ~ below 0.65**: May indicate lower faithfulness (potential hallucination)

Actual interpretation depends on the specific use case and should be calibrated based on your domain.

### Semantic Entropy Production (SEP)
Measures irreversibility in the answer generation process.
- **Ṡ (System Entropy Change)**: H(A) - H(C), semantic expansion/compression
- **Ṡ_m (Dissipated Heat)**: SEP - Ṡ (see paper for formula)

## Technical Details

### Frontend: NiceGUI

[NiceGUI](https://nicegui.io/) is a Python-based web UI framework that allows building interactive web applications entirely in Python. Key characteristics:

- **Pure Python**: No JavaScript, HTML, or CSS required (though customization is supported)
- **Built on Vue.js + Quasar**: Under the hood, NiceGUI uses Vue.js for reactivity and Quasar for UI components, providing a rich set of Material Design widgets
- **Auto-refresh**: UI updates automatically when Python state changes
- **Async support**: Native support for async/await, enabling non-blocking LLM API calls

This application uses NiceGUI's component library for forms, cards, buttons, tabs, and interactive elements, styled with custom CSS for an Apple-inspired look.

### Backend: FastAPI

NiceGUI includes FastAPI as its built-in web server, providing:

- **High performance**: Async request handling with uvicorn
- **Session management**: User state persistence via `app.storage.user`
- **WebSocket communication**: Real-time UI updates between browser and server
- **Static file serving**: Automatic handling of CSS, JavaScript, and assets

The backend orchestrates:
- LLM API calls (OpenAI, Anthropic, Google Gemini) for paraphrase/answer generation
- SDM analysis engine for computing semantic faithfulness metrics
- Caching of embeddings, distributions, and LLM responses

### Core SDM Package

The `sdm_package/` directory contains the computational engine:

- **`SDM.py`**: Main `SemanticFaithfulnessAnalyzer` class that handles tokenization, embedding, and distribution computation
- **`compute_semantic_faithfulness.py`**: Implements the SF and SEP metric calculations using information-theoretic methods
- **`DIB_with_KL_upper_bound.py`**: Upper-Bounded Deterministic Information Bottleneck (UDIB) clustering algorithm for semantic topic discovery

This package can be used independently of the GUI for programmatic analysis (see main README for API examples).

### Visualization: Plotly

Interactive charts are rendered using [Plotly](https://plotly.com/python/), including:
- Probability distribution bar charts
- Transition matrix heatmaps
- F_S score comparisons across paraphrases

### Styling

Apple-inspired design implemented with CSS custom properties:
- Clean typography using system fonts (SF Pro-like)
- Subtle shadows and rounded corners
- Light/dark theme toggle with smooth transitions
- Frosted glass effect on navigation bar

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
  title={Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations},
  author={Halperin, Igor},
  journal={arXiv preprint arXiv:2512.05156},
  year={2025},
  url={https://arxiv.org/abs/2512.05156}
}
```

- arXiv: [https://arxiv.org/abs/2512.05156](https://arxiv.org/abs/2512.05156)
- SSRN: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5858022](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5858022)

## License

MIT License - See parent directory for details.

## Support

- **GitHub Issues**: https://github.com/ighalp/semantic-faithfulness-sdm/issues
- **Documentation**: https://github.com/ighalp/semantic-faithfulness-sdm/blob/main/docs/methodology.md

## Version

Current version: 2.1.0

### Changelog

**v2.1.0** (Current)
- Added OAuth2/OIDC authentication module for enterprise deployments (EXPERIMENTAL - not yet tested in production)
- Support for Okta, Azure AD, and custom PyAuth providers
- Pluggable session storage (Memory, Redis, Oracle)
- API keys injection from OAuth session
- Light/dark theme toggle with persistence
- Custom judge prompt editor for LLM-as-a-Judge
- Judge model selection (independent of answer generation model)

**v2.0.0**
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
