# Semantic Faithfulness Analyzer - GUI Application

A browser-based interface for computing semantic faithfulness metrics for Large Language Model (LLM) responses using information-theoretic methods.

## Features

- **Interactive Web Interface** - Clean, modern UI built with NiceGUI 3.x
- **Complete Analysis Pipeline** - From text input to computed metrics
- **Real-time Progress Tracking** - 4-stage progress visualization (Tokenization → Embedding → Clustering → Optimization)
- **Interactive Visualizations** - Plotly charts for probability distributions and transition matrices
- **Multiple Input Methods** - Manual text input or JSON file upload
- **Export Functionality** - Download results as JSON
- **Example Data** - Built-in example for quick testing
- **Configurable Parameters** - Customize embedding models, clustering methods, and optimization settings

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

   This will install:
   - NiceGUI 3.x (web framework)
   - Plotly (interactive visualizations)
   - SentenceTransformers (embeddings)
   - scikit-learn (clustering)
   - All SDM package dependencies

## Usage

### Starting the Application

```bash
python app.py
```

The application will:
- Start a local web server on **http://localhost:8080**
- Automatically open your default browser
- Be accessible from other devices on your local network

### Workflow

1. **Input Page** (`/input`)
   - Enter Question, Context, and Answer text manually
   - OR upload a JSON file with QCA triplet
   - OR click "Load Example" for sample data
   - Configure advanced settings (optional):
     - Embedding model
     - Clustering method (UDIB/k-means/agglomerative)
     - Tolerance and max iterations
   - Click "Analyze" to proceed

2. **Analysis Page** (`/analyze`)
   - Analysis runs automatically
   - Watch real-time progress through 4 stages:
     - Tokenization
     - Embedding
     - Clustering
     - Optimization
   - View detailed log output
   - Auto-navigates to results when complete

3. **Results Page** (`/results`)
   - View key metrics:
     - **F_S** (Semantic Faithfulness): 0-1 scale, higher is better
     - **SEP_total** (Total Entropy Production)
     - **SEP_system** (System Entropy Production)
     - **H(Q), H(C), H(A)** (Entropy values in bits)
   - Explore visualizations:
     - **Distributions** tab: Probability distributions over semantic topics
     - **Matrices** tab: Q* and A* transition matrix heatmaps
     - **Statistics** tab: Convergence details and sentence counts
   - Export results as JSON
   - Start new analysis

## Input Format

### Manual Input
- **Question**: The query or question text
- **Context**: The source context or document
- **Answer**: The LLM-generated answer to evaluate

### JSON File Upload

```json
{
  "question": "What were the company's main financial highlights in Q3 2023?",
  "context": "Apple Inc. today announced...",
  "answer": "Apple reported Q3 2023 revenue of $81.8 billion..."
}
```

## Configuration Options

### Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2` (Default, fast)
- `sentence-transformers/all-mpnet-base-v2` (More accurate)
- `Qwen/Qwen3-Embedding-0.6B` (Larger model)

### Clustering Methods
- **udib** (Default): Upper-Bounded Deterministic Information Bottleneck
- **kmeans**: K-means clustering
- **agglomerative**: Hierarchical agglomerative clustering

### Optimization Parameters
- **Tolerance**: Convergence threshold (default: 1e-7)
- **Max Iterations**: Maximum optimization iterations (default: 100)

## Output Metrics

### Semantic Faithfulness (F_S)
Value between 0 and 1 indicating how well the answer aligns with the optimal information channel.
- **Higher is better** (closer to 1.0 = more faithful)

### Semantic Entropy Production (SEP)
Measures irreversibility in the answer generation process.
- **SEP_total**: KL divergence from optimal channel
- **SEP_system**: H(A) - H(C), difference between answer and context entropy

### Entropy Values
- **H(Q)**: Information content of question (bits)
- **H(C)**: Information content of context (bits)
- **H(A)**: Information content of answer (bits)

## Technical Details

### Architecture
- **Frontend**: NiceGUI (Vue.js 3 + Tailwind CSS 4)
- **Backend**: FastAPI (built into NiceGUI)
- **Visualization**: Plotly
- **Analysis**: Async/await for non-blocking execution

### Processing Pipeline
1. **Tokenization**: Split text into sentences using NLTK
2. **Embedding**: Convert sentences to vectors using SentenceTransformer
3. **Clustering**: Group sentences into semantic topics
4. **Optimization**: Compute optimal transition matrices (Q*, A*) via convex optimization

### Session Management
- User data stored in browser sessions
- Each analysis isolated per session
- No persistent storage (sessions cleared on browser close)

## Troubleshooting

### Port Already in Use
If port 8080 is occupied, edit `app.py` and change:
```python
APP_PORT = 8080  # Change to another port like 8081
```

### Memory Issues
For very long documents:
- Break context into smaller chunks
- Reduce number of clusters (manually set in advanced config)
- Use a smaller embedding model

### Slow Performance
- First run downloads embedding model (~90MB)
- Subsequent runs use cached model
- Analysis time depends on text length and cluster count
- Typical: 10-30 seconds for moderate-length texts

## Development

### Project Structure
```
gui/
├── app.py                 # Main application entry point
├── requirements-gui.txt   # Python dependencies
├── pages/                 # Page modules
│   ├── home.py           # Landing page
│   ├── input_page.py     # Input form
│   ├── analyze.py        # Analysis execution
│   └── results.py        # Results display
├── services/             # Business logic
│   └── analysis_service.py  # SDM pipeline wrapper
├── components/           # Reusable UI components (future)
├── utils/               # Helper functions (future)
└── static/              # CSS and assets (future)
```

### Running in Development Mode
The app runs with auto-reload enabled by default. Code changes are reflected automatically.

## Cross-Platform Compatibility

Tested on:
- macOS (Apple Silicon & Intel)
- Linux
- Windows 10/11

## Citation

If you use this tool in your research, please cite:

```
Halperin, I. (2025). Information-Theoretic Faithfulness Metrics for Large Language Models.
arXiv preprint arXiv:XXXX.XXXXX
```

## License

See parent directory for license information.

## Support

For issues or questions:
- GitHub Issues: https://github.com/ighalp/semantic-faithfulness-sdm/issues
- Documentation: [Coming soon]

## Version

Current version: 1.0.0 (MVP)

## Roadmap

Future enhancements:
- Batch processing for multiple QCA triplets
- Multi-triplet comparison view
- CSV and PDF export options
- Configuration presets (save/load settings)
- Topic analysis display
- Convergence plots
- Enhanced error handling and validation
