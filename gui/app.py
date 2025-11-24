"""
Semantic Faithfulness GUI Application
Main entry point for the browser-based interface
"""

# CRITICAL: Set multiprocessing start method to 'spawn' BEFORE any other imports
# This fixes PyTorch threading issues on macOS by avoiding fork()-based multiprocessing
# which corrupts PyTorch's internal thread pools
import multiprocessing as mp
import sys

# Only set start method if running as main process (not in spawned worker)
if __name__ in {"__main__", "__mp_main__"}:
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set (e.g., in tests or reloads)
        pass

from pathlib import Path

# Load environment variables from .env files
from dotenv import load_dotenv
# Try to load .env from multiple locations
# 1. Current directory (gui/)
load_dotenv(Path(__file__).parent / '.env')
# 2. Parent directory (semantic-faithfulness-sdm/)
load_dotenv(Path(__file__).parent.parent / '.env')
# 3. Parent's parent directory (LLMs/)
load_dotenv(Path(__file__).parent.parent.parent / '.env')

# Add parent directory to path to import sdm_package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables to optimize PyTorch performance (macOS compatibility)
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import NiceGUI framework
from nicegui import ui, app

print("="*70)
print("PARAPHRASE ME - LLM SEMANTIC ASSISTANT")
print("="*70)
print("Starting web server on http://localhost:8080")
print("\nEmbedding models will be loaded on-demand during analysis.")
print("First analysis with Qwen3 takes ~30 seconds, subsequent runs use cache.\n")
sys.stdout.flush()

# Pages imported lazily in route functions

# Application configuration
APP_TITLE = "Paraphrase Me - LLM Semantic Assistant"
APP_PORT = 8080
STORAGE_SECRET = "semantic-faithfulness-secret-key-2024"  # For session management

# Main navigation bar component
def create_navbar():
    """Create the main navigation bar"""
    with ui.header().classes('items-center justify-between'):
        ui.label(APP_TITLE).classes('text-h5 font-bold')
        with ui.row().classes('gap-4'):
            ui.link('Home', '/').classes('text-white no-underline')
            ui.link('Input', '/input').classes('text-white no-underline')
            ui.link('Analyze', '/analyze').classes('text-white no-underline')
            ui.link('Results', '/results').classes('text-white no-underline')
            ui.link('Compare', '/compare').classes('text-white no-underline')

# Route definitions
@ui.page('/')
def index():
    """Home page"""
    create_navbar()
    from pages import home
    home.create()

@ui.page('/input')
def input_route():
    """Input page for QCA triplets"""
    create_navbar()
    from pages import input_page
    input_page.create()

@ui.page('/analyze')
def analyze_route():
    """Analysis execution page"""
    create_navbar()
    from pages import analyze
    analyze.create()

@ui.page('/results')
def results_route():
    """Results visualization page"""
    create_navbar()
    from pages import results
    results.create()

@ui.page('/compare')
def compare_route():
    """Multi-triplet comparison page"""
    create_navbar()
    from pages import compare
    compare.create()

# Application startup
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title=APP_TITLE,
        port=APP_PORT,
        storage_secret=STORAGE_SECRET,  # Enable user session storage
        reload=False,  # Disable auto-reload (not in IDE)
        show=False     # Don't auto-open browser (manual navigation to localhost:8080)
    )
