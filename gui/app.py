"""
Semantic Faithfulness GUI Application
Main entry point for the browser-based interface
"""

import sys
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

print("="*70)
print("PRE-LOADING MODULES FOR CACHE MODE")
print("="*70)
print("Loading only modules needed for cached analysis...")
sys.stdout.flush()

# For cache mode, we only need:
# - numpy, scipy (for probability distributions and entropy)
# - compute_semantic_faithfulness (for optimization)
# - matplotlib (for plotting)
# - NiceGUI (web framework)

print("\n[1/4] Importing numpy and scipy...")
sys.stdout.flush()
import numpy as np
from scipy.stats import entropy
import json
print("      ✓ Core libraries imported")
sys.stdout.flush()

print("[2/4] Importing compute_semantic_faithfulness (optimization only)...")
sys.stdout.flush()
# Import the function directly without loading sdm_package/__init__.py
# This avoids triggering imports of SDM.py which has heavy dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "compute_semantic_faithfulness_module",
    str(Path(__file__).parent.parent / "sdm_package" / "compute_semantic_faithfulness.py")
)
csf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(csf_module)
compute_semantic_faithfulness = csf_module.compute_semantic_faithfulness
print("      ✓ Optimization function imported (bypassing package __init__.py)")
sys.stdout.flush()

print("[3/4] Importing matplotlib...")
sys.stdout.flush()
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side
import matplotlib.pyplot as plt
print("      ✓ Plotting library imported")
sys.stdout.flush()

print("[4/4] Importing NiceGUI framework...")
sys.stdout.flush()
from nicegui import ui, app
print("      ✓ Web framework imported")
sys.stdout.flush()

print("\n" + "="*70)
print("CACHE MODE STARTUP COMPLETE")
print("="*70)
print("Note: PyTorch/transformers NOT loaded (cache mode uses pre-computed data)")
print("Starting web server...\n")
sys.stdout.flush()

# Pages imported lazily in route functions (lightweight imports only)

# Application configuration
APP_TITLE = "Semantic Faithfulness Analyzer"
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
