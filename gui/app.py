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

# =============================================================================
# APPLE-STYLE THEME
# =============================================================================
# Clean, minimal design inspired by Apple's website aesthetics
# - SF Pro-like fonts (system fonts)
# - Subtle gray palette with blue accents
# - Rounded corners, soft shadows
# - Generous whitespace

APPLE_STYLE_CSS = """
<style>
/* Import system fonts similar to SF Pro */
/* Light theme (default) */
:root {
    --apple-bg: #fbfbfd;
    --apple-bg-secondary: #f5f5f7;
    --apple-text: #1d1d1f;
    --apple-text-secondary: #86868b;
    --apple-blue: #0071e3;
    --apple-blue-hover: #0077ed;
    --apple-border: #d2d2d7;
    --apple-card-bg: #ffffff;
    --apple-nav-bg: rgba(251, 251, 253, 0.8);
    --apple-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    --apple-shadow-hover: 0 8px 24px rgba(0, 0, 0, 0.12);
    --apple-radius: 12px;
    --apple-radius-sm: 8px;
}

/* Dark theme */
body.body--dark {
    --apple-bg: #1d1d1f;
    --apple-bg-secondary: #2d2d2f;
    --apple-text: #f5f5f7;
    --apple-text-secondary: #a1a1a6;
    --apple-blue: #2997ff;
    --apple-blue-hover: #40a9ff;
    --apple-border: #424245;
    --apple-card-bg: #2d2d2f;
    --apple-nav-bg: rgba(29, 29, 31, 0.8);
    --apple-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    --apple-shadow-hover: 0 8px 24px rgba(0, 0, 0, 0.4);
}

body.body--dark {
    background-color: var(--apple-bg) !important;
    color: var(--apple-text) !important;
}

body.body--dark .q-header {
    background: var(--apple-nav-bg) !important;
    border-bottom-color: var(--apple-border) !important;
}

body.body--dark .q-card {
    background: var(--apple-card-bg) !important;
    border-color: var(--apple-border) !important;
}

body.body--dark .text-h4,
body.body--dark .text-h5,
body.body--dark .text-h6,
body.body--dark .q-header .text-h5 {
    color: var(--apple-text) !important;
}

body.body--dark .q-header a {
    color: var(--apple-text) !important;
}

body.body--dark .text-subtitle1,
body.body--dark .text-subtitle2,
body.body--dark .text-caption {
    color: var(--apple-text-secondary) !important;
}

body.body--dark .bg-blue-50,
body.body--dark .bg-blue-100 {
    background: #1a3a5c !important;
}

body.body--dark .bg-green-50 {
    background: #1a3d2e !important;
}

body.body--dark .bg-yellow-50 {
    background: #3d3a1a !important;
}

body.body--dark .bg-amber-50 {
    background: #3d351a !important;
}

body.body--dark .bg-purple-50 {
    background: #2d1a3d !important;
}

body.body--dark .bg-grey-1,
body.body--dark .bg-grey-50 {
    background: var(--apple-bg-secondary) !important;
}

body.body--dark .q-field--outlined .q-field__control:before {
    border-color: var(--apple-border) !important;
}

body.body--dark .q-select .q-field__native,
body.body--dark .q-input .q-field__native {
    color: var(--apple-text) !important;
}

body.body--dark .q-separator {
    background: var(--apple-border) !important;
}

body.body--dark .q-tab {
    color: var(--apple-text-secondary) !important;
}

body.body--dark .q-stepper__dot {
    background: var(--apple-border) !important;
}

body.body--dark .q-stepper__tab--active .q-stepper__dot {
    background: var(--apple-blue) !important;
}

body.body--dark ::-webkit-scrollbar-thumb {
    background: var(--apple-border);
}

body.body--dark ::-webkit-scrollbar-thumb:hover {
    background: var(--apple-text-secondary);
}

/* Global font and background */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    background-color: var(--apple-bg) !important;
    color: var(--apple-text) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Navigation bar - Apple frosted glass effect */
.q-header {
    background: var(--apple-nav-bg) !important;
    backdrop-filter: saturate(180%) blur(20px) !important;
    -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
    border-bottom: 1px solid var(--apple-border) !important;
    box-shadow: none !important;
}

.q-header .q-toolbar {
    min-height: 48px !important;
}

/* Nav links */
.q-header a {
    color: var(--apple-text) !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    opacity: 0.8;
    transition: opacity 0.2s ease !important;
}

.q-header a:hover {
    opacity: 1;
    text-decoration: none !important;
}

/* App title in nav */
.q-header .text-h5 {
    color: var(--apple-text) !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}

/* Cards - Apple style */
.q-card {
    background: var(--apple-card-bg) !important;
    border-radius: var(--apple-radius) !important;
    box-shadow: var(--apple-shadow) !important;
    border: 1px solid var(--apple-border) !important;
    transition: box-shadow 0.3s ease, transform 0.3s ease !important;
}

.q-card:hover {
    box-shadow: var(--apple-shadow-hover) !important;
}

/* Buttons - Apple style */
.q-btn {
    border-radius: var(--apple-radius-sm) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    letter-spacing: -0.01em;
    text-transform: none !important;
    transition: all 0.2s ease !important;
}

/* Primary buttons */
.q-btn--standard.bg-primary {
    background: var(--apple-blue) !important;
    box-shadow: none !important;
}

.q-btn--standard.bg-primary:hover {
    background: var(--apple-blue-hover) !important;
    transform: scale(1.02);
}

/* Outline buttons */
.q-btn--outline {
    border-color: var(--apple-border) !important;
    color: var(--apple-blue) !important;
}

.q-btn--outline:hover {
    background: var(--apple-bg-secondary) !important;
}

/* Input fields */
.q-field__control {
    border-radius: var(--apple-radius-sm) !important;
}

.q-field--outlined .q-field__control:before {
    border-color: var(--apple-border) !important;
}

.q-field--outlined .q-field__control:hover:before {
    border-color: var(--apple-blue) !important;
}

.q-field--focused .q-field__control:after {
    border-color: var(--apple-blue) !important;
}

/* Select dropdowns */
.q-select .q-field__native {
    color: var(--apple-text) !important;
}

/* Labels and text */
.text-h4, .text-h5, .text-h6 {
    color: var(--apple-text) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

.text-subtitle1, .text-subtitle2 {
    color: var(--apple-text-secondary) !important;
    font-weight: 400 !important;
}

.text-caption {
    color: var(--apple-text-secondary) !important;
    font-size: 12px !important;
}

/* Colored backgrounds - softer Apple palette */
.bg-blue-50, .bg-blue-100 {
    background: #e8f4fd !important;
}

.bg-green-50 {
    background: #e8f8ef !important;
}

.bg-yellow-50 {
    background: #fef8e8 !important;
}

.bg-amber-50 {
    background: #fef6e8 !important;
}

.bg-purple-50 {
    background: #f3e8fd !important;
}

.bg-grey-1, .bg-grey-50 {
    background: var(--apple-bg-secondary) !important;
}

/* Separators */
.q-separator {
    background: var(--apple-border) !important;
}

/* Tabs */
.q-tab {
    color: var(--apple-text-secondary) !important;
    font-weight: 500 !important;
}

.q-tab--active {
    color: var(--apple-blue) !important;
}

/* Badges */
.q-badge {
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 11px !important;
}

/* Notifications */
.q-notification {
    border-radius: var(--apple-radius) !important;
    font-family: inherit !important;
}

/* Scrollbars - minimal Apple style */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: var(--apple-border);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--apple-text-secondary);
}

/* Stepper */
.q-stepper {
    background: transparent !important;
    box-shadow: none !important;
}

.q-stepper__tab--active .q-stepper__dot {
    background: var(--apple-blue) !important;
}

/* Spinner */
.q-spinner {
    color: var(--apple-blue) !important;
}

/* Links in content */
a:not(.q-btn):not(.q-tab) {
    color: var(--apple-blue) !important;
    text-decoration: none !important;
}

a:not(.q-btn):not(.q-tab):hover {
    text-decoration: underline !important;
}

/* Grid improvements */
.q-grid {
    gap: 16px !important;
}

/* Page content container */
.max-w-6xl, .max-w-7xl {
    max-width: 1200px !important;
}

/* Plotly chart containers */
.js-plotly-plot {
    border-radius: var(--apple-radius) !important;
    overflow: hidden;
}
</style>
"""

# =============================================================================
# ORIGINAL THEME (commented out for reference)
# =============================================================================
# # Main navigation bar component - ORIGINAL
# def create_navbar_original():
#     """Create the main navigation bar - Original style"""
#     with ui.header().classes('items-center justify-between'):
#         ui.label(APP_TITLE).classes('text-h5 font-bold')
#         with ui.row().classes('gap-4'):
#             ui.link('Home', '/').classes('text-white no-underline')
#             ui.link('Input', '/input').classes('text-white no-underline')
#             ui.link('Analyze', '/analyze').classes('text-white no-underline')
#             ui.link('Results', '/results').classes('text-white no-underline')
#             ui.link('Compare', '/compare').classes('text-white no-underline')
#             ui.link('LLM Judge', '/judge').classes('text-white no-underline')

# Main navigation bar component - Apple Style
def create_navbar():
    """Create the main navigation bar - Apple style"""
    # Inject Apple CSS on first page load
    ui.add_head_html(APPLE_STYLE_CSS)

    # Initialize dark mode from user storage (persists across sessions)
    dark_mode = ui.dark_mode()

    # Restore user's theme preference
    stored_dark = app.storage.user.get('dark_mode', False)
    if stored_dark:
        dark_mode.enable()

    with ui.header().classes('items-center justify-between px-8'):
        ui.label(APP_TITLE).classes('text-h5 font-bold')
        with ui.row().classes('gap-6 items-center'):
            ui.link('Home', '/').classes('no-underline')
            ui.link('Input', '/input').classes('no-underline')
            ui.link('Analyze', '/analyze').classes('no-underline')
            ui.link('Results', '/results').classes('no-underline')
            ui.link('Compare', '/compare').classes('no-underline')
            ui.link('LLM Judge', '/judge').classes('no-underline')

            # Theme toggle with sun/moon icons
            with ui.row().classes('items-center gap-1 ml-4'):
                ui.icon('light_mode', size='sm').classes('text-amber-500')

                def toggle_theme(e):
                    if e.value:
                        dark_mode.enable()
                    else:
                        dark_mode.disable()
                    # Persist preference
                    app.storage.user['dark_mode'] = e.value

                ui.switch(value=stored_dark, on_change=toggle_theme).props('dense color=grey-8')
                ui.icon('dark_mode', size='sm').classes('text-blue-300')

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

@ui.page('/judge')
def judge_route():
    """LLM-as-a-Judge comparison page"""
    create_navbar()
    from pages import judge
    judge.create()

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
