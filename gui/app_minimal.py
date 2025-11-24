"""
Minimal test app
"""
import sys
from pathlib import Path

# Add parent directory to path to import sdm_package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nicegui import ui

# Application configuration
APP_TITLE = "Semantic Faithfulness Analyzer"
APP_PORT = 8080
STORAGE_SECRET = "semantic-faithfulness-secret-key-2024"

# Route definitions
@ui.page('/')
def index():
    """Home page"""
    with ui.column().classes('w-full max-w-6xl mx-auto p-8'):
        ui.label('Semantic Faithfulness Analyzer').classes('text-h4 mb-6')
        ui.label('Minimal test - Server is working!').classes('text-subtitle1')

# Application startup
if __name__ in {"__main__", "__mp_main__"}:
    print("Starting minimal app...")
    ui.run(
        title=APP_TITLE,
        port=APP_PORT,
        storage_secret=STORAGE_SECRET,
        reload=False,
        show=False
    )
