"""
Home page - Landing page with welcome message and overview
"""

from nicegui import ui

def create():
    """Create the home page content"""
    with ui.column().classes('w-full max-w-4xl mx-auto p-8'):
        # Welcome header
        ui.label('Semantic Faithfulness Analyzer').classes('text-h3 font-bold mb-4')
        ui.label('Information-Theoretic Metrics for LLM Evaluation').classes('text-h6 text-grey-7 mb-8')

        # Overview card
        with ui.card().classes('w-full p-6 mb-6'):
            ui.label('Overview').classes('text-h5 mb-4')
            ui.label(
                'This application provides tools to evaluate the faithfulness of Large Language Model (LLM) '
                'responses to provided context using information-theoretic metrics.'
            ).classes('mb-4')

            with ui.column().classes('gap-2'):
                ui.label('Key Features:').classes('font-bold')
                ui.label('• Semantic Faithfulness (F_S): Measures alignment with optimal information channel')
                ui.label('• Semantic Entropy Production (SEP): Quantifies irreversibility in answer generation')
                ui.label('• Topic Discovery: Automatic semantic topic identification via UDIB clustering')
                ui.label('• Interactive Visualizations: Explore distributions, matrices, and metrics')

        # Quick start guide
        with ui.card().classes('w-full p-6 mb-6'):
            ui.label('Quick Start').classes('text-h5 mb-4')
            with ui.stepper().props('vertical').classes('w-full') as stepper:
                with ui.step('Input'):
                    ui.label('Enter or upload your Question-Context-Answer triplet')
                    ui.button('Go to Input', on_click=lambda: ui.navigate.to('/input')).props('outline')

                with ui.step('Configure'):
                    ui.label('Adjust analysis parameters (optional)')

                with ui.step('Analyze'):
                    ui.label('Run the semantic faithfulness analysis')

                with ui.step('Results'):
                    ui.label('View metrics and visualizations')

        # Action buttons
        with ui.row().classes('gap-4 mt-8'):
            ui.button('Start New Analysis', on_click=lambda: ui.navigate.to('/input')).props('color=primary size=lg')
            ui.button('View Documentation', on_click=lambda: ui.notify('Documentation coming soon')).props('outline size=lg')

        # Footer info
        ui.separator().classes('my-8')
        with ui.row().classes('justify-center gap-8 text-grey-6'):
            ui.label('Version 1.0.0')
            ui.label('•')
            ui.label('Python 3.10+')
            ui.label('•')
            ui.link('GitHub', 'https://github.com/ighalp/semantic-faithfulness-sdm', new_tab=True)
