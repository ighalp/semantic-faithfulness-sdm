"""
Home page - Landing page with welcome message and overview
"""

from nicegui import ui

def create():
    """Create the home page content"""
    with ui.column().classes('w-full max-w-6xl mx-auto p-8'):
        # Welcome header with logo
        with ui.row().classes('w-full items-center gap-8 mb-8'):
            with ui.column().classes('flex-1 gap-2'):
                ui.label('Paraphrase Me').classes('text-h2 font-bold')
                ui.label('LLM Semantic Assistant - Information-Theoretic Evaluation').classes('text-h5 text-grey-7')
            ui.image('/static/LLM_Maxwell_demon.png').classes('w-56 h-auto rounded-lg shadow-md')

        # What is this tool card
        with ui.card().classes('w-full p-6 mb-6 bg-blue-50'):
            ui.label('What Does This Tool Do?').classes('text-h5 mb-4')
            ui.label(
                'Paraphrase Me helps you evaluate faithfulness of LLM answer or answers to your questions and context '
                'documents - in short, faithfulness of an LLM to its task. Our tool allows you to find answers to such '
                'questions while operating with the LLM as a black box.'
            ).classes('mb-4')
            ui.label(
                'Our tool uses paraphrasings of the initial user question (prompt) in various semantically equivalent ways '
                'to explore the amount of variability produced in answers by paraphrasing of questions. This tool uses '
                'methods from information theory and physics to find the most faithful answer from the set of generated answers.'
            ).classes('mb-4')
            ui.label(
                'The recommendation obtained by our method using our quantitative metrics called Semantic Faithfulness (F_S) '
                'and Semantic Entropy Production (SEP) can then be validated qualitatively using the LLM-as-a-Judge technique, '
                'or human evaluation. The tool can be used for tasks such as LLM summarizations as an automatic selector of '
                'the best answer among a set of generated or externally provided multiple answers.'
            ).classes('mb-4')

        # Overview card
        with ui.card().classes('w-full p-6 mb-6'):
            ui.label('Key Features').classes('text-h5 mb-4')

            with ui.column().classes('gap-2'):
                ui.label('• Semantic Faithfulness F_S: Measures alignment with optimal information channel')
                ui.label('• Semantic Entropy Production (SEP): Quantifies irreversibility in answer generation')
                ui.label('• Topic Discovery: Automatic semantic topic identification via UDIB clustering')
                ui.label('• Interactive Visualizations: Explore distributions, matrices, and metrics')

        # Method explanation card
        with ui.card().classes('w-full p-6 mb-6 bg-green-50'):
            ui.label('How It Works').classes('text-h5 mb-4')
            ui.label(
                'The method embeds your Question, Context, and Answer into a semantic space, then discovers latent topics '
                'using UDIB clustering. It computes probability distributions over these topics and measures how well '
                'the joint distribution P(Q,C,A) can be compressed into an optimal channel P(Q)→P(C|Q)→P(A|C).'
            ).classes('mb-2')
            ui.label(
                'A high F_S score means the answer is informationally consistent with the context given the question, '
                'indicating faithful paraphrasing. A low F_S score suggests the LLM may have hallucinated or ignored context.'
            ).classes('mb-2')

        # Quick start guide
        with ui.card().classes('w-full p-6 mb-6'):
            ui.label('Quick Start Guide').classes('text-h5 mb-4')
            with ui.stepper().props('vertical').classes('w-full') as stepper:
                with ui.step('Input'):
                    ui.label('Enter or paste your Question, Context, and Answer. You can also provide a URL as context.')
                    ui.button('Go to Input', on_click=lambda: ui.navigate.to('/input')).props('outline')

                with ui.step('Configure'):
                    ui.label('Choose embedding model and adjust clustering parameters (optional - defaults work well)')

                with ui.step('Analyze'):
                    ui.label('Click "Start Analysis" to run semantic embedding, clustering, and metric computation')

                with ui.step('Results'):
                    ui.label('View F_S metric, visualizations, and detailed topic distributions')

                with ui.step('Compare'):
                    ui.label('Compare answer variants side-by-side (Initial vs Lowest F_S by default)')
                    ui.button('Go to Compare', on_click=lambda: ui.navigate.to('/compare')).props('outline')

                with ui.step('LLM Judge'):
                    ui.label('Use LLM-as-a-Judge to evaluate and select the best answer, then export')
                    ui.button('Go to LLM Judge', on_click=lambda: ui.navigate.to('/judge')).props('outline')

        # Action buttons
        with ui.row().classes('gap-4 mt-8'):
            ui.button('Start New Analysis', on_click=lambda: ui.navigate.to('/input')).props('color=primary size=lg')
            ui.link(
                'View Documentation',
                'https://github.com/ighalp/semantic-faithfulness-sdm/blob/main/gui/README.md',
                new_tab=True
            ).classes('q-btn q-btn--outline q-btn--rectangle text-primary q-btn--standard q-btn--size-lg no-underline')

        # Documentation links card
        with ui.card().classes('w-full p-6 mt-6 bg-purple-50'):
            ui.label('Documentation & Resources').classes('text-h6 mb-4')
            with ui.column().classes('gap-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('description', color='purple')
                    ui.link(
                        'GUI Application Guide',
                        'https://github.com/ighalp/semantic-faithfulness-sdm/blob/main/gui/README.md',
                        new_tab=True
                    ).classes('text-primary')
                with ui.row().classes('items-center gap-2'):
                    ui.icon('architecture', color='purple')
                    ui.link(
                        'System Architecture',
                        'https://github.com/ighalp/semantic-faithfulness-sdm/blob/main/docs/architecture.md',
                        new_tab=True
                    ).classes('text-primary')
                with ui.row().classes('items-center gap-2'):
                    ui.icon('science', color='purple')
                    ui.link(
                        'Methodology & Theory',
                        'https://github.com/ighalp/semantic-faithfulness-sdm/blob/main/docs/methodology.md',
                        new_tab=True
                    ).classes('text-primary')
                with ui.row().classes('items-center gap-2'):
                    ui.icon('code', color='purple')
                    ui.link(
                        'API Reference & Examples',
                        'https://github.com/ighalp/semantic-faithfulness-sdm/blob/main/README.md',
                        new_tab=True
                    ).classes('text-primary')

        # Footer info
        ui.separator().classes('my-8')
        with ui.row().classes('justify-center gap-8 text-grey-6'):
            ui.label('Version 2.0.0')
            ui.label('•')
            ui.label('Python 3.10+')
            ui.label('•')
            ui.link('GitHub', 'https://github.com/ighalp/semantic-faithfulness-sdm', new_tab=True)
