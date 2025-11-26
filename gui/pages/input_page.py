"""
Input page - QCA triplet input and configuration
"""

from nicegui import ui, app
import json
import sys
from pathlib import Path

# Add parent directory to path (imports moved to function level to avoid blocking page load)
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_cached_triplets():
    """Load available triplets from the cache file"""
    cache_file = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'distributions' / 'distributions_v2.json'
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return data.get('triplets', [])
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f'Error loading cache: {e}')
        return []


def create():
    """Create the input page content"""

    print("[DEBUG] input_page.create() START")
    import sys
    sys.stdout.flush()

    # Form state storage
    form_data = {
        'question': None,
        'context': None,
        'answer': None,
        'embedding_model': None,
        'clustering_method': None,
        'tolerance': None,
        'max_iter': None
    }

    with ui.column().classes('w-full max-w-6xl mx-auto p-8'):
        ui.label('Question-Context-Answer Input').classes('text-h4 mb-6')

        # Input method tabs
        with ui.tabs().classes('w-full') as tabs:
            tab_manual = ui.tab('Manual Input')
            tab_llm = ui.tab('LLM Generation')
            tab_file = ui.tab('File Upload')
            tab_cache = ui.tab('Load from Cache')

        with ui.tab_panels(tabs, value=tab_manual).classes('w-full'):
            # Manual input panel
            with ui.tab_panel(tab_manual):
                with ui.card().classes('w-full p-6'):
                    ui.label('Enter QCA Triplet').classes('text-h6 mb-4')

                    # Question input
                    form_data['question'] = ui.textarea(
                        'Question',
                        placeholder='Enter the question or query...'
                    ).classes('w-full').props('rows=3')

                    # Context input
                    form_data['context'] = ui.textarea(
                        'Context',
                        placeholder='Enter the source context or document...'
                    ).classes('w-full').props('rows=8')

                    # Answer input
                    form_data['answer'] = ui.textarea(
                        'Answer',
                        placeholder='Enter the LLM-generated answer...'
                    ).classes('w-full').props('rows=6')

            # LLM Generation panel
            with ui.tab_panel(tab_llm):
                with ui.card().classes('w-full p-6'):
                    ui.label('LLM-Powered Analysis').classes('text-h6 mb-2')
                    ui.label('Generate paraphrases and answers using LLM APIs').classes('text-subtitle2 text-grey-7 mb-4')

                    # Load previous session data from user storage
                    last_question = app.storage.user.get('llm_last_question', '')
                    last_url = app.storage.user.get('llm_last_url', '')
                    last_provider = app.storage.user.get('llm_last_provider', 'anthropic')
                    last_model = app.storage.user.get('llm_last_model', 'claude-sonnet-4-5-20250929')
                    last_context_source = app.storage.user.get('llm_last_context_source', 'text')
                    last_num_paraphrases = app.storage.user.get('llm_last_num_paraphrases', 3)
                    question_history = app.storage.user.get('llm_question_history', [])

                    # LLM Provider selection
                    with ui.row().classes('w-full gap-4'):
                        with ui.column().classes('flex-1'):
                            ui.label('LLM Provider').classes('text-subtitle2 font-bold mb-2')
                            form_data['llm_provider'] = ui.select(
                                options={'openai': 'OpenAI', 'anthropic': 'Anthropic', 'gemini': 'Google Gemini'},
                                value=last_provider
                            ).classes('w-full')

                        with ui.column().classes('flex-1'):
                            ui.label('Model').classes('text-subtitle2 font-bold mb-2')
                            form_data['llm_model'] = ui.select(
                                options={
                                    # OpenAI models (2025)
                                    'gpt-5': 'GPT-5 (Latest)',
                                    'gpt-5-codex': 'GPT-5 Codex',
                                    'gpt-4.5': 'GPT-4.5',
                                    'gpt-4o': 'GPT-4o',
                                    'gpt-4o-mini': 'GPT-4o Mini',
                                    'o1-preview': 'o1-preview (Reasoning)',
                                    'o1-mini': 'o1-mini (Reasoning)',
                                    # Anthropic models (2025)
                                    'claude-sonnet-4-5-20250929': 'Claude Sonnet 4.5 (Latest)',
                                    'claude-opus-4-1': 'Claude Opus 4.1',
                                    'claude-sonnet-4': 'Claude Sonnet 4',
                                    'claude-haiku-4-5': 'Claude Haiku 4.5',
                                    'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet',
                                    # Google Gemini models (2025)
                                    'gemini-3-pro-image': 'Gemini 3 Pro Image (Latest)',
                                    'gemini-2.5-pro': 'Gemini 2.5 Pro',
                                    'gemini-2.5-flash': 'Gemini 2.5 Flash',
                                    'gemini-2.5-flash-lite': 'Gemini 2.5 Flash-Lite',
                                    'gemini-2.0-flash-exp': 'Gemini 2.0 Flash (Experimental)'
                                },
                                value=last_model
                            ).classes('w-full')

                    # API Key input
                    with ui.row().classes('w-full gap-4 mt-4'):
                        with ui.column().classes('flex-1'):
                            ui.label('API Key (Optional)').classes('text-subtitle2 font-bold mb-2')
                            form_data['llm_api_key'] = ui.input(
                                placeholder='Enter your API key or leave blank to use environment variable...',
                                password=True,
                                password_toggle_button=True
                            ).classes('w-full')
                            ui.label('Leave blank to use OPENAI_API_KEY, ANTHROPIC_API_KEY/CLAUDE_API_KEY, or GEMINI_API_KEY from environment variables or .env file').classes('text-caption text-grey-6')

                        with ui.column().classes('flex-1'):
                            ui.label('Number of Paraphrases').classes('text-subtitle2 font-bold mb-2')
                            form_data['num_paraphrases'] = ui.number(
                                value=last_num_paraphrases,
                                min=1,
                                max=10,
                                step=1
                            ).classes('w-full')
                            ui.label('Will generate N+1 total prompts (original + paraphrases)').classes('text-caption text-grey-6')

                    # Cache control
                    with ui.row().classes('w-full gap-4 mt-4'):
                        with ui.column().classes('w-full'):
                            form_data['force_regenerate'] = ui.checkbox(
                                'Force regenerate all data (bypass cache)',
                                value=False
                            )
                            ui.label('When checked, all paraphrases, answers, embeddings, and F_S scores will be regenerated even if cached versions exist').classes('text-caption text-grey-6')

                    ui.separator().classes('my-4')

                    # Embedding Model Selection (moved from Advanced Config for better visibility)
                    ui.label('Embedding Model').classes('text-subtitle2 font-bold mb-2')
                    form_data['embedding_model'] = ui.select(
                        options={
                            'Qwen/Qwen3-Embedding-0.6B': 'Qwen3 0.6B (Default - High Quality)',
                            'sentence-transformers/all-mpnet-base-v2': 'MPNet Base v2 (Fast & Accurate)',
                            'sentence-transformers/all-MiniLM-L6-v2': 'MiniLM L6 v2 (Fastest, Lower Quality)',
                            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2': 'Multilingual MPNet (Multi-language Support)',
                            'BAAI/bge-large-en-v1.5': 'BGE Large EN v1.5 (High Quality, Slower)',
                        },
                        value='Qwen/Qwen3-Embedding-0.6B'
                    ).classes('w-full')
                    ui.label('Model affects embedding quality and speed. Qwen3 (default) provides high quality embeddings. First load takes ~30 seconds.').classes('text-caption text-grey-6')

                    ui.separator().classes('my-4')

                    # Original Question
                    ui.label('Original Question').classes('text-subtitle2 font-bold mb-2')

                    # Recent questions dropdown (if history exists)
                    if question_history:
                        # Create dropdown options with truncated display text
                        history_options = {}
                        for i, q in enumerate(question_history[:10]):  # Show last 10 questions
                            # Truncate long questions for display
                            display_text = q if len(q) <= 80 else q[:77] + '...'
                            history_options[q] = f"{display_text}"

                        with ui.row().classes('w-full items-center gap-2 mb-2'):
                            ui.label('Recent questions:').classes('text-caption text-grey-6')
                            recent_select = ui.select(
                                options=history_options,
                                value=None,
                                with_input=False
                            ).classes('flex-1').props('dense outlined').style('font-size: 0.875rem')

                            def load_from_history(e):
                                # Use recent_select.value directly instead of event args
                                selected_question = recent_select.value
                                if selected_question:
                                    form_data['llm_question'].value = selected_question
                                    recent_select.value = None  # Reset dropdown

                            recent_select.on('update:model-value', load_from_history)

                    form_data['llm_question'] = ui.textarea(
                        placeholder='Enter the original question...',
                        value=last_question
                    ).classes('w-full').props('rows=3')

                    # Context Source Selection
                    ui.label('Context Source').classes('text-subtitle2 font-bold mt-4 mb-2')
                    form_data['context_source'] = ui.select(
                        options={
                            'url': 'Fetch from URL(s)',
                            'pdf': 'Upload PDF Document(s)',
                            'text': 'Manual Text Input'
                        },
                        value=last_context_source if last_context_source in ['url', 'pdf', 'text'] else 'url'
                    ).classes('w-full')

                    # Context input containers (shown/hidden based on selection)
                    # URL context (default - shown first)
                    context_url_container = ui.column().classes('w-full')
                    with context_url_container:
                        ui.label('Context URLs').classes('text-subtitle2 font-bold mt-4 mb-2')
                        form_data['context_url'] = ui.textarea(
                            placeholder='Enter one or more URLs (one per line):\nhttps://example.com/article1\nhttps://example.com/article2',
                            value=last_url
                        ).classes('w-full').props('rows=4')
                        ui.label('Enter one URL per line. Content from all URLs will be fetched and combined.').classes('text-caption text-grey-6')

                    # PDF context
                    context_pdf_container = ui.column().classes('w-full').style('display: none')
                    with context_pdf_container:
                        ui.label('Upload PDFs').classes('text-subtitle2 font-bold mt-4 mb-2')

                        # Store uploaded PDF texts
                        form_data['pdf_contexts'] = []
                        pdf_status_label = ui.label('No PDFs uploaded yet').classes('text-caption text-grey-6')

                        def handle_pdf_upload(e):
                            try:
                                import PyPDF2
                                import io
                                pdf_file = io.BytesIO(e.content.read())
                                pdf_reader = PyPDF2.PdfReader(pdf_file)
                                text = ''
                                for page in pdf_reader.pages:
                                    text += page.extract_text() + '\n'
                                form_data['pdf_contexts'].append(text.strip())
                                pdf_status_label.text = f'{len(form_data["pdf_contexts"])} PDF(s) uploaded, {sum(len(t) for t in form_data["pdf_contexts"])} total characters'
                                ui.notify(f'PDF loaded: {len(text)} characters', type='positive')
                            except Exception as ex:
                                ui.notify(f'PDF error: {str(ex)}', type='negative')

                        ui.upload(
                            on_upload=handle_pdf_upload,
                            auto_upload=True,
                            multiple=True
                        ).props('accept=.pdf').classes('w-full')

                        def clear_pdfs():
                            form_data['pdf_contexts'] = []
                            pdf_status_label.text = 'No PDFs uploaded yet'
                            ui.notify('PDFs cleared', type='info')

                        ui.button('Clear PDFs', on_click=clear_pdfs).props('flat size=sm')

                    # Text context
                    context_text_container = ui.column().classes('w-full').style('display: none')
                    with context_text_container:
                        ui.label('Context').classes('text-subtitle2 font-bold mt-4 mb-2')
                        form_data['llm_context'] = ui.textarea(
                            placeholder='Enter the source context or document...'
                        ).classes('w-full').props('rows=8')

                    # Update visibility based on selection
                    def update_context_source(e):
                        # Extract the actual value from the event args
                        # NiceGUI select can pass either a string or a dict
                        if isinstance(e.args, dict):
                            source = e.args.get('value')  # Get the key value
                        else:
                            source = e.args

                        print(f"[DEBUG] Context source changed to: {e.args}, extracted: {source}")
                        import sys
                        sys.stdout.flush()

                        # Since the select uses numeric indices (0, 1, 2), map them back to keys
                        source_map = {0: 'url', 1: 'pdf', 2: 'text'}
                        if isinstance(source, int):
                            source = source_map.get(source, 'url')

                        if source == 'url':
                            context_url_container.style('display: block')
                            context_pdf_container.style('display: none')
                            context_text_container.style('display: none')
                        elif source == 'pdf':
                            context_url_container.style('display: none')
                            context_pdf_container.style('display: block')
                            context_text_container.style('display: none')
                        elif source == 'text':
                            context_url_container.style('display: none')
                            context_pdf_container.style('display: none')
                            context_text_container.style('display: block')

                    form_data['context_source'].on('update:model-value', update_context_source)

                    # Initialize visibility based on last_context_source
                    if last_context_source == 'url' or last_context_source not in ['url', 'pdf', 'text']:
                        context_url_container.style('display: block')
                        context_pdf_container.style('display: none')
                        context_text_container.style('display: none')
                    elif last_context_source == 'pdf':
                        context_url_container.style('display: none')
                        context_pdf_container.style('display: block')
                        context_text_container.style('display: none')
                    elif last_context_source == 'text':
                        context_url_container.style('display: none')
                        context_pdf_container.style('display: none')
                        context_text_container.style('display: block')

                    # Info box
                    with ui.card().classes('w-full mt-4 bg-blue-50 border-l-4 border-blue-500'):
                        with ui.row().classes('items-start gap-4 p-4'):
                            ui.icon('info', size='sm').classes('text-blue-500')
                            with ui.column():
                                ui.label('Pipeline Overview').classes('text-subtitle2 font-bold')
                                ui.label('1. Generate N paraphrases of your question').classes('text-body2')
                                ui.label('2. Generate answers for each paraphrase').classes('text-body2')
                                ui.label('3. Compute embeddings and cluster').classes('text-body2')
                                ui.label('4. Compute semantic faithfulness F_S').classes('text-body2')
                                ui.label('Estimated time: 3-5 minutes for 3 paraphrases').classes('text-caption text-grey-7 mt-2')

                    # Generate & Analyze button
                    with ui.row().classes('justify-end mt-6'):
                        ui.button('Generate & Analyze', on_click=lambda: start_llm_pipeline(form_data)).props('color=primary size=lg icon=auto_awesome')

            # File upload panel
            with ui.tab_panel(tab_file):
                with ui.card().classes('w-full p-6'):
                    ui.label('Upload QCA Triplet (JSON)').classes('text-h6 mb-4')
                    ui.label('Upload a JSON file containing QCA triplet(s)').classes('text-subtitle2 text-grey-7 mb-4')

                    def handle_upload(e):
                        try:
                            content = e.content.read().decode('utf-8')
                            data = json.loads(content)

                            # Load data into form
                            if 'question' in data:
                                form_data['question'].value = data['question']
                            if 'context' in data:
                                form_data['context'].value = data['context']
                            if 'answer' in data:
                                form_data['answer'].value = data['answer']

                            ui.notify(f'Loaded: {e.name}', type='positive')
                        except json.JSONDecodeError:
                            ui.notify('Invalid JSON file', type='negative')
                        except Exception as ex:
                            ui.notify(f'Error: {str(ex)}', type='negative')

                    upload = ui.upload(
                        on_upload=handle_upload,
                        auto_upload=True
                    ).props('accept=.json').classes('w-full')

                    ui.label('Expected format:').classes('text-subtitle2 font-bold mt-4')
                    ui.code('''{
  "question": "Your question here",
  "context": "Your context here",
  "answer": "Your answer here"
}''', language='json').classes('w-full')

            # Cache loading panel
            with ui.tab_panel(tab_cache):
                with ui.card().classes('w-full p-6'):
                    ui.label('Load from Cached Distributions').classes('text-h6 mb-4')
                    ui.label('Select a pre-computed triplet (skip embedding/clustering)').classes('text-subtitle2 text-grey-7 mb-4')

                    # Load available triplets from cache (with error handling)
                    try:
                        cache_triplets = load_cached_triplets()
                    except Exception as e:
                        cache_triplets = []
                        print(f'[ERROR] Failed to load cached triplets: {e}')

                    if cache_triplets:
                        form_data['cached_triplet'] = ui.select(
                            label='Select Triplet',
                            options={t['prompt_id']: f"{t['prompt_id']} ({t['group']}, k={t['k']})" for t in cache_triplets},
                            value=cache_triplets[0]['prompt_id']
                        ).classes('w-full')

                        ui.label(f'Available: {len(cache_triplets)} cached triplets').classes('text-caption text-grey-6 mt-2')
                    else:
                        form_data['cached_triplet'] = ui.select(
                            label='Select Triplet',
                            options=[],
                            value=None
                        ).classes('w-full')
                        ui.label('No cached distributions found').classes('text-negative')
                        ui.label('Cache file path: ../data/cache/distributions/distributions_v2.json').classes('text-caption')

                    ui.label('Note: Using cached distributions - analysis will be fast (metrics computation only)').classes('text-body2 text-blue-7 mt-4')

        # Configuration panel
        with ui.expansion('Advanced Configuration', icon='settings').classes('w-full mt-6'):
            with ui.card().classes('w-full p-6'):
                with ui.row().classes('w-full gap-4'):
                    # Clustering method
                    with ui.column().classes('flex-1'):
                        ui.label('Clustering Method').classes('text-subtitle2 font-bold mb-2')
                        form_data['clustering_method'] = ui.select(
                            options=['udib', 'kmeans', 'agglomerative'],
                            value='udib'
                        ).classes('w-full')
                        ui.label('Algorithm for clustering embeddings into semantic concepts').classes('text-caption text-grey-6')

                # Optimization parameters
                with ui.row().classes('w-full gap-4 mt-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('Tolerance').classes('text-subtitle2 font-bold mb-2')
                        form_data['tolerance'] = ui.number(value=1e-7, format='%.1e').classes('w-full')

                    with ui.column().classes('flex-1'):
                        ui.label('Max Iterations').classes('text-subtitle2 font-bold mb-2')
                        form_data['max_iter'] = ui.number(value=100, min=1, max=500).classes('w-full')

        # Action buttons
        with ui.row().classes('gap-4 mt-8 justify-end'):
            ui.button('Clear', on_click=lambda: clear_form(form_data)).props('outline')
            ui.button('Load Example', on_click=lambda: load_example(form_data)).props('outline')
            ui.button('Analyze', on_click=lambda: start_analysis(form_data)).props('color=primary')


def clear_form(form_data):
    """Clear all input fields"""
    form_data['question'].value = ''
    form_data['context'].value = ''
    form_data['answer'].value = ''
    ui.notify('Form cleared')


def load_example(form_data):
    """Load an example QCA triplet"""
    # Example from a corporate filing summarization task
    form_data['question'].value = "What were the company's main financial highlights in Q3 2023?"

    form_data['context'].value = """Apple Inc. today announced financial results for its fiscal 2023 third quarter ended July 1, 2023.
The Company posted quarterly revenue of $81.8 billion, down 1 percent year over year, and quarterly earnings per
diluted share of $1.26, up 5 percent year over year. "We are happy to report that we had an all-time revenue record
in Services during the June quarter, driven by over 1 billion paid subscriptions, and we saw continued strength in
emerging markets, with June quarter revenue records in both India and Indonesia," said Tim Cook, Apple's CEO.
"From education to the environment, we are continuing to advance our values, while championing innovation that
enriches the lives of our customers and leaves the world better than we found it." The Company's board of directors
has declared a cash dividend of $0.24 per share of the Company's common stock."""

    form_data['answer'].value = """Apple reported Q3 2023 revenue of $81.8 billion (down 1% YoY) and EPS of $1.26
(up 5% YoY). The Services segment achieved an all-time revenue record with over 1 billion paid subscriptions.
Emerging markets showed strength with revenue records in India and Indonesia. The board declared a dividend of $0.24 per share."""

    ui.notify('Example loaded', type='positive')


def start_analysis(form_data):
    """Validate and store data, then navigate to analysis page"""

    # Debug logging
    print(f"[DEBUG] start_analysis called")
    print(f"[DEBUG] form_data keys: {list(form_data.keys())}")
    print(f"[DEBUG] 'cached_triplet' in form_data: {'cached_triplet' in form_data}")

    # Check if we're loading from cache
    print(f"[DEBUG] Checking cache mode...")
    is_cache_mode = 'cached_triplet' in form_data and form_data['cached_triplet']
    print(f"[DEBUG] is_cache_mode: {is_cache_mode}")

    # For cache mode, we don't need the services import at all!
    if not is_cache_mode:
        print(f"[DEBUG] Normal mode - importing services...")
        # Lazy import to avoid blocking page load
        from services import get_analysis_service
        print(f"[DEBUG] Services imported successfully")

    # Check if we're loading from cache
    if 'cached_triplet' in form_data and form_data['cached_triplet']:
        # Cache mode - load selected triplet
        selected_id = form_data['cached_triplet'].value
        cache_triplets = load_cached_triplets()
        selected_triplet = next((t for t in cache_triplets if t['prompt_id'] == selected_id), None)

        if not selected_triplet:
            ui.notify('Cached triplet not found', type='negative')
            return

        # Store cached distributions directly
        app.storage.user['cached_distributions'] = {
            'prompt_id': selected_triplet['prompt_id'],
            'p_q': selected_triplet['p_q'],
            'p_c': selected_triplet['p_c'],
            'p_a': selected_triplet['p_a'],
            'k': selected_triplet['k']
        }

        # Store analysis config (used for display only in cache mode)
        app.storage.user['analysis_config'] = {
            'embedding_model': 'cached',
            'clustering_method': 'cached',
            'tolerance': form_data['tolerance'].value,
            'max_iterations': int(form_data['max_iter'].value)
        }

        # Clear QCA triplet to signal cache mode
        app.storage.user['qca_triplet'] = None
        app.storage.user['analysis_results'] = None
        app.storage.user['analysis_status'] = 'pending_cache'

        print(f"[DEBUG] About to navigate to /analyze")
        ui.notify(f'Loaded cached distributions for {selected_id}', type='positive')
        ui.navigate.to('/analyze')
        print(f"[DEBUG] Navigation triggered")
        return

    # Normal mode - get text input
    print(f"[DEBUG] Normal mode - getting text input")
    question = (form_data['question'].value or '').strip()
    context = (form_data['context'].value or '').strip()
    answer = (form_data['answer'].value or '').strip()

    # Validate input
    from services import get_analysis_service
    service = get_analysis_service()
    is_valid, error_message = service.validate_input(question, context, answer)

    if not is_valid:
        ui.notify(error_message, type='negative')
        return

    # Store in session
    app.storage.user['qca_triplet'] = {
        'question': question,
        'context': context,
        'answer': answer
    }

    app.storage.user['analysis_config'] = {
        'embedding_model': form_data['embedding_model'].value,
        'clustering_method': form_data['clustering_method'].value,
        'tolerance': form_data['tolerance'].value,
        'max_iterations': int(form_data['max_iter'].value)
    }

    # Clear cache flag
    app.storage.user['cached_distributions'] = None
    app.storage.user['analysis_results'] = None
    app.storage.user['analysis_status'] = 'pending'

    ui.notify('Input validated', type='positive')
    ui.navigate.to('/analyze')


async def start_llm_pipeline(form_data):
    """Run the complete LLM-powered pipeline"""
    from pathlib import Path
    import sys
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Extract form data
    provider_value = form_data['llm_provider'].value
    model_value = form_data['llm_model'].value
    num_paraphrases = int(form_data['num_paraphrases'].value)

    # Save current inputs to user storage for next session
    app.storage.user['llm_last_provider'] = provider_value
    app.storage.user['llm_last_model'] = model_value
    app.storage.user['llm_last_num_paraphrases'] = num_paraphrases

    current_question = (form_data['llm_question'].value or '').strip()
    # DISABLED: Don't save to session to prevent auto-trigger on page load
    # app.storage.user['llm_last_question'] = current_question
    # app.storage.user['llm_last_context_source'] = form_data['context_source'].value
    if form_data['context_source'].value == 'url':
        app.storage.user['llm_last_url'] = (form_data['context_url'].value or '').strip()

    # Update question history (keep last 10 unique questions)
    if current_question:
        question_history = app.storage.user.get('llm_question_history', [])
        # Remove if already in history to avoid duplicates
        if current_question in question_history:
            question_history.remove(current_question)
        # Add to front of list
        question_history.insert(0, current_question)
        # Keep only last 10
        question_history = question_history[:10]
        app.storage.user['llm_question_history'] = question_history

    # Get API key - try form first, then environment variables
    api_key = (form_data['llm_api_key'].value or '').strip()
    api_key_source = 'form input'

    if not api_key:
        # Try environment variables based on provider
        if provider_value == 'openai':
            api_key = os.getenv('OPENAI_API_KEY', '')
            if api_key:
                api_key_source = 'OPENAI_API_KEY environment variable'
        elif provider_value == 'anthropic':
            # Try both ANTHROPIC_API_KEY and CLAUDE_API_KEY
            api_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY', '')
            if api_key:
                if os.getenv('ANTHROPIC_API_KEY'):
                    api_key_source = 'ANTHROPIC_API_KEY environment variable'
                else:
                    api_key_source = 'CLAUDE_API_KEY environment variable'

    # Get context - either from text or URL/PDF
    context_source = form_data['context_source'].value

    print(f"[DEBUG] start_llm_pipeline - context_source: {context_source}")
    import sys
    sys.stdout.flush()

    if context_source == 'text':
        context = (form_data['llm_context'].value or '').strip()
        question = (form_data['llm_question'].value or '').strip()
    elif context_source == 'url':
        # Support multiple URLs (one per line)
        context_urls_raw = (form_data['context_url'].value or '').strip()
        print(f"[DEBUG] context_urls value: '{context_urls_raw}'")
        sys.stdout.flush()
        if not context_urls_raw:
            ui.notify('Please provide at least one URL for the context source', type='negative')
            return

        # Parse URLs (one per line, skip empty lines)
        context_urls = [url.strip() for url in context_urls_raw.split('\n') if url.strip()]
        if not context_urls:
            ui.notify('Please provide at least one valid URL', type='negative')
            return

        # Fetch content from all URLs
        all_contexts = []
        try:
            import aiohttp
            from bs4 import BeautifulSoup

            # Use proper headers to avoid being blocked by sites like Wikipedia
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }

            async with aiohttp.ClientSession(headers=headers) as session:
                for i, url in enumerate(context_urls):
                    ui.notify(f'Fetching URL {i+1}/{len(context_urls)}...', type='info')
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                            if response.status == 200:
                                html_content = await response.text()
                                soup = BeautifulSoup(html_content, 'html.parser')
                                # Remove script and style elements for cleaner text
                                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                                    element.decompose()
                                page_text = soup.get_text(separator='\n', strip=True)
                                all_contexts.append(f"--- Content from {url} ---\n{page_text}")
                            else:
                                ui.notify(f'Failed to fetch {url}: {response.status}', type='warning')
                    except Exception as url_error:
                        ui.notify(f'Error fetching {url}: {str(url_error)}', type='warning')

            if not all_contexts:
                ui.notify('Failed to fetch any URL content', type='negative')
                return

            context = '\n\n'.join(all_contexts)
            ui.notify(f'Context loaded from {len(all_contexts)} URL(s)', type='positive')
        except Exception as e:
            ui.notify(f'Error fetching URLs: {str(e)}', type='negative')
            return
        question = (form_data['llm_question'].value or '').strip()
    elif context_source == 'pdf':
        # PDF upload handled separately via file upload - now supports multiple PDFs
        if 'pdf_contexts' not in form_data or not form_data['pdf_contexts']:
            ui.notify('Please upload at least one PDF file', type='negative')
            return
        # Combine all PDF contexts
        context = '\n\n--- Next PDF Document ---\n\n'.join(form_data['pdf_contexts'])
        question = (form_data['llm_question'].value or '').strip()
    else:
        context = (form_data['llm_context'].value or '').strip()
        question = (form_data['llm_question'].value or '').strip()

    # Automatic context truncation to prevent memory issues
    # Limit to ~300K characters which should handle 3-5 Wikipedia pages
    MAX_CONTEXT_CHARS = 300000
    if len(context) > MAX_CONTEXT_CHARS:
        original_len = len(context)
        context = context[:MAX_CONTEXT_CHARS]
        # Try to truncate at a sentence boundary
        last_period = context.rfind('.')
        if last_period > MAX_CONTEXT_CHARS * 0.8:  # Only truncate at period if reasonably close to limit
            context = context[:last_period + 1]
        ui.notify(
            f'⚠️ Context truncated from {original_len:,} to {len(context):,} characters to prevent memory issues',
            type='warning',
            timeout=10000
        )
        print(f"[WARNING] Context truncated from {original_len:,} to {len(context):,} characters")

    # Validate inputs
    if not api_key:
        if provider_value == 'openai':
            ui.notify('Please provide an API key (form input, OPENAI_API_KEY environment variable, or .env file)', type='negative')
        else:
            ui.notify('Please provide an API key (form input, ANTHROPIC_API_KEY/CLAUDE_API_KEY environment variable, or .env file)', type='negative')
        return

    # Notify user about API key source
    if api_key_source != 'form input':
        ui.notify(f'Using API key from {api_key_source}', type='info')

    if not question:
        ui.notify('Please provide a question', type='negative')
        return

    if not context:
        ui.notify('Please provide context', type='negative')
        return

    # Import pipeline modules
    try:
        from llm_client import LLMClient, LLMProvider, LLMModel
        from pipeline import SemanticFaithfulnessPipeline
    except ImportError as e:
        ui.notify(f'Import error: {str(e)}', type='negative')
        return

    # Map provider and model
    # Map provider string to enum
    provider_map = {
        'openai': LLMProvider.OPENAI,
        'anthropic': LLMProvider.ANTHROPIC,
        'gemini': LLMProvider.GEMINI
    }
    provider = provider_map.get(provider_value, LLMProvider.ANTHROPIC)

    # Map model string to enum (2025 models)
    model_map = {
        # OpenAI models (2025)
        'gpt-5': LLMModel.GPT5,
        'gpt-5-codex': LLMModel.GPT5_CODEX,
        'gpt-4.5': LLMModel.GPT4_5,
        'gpt-4o': LLMModel.GPT4O,
        'gpt-4o-mini': LLMModel.GPT4O_MINI,
        'o1-preview': LLMModel.O1_PREVIEW,
        'o1-mini': LLMModel.O1_MINI,
        # Anthropic models (2025)
        'claude-sonnet-4-5-20250929': LLMModel.CLAUDE_SONNET_4_5,
        'claude-opus-4-1': LLMModel.CLAUDE_OPUS_4_1,
        'claude-sonnet-4': LLMModel.CLAUDE_SONNET_4,
        'claude-haiku-4-5': LLMModel.CLAUDE_HAIKU_4_5,
        'claude-3-5-sonnet-20241022': LLMModel.CLAUDE_SONNET_3_5,
        # Google Gemini models (2025)
        'gemini-3-pro-image': LLMModel.GEMINI_3_PRO_IMAGE,
        'gemini-2.5-pro': LLMModel.GEMINI_2_5_PRO,
        'gemini-2.5-flash': LLMModel.GEMINI_2_5_FLASH,
        'gemini-2.5-flash-lite': LLMModel.GEMINI_2_5_FLASH_LITE,
        'gemini-2.0-flash-exp': LLMModel.GEMINI_2_0_FLASH_EXP
    }
    model = model_map.get(model_value, LLMModel.CLAUDE_SONNET_4_5)  # Default to Claude Sonnet 4.5

    # Create LLM client
    llm_client = LLMClient(provider=provider, model=model, api_key=api_key)

    # Create pipeline
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'llm_runs'
    pipeline = SemanticFaithfulnessPipeline(
        llm_client=llm_client,
        output_dir=output_dir
    )

    # Create progress dialog
    progress_state = {
        'step': 'Initializing...',
        'current': 0,
        'total': 5,
        'message': ''
    }

    with ui.dialog() as progress_dialog, ui.card().classes('p-6 min-w-96'):
        ui.label('LLM Pipeline Progress').classes('text-h6 mb-4')
        step_label = ui.label(progress_state['step']).classes('text-subtitle1 mb-2')
        message_label = ui.label(progress_state['message']).classes('text-body2 text-grey-7 mb-4')
        progress_bar = ui.linear_progress(value=0).classes('w-full')
        ui.label('This may take 3-5 minutes...').classes('text-caption text-grey-6 mt-2')

    progress_dialog.open()

    async def update_progress(step: str, current: int, total: int, message: str = ""):
        """Update progress dialog"""
        import asyncio
        progress_state['step'] = step
        progress_state['current'] = current
        progress_state['total'] = total
        progress_state['message'] = message

        step_label.set_text(step)
        message_label.set_text(message)
        progress_bar.set_value(current / total if total > 0 else 0)

        # Force UI update by giving control back to event loop
        await asyncio.sleep(0.1)

    # Run pipeline
    try:
        embedding_model = form_data['embedding_model'].value
        clustering_method = form_data['clustering_method'].value
        force_regenerate = form_data['force_regenerate'].value

        # Set progress callback
        pipeline.progress_callback = update_progress

        # Run the full pipeline
        results = await pipeline.run_full_pipeline(
            original_question=question,
            context=context,
            num_paraphrases=num_paraphrases,
            embedding_model=embedding_model,
            clustering_method=clustering_method,
            force_regenerate=force_regenerate
        )

        # Store results in session
        app.storage.user['llm_pipeline_results'] = results
        app.storage.user['analysis_status'] = 'llm_complete'
        # Clear previous analysis results to avoid stale data on results page
        app.storage.user['analysis_results'] = None

        # Store first triplet for analysis
        if results.get('triplets'):
            first_triplet = results['triplets'][0]
            app.storage.user['qca_triplet'] = {
                'question': first_triplet['question'],
                'context': first_triplet['context'],
                'answer': first_triplet['answer']
            }

        # Store distributions for display
        if results.get('distributions'):
            app.storage.user['cached_distributions'] = results['distributions'][0]

        progress_dialog.close()
        ui.notify('Pipeline completed successfully!', type='positive')
        ui.navigate.to('/analyze')

    except Exception as e:
        progress_dialog.close()
        ui.notify(f'Pipeline error: {str(e)}', type='negative')
        print(f'[ERROR] Pipeline failed: {e}')
        import traceback
        traceback.print_exc()
