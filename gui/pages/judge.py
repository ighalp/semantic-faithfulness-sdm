"""
LLM Judge page - LLM-as-a-Judge for comparing and evaluating answers
Same layout as Compare page, plus LLM evaluation results and export options
"""

from nicegui import ui, app
import sys
import os
import re
from pathlib import Path
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_client import LLMClient, LLMProvider, LLMModel
from pages.markdown_utils import generate_diff_html, markdown_to_html, apply_llm_highlights


def create():
    """Create the LLM Judge page content"""

    # Get LLM pipeline results
    llm_results = app.storage.user.get('llm_pipeline_results')

    if llm_results and llm_results.get('triplets'):
        triplets = llm_results['triplets']
        fs_scores_raw = llm_results.get('fs_scores', {})

        # Build lookups
        prompts_dict = {}
        answers_dict = {}
        for i, t in enumerate(triplets):
            pid = t.get('prompt_id', f'prompt_{i}')
            prompts_dict[pid] = {
                'prompt_id': pid,
                'text': t.get('question', ''),
                'context': t.get('context', ''),
            }
            answers_dict[pid] = {
                'prompt_id': pid,
                'answer': t.get('answer', ''),
            }

        # Normalize fs_scores
        fs_scores = {}
        for pid, score_data in dict(fs_scores_raw).items():
            if isinstance(score_data, dict):
                fs_scores[pid] = float(score_data.get('F_S', 0))
            else:
                fs_scores[pid] = float(score_data) if score_data is not None else 0.0

        data_loaded = True
    else:
        data_loaded = False
        fs_scores = {}
        prompts_dict = {}
        answers_dict = {}
        triplets = []

    with ui.column().classes('w-full max-w-7xl mx-auto p-8'):
        ui.label('LLM Judge').classes('text-h4 mb-2')
        ui.label('Evaluate answers using LLM-as-a-Judge').classes('text-subtitle1 text-grey-7 mb-6')

        if not data_loaded or len(triplets) < 2:
            with ui.card().classes('w-full p-6 bg-yellow-50'):
                ui.label('Not Enough Data').classes('text-h6 text-yellow-800')
                ui.label(
                    'LLM Judge requires at least 2 answer variants. '
                    'Please run the pipeline with 2 or more paraphrases first.'
                ).classes('text-body1 text-yellow-700 mt-2')
                with ui.row().classes('gap-4 mt-4'):
                    ui.button('Go to Input', on_click=lambda: ui.navigate.to('/input')).props('color=primary')
                    ui.button('Go to Compare', on_click=lambda: ui.navigate.to('/compare')).props('outline')
            return

        # Determine default selections from Compare page or use defaults
        compare_selection = app.storage.user.get('compare_selection', {})
        prompt_ids_list = list(prompts_dict.keys())

        # Default: Initial vs Lowest F_S
        initial_pid = 'prompt_0' if 'prompt_0' in prompt_ids_list else prompt_ids_list[0]
        if fs_scores:
            sorted_by_fs = sorted(fs_scores.items(), key=lambda x: x[1])
            lowest_fs_pid = sorted_by_fs[0][0]
            if lowest_fs_pid == initial_pid and len(sorted_by_fs) > 1:
                lowest_fs_pid = sorted_by_fs[1][0]
        else:
            lowest_fs_pid = prompt_ids_list[1] if len(prompt_ids_list) > 1 else prompt_ids_list[0]

        # Use selection from Compare page if available
        default_left = compare_selection.get('left_pid', initial_pid)
        default_right = compare_selection.get('right_pid', lowest_fs_pid)

        # Validate defaults exist
        if default_left not in prompt_ids_list:
            default_left = initial_pid
        if default_right not in prompt_ids_list:
            default_right = lowest_fs_pid

        # Build dropdown options
        triplet_options = {}
        for pid in prompt_ids_list:
            fs = fs_scores.get(pid)
            if fs is not None:
                triplet_options[pid] = f"{pid} (F_S: {fs:.4f})"
            else:
                triplet_options[pid] = pid

        # State
        judge_state = {
            'left_pid': default_left,
            'right_pid': default_right,
            'result': None,
            'is_running': False,
            'recommended_pid': None,
            'key_differences': None  # Will store {'A': [...], 'B': [...]} after judge runs
        }

        # Explanation card
        with ui.card().classes('w-full p-4 mb-4 bg-blue-50'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('gavel', color='blue')
                ui.label('LLM-as-a-Judge evaluates answers on: Faithfulness, Completeness, Coherence, Relevance').classes('text-subtitle2')

        # Store references in a dict for access from nested functions
        selectors = {}
        comparison_container = None  # Will be set after UI is created

        def update_comparison(e=None):
            """Update the side-by-side answer display"""
            nonlocal comparison_container
            if comparison_container is None:
                return
            comparison_container.clear()

            left_pid = selectors.get('left')
            right_pid = selectors.get('right')
            if left_pid:
                left_pid = left_pid.value
            if right_pid:
                right_pid = right_pid.value

            if not left_pid or not right_pid or left_pid == right_pid:
                with comparison_container:
                    ui.label('Please select two different answers to compare').classes('text-orange-600 p-4')
                return

            judge_state['left_pid'] = left_pid
            judge_state['right_pid'] = right_pid

            left_answer = answers_dict[left_pid]['answer']
            right_answer = answers_dict[right_pid]['answer']
            fs_left = fs_scores.get(left_pid)
            fs_right = fs_scores.get(right_pid)

            with comparison_container:
                # Check if we have LLM-provided key differences for the current answer pair
                use_llm_highlights = (
                    judge_state.get('key_differences') is not None and
                    judge_state.get('compared_left') == left_pid and
                    judge_state.get('compared_right') == right_pid
                )

                with ui.row().classes('w-full gap-4'):
                    # Left pane
                    with ui.card().classes('flex-1 p-6'):
                        with ui.row().classes('items-center gap-2 mb-2'):
                            ui.label(f'Answer A: {left_pid}').classes('text-h6')
                            if left_pid == initial_pid:
                                ui.badge('Initial', color='blue').props('outline')
                            if use_llm_highlights:
                                ui.badge('LLM Highlights', color='purple').props('outline')
                        if fs_left is not None:
                            color = 'text-red-600' if fs_scores and fs_left == min(fs_scores.values()) else 'text-primary'
                            ui.label(f"F_S: {fs_left:.4f}").classes(f'text-subtitle2 {color} mb-2')
                        ui.label(f"{len(left_answer)} characters").classes('text-caption text-grey-6 mb-4')

                        # Use LLM highlights if available, otherwise fall back to diff-based highlights
                        if use_llm_highlights:
                            key_diffs_a = judge_state['key_differences'].get('A', [])
                            html_left = apply_llm_highlights(left_answer, key_diffs_a)
                        else:
                            html_left = generate_diff_html(left_answer, right_answer, side='left')
                        ui.html(html_left, sanitize=False).classes('text-body2 whitespace-pre-wrap max-h-96 overflow-auto')

                    # Right pane
                    with ui.card().classes('flex-1 p-6'):
                        with ui.row().classes('items-center gap-2 mb-2'):
                            ui.label(f'Answer B: {right_pid}').classes('text-h6')
                            if fs_scores and fs_right is not None and fs_right == min(fs_scores.values()):
                                ui.badge('Lowest F_S', color='red').props('outline')
                            if use_llm_highlights:
                                ui.badge('LLM Highlights', color='purple').props('outline')
                        if fs_right is not None:
                            color = 'text-red-600' if fs_scores and fs_right == min(fs_scores.values()) else 'text-primary'
                            ui.label(f"F_S: {fs_right:.4f}").classes(f'text-subtitle2 {color} mb-2')
                        ui.label(f"{len(right_answer)} characters").classes('text-caption text-grey-6 mb-4')

                        # Use LLM highlights if available, otherwise fall back to diff-based highlights
                        if use_llm_highlights:
                            key_diffs_b = judge_state['key_differences'].get('B', [])
                            html_right = apply_llm_highlights(right_answer, key_diffs_b)
                        else:
                            html_right = generate_diff_html(left_answer, right_answer, side='right')
                        ui.html(html_right, sanitize=False).classes('text-body2 whitespace-pre-wrap max-h-96 overflow-auto')

        # Selection controls
        with ui.card().classes('w-full p-6 mb-4'):
            ui.label('Select Answers to Evaluate').classes('text-h6 mb-4')

            with ui.row().classes('w-full gap-4 items-end'):
                selectors['left'] = ui.select(
                    label='Answer A (Left)',
                    options=triplet_options,
                    value=default_left
                ).classes('flex-1')

                selectors['right'] = ui.select(
                    label='Answer B (Right)',
                    options=triplet_options,
                    value=default_right
                ).classes('flex-1')

            # Quick selection buttons
            with ui.row().classes('w-full gap-4 mt-4'):
                def select_initial_vs_highest():
                    if fs_scores:
                        sorted_by_fs = sorted(fs_scores.items(), key=lambda x: x[1], reverse=True)
                        highest_fs_pid = sorted_by_fs[0][0]
                        selectors['left'].value = initial_pid
                        selectors['right'].value = highest_fs_pid
                        selectors['left'].update()
                        selectors['right'].update()
                        update_comparison()

                def select_initial_vs_lowest():
                    selectors['left'].value = initial_pid
                    selectors['right'].value = lowest_fs_pid
                    selectors['left'].update()
                    selectors['right'].update()
                    update_comparison()

                def select_highest_vs_lowest():
                    if fs_scores:
                        sorted_by_fs = sorted(fs_scores.items(), key=lambda x: x[1])
                        lowest = sorted_by_fs[0][0]
                        highest = sorted_by_fs[-1][0]
                        selectors['left'].value = highest
                        selectors['right'].value = lowest
                        selectors['left'].update()
                        selectors['right'].update()
                        update_comparison()

                ui.button('Initial vs Highest F_S', on_click=select_initial_vs_highest, icon='arrow_upward').props('outline size=sm')
                ui.button('Initial vs Lowest F_S', on_click=select_initial_vs_lowest, icon='arrow_downward').props('outline size=sm')
                ui.button('Highest vs Lowest F_S', on_click=select_highest_vs_lowest, icon='swap_vert').props('outline size=sm')

        # Side-by-side answer display container (after controls)
        comparison_container = ui.column().classes('w-full')

        # Bind on_change handlers
        selectors['left'].on('update:model-value', update_comparison)
        selectors['right'].on('update:model-value', update_comparison)

        # Initialize comparison display on page load
        ui.timer(0.1, update_comparison, once=True)

        # LLM Judge Model Selection
        ui.separator().classes('my-4')

        with ui.card().classes('w-full p-6 mb-4'):
            ui.label('Judge Model Selection').classes('text-h6 mb-4')

            # Get the model used for answer generation
            default_provider = app.storage.user.get('llm_last_provider', 'anthropic')
            default_model = app.storage.user.get('llm_last_model', 'claude-sonnet-4-5-20250929')

            # Build model options organized by provider
            model_options = {
                'OpenAI': {
                    'gpt-4o': 'GPT-4o',
                    'gpt-4o-mini': 'GPT-4o Mini',
                    'o1-preview': 'o1-preview',
                    'o1-mini': 'o1-mini',
                },
                'Anthropic': {
                    'claude-sonnet-4-5-20250929': 'Claude Sonnet 4.5',
                    'claude-opus-4-1': 'Claude Opus 4.1',
                    'claude-sonnet-4': 'Claude Sonnet 4',
                    'claude-haiku-4-5': 'Claude Haiku 4.5',
                    'claude-3-5-sonnet-20241022': 'Claude Sonnet 3.5',
                },
                'Google Gemini': {
                    'gemini-2.5-pro': 'Gemini 2.5 Pro',
                    'gemini-2.5-flash': 'Gemini 2.5 Flash',
                    'gemini-2.0-flash-exp': 'Gemini 2.0 Flash Exp',
                }
            }

            # Flatten for dropdown with provider prefix
            flat_options = {}
            for provider_name, models in model_options.items():
                for model_id, model_name in models.items():
                    flat_options[model_id] = f"{model_name} ({provider_name})"

            # State for judge model
            judge_model_state = {'model': default_model, 'provider': default_provider}

            with ui.row().classes('w-full gap-4 items-end'):
                with ui.column().classes('flex-1'):
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('Judge Model').classes('text-subtitle2')
                        ui.badge('Default: Same as answer generation', color='blue').props('outline')

                    def on_judge_model_change(e):
                        judge_model_state['model'] = e.value
                        # Determine provider from model
                        if e.value.startswith('gpt') or e.value.startswith('o1'):
                            judge_model_state['provider'] = 'openai'
                        elif e.value.startswith('claude'):
                            judge_model_state['provider'] = 'anthropic'
                        elif e.value.startswith('gemini'):
                            judge_model_state['provider'] = 'gemini'

                    judge_model_select = ui.select(
                        label='Select Judge Model',
                        options=flat_options,
                        value=default_model,
                        on_change=on_judge_model_change
                    ).classes('w-full')

                with ui.column():
                    ui.label('Tip').classes('text-caption text-grey-6')
                    ui.label('Use a different model for independent evaluation').classes('text-caption text-grey-6')

        # Custom prompt section
        with ui.card().classes('w-full p-6 mb-4'):
            with ui.row().classes('items-center gap-2 mb-4'):
                ui.label('Judge Instructions').classes('text-h6')
                ui.badge('Customizable', color='green').props('outline')

            # Default prompt text
            default_judge_prompt = """You are an expert evaluator assessing the quality of answers to questions based on provided context.

Your task is to compare two answers and determine which one is better. Evaluate based on these criteria:

1. **Faithfulness to Context**: Does the answer accurately reflect information from the context without hallucinations?
2. **Completeness**: Does the answer address all aspects of the question?
3. **Coherence**: Is the answer well-organized and logically structured?
4. **Relevance**: Does the answer focus on what was asked without unnecessary tangents?

IMPORTANT: Base your evaluation ONLY on how well each answer represents the information in the context. Do not prefer answers simply because they are longer or more detailed if that additional detail is not supported by the context.

Provide scores from 1-10 for each answer on each criterion, overall scores, determine the winner (A, B, or TIE), and provide a detailed explanation of your judgment."""

            # State for custom prompt
            prompt_state = {'prompt': default_judge_prompt, 'is_custom': False}

            with ui.expansion('Edit Judge Prompt', icon='edit').classes('w-full'):
                ui.label('Customize the instructions given to the LLM judge. You can modify the evaluation criteria or add specific questions.').classes('text-caption text-grey-6 mb-4')

                prompt_textarea = ui.textarea(
                    label='System Prompt for Judge',
                    value=default_judge_prompt
                ).classes('w-full').props('outlined rows=12')

                def on_prompt_change(e):
                    prompt_state['prompt'] = e.value
                    prompt_state['is_custom'] = (e.value.strip() != default_judge_prompt.strip())

                prompt_textarea.on('update:model-value', on_prompt_change)

                with ui.row().classes('w-full gap-4 mt-4'):
                    def reset_prompt():
                        prompt_textarea.value = default_judge_prompt
                        prompt_state['prompt'] = default_judge_prompt
                        prompt_state['is_custom'] = False
                        ui.notify('Prompt reset to default', type='info')

                    ui.button('Reset to Default', on_click=reset_prompt, icon='restore').props('outline')
                    ui.label('Changes are applied when you run the judge').classes('text-caption text-grey-6 self-center')

        with ui.row().classes('w-full items-center gap-4'):
            run_button = ui.button('Run LLM-as-a-Judge', icon='gavel').props('color=primary size=lg')
            spinner = ui.spinner('dots', size='lg').classes('hidden')

        # LLM Judge Results container
        results_container = ui.column().classes('w-full mt-4')

        # Export section container (hidden until evaluation is done)
        export_container = ui.column().classes('w-full mt-6 hidden')

        async def run_judge():
            """Execute the LLM-as-a-Judge evaluation"""
            if judge_state['is_running']:
                return

            left_pid = selectors['left'].value
            right_pid = selectors['right'].value

            if not left_pid or not right_pid:
                ui.notify('Please select both answers', type='warning')
                return

            if left_pid == right_pid:
                ui.notify('Please select two different answers', type='warning')
                return

            # Get LLM settings from judge model selection
            provider_str = judge_model_state['provider']
            model_str = judge_model_state['model']
            api_key = ''

            if provider_str == 'openai':
                api_key = os.environ.get('OPENAI_API_KEY', '')
            elif provider_str == 'anthropic':
                api_key = os.environ.get('ANTHROPIC_API_KEY', '') or os.environ.get('CLAUDE_API_KEY', '')
            elif provider_str == 'gemini':
                api_key = os.environ.get('GOOGLE_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')

            if not api_key:
                ui.notify('No API key found. Please configure LLM settings on the Input page.', type='negative')
                return

            # Map provider
            provider_map = {
                'openai': LLMProvider.OPENAI,
                'anthropic': LLMProvider.ANTHROPIC,
                'gemini': LLMProvider.GEMINI
            }
            provider = provider_map.get(provider_str, LLMProvider.ANTHROPIC)

            # Find model
            model = None
            for m in LLMModel:
                if m.value == model_str:
                    model = m
                    break
            if not model:
                model = LLMModel.CLAUDE_SONNET_4_5

            # Update UI
            judge_state['is_running'] = True
            run_button.disable()
            spinner.classes(remove='hidden')

            try:
                llm_client = LLMClient(provider, model, api_key)

                question = prompts_dict[left_pid]['text']
                context = prompts_dict[left_pid]['context']

                # Use custom prompt if modified, otherwise None (will use default)
                custom_prompt = prompt_state['prompt'] if prompt_state['is_custom'] else None

                result = await llm_client.judge_answers(
                    question=question,
                    context=context,
                    answer_a=answers_dict[left_pid]['answer'],
                    answer_b=answers_dict[right_pid]['answer'],
                    answer_a_label=f"Answer A ({left_pid})",
                    answer_b_label=f"Answer B ({right_pid})",
                    custom_system_prompt=custom_prompt
                )

                judge_state['result'] = result
                # Store the compared prompt IDs for reference
                judge_state['compared_left'] = left_pid
                judge_state['compared_right'] = right_pid
                # Store which model was used as judge
                judge_state['judge_model'] = model_str
                judge_state['judge_provider'] = provider_str
                # Store key differences for LLM-based highlighting
                judge_state['key_differences'] = result.get('key_differences', {'A': [], 'B': []})

                # Determine recommended answer
                if result['winner'] == 'A':
                    judge_state['recommended_pid'] = left_pid
                elif result['winner'] == 'B':
                    judge_state['recommended_pid'] = right_pid
                else:
                    # TIE - recommend higher F_S
                    fs_left = fs_scores.get(left_pid, 0)
                    fs_right = fs_scores.get(right_pid, 0)
                    judge_state['recommended_pid'] = left_pid if fs_left >= fs_right else right_pid

                # Refresh the comparison display with LLM-based highlights
                update_comparison()

                # Display results using the actual compared prompts
                display_results(result, judge_state['compared_left'], judge_state['compared_right'], model_str)

                # Show export section
                export_container.classes(remove='hidden')
                setup_export_section(left_pid, right_pid)

                ui.notify('Evaluation complete!', type='positive')

            except Exception as e:
                ui.notify(f'Evaluation failed: {str(e)}', type='negative')

            finally:
                judge_state['is_running'] = False
                run_button.enable()
                spinner.classes(add='hidden')

        def display_results(result, compared_left_pid, compared_right_pid, judge_model_used):
            """Display the LLM-as-a-Judge results"""
            display_left = str(compared_left_pid)
            display_right = str(compared_right_pid)

            # Get friendly model name
            model_display_name = flat_options.get(judge_model_used, judge_model_used)

            results_container.clear()

            with results_container:
                with ui.row().classes('items-center justify-between w-full mb-4'):
                    with ui.row().classes('items-center gap-4'):
                        ui.label('LLM Judge Evaluation Results').classes('text-h5')
                        ui.badge(f'Judge: {model_display_name}', color='purple').props('outline')

                    # Export Verdict buttons
                    async def export_verdict_md():
                        left_answer = answers_dict[display_left]['answer']
                        right_answer = answers_dict[display_right]['answer']

                        md_content = generate_verdict_export(
                            judge_model_name=model_display_name,
                            left_pid=display_left,
                            right_pid=display_right,
                            judge_result=result,
                            left_answer=left_answer,
                            right_answer=right_answer,
                            fs_scores=fs_scores
                        )

                        filename = f"verdict_{display_left}_vs_{display_right}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                        ui.download(md_content.encode('utf-8'), filename)
                        ui.notify(f'Verdict exported as {filename}', type='positive')

                    async def export_verdict_pdf():
                        try:
                            left_answer = answers_dict[display_left]['answer']
                            right_answer = answers_dict[display_right]['answer']

                            pdf_content = generate_verdict_pdf(
                                judge_model_name=model_display_name,
                                left_pid=display_left,
                                right_pid=display_right,
                                judge_result=result,
                                left_answer=left_answer,
                                right_answer=right_answer,
                                fs_scores=fs_scores
                            )

                            filename = f"verdict_{display_left}_vs_{display_right}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            ui.download(pdf_content, filename)
                            ui.notify(f'Verdict exported as {filename}', type='positive')
                        except ImportError:
                            ui.notify('PDF export requires reportlab. Install with: pip install reportlab', type='negative')
                        except Exception as e:
                            ui.notify(f'PDF export error: {str(e)}', type='negative')

                    with ui.row().classes('gap-2'):
                        ui.button('Export Verdict (MD)', icon='description', on_click=export_verdict_md).props('color=purple outline size=sm')
                        ui.button('Export Verdict (PDF)', icon='picture_as_pdf', on_click=export_verdict_pdf).props('color=purple outline size=sm')

                # Winner card
                with ui.card().classes('w-full p-6 mb-4 bg-amber-50'):
                    winner = result.get('winner', 'TIE')
                    scores = result.get('scores', {})

                    if winner == 'A':
                        winner_text = f'Answer A ({display_left})'
                        winner_color = 'text-blue-700'
                    elif winner == 'B':
                        winner_text = f'Answer B ({display_right})'
                        winner_color = 'text-green-700'
                    else:
                        winner_text = 'TIE'
                        winner_color = 'text-grey-700'

                    with ui.row().classes('items-center gap-4 mb-4'):
                        ui.icon('emoji_events', size='xl').classes('text-amber-500')
                        ui.label(f'Winner: {winner_text}').classes(f'text-h4 font-bold {winner_color}')

                    # Scores side by side
                    with ui.row().classes('w-full gap-8'):
                        with ui.card().classes('flex-1 p-4 bg-blue-50'):
                            ui.label(f'Answer A ({display_left})').classes('text-subtitle2 font-bold text-blue-700')
                            ui.label(f"LLM Score: {scores.get('A', 'N/A')}/10").classes('text-h5')
                            fs_left_val = fs_scores.get(display_left)
                            if fs_left_val is not None:
                                ui.label(f"F_S: {fs_left_val:.4f}").classes('text-body2 text-grey-6')

                        with ui.card().classes('flex-1 p-4 bg-green-50'):
                            ui.label(f'Answer B ({display_right})').classes('text-subtitle2 font-bold text-green-700')
                            ui.label(f"LLM Score: {scores.get('B', 'N/A')}/10").classes('text-h5')
                            fs_right_val = fs_scores.get(display_right)
                            if fs_right_val is not None:
                                ui.label(f"F_S: {fs_right_val:.4f}").classes('text-body2 text-grey-6')

                # Criteria breakdown
                criteria = result.get('criteria_breakdown', {})
                if criteria:
                    with ui.card().classes('w-full p-6 mb-4'):
                        ui.label('Criteria Breakdown').classes('text-h6 mb-4')

                        with ui.grid(columns=5).classes('w-full gap-2'):
                            # Header
                            ui.label('Criterion').classes('font-bold')
                            ui.label('A').classes('font-bold text-center')
                            ui.label('B').classes('font-bold text-center')
                            ui.label('Diff').classes('font-bold text-center')
                            ui.label('Better').classes('font-bold text-center')

                            for criterion, scores_dict in criteria.items():
                                score_a = scores_dict.get('A', 0)
                                score_b = scores_dict.get('B', 0)
                                diff = score_a - score_b

                                ui.label(criterion.replace('_', ' ').title())
                                ui.label(f'{score_a}/10').classes('text-center')
                                ui.label(f'{score_b}/10').classes('text-center')

                                diff_color = 'text-blue-600' if diff > 0 else ('text-green-600' if diff < 0 else 'text-grey-600')
                                ui.label(f'{diff:+d}').classes(f'text-center {diff_color}')

                                if diff > 0:
                                    ui.label('A').classes('text-center text-blue-600 font-bold')
                                elif diff < 0:
                                    ui.label('B').classes('text-center text-green-600 font-bold')
                                else:
                                    ui.label('=').classes('text-center text-grey-600')

                # Key Phrases Discussed (from LLM)
                key_diffs = result.get('key_differences', {})
                diffs_a = key_diffs.get('A', [])
                diffs_b = key_diffs.get('B', [])
                if diffs_a or diffs_b:
                    with ui.card().classes('w-full p-6 mb-4'):
                        with ui.row().classes('items-center gap-2 mb-4'):
                            ui.label('Key Phrases Discussed').classes('text-h6')
                            ui.badge('Referenced in Analysis', color='purple').props('outline')
                        ui.label('These exact phrases from each answer are discussed in the detailed analysis above:').classes('text-caption text-grey-6 mb-4')

                        with ui.row().classes('w-full gap-4'):
                            # Answer A differences
                            with ui.column().classes('flex-1'):
                                ui.label(f'Answer A ({display_left})').classes('text-subtitle2 font-bold text-blue-700 mb-2')
                                if diffs_a:
                                    for diff in diffs_a:
                                        with ui.row().classes('items-start gap-2 mb-2'):
                                            ui.icon('format_quote', size='sm').classes('text-blue-400')
                                            ui.label(f'"{diff}"').classes('text-body2 italic bg-blue-50 p-2 rounded')
                                else:
                                    ui.label('No key differences identified').classes('text-caption text-grey-5')

                            # Answer B differences
                            with ui.column().classes('flex-1'):
                                ui.label(f'Answer B ({display_right})').classes('text-subtitle2 font-bold text-green-700 mb-2')
                                if diffs_b:
                                    for diff in diffs_b:
                                        with ui.row().classes('items-start gap-2 mb-2'):
                                            ui.icon('format_quote', size='sm').classes('text-green-400')
                                            ui.label(f'"{diff}"').classes('text-body2 italic bg-green-50 p-2 rounded')
                                else:
                                    ui.label('No key differences identified').classes('text-caption text-grey-5')

                # Explanation
                with ui.card().classes('w-full p-6 mb-4'):
                    ui.label('Detailed Explanation').classes('text-h6 mb-4')
                    explanation_text = result.get('explanation', 'No explanation provided')
                    explanation_html = markdown_to_html(explanation_text)
                    ui.html(explanation_html, sanitize=False).classes('text-body1')

                # F_S vs LLM comparison
                with ui.card().classes('w-full p-6 bg-purple-50'):
                    ui.label('F_S vs LLM Judge Comparison').classes('text-h6 mb-4')

                    fs_left_cmp = fs_scores.get(display_left, 0)
                    fs_right_cmp = fs_scores.get(display_right, 0)
                    fs_winner = 'A' if fs_left_cmp > fs_right_cmp else ('B' if fs_right_cmp > fs_left_cmp else 'TIE')
                    llm_winner = result.get('winner', 'TIE')

                    if fs_winner == llm_winner:
                        ui.label('Both metrics agree!').classes('text-body1 text-green-700 font-bold')
                        ui.label(
                            'The F_S score and LLM Judge reached the same conclusion, '
                            'providing strong confidence in this recommendation.'
                        ).classes('text-body2 mt-2')
                    else:
                        ui.label(f'Metrics disagree: F_S favors {fs_winner}, LLM favors {llm_winner}').classes('text-body1 text-amber-700 font-bold')
                        ui.label(
                            'The answers may have different strengths - '
                            'one may be more informationally faithful while the other is more coherent.'
                        ).classes('text-body2 mt-2')

        def setup_export_section(left_pid, right_pid):
            """Setup the export section"""
            export_container.clear()

            with export_container:
                ui.separator().classes('my-6')
                ui.label('Export Selected Answer').classes('text-h5 mb-4')

                # Export state
                export_state = {
                    'selected_pid': judge_state['recommended_pid'],
                    'selection_method': 'LLM Judge Recommendation'
                }

                with ui.card().classes('w-full p-6'):
                    # Recommendation banner
                    with ui.card().classes('w-full p-4 mb-4 bg-green-50'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('recommend', color='green')
                            ui.label(f"Recommended: {judge_state['recommended_pid']}").classes('text-subtitle1 font-bold text-green-700')
                        ui.label('Based on LLM-as-a-Judge evaluation').classes('text-caption text-grey-6 ml-8')

                    ui.label('Select Answer to Export').classes('text-subtitle1 font-bold mb-4')

                    # Build export options
                    all_options = {}
                    for pid in [left_pid, right_pid]:
                        fs = fs_scores.get(pid)
                        label = f"{pid}"
                        if pid == judge_state['recommended_pid']:
                            label += " (Recommended)"
                        if fs is not None:
                            label += f" - F_S: {fs:.4f}"
                        all_options[pid] = label

                    # Add option for any other answer
                    for pid in prompt_ids_list:
                        if pid not in all_options:
                            fs = fs_scores.get(pid)
                            label = f"{pid}"
                            if fs is not None:
                                label += f" - F_S: {fs:.4f}"
                            all_options[pid] = label

                    def on_export_change(e):
                        export_state['selected_pid'] = e.value
                        if e.value == judge_state['recommended_pid']:
                            export_state['selection_method'] = 'LLM Judge Recommendation'
                        else:
                            export_state['selection_method'] = 'Manual Override'

                    export_select = ui.select(
                        options=all_options,
                        value=judge_state['recommended_pid'],
                        label='Answer to Export',
                        on_change=on_export_change
                    ).classes('w-full mb-4')

                    # Export buttons
                    with ui.row().classes('w-full gap-4 mt-4'):
                        async def export_markdown():
                            pid = export_state['selected_pid']
                            if not pid:
                                ui.notify('Please select an answer', type='warning')
                                return

                            question = prompts_dict[pid]['text']
                            context = prompts_dict[pid]['context']
                            answer = answers_dict[pid]['answer']
                            fs = fs_scores.get(pid, 0)

                            md_content = generate_markdown_export(
                                question=question,
                                context=context,
                                answer=answer,
                                prompt_id=pid,
                                fs_score=fs,
                                selection_method=export_state['selection_method'],
                                judge_result=judge_state['result']
                            )

                            filename = f"answer_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                            ui.download(md_content.encode('utf-8'), filename)
                            ui.notify(f'Exported as {filename}', type='positive')

                        async def export_pdf():
                            pid = export_state['selected_pid']
                            if not pid:
                                ui.notify('Please select an answer', type='warning')
                                return

                            try:
                                from reportlab.lib.pagesizes import letter
                                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
                                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                                from reportlab.lib.units import inch
                                from reportlab.lib.enums import TA_LEFT
                                import io

                                question = prompts_dict[pid]['text']
                                context = prompts_dict[pid]['context']
                                answer = answers_dict[pid]['answer']
                                fs = fs_scores.get(pid, 0)

                                buffer = io.BytesIO()
                                doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)

                                styles = getSampleStyleSheet()

                                # Define styles for different markdown elements
                                title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=18, spaceAfter=16, fontName='Helvetica-Bold')
                                h1_style = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=16, spaceAfter=10, spaceBefore=16, fontName='Helvetica-Bold')
                                h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, spaceAfter=8, spaceBefore=14, fontName='Helvetica-Bold')
                                h3_style = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=12, spaceAfter=6, spaceBefore=12, fontName='Helvetica-Bold')
                                body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=6, leading=14)
                                bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], fontSize=10, spaceAfter=4, leftIndent=20, leading=14)
                                meta_style = ParagraphStyle('Meta', parent=styles['Normal'], fontSize=9, textColor='gray', spaceAfter=4)

                                story = []

                                # Title and metadata
                                story.append(Paragraph("LLM Judge - Selected Answer Report", title_style))
                                story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
                                story.append(Paragraph(f"Prompt ID: {pid} | F_S Score: {fs:.4f} | Selection: {export_state['selection_method']}", meta_style))
                                story.append(Spacer(1, 16))

                                # Question section
                                story.append(Paragraph("Question", h2_style))
                                safe_question = question.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                story.append(Paragraph(safe_question, body_style))
                                story.append(Spacer(1, 12))

                                # Answer section - parse markdown
                                story.append(Paragraph("Selected Answer", h2_style))
                                story.extend(parse_markdown_to_pdf(answer, h1_style, h2_style, h3_style, body_style, bullet_style))

                                doc.build(story)

                                filename = f"answer_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                ui.download(buffer.getvalue(), filename)
                                ui.notify(f'Exported as {filename}', type='positive')

                            except ImportError:
                                ui.notify('PDF export requires reportlab. Install with: pip install reportlab', type='negative')
                            except Exception as e:
                                ui.notify(f'PDF export error: {str(e)}', type='negative')

                        ui.button('Export as Markdown', icon='description', on_click=export_markdown).props('color=primary')
                        ui.button('Export as PDF', icon='picture_as_pdf', on_click=export_pdf).props('outline')

        # Bind run button
        run_button.on_click(run_judge)


def parse_markdown_to_pdf(text: str, h1_style, h2_style, h3_style, body_style, bullet_style):
    """
    Parse markdown text and convert to reportlab Paragraph elements.
    Handles headers (##, ###), bold (**text**), and bullet points (- text).
    """
    from reportlab.platypus import Paragraph, Spacer

    elements = []
    lines = text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Escape HTML special characters
        def escape_html(s):
            return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Convert markdown bold **text** to reportlab bold <b>text</b>
        def convert_bold(s):
            import re
            # Handle **bold** syntax
            s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
            return s

        # Check for headers
        if line.startswith('### '):
            header_text = escape_html(line[4:])
            header_text = convert_bold(header_text)
            elements.append(Paragraph(header_text, h3_style))

        elif line.startswith('## '):
            header_text = escape_html(line[3:])
            header_text = convert_bold(header_text)
            elements.append(Paragraph(header_text, h2_style))

        elif line.startswith('# '):
            header_text = escape_html(line[2:])
            header_text = convert_bold(header_text)
            elements.append(Paragraph(header_text, h1_style))

        # Check for bullet points
        elif line.startswith('- '):
            bullet_text = escape_html(line[2:])
            bullet_text = convert_bold(bullet_text)
            elements.append(Paragraph(f"• {bullet_text}", bullet_style))

        # Check for numbered list
        elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
            # Extract the number and text
            import re
            match = re.match(r'^(\d+)[.):\s]+(.*)$', line)
            if match:
                num, text_content = match.groups()
                text_content = escape_html(text_content)
                text_content = convert_bold(text_content)
                elements.append(Paragraph(f"{num}. {text_content}", bullet_style))
            else:
                para_text = escape_html(line)
                para_text = convert_bold(para_text)
                elements.append(Paragraph(para_text, body_style))

        # Regular paragraph
        else:
            para_text = escape_html(line)
            para_text = convert_bold(para_text)
            elements.append(Paragraph(para_text, body_style))

        i += 1

    return elements


def generate_markdown_export(question, context, answer, prompt_id, fs_score, selection_method, judge_result):
    """Generate Markdown export with judge results"""
    md = f"""# LLM Judge - Selected Answer Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Selection Details

| Field | Value |
|-------|-------|
| Prompt ID | {prompt_id} |
| F_S Score | {fs_score:.4f} |
| Selection Method | {selection_method} |

"""

    if judge_result:
        md += f"""
## LLM Judge Evaluation

**Winner:** {judge_result.get('winner', 'N/A')}

**Scores:**
- Answer A: {judge_result.get('scores', {}).get('A', 'N/A')}/10
- Answer B: {judge_result.get('scores', {}).get('B', 'N/A')}/10

**Explanation:**
{judge_result.get('explanation', 'No explanation provided')}

"""

    md += f"""---

## Question

{question}

---

## Selected Answer

{answer}

---

## Context Summary

*Context length: {len(context)} characters*

{context[:1000]}{'...' if len(context) > 1000 else ''}

---

*Report generated by Semantic Faithfulness Analyzer - LLM Judge*
"""
    return md


def generate_verdict_export(
    judge_model_name: str,
    left_pid: str,
    right_pid: str,
    judge_result: dict,
    left_answer: str,
    right_answer: str,
    fs_scores: dict
) -> str:
    """
    Generate a comprehensive LLM Judge Verdict export in Markdown format.

    Args:
        judge_model_name: Display name of the judge model
        left_pid: Prompt ID for Answer A
        right_pid: Prompt ID for Answer B
        judge_result: The full judge result dictionary
        left_answer: Full text of Answer A
        right_answer: Full text of Answer B
        fs_scores: Dictionary of F_S scores by prompt_id

    Returns:
        Markdown formatted verdict document
    """
    from datetime import datetime

    winner = judge_result.get('winner', 'TIE')
    scores = judge_result.get('scores', {})
    criteria = judge_result.get('criteria_breakdown', {})
    key_diffs = judge_result.get('key_differences', {})
    explanation = judge_result.get('explanation', 'No explanation provided')

    # Determine winner text
    if winner == 'A':
        winner_text = f"{left_pid} (Answer A)"
    elif winner == 'B':
        winner_text = f"{right_pid} (Answer B)"
    else:
        winner_text = "TIE - No clear winner"

    fs_left = fs_scores.get(left_pid, 'N/A')
    fs_right = fs_scores.get(right_pid, 'N/A')
    if isinstance(fs_left, float):
        fs_left = f"{fs_left:.4f}"
    if isinstance(fs_right, float):
        fs_right = f"{fs_right:.4f}"

    md = f"""# LLM Judge Verdict

---

## Summary

| Field | Value |
|-------|-------|
| **LLM Judge** | {judge_model_name} |
| **Date** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| **Comparison** | {left_pid} vs {right_pid} |
| **Verdict** | **The Winner is {winner_text}** |

---

## Scores Overview

| Answer | LLM Score | F_S Score |
|--------|-----------|-----------|
| **{left_pid}** (Answer A) | {scores.get('A', 'N/A')}/10 | {fs_left} |
| **{right_pid}** (Answer B) | {scores.get('B', 'N/A')}/10 | {fs_right} |

"""

    # Criteria breakdown
    if criteria:
        md += """## Criteria Breakdown

| Criterion | Answer A | Answer B | Difference | Better |
|-----------|----------|----------|------------|--------|
"""
        for criterion, scores_dict in criteria.items():
            score_a = scores_dict.get('A', 0)
            score_b = scores_dict.get('B', 0)
            diff = score_a - score_b
            if diff > 0:
                better = "A"
                diff_str = f"+{diff}"
            elif diff < 0:
                better = "B"
                diff_str = str(diff)
            else:
                better = "="
                diff_str = "0"
            md += f"| {criterion.replace('_', ' ').title()} | {score_a}/10 | {score_b}/10 | {diff_str} | {better} |\n"
        md += "\n"

    # Key semantic differences
    diffs_a = key_diffs.get('A', [])
    diffs_b = key_diffs.get('B', [])

    if diffs_a or diffs_b:
        md += """---

## Key Semantic Differences

*These phrases were identified by the LLM Judge as representing meaningful differences between the answers.*

"""
        if diffs_a:
            md += f"### Distinctive phrases in {left_pid} (Answer A):\n\n"
            for i, diff in enumerate(diffs_a, 1):
                md += f'{i}. > "{diff}"\n\n'

        if diffs_b:
            md += f"### Distinctive phrases in {right_pid} (Answer B):\n\n"
            for i, diff in enumerate(diffs_b, 1):
                md += f'{i}. > "{diff}"\n\n'

    # Detailed explanation
    md += f"""---

## Detailed Analysis

{explanation}

---

## Full Answer Texts

### {left_pid} (Answer A)

{left_answer}

---

### {right_pid} (Answer B)

{right_answer}

---

*Verdict generated by Semantic Faithfulness Analyzer - LLM Judge*
*Model: {judge_model_name}*
"""
    return md


def generate_verdict_pdf(
    judge_model_name: str,
    left_pid: str,
    right_pid: str,
    judge_result: dict,
    left_answer: str,
    right_answer: str,
    fs_scores: dict
) -> bytes:
    """
    Generate a comprehensive LLM Judge Verdict export in PDF format.

    Args:
        judge_model_name: Display name of the judge model
        left_pid: Prompt ID for Answer A
        right_pid: Prompt ID for Answer B
        judge_result: The full judge result dictionary
        left_answer: Full text of Answer A
        right_answer: Full text of Answer B
        fs_scores: Dictionary of F_S scores by prompt_id

    Returns:
        PDF content as bytes
    """
    from datetime import datetime
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    import io

    winner = judge_result.get('winner', 'TIE')
    scores = judge_result.get('scores', {})
    criteria = judge_result.get('criteria_breakdown', {})
    key_diffs = judge_result.get('key_differences', {})
    explanation = judge_result.get('explanation', 'No explanation provided')

    # Determine winner text
    if winner == 'A':
        winner_text = f"{left_pid} (Answer A)"
    elif winner == 'B':
        winner_text = f"{right_pid} (Answer B)"
    else:
        winner_text = "TIE - No clear winner"

    fs_left = fs_scores.get(left_pid, 'N/A')
    fs_right = fs_scores.get(right_pid, 'N/A')
    if isinstance(fs_left, float):
        fs_left = f"{fs_left:.4f}"
    if isinstance(fs_right, float):
        fs_right = f"{fs_right:.4f}"

    # Create PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()

    # Define custom styles
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, spaceAfter=20, fontName='Helvetica-Bold', textColor=colors.HexColor('#4a148c'))
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=14, spaceAfter=10, spaceBefore=16, fontName='Helvetica-Bold', textColor=colors.HexColor('#1a237e'))
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, spaceAfter=8, spaceBefore=12, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=6, leading=14)
    quote_style = ParagraphStyle('Quote', parent=styles['Normal'], fontSize=9, spaceAfter=4, leftIndent=20, textColor=colors.HexColor('#37474f'), fontName='Helvetica-Oblique')
    meta_style = ParagraphStyle('Meta', parent=styles['Normal'], fontSize=9, textColor=colors.gray, spaceAfter=4)
    verdict_style = ParagraphStyle('Verdict', parent=styles['Normal'], fontSize=14, fontName='Helvetica-Bold', textColor=colors.HexColor('#1b5e20'), spaceAfter=12)

    story = []

    # Title
    story.append(Paragraph("LLM Judge Verdict", title_style))
    story.append(Spacer(1, 12))

    # Summary table
    summary_data = [
        ['Field', 'Value'],
        ['LLM Judge', judge_model_name],
        ['Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Comparison', f'{left_pid} vs {right_pid}'],
    ]
    summary_table = Table(summary_data, colWidths=[1.5*inch, 4.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eaf6')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    # Verdict
    story.append(Paragraph(f"Verdict: The Winner is {winner_text}", verdict_style))
    story.append(Spacer(1, 16))

    # Scores Overview
    story.append(Paragraph("Scores Overview", h1_style))
    scores_data = [
        ['Answer', 'LLM Score', 'F_S Score'],
        [f'{left_pid} (Answer A)', f"{scores.get('A', 'N/A')}/10", str(fs_left)],
        [f'{right_pid} (Answer B)', f"{scores.get('B', 'N/A')}/10", str(fs_right)],
    ]
    scores_table = Table(scores_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    scores_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e3f2fd')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(scores_table)
    story.append(Spacer(1, 16))

    # Criteria Breakdown
    if criteria:
        story.append(Paragraph("Criteria Breakdown", h1_style))
        criteria_data = [['Criterion', 'Answer A', 'Answer B', 'Diff', 'Better']]
        for criterion, scores_dict in criteria.items():
            score_a = scores_dict.get('A', 0)
            score_b = scores_dict.get('B', 0)
            diff = score_a - score_b
            if diff > 0:
                better = "A"
                diff_str = f"+{diff}"
            elif diff < 0:
                better = "B"
                diff_str = str(diff)
            else:
                better = "="
                diff_str = "0"
            criteria_data.append([criterion.replace('_', ' ').title(), f'{score_a}/10', f'{score_b}/10', diff_str, better])

        criteria_table = Table(criteria_data, colWidths=[1.8*inch, 1*inch, 1*inch, 0.8*inch, 0.8*inch])
        criteria_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fff3e0')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(criteria_table)
        story.append(Spacer(1, 16))

    # Key Semantic Differences
    diffs_a = key_diffs.get('A', [])
    diffs_b = key_diffs.get('B', [])

    if diffs_a or diffs_b:
        story.append(Paragraph("Key Semantic Differences", h1_style))
        story.append(Paragraph("These phrases were identified by the LLM Judge as representing meaningful differences between the answers.", meta_style))
        story.append(Spacer(1, 8))

        if diffs_a:
            story.append(Paragraph(f"Distinctive phrases in {left_pid} (Answer A):", h2_style))
            for i, diff in enumerate(diffs_a, 1):
                safe_diff = diff.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(f'{i}. "{safe_diff}"', quote_style))
            story.append(Spacer(1, 8))

        if diffs_b:
            story.append(Paragraph(f"Distinctive phrases in {right_pid} (Answer B):", h2_style))
            for i, diff in enumerate(diffs_b, 1):
                safe_diff = diff.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(f'{i}. "{safe_diff}"', quote_style))
            story.append(Spacer(1, 8))

    # Detailed Analysis
    story.append(Paragraph("Detailed Analysis", h1_style))
    # Split explanation into paragraphs and escape HTML
    for para in explanation.split('\n\n'):
        if para.strip():
            safe_para = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            # Handle basic markdown bold
            import re
            safe_para = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', safe_para)
            story.append(Paragraph(safe_para, body_style))
    story.append(Spacer(1, 16))

    # Full Answer Texts
    story.append(Paragraph(f"{left_pid} (Answer A)", h1_style))
    for para in left_answer.split('\n\n'):
        if para.strip():
            safe_para = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe_para, body_style))
    story.append(Spacer(1, 16))

    story.append(Paragraph(f"{right_pid} (Answer B)", h1_style))
    for para in right_answer.split('\n\n'):
        if para.strip():
            safe_para = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe_para, body_style))
    story.append(Spacer(1, 16))

    # Footer
    story.append(Paragraph(f"Verdict generated by Semantic Faithfulness Analyzer - LLM Judge | Model: {judge_model_name}", meta_style))

    doc.build(story)
    return buffer.getvalue()
