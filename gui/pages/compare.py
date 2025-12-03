"""
Comparison Page for Semantic Faithfulness GUI
Allows side-by-side comparison of prompts and answers with diff highlighting
Default: Initial answer (left) vs Lowest F_S answer (right)
"""

from pathlib import Path
import json
from nicegui import ui, app
from pages.markdown_utils import generate_diff_html, markdown_to_html


def create():
    """Create the comparison page"""

    # Check for LLM pipeline results first (from current session)
    llm_results = app.storage.user.get('llm_pipeline_results')

    if llm_results and llm_results.get('triplets'):
        # Use results from current LLM pipeline run
        triplets = llm_results['triplets']
        fs_scores_raw = llm_results.get('fs_scores', {})

        # Build prompts and answers from triplets
        prompts_list = []
        answers_list = []
        for i, t in enumerate(triplets):
            prompt_id = t.get('prompt_id', f'prompt_{i}')
            prompts_list.append({
                'prompt_id': prompt_id,
                'text': t.get('question', ''),
                'context': t.get('context', ''),
                'group': t.get('group', 'default')
            })
            answers_list.append({
                'prompt_id': prompt_id,
                'answer': t.get('answer', ''),
                'group': t.get('group', 'default')
            })

        # Normalize fs_scores - handle nested dict structure from pipeline
        fs_scores = {}
        for pid, score_data in dict(fs_scores_raw).items():
            if isinstance(score_data, dict):
                fs_scores[pid] = float(score_data.get('F_S', 0))
            else:
                fs_scores[pid] = float(score_data) if score_data is not None else 0.0

        data_loaded = True
        data_source = "current session"
    else:
        data_loaded = False
        fs_scores = {}
        prompts_list = []
        answers_list = []

    # Main container
    with ui.column().classes('w-full max-w-7xl mx-auto p-8'):
        ui.label('Answer Comparison').classes('text-h4 mb-2')
        ui.label('Compare different answer variants side-by-side').classes('text-subtitle1 text-grey-7 mb-6')

        if not data_loaded:
            with ui.card().classes('w-full p-6 bg-yellow-50'):
                ui.label('No Comparison Data Available').classes('text-h6 text-yellow-800')
                ui.label(
                    'The comparison page requires data from an LLM pipeline run with multiple paraphrases. '
                    'Please go to the Input page and run the pipeline with 2 or more paraphrases.'
                ).classes('text-body1 text-yellow-700 mt-2')
                ui.button('Go to Input', on_click=lambda: ui.navigate.to('/input')).props('color=primary').classes('mt-4')
            return

        if len(prompts_list) < 2:
            with ui.card().classes('w-full p-6 bg-yellow-50'):
                ui.label('Not Enough Data').classes('text-h6 text-yellow-800')
                ui.label(
                    'You need at least 2 answer variants to compare. '
                    'Please run the pipeline with more paraphrases.'
                ).classes('text-body1 text-yellow-700 mt-2')
                ui.button('Go to Input', on_click=lambda: ui.navigate.to('/input')).props('color=primary').classes('mt-4')
            return

        # Create lookups
        prompts_dict = {p['prompt_id']: p for p in prompts_list}
        answers_dict = {a['prompt_id']: a for a in answers_list}

        # Determine default selections:
        # Left: Initial answer (prompt_0)
        # Right: Lowest F_S score answer
        prompt_ids_list = list(prompts_dict.keys())
        initial_pid = 'prompt_0' if 'prompt_0' in prompt_ids_list else prompt_ids_list[0]

        # Find lowest F_S (worst faithfulness)
        if fs_scores:
            sorted_by_fs = sorted(fs_scores.items(), key=lambda x: x[1])
            lowest_fs_pid = sorted_by_fs[0][0]
            # If initial is also the lowest, pick second lowest
            if lowest_fs_pid == initial_pid and len(sorted_by_fs) > 1:
                lowest_fs_pid = sorted_by_fs[1][0]
        else:
            lowest_fs_pid = prompt_ids_list[1] if len(prompt_ids_list) > 1 else prompt_ids_list[0]

        # Build dropdown options with F_S scores
        def make_options():
            options = {}
            for pid in prompt_ids_list:
                fs = fs_scores.get(pid)
                if fs is not None:
                    options[pid] = f"{pid} (F_S: {fs:.4f})"
                else:
                    options[pid] = pid
            return options

        triplet_options = make_options()

        # State for comparison
        state = {
            'left_pid': initial_pid,
            'right_pid': lowest_fs_pid
        }

        # Info card about default selection
        with ui.card().classes('w-full p-4 mb-4 bg-blue-50'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('info', color='blue')
                ui.label('Default comparison: Initial answer vs Lowest F_S answer').classes('text-subtitle2')
            if fs_scores:
                initial_fs = fs_scores.get(initial_pid, 0)
                lowest_fs = fs_scores.get(lowest_fs_pid, 0)
                ui.label(
                    f"Initial ({initial_pid}): F_S = {initial_fs:.4f} | "
                    f"Lowest ({lowest_fs_pid}): F_S = {lowest_fs:.4f}"
                ).classes('text-caption text-grey-7 ml-8')

        # Store references in a dict for access from nested functions
        selectors = {}
        comparison_container = None  # Will be set after UI is created

        # Define do_compare function that will be referenced by buttons
        async def do_compare():
            """Perform the comparison"""
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

            if not left_pid or not right_pid:
                with comparison_container:
                    ui.label('Please select answers to compare').classes('text-orange-600')
                return

            if left_pid == right_pid:
                with comparison_container:
                    ui.label('Please select two different answers to compare').classes('text-orange-600')
                return

            # Update state
            state['left_pid'] = left_pid
            state['right_pid'] = right_pid

            left_answer = answers_dict[left_pid]
            right_answer = answers_dict[right_pid]
            left_prompt = prompts_dict[left_pid]
            right_prompt = prompts_dict[right_pid]

            fs_left = fs_scores.get(left_pid)
            fs_right = fs_scores.get(right_pid)

            with comparison_container:
                # Tabs for prompts vs answers comparison
                with ui.tabs().classes('w-full') as tabs:
                    answers_tab = ui.tab('Answers')
                    prompts_tab = ui.tab('Prompts')

                with ui.tab_panels(tabs, value=answers_tab).classes('w-full'):
                    # Answers comparison
                    with ui.tab_panel(answers_tab):
                        with ui.row().classes('w-full gap-4'):
                            # Left pane
                            with ui.card().classes('flex-1 p-6'):
                                with ui.row().classes('items-center gap-2 mb-2'):
                                    ui.label(left_pid).classes('text-h6')
                                    if left_pid == initial_pid:
                                        ui.badge('Initial', color='blue').props('outline')
                                if fs_left is not None:
                                    color = 'text-red-600' if fs_left == min(fs_scores.values()) else 'text-primary'
                                    ui.label(f"F_S: {fs_left:.4f}").classes(f'text-subtitle2 {color} mb-2')
                                ui.label(f"{len(left_answer['answer'])} characters").classes('text-caption text-grey-6 mb-4')

                                # Diff-highlighted text
                                html_left = generate_diff_html(left_answer['answer'], right_answer['answer'], side='left')
                                ui.html(html_left, sanitize=False).classes('text-body2 whitespace-pre-wrap')

                            # Right pane
                            with ui.card().classes('flex-1 p-6'):
                                with ui.row().classes('items-center gap-2 mb-2'):
                                    ui.label(right_pid).classes('text-h6')
                                    if fs_right is not None and fs_right == min(fs_scores.values()):
                                        ui.badge('Lowest F_S', color='red').props('outline')
                                if fs_right is not None:
                                    color = 'text-red-600' if fs_right == min(fs_scores.values()) else 'text-primary'
                                    ui.label(f"F_S: {fs_right:.4f}").classes(f'text-subtitle2 {color} mb-2')
                                ui.label(f"{len(right_answer['answer'])} characters").classes('text-caption text-grey-6 mb-4')

                                # Diff-highlighted text
                                html_right = generate_diff_html(left_answer['answer'], right_answer['answer'], side='right')
                                ui.html(html_right, sanitize=False).classes('text-body2 whitespace-pre-wrap')

                    # Prompts comparison
                    with ui.tab_panel(prompts_tab):
                        with ui.row().classes('w-full gap-4'):
                            # Left pane
                            with ui.card().classes('flex-1 p-6'):
                                ui.label(left_pid).classes('text-h6 mb-2')
                                left_text = left_prompt.get('text') or left_prompt.get('question', '')
                                ui.label(f"{len(left_text)} characters").classes('text-caption text-grey-6 mb-4')

                                html_left = generate_diff_html(left_text, right_prompt.get('text') or right_prompt.get('question', ''), side='left')
                                ui.html(html_left, sanitize=False).classes('text-body2 whitespace-pre-wrap')

                            # Right pane
                            with ui.card().classes('flex-1 p-6'):
                                ui.label(right_pid).classes('text-h6 mb-2')
                                right_text = right_prompt.get('text') or right_prompt.get('question', '')
                                ui.label(f"{len(right_text)} characters").classes('text-caption text-grey-6 mb-4')

                                html_right = generate_diff_html(left_prompt.get('text') or left_prompt.get('question', ''), right_text, side='right')
                                ui.html(html_right, sanitize=False).classes('text-body2 whitespace-pre-wrap')

                # Legend
                ui.separator().classes('my-4')
                with ui.row().classes('items-center gap-4'):
                    ui.html('<span style="background-color: #ffeb3b; padding: 2px 8px;">Highlighted</span>', sanitize=False)
                    ui.label('= Text unique to this version or substantially different').classes('text-caption text-grey-6')

        # Selection controls card
        with ui.card().classes('w-full p-6 mb-4'):
            ui.label('Select Answers to Compare').classes('text-h6 mb-4')

            with ui.row().classes('w-full gap-4 items-end'):
                selectors['left'] = ui.select(
                    label='Left Pane (Baseline)',
                    options=triplet_options,
                    value=initial_pid
                ).classes('flex-1')

                selectors['right'] = ui.select(
                    label='Right Pane (Comparison)',
                    options=triplet_options,
                    value=lowest_fs_pid
                ).classes('flex-1')

            compare_btn = ui.button('Compare', icon='compare_arrows').props('color=primary')
            compare_btn.on('click', do_compare)

            # Quick selection buttons
            with ui.row().classes('w-full gap-4 mt-4'):
                async def select_initial_vs_highest():
                    """Select Initial vs Highest F_S (best)"""
                    if fs_scores:
                        sorted_by_fs = sorted(fs_scores.items(), key=lambda x: x[1], reverse=True)
                        highest_fs_pid = sorted_by_fs[0][0]
                        selectors['left'].value = initial_pid
                        selectors['right'].value = highest_fs_pid
                        selectors['left'].update()
                        selectors['right'].update()
                        await do_compare()

                async def select_initial_vs_lowest():
                    """Select Initial vs Lowest F_S"""
                    selectors['left'].value = initial_pid
                    selectors['right'].value = lowest_fs_pid
                    selectors['left'].update()
                    selectors['right'].update()
                    await do_compare()

                async def select_highest_vs_lowest():
                    """Select Highest F_S vs Lowest F_S"""
                    if fs_scores:
                        sorted_by_fs = sorted(fs_scores.items(), key=lambda x: x[1])
                        lowest = sorted_by_fs[0][0]
                        highest = sorted_by_fs[-1][0]
                        selectors['left'].value = highest
                        selectors['right'].value = lowest
                        selectors['left'].update()
                        selectors['right'].update()
                        await do_compare()

                ui.button('Initial vs Highest F_S', on_click=select_initial_vs_highest, icon='arrow_upward').props('outline size=sm')
                ui.button('Initial vs Lowest F_S', on_click=select_initial_vs_lowest, icon='arrow_downward').props('outline size=sm')
                ui.button('Highest vs Lowest F_S', on_click=select_highest_vs_lowest, icon='swap_vert').props('outline size=sm')

        # Now create the comparison container (after controls)
        comparison_container = ui.column().classes('w-full')

        # Auto-compare on load
        async def init_compare():
            await do_compare()
        ui.timer(0.1, init_compare, once=True)

        # Action buttons - Proceed to LLM Judge
        ui.separator().classes('my-6')

        with ui.card().classes('w-full p-6 bg-green-50'):
            ui.label('Next Step: LLM-as-a-Judge Evaluation').classes('text-h6 mb-2')
            ui.label(
                'Use an LLM to evaluate which answer is better based on faithfulness, completeness, '
                'coherence, and relevance to the context.'
            ).classes('text-body2 text-grey-7 mb-4')

            async def go_to_judge():
                """Save current selection and go to Judge page"""
                # Store the current comparison selection for the Judge page
                app.storage.user['compare_selection'] = {
                    'left_pid': selectors['left'].value,
                    'right_pid': selectors['right'].value
                }
                ui.navigate.to('/judge')

            with ui.row().classes('gap-4'):
                ui.button('Analyze with LLM Judge', icon='gavel', on_click=go_to_judge).props('color=primary size=lg')
                ui.button('Go to Results', on_click=lambda: ui.navigate.to('/results')).props('outline')
