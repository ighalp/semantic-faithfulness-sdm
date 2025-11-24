"""
Comparison Page for Semantic Faithfulness GUI
Allows side-by-side comparison of prompts and answers with diff highlighting
"""

from pathlib import Path
import json
import difflib
import asyncio
import numpy as np
from nicegui import ui, app


def create():
    """Create the comparison page"""

    # Load data from external directory
    try:
        nvidia_dir = Path.home() / 'home' / 'Python' / 'LLMs' / 'nvidia_rich_qca_results'

        # Load prompts
        prompts_file = nvidia_dir / 'prompts_v2.json'
        with open(prompts_file, 'r') as f:
            prompts_data = json.load(f)
        prompts_list = prompts_data['prompts']

        # Load answers
        answers_file = nvidia_dir / 'answers_v2.json'
        with open(answers_file, 'r') as f:
            answers_data = json.load(f)
        answers_list = answers_data['answers']

        # Load distributions for F_S computation
        dist_file = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'distributions' / 'distributions_v2.json'
        with open(dist_file, 'r') as f:
            dist_data = json.load(f)
        distributions = {t['prompt_id']: t for t in dist_data['triplets']}

        data_loaded = True
    except FileNotFoundError as e:
        data_loaded = False
        error_msg = f"Data files not found: {e}"
    except Exception as e:
        data_loaded = False
        error_msg = f"Error loading data: {e}"

    # Main container
    with ui.column().classes('w-full max-w-7xl mx-auto p-8'):
        ui.label('Text Comparison').classes('text-h4 mb-6')

        if not data_loaded:
            ui.label(f'Error: {error_msg}').classes('text-red-600')
            return

        # Create prompt and answer lookups
        prompts_dict = {p['prompt_id']: p for p in prompts_list}
        answers_dict = {a['prompt_id']: a for a in answers_list}

        # Dropdown options - format as dict {value: label}
        triplet_options = {
            p['prompt_id']: f"{p['prompt_id']} (Group {p['group']})"
            for p in prompts_list
        }

        # State for comparison
        state = {
            'fs_scores': app.storage.user.get('comparison_fs_scores'),
            'computing_fs': False,
            'selected_1': None,
            'selected_2': None,
            'comparison_mode': 'prompts'  # 'prompts' or 'answers'
        }

        # Create tabs for prompts vs answers
        with ui.tabs().classes('w-full') as tabs:
            prompts_tab = ui.tab('Prompts Comparison')
            answers_tab = ui.tab('Answers Comparison')

        with ui.tab_panels(tabs, value=prompts_tab).classes('w-full'):
            # PROMPTS COMPARISON TAB
            with ui.tab_panel(prompts_tab):
                with ui.card().classes('w-full p-6 mb-4'):
                    ui.label('Select Prompts to Compare').classes('text-h6 mb-4')

                    with ui.row().classes('w-full gap-4 items-end'):
                        # Get first two prompt IDs for default values
                        prompt_ids_list = list(triplet_options.keys())

                        select1_prompts = ui.select(
                            label='First Prompt',
                            options=triplet_options,
                            value=prompt_ids_list[0]
                        ).classes('flex-1')

                        select2_prompts = ui.select(
                            label='Second Prompt',
                            options=triplet_options,
                            value=prompt_ids_list[1] if len(prompt_ids_list) > 1 else prompt_ids_list[0]
                        ).classes('flex-1')

                        compare_btn_prompts = ui.button('Compare', icon='compare_arrows').props('color=primary')
                        best_worst_btn_prompts = ui.button('Best vs Worst', icon='filter_alt').props('outline')

                # Comparison results container for prompts
                comparison_container_prompts = ui.column().classes('w-full')

                async def compare_prompts():
                    """Compare selected prompts"""
                    comparison_container_prompts.clear()

                    id1 = select1_prompts.value
                    id2 = select2_prompts.value

                    if id1 == id2:
                        with comparison_container_prompts:
                            ui.label('Please select two different prompts to compare').classes('text-orange-600')
                        return

                    prompt1 = prompts_dict[id1]
                    prompt2 = prompts_dict[id2]

                    # Get F_S scores if available
                    fs1 = state['fs_scores'].get(id1) if state['fs_scores'] else None
                    fs2 = state['fs_scores'].get(id2) if state['fs_scores'] else None

                    with comparison_container_prompts:
                        # Display side-by-side
                        with ui.row().classes('w-full gap-4'):
                            # Left side
                            with ui.card().classes('flex-1 p-6'):
                                ui.label(f"{id1} (Group {prompt1['group']})").classes('text-h6 mb-2')
                                if fs1 is not None:
                                    ui.label(f"F_S: {fs1:.3f}").classes('text-subtitle2 text-primary mb-2')
                                ui.label(f"{len(prompt1['text'])} characters").classes('text-caption text-grey-6 mb-4')

                                # Highlighted text
                                html1 = generate_diff_html(prompt1['text'], prompt2['text'], side='left')
                                ui.html(html1, sanitize=False).classes('text-body2 whitespace-pre-wrap')

                            # Right side
                            with ui.card().classes('flex-1 p-6'):
                                ui.label(f"{id2} (Group {prompt2['group']})").classes('text-h6 mb-2')
                                if fs2 is not None:
                                    ui.label(f"F_S: {fs2:.3f}").classes('text-subtitle2 text-primary mb-2')
                                ui.label(f"{len(prompt2['text'])} characters").classes('text-caption text-grey-6 mb-4')

                                # Highlighted text
                                html2 = generate_diff_html(prompt1['text'], prompt2['text'], side='right')
                                ui.html(html2, sanitize=False).classes('text-body2 whitespace-pre-wrap')

                async def best_worst_prompts():
                    """Select best and worst prompts based on F_S scores"""
                    # Check if F_S scores are computed
                    if not state['fs_scores']:
                        # Ask user to compute
                        with ui.dialog() as dialog, ui.card():
                            ui.label('Compute F_S Scores?').classes('text-h6 mb-4')
                            ui.label('Computing F_S scores for all 10 triplets will take approximately 2 minutes.').classes('mb-4')
                            ui.label('Do you want to proceed?').classes('mb-4')
                            with ui.row().classes('w-full justify-end gap-2'):
                                ui.button('Cancel', on_click=dialog.close).props('outline')
                                ui.button('Compute', on_click=lambda: [dialog.close(), compute_all_fs()]).props('color=primary')
                        dialog.open()
                    else:
                        # Select best and worst
                        sorted_ids = sorted(state['fs_scores'].items(), key=lambda x: x[1])
                        worst_id = sorted_ids[0][0]
                        best_id = sorted_ids[-1][0]

                        select1_prompts.value = best_id
                        select2_prompts.value = worst_id
                        await compare_prompts()

                compare_btn_prompts.on_click(compare_prompts)
                best_worst_btn_prompts.on_click(best_worst_prompts)

                # Auto-compare on page load
                ui.timer(0.1, lambda: compare_prompts(), once=True)

            # ANSWERS COMPARISON TAB
            with ui.tab_panel(answers_tab):
                with ui.card().classes('w-full p-6 mb-4'):
                    ui.label('Select Answers to Compare').classes('text-h6 mb-4')

                    with ui.row().classes('w-full gap-4 items-end'):
                        select1_answers = ui.select(
                            label='First Answer',
                            options=triplet_options,
                            value=prompt_ids_list[0]
                        ).classes('flex-1')

                        select2_answers = ui.select(
                            label='Second Answer',
                            options=triplet_options,
                            value=prompt_ids_list[1] if len(prompt_ids_list) > 1 else prompt_ids_list[0]
                        ).classes('flex-1')

                        compare_btn_answers = ui.button('Compare', icon='compare_arrows').props('color=primary')
                        best_worst_btn_answers = ui.button('Best vs Worst', icon='filter_alt').props('outline')

                # Comparison results container for answers
                comparison_container_answers = ui.column().classes('w-full')

                async def compare_answers():
                    """Compare selected answers"""
                    comparison_container_answers.clear()

                    id1 = select1_answers.value
                    id2 = select2_answers.value

                    if id1 == id2:
                        with comparison_container_answers:
                            ui.label('Please select two different answers to compare').classes('text-orange-600')
                        return

                    answer1 = answers_dict[id1]
                    answer2 = answers_dict[id2]

                    # Get F_S scores if available
                    fs1 = state['fs_scores'].get(id1) if state['fs_scores'] else None
                    fs2 = state['fs_scores'].get(id2) if state['fs_scores'] else None

                    with comparison_container_answers:
                        # Display side-by-side
                        with ui.row().classes('w-full gap-4'):
                            # Left side
                            with ui.card().classes('flex-1 p-6'):
                                ui.label(f"{id1}").classes('text-h6 mb-2')
                                if fs1 is not None:
                                    ui.label(f"F_S: {fs1:.3f}").classes('text-subtitle2 text-primary mb-2')
                                ui.label(f"{len(answer1['answer'])} characters").classes('text-caption text-grey-6 mb-4')

                                # Highlighted text
                                html1 = generate_diff_html(answer1['answer'], answer2['answer'], side='left')
                                ui.html(html1, sanitize=False).classes('text-body2 whitespace-pre-wrap')

                            # Right side
                            with ui.card().classes('flex-1 p-6'):
                                ui.label(f"{id2}").classes('text-h6 mb-2')
                                if fs2 is not None:
                                    ui.label(f"F_S: {fs2:.3f}").classes('text-subtitle2 text-primary mb-2')
                                ui.label(f"{len(answer2['answer'])} characters").classes('text-caption text-grey-6 mb-4')

                                # Highlighted text
                                html2 = generate_diff_html(answer1['answer'], answer2['answer'], side='right')
                                ui.html(html2, sanitize=False).classes('text-body2 whitespace-pre-wrap')

                async def best_worst_answers():
                    """Select best and worst answers based on F_S scores"""
                    # Check if F_S scores are computed
                    if not state['fs_scores']:
                        # Ask user to compute
                        with ui.dialog() as dialog, ui.card():
                            ui.label('Compute F_S Scores?').classes('text-h6 mb-4')
                            ui.label('Computing F_S scores for all 10 triplets will take approximately 2 minutes.').classes('mb-4')
                            ui.label('Do you want to proceed?').classes('mb-4')
                            with ui.row().classes('w-full justify-end gap-2'):
                                ui.button('Cancel', on_click=dialog.close).props('outline')
                                ui.button('Compute', on_click=lambda: [dialog.close(), compute_all_fs()]).props('color=primary')
                        dialog.open()
                    else:
                        # Select best and worst
                        sorted_ids = sorted(state['fs_scores'].items(), key=lambda x: x[1])
                        worst_id = sorted_ids[0][0]
                        best_id = sorted_ids[-1][0]

                        select1_answers.value = best_id
                        select2_answers.value = worst_id
                        await compare_answers()

                compare_btn_answers.on_click(compare_answers)
                best_worst_btn_answers.on_click(best_worst_answers)

        async def compute_all_fs():
            """Compute F_S scores for all triplets"""
            if state['computing_fs']:
                return

            state['computing_fs'] = True

            # Show progress dialog
            with ui.dialog() as progress_dialog, ui.card():
                ui.label('Computing F_S Scores').classes('text-h6 mb-4')
                progress_label = ui.label('Starting...').classes('mb-2')
                progress_bar = ui.linear_progress(value=0).classes('w-full')
                ui.label('This will take approximately 2 minutes...').classes('text-caption text-grey-6')

            progress_dialog.open()

            try:
                # Import compute_semantic_faithfulness directly
                import importlib.util
                csf_path = Path(__file__).parent.parent.parent / "sdm_package" / "compute_semantic_faithfulness.py"
                spec = importlib.util.spec_from_file_location("compute_semantic_faithfulness_module", str(csf_path))
                csf_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(csf_module)
                compute_semantic_faithfulness = csf_module.compute_semantic_faithfulness

                fs_scores = {}
                total = len(distributions)

                for i, (prompt_id, dist) in enumerate(distributions.items()):
                    progress_label.text = f'Computing {prompt_id}... ({i+1}/{total})'
                    progress_bar.value = i / total
                    await asyncio.sleep(0.01)  # Allow UI to update

                    # Run in thread to avoid blocking
                    p_q = np.array(dist['p_q'])
                    p_c = np.array(dist['p_c'])
                    p_a = np.array(dist['p_a'])

                    result = await asyncio.to_thread(
                        compute_semantic_faithfulness,
                        p_c=p_c,
                        p_q=p_q,
                        p_a=p_a,
                        tol_outer=1e-7,
                        max_outer_iter=100,
                        debug=False
                    )

                    fs_scores[prompt_id] = result['F_S']

                progress_bar.value = 1.0
                progress_label.text = 'Complete!'

                # Store in state and app storage
                state['fs_scores'] = fs_scores
                app.storage.user['comparison_fs_scores'] = fs_scores

                await asyncio.sleep(0.5)
                progress_dialog.close()

                ui.notify('F_S scores computed successfully!', type='positive')

            except Exception as e:
                progress_dialog.close()
                ui.notify(f'Error computing F_S scores: {e}', type='negative')
            finally:
                state['computing_fs'] = False


def generate_diff_html(text1: str, text2: str, side: str, similarity_threshold: float = 0.3) -> str:
    """
    Generate HTML with diff highlighting for substantially different sentences

    Args:
        text1: First text
        text2: Second text
        side: 'left' or 'right' - which side to highlight
        similarity_threshold: Only highlight sentences with similarity below this threshold (0-1)
                            Default 0.3 = only highlight very different sentences (<30% similar)

    Returns:
        HTML string with highlighted differences
    """
    # Split into sentences for better diff granularity
    # Handle multiple sentence endings
    import re
    sentences1 = re.split(r'(?<=[.!?])\s+', text1.strip())
    sentences2 = re.split(r'(?<=[.!?])\s+', text2.strip())

    # Use difflib to find matching blocks
    matcher = difflib.SequenceMatcher(None, sentences1, sentences2)

    html_parts = []

    if side == 'left':
        # Build a map of which sentences should be highlighted
        highlight_indices = set()

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Sentences were changed - check similarity
                for i in range(i1, i2):
                    # Find best match in the corresponding range
                    best_ratio = 0
                    for j in range(j1, j2):
                        ratio = difflib.SequenceMatcher(None, sentences1[i], sentences2[j]).ratio()
                        best_ratio = max(best_ratio, ratio)

                    # Highlight if substantially different
                    if best_ratio < similarity_threshold:
                        highlight_indices.add(i)
            elif tag == 'delete':
                # Sentence only in text1 - highlight
                for i in range(i1, i2):
                    highlight_indices.add(i)
            # 'insert' and 'equal' don't apply to left side

        for i, sentence in enumerate(sentences1):
            if i in highlight_indices:
                html_parts.append(f'<span style="background-color: #ffeb3b; padding: 2px;">{sentence}</span>')
            else:
                html_parts.append(sentence)

    else:  # side == 'right'
        # Build a map of which sentences should be highlighted
        highlight_indices = set()

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Sentences were changed - check similarity
                for j in range(j1, j2):
                    # Find best match in the corresponding range
                    best_ratio = 0
                    for i in range(i1, i2):
                        ratio = difflib.SequenceMatcher(None, sentences1[i], sentences2[j]).ratio()
                        best_ratio = max(best_ratio, ratio)

                    # Highlight if substantially different
                    if best_ratio < similarity_threshold:
                        highlight_indices.add(j)
            elif tag == 'insert':
                # Sentence only in text2 - highlight
                for j in range(j1, j2):
                    highlight_indices.add(j)
            # 'delete' and 'equal' don't apply to right side

        for j, sentence in enumerate(sentences2):
            if j in highlight_indices:
                html_parts.append(f'<span style="background-color: #ffeb3b; padding: 2px;">{sentence}</span>')
            else:
                html_parts.append(sentence)

    return ' '.join(html_parts)
