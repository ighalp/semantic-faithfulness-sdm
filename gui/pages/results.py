"""
Results page - Display metrics and visualizations
"""

from nicegui import ui, app
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def create():
    """Create the results page content"""

    # Get analysis results (set by analyze page after optimization completes)
    try:
        results = app.storage.user.get('analysis_results')
    except Exception:
        results = None

    # Validate that all required keys are present
    required_keys = ['F_S', 'SEP_total', 'SEP_system', 'H_Q', 'H_C', 'H_A', 'p_q', 'p_c', 'p_a', 'n_clusters']
    if not results or not isinstance(results, dict) or not all(k in results and results[k] is not None for k in required_keys):
        with ui.column().classes('w-full max-w-6xl mx-auto p-8'):
            ui.label('No Results Available').classes('text-h4 mb-6')
            ui.label('Please run an analysis first from the Analyze page.').classes('text-subtitle1 text-grey-7 mb-4')
            ui.button('Go to Analyze', on_click=lambda: ui.navigate.to('/analyze')).props('color=primary')
        return

    display_results = results

    with ui.column().classes('w-full max-w-6xl mx-auto p-8'):
        ui.label('Analysis Results').classes('text-h4 mb-6')

        # Metric cards
        ui.label('Key Metrics').classes('text-h6 mb-4')
        with ui.row().classes('w-full gap-4 mb-8'):
            # Semantic Faithfulness card
            with ui.card().classes('flex-1 p-6'):
                ui.label('Semantic Faithfulness').classes('text-subtitle1 text-grey-7')
                ui.label(f"{display_results['F_S']:.4f}").classes('text-h3 font-bold text-primary')
                ui.label('F_S').classes('text-caption text-grey-6')

            # SEP Total card
            with ui.card().classes('flex-1 p-6'):
                ui.label('Total Entropy Production').classes('text-subtitle1 text-grey-7')
                ui.label(f"{display_results['SEP_total']:.4f} bits").classes('text-h3 font-bold')
                ui.label('SEP_total').classes('text-caption text-grey-6')

            # SEP System card
            with ui.card().classes('flex-1 p-6'):
                ui.label('System Entropy Production').classes('text-subtitle1 text-grey-7')
                ui.label(f"{display_results['SEP_system']:.4f} bits").classes('text-h3 font-bold')
                ui.label('SEP_system').classes('text-caption text-grey-6')

        # Entropy metrics
        ui.label('Entropy Metrics').classes('text-h6 mb-4')
        with ui.row().classes('w-full gap-4 mb-8'):
            with ui.card().classes('flex-1 p-4'):
                ui.label('H(Q)').classes('text-subtitle2')
                ui.label(f"{display_results['H_Q']:.4f} bits").classes('text-h5')

            with ui.card().classes('flex-1 p-4'):
                ui.label('H(C)').classes('text-subtitle2')
                ui.label(f"{display_results['H_C']:.4f} bits").classes('text-h5')

            with ui.card().classes('flex-1 p-4'):
                ui.label('H(A)').classes('text-subtitle2')
                ui.label(f"{display_results['H_A']:.4f} bits").classes('text-h5')

        # Cache Statistics (only if available)
        if 'cache_stats' in display_results:
            cache_stats = display_results['cache_stats']
            ui.label('Cache Statistics').classes('text-h6 mb-4 mt-6')
            with ui.card().classes('w-full p-6'):
                ui.label('Performance Optimization').classes('text-subtitle1 mb-4')

                # Calculate totals
                total_answers = cache_stats.get('answers_cached', 0) + cache_stats.get('answers_generated', 0)
                total_distributions = cache_stats.get('distributions_cached', 0) + cache_stats.get('distributions_computed', 0)
                total_fs_scores = cache_stats.get('fs_scores_cached', 0) + cache_stats.get('fs_scores_computed', 0)

                with ui.row().classes('w-full gap-4 mb-4'):
                    # Paraphrases
                    with ui.column().classes('flex-1'):
                        ui.label('Paraphrases').classes('text-subtitle2 font-bold mb-2')
                        if cache_stats.get('paraphrases_cached', False):
                            ui.label('Loaded from cache').classes('text-body2 text-green-600')
                            ui.icon('check_circle', color='green').classes('text-sm')
                        else:
                            ui.label('Newly generated').classes('text-body2 text-blue-600')
                            ui.icon('auto_awesome', color='blue').classes('text-sm')

                    # Answers
                    with ui.column().classes('flex-1'):
                        ui.label('Answers').classes('text-subtitle2 font-bold mb-2')
                        if total_answers > 0:
                            cached_pct = (cache_stats.get('answers_cached', 0) / total_answers) * 100
                            ui.label(f"{cache_stats.get('answers_cached', 0)}/{total_answers} cached ({cached_pct:.0f}%)").classes('text-body2')
                            ui.label(f"{cache_stats.get('answers_generated', 0)} newly generated").classes('text-caption text-grey-6')

                with ui.row().classes('w-full gap-4'):
                    # Distributions
                    with ui.column().classes('flex-1'):
                        ui.label('Distributions').classes('text-subtitle2 font-bold mb-2')
                        if total_distributions > 0:
                            cached_pct = (cache_stats.get('distributions_cached', 0) / total_distributions) * 100
                            ui.label(f"{cache_stats.get('distributions_cached', 0)}/{total_distributions} cached ({cached_pct:.0f}%)").classes('text-body2')
                            ui.label(f"{cache_stats.get('distributions_computed', 0)} newly computed").classes('text-caption text-grey-6')

                    # F_S Scores
                    with ui.column().classes('flex-1'):
                        ui.label('F_S Scores').classes('text-subtitle2 font-bold mb-2')
                        if total_fs_scores > 0:
                            cached_pct = (cache_stats.get('fs_scores_cached', 0) / total_fs_scores) * 100
                            ui.label(f"{cache_stats.get('fs_scores_cached', 0)}/{total_fs_scores} cached ({cached_pct:.0f}%)").classes('text-body2')
                            ui.label(f"{cache_stats.get('fs_scores_computed', 0)} newly computed").classes('text-caption text-grey-6')

                # Overall summary
                ui.separator().classes('my-4')
                total_cached = sum([
                    1 if cache_stats.get('paraphrases_cached', False) else 0,
                    cache_stats.get('answers_cached', 0),
                    cache_stats.get('distributions_cached', 0),
                    cache_stats.get('fs_scores_cached', 0)
                ])
                total_items = sum([
                    1,  # paraphrases
                    total_answers,
                    total_distributions,
                    total_fs_scores
                ])
                if total_items > 0:
                    overall_pct = (total_cached / total_items) * 100
                    ui.label(f'Overall cache utilization: {overall_pct:.1f}%').classes('text-subtitle2 font-bold text-primary')
                    if overall_pct > 50:
                        ui.label('Significant time and API cost savings from caching!').classes('text-caption text-green-600')

        # Visualizations
        ui.label('Visualizations').classes('text-h6 mb-4')
        with ui.tabs().classes('w-full') as viz_tabs:
            dist_tab = ui.tab('Distributions')
            matrix_tab = ui.tab('Matrices')
            stats_tab = ui.tab('Statistics')

        with ui.tab_panels(viz_tabs, value=dist_tab).classes('w-full'):
            with ui.tab_panel(dist_tab):
                with ui.card().classes('w-full p-6'):
                    ui.label('Topic Distributions').classes('text-h6 mb-4')
                    plot_distributions(display_results)

            with ui.tab_panel(matrix_tab):
                with ui.card().classes('w-full p-6'):
                    ui.label('Transition Matrices').classes('text-h6 mb-4')
                    plot_matrices(display_results)

            with ui.tab_panel(stats_tab):
                with ui.card().classes('w-full p-6'):
                    ui.label('Analysis Statistics').classes('text-h6 mb-4')
                    display_statistics(display_results)

        # Export options
        with ui.row().classes('gap-4 mt-8'):
            ui.button('Export JSON', icon='download', on_click=lambda: export_json(display_results)).props('outline')
            ui.button('New Analysis', on_click=lambda: ui.navigate.to('/input')).props('color=primary')


def plot_distributions(results):
    """Plot probability distributions using Plotly"""
    p_q = np.array(results['p_q'])
    p_c = np.array(results['p_c'])
    p_a = np.array(results['p_a'])
    n_clusters = results['n_clusters']

    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('p(Q) - Question', 'p(C) - Context', 'p(A) - Answer'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )

    # X-axis labels (cluster indices)
    x = list(range(n_clusters))

    # Add bars for each distribution
    fig.add_trace(
        go.Bar(x=x, y=p_q, name='p(Q)', marker_color='#1f77b4'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=x, y=p_c, name='p(C)', marker_color='#ff7f0e'),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(x=x, y=p_a, name='p(A)', marker_color='#2ca02c'),
        row=1, col=3
    )

    # Update layout
    fig.update_xaxes(title_text="Cluster", row=1, col=1)
    fig.update_xaxes(title_text="Cluster", row=1, col=2)
    fig.update_xaxes(title_text="Cluster", row=1, col=3)

    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=2)
    fig.update_yaxes(title_text="Probability", row=1, col=3)

    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Probability Distributions Over Semantic Topics"
    )

    ui.plotly(fig).classes('w-full')


def plot_matrices(results):
    """Plot transition matrices using Plotly"""
    Q_star = np.array(results['Q_star'])
    A_star = np.array(results['A_star'])

    # Create subplots for heatmaps
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Q* Matrix (Context → Question)', 'A* Matrix (Context → Answer)'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
    )

    # Q* heatmap
    fig.add_trace(
        go.Heatmap(
            z=Q_star,
            colorscale='Blues',
            showscale=True,
            hovertemplate='Context: %{y}<br>Question: %{x}<br>Prob: %{z:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # A* heatmap
    fig.add_trace(
        go.Heatmap(
            z=A_star,
            colorscale='Greens',
            showscale=True,
            hovertemplate='Context: %{y}<br>Answer: %{x}<br>Prob: %{z:.4f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="Question Topic", row=1, col=1)
    fig.update_yaxes(title_text="Context Topic", row=1, col=1)

    fig.update_xaxes(title_text="Answer Topic", row=1, col=2)
    fig.update_yaxes(title_text="Context Topic", row=1, col=2)

    fig.update_layout(
        height=500,
        title_text="Optimal Transition Matrices"
    )

    ui.plotly(fig).classes('w-full')


def display_statistics(results):
    """Display analysis statistics"""
    with ui.column().classes('w-full gap-4'):
        # Convergence info
        with ui.card().classes('w-full p-4'):
            ui.label('Optimization Details').classes('text-subtitle1 font-bold mb-2')
            with ui.grid(columns=2).classes('w-full gap-4'):
                ui.label('Iterations:').classes('text-grey-7')
                ui.label(f"{results['iterations']}").classes('font-bold')

                ui.label('Converged:').classes('text-grey-7')
                converged_icon = 'check_circle' if results['converged'] else 'warning'
                converged_color = 'positive' if results['converged'] else 'warning'
                ui.icon(converged_icon).props(f'color={converged_color}')

                ui.label('Clusters:').classes('text-grey-7')
                ui.label(f"{results['n_clusters']}").classes('font-bold')

        # Sentence counts
        with ui.card().classes('w-full p-4'):
            ui.label('Sentence Counts').classes('text-subtitle1 font-bold mb-2')
            with ui.grid(columns=2).classes('w-full gap-4'):
                ui.label('Question sentences:').classes('text-grey-7')
                ui.label(f"{len(results['question_sentences'])}").classes('font-bold')

                ui.label('Context sentences:').classes('text-grey-7')
                ui.label(f"{len(results['context_sentences'])}").classes('font-bold')

                ui.label('Answer sentences:').classes('text-grey-7')
                ui.label(f"{len(results['answer_sentences'])}").classes('font-bold')

        # Interpretation guide
        with ui.card().classes('w-full p-4'):
            ui.label('Interpretation Guide').classes('text-subtitle1 font-bold mb-2')
            with ui.column().classes('gap-2'):
                ui.label('• F_S (Semantic Faithfulness): Higher is better (max = 1.0)').classes('text-body2')
                ui.label('• SEP_total: Lower indicates closer alignment with optimal channel').classes('text-body2')
                ui.label('• SEP_system: Difference between answer and context entropy').classes('text-body2')
                ui.label('• H(Q), H(C), H(A): Information content in bits').classes('text-body2')


def export_json(results):
    """Export results as JSON"""
    # Prepare export data
    export_data = {
        'metrics': {
            'F_S': results['F_S'],
            'SEP_total': results['SEP_total'],
            'SEP_system': results['SEP_system'],
            'H_Q': results['H_Q'],
            'H_C': results['H_C'],
            'H_A': results['H_A']
        },
        'distributions': {
            'p_q': results['p_q'],
            'p_c': results['p_c'],
            'p_a': results['p_a']
        },
        'matrices': {
            'Q_star': results['Q_star'],
            'A_star': results['A_star']
        },
        'metadata': {
            'n_clusters': results['n_clusters'],
            'iterations': results['iterations'],
            'converged': results['converged']
        }
    }

    # Convert to JSON string
    json_str = json.dumps(export_data, indent=2)

    # Create download
    ui.download(json_str.encode(), 'semantic_faithfulness_results.json')
    ui.notify('Results exported', type='positive')
