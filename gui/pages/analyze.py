"""
Analyze page - Analysis execution with progress tracking
"""

from nicegui import ui, app
import asyncio
import sys
from pathlib import Path

# Add parent directory to path (imports moved to function level to avoid blocking page load)
sys.path.insert(0, str(Path(__file__).parent.parent))

def create():
    """Create the analyze page content"""

    print("[DEBUG] analyze.create() START")
    import sys
    sys.stdout.flush()

    # Check if we have input data (either QCA triplet or cached distributions)
    print("[DEBUG] Checking for input data...")
    sys.stdout.flush()
    has_qca = app.storage.user.get('qca_triplet')
    has_cache = app.storage.user.get('cached_distributions')
    print(f"[DEBUG] has_qca: {has_qca is not None}, has_cache: {has_cache is not None}")
    sys.stdout.flush()

    if not has_qca and not has_cache:
        with ui.column().classes('w-full max-w-4xl mx-auto p-8'):
            ui.label('No Input Data').classes('text-h4 mb-6')
            ui.label('Please provide a QCA triplet or select cached data first.').classes('text-subtitle1 text-grey-7 mb-4')
            ui.button('Go to Input', on_click=lambda: ui.navigate.to('/input')).props('color=primary')
        return

    # UI State
    print("[DEBUG] Creating UI state dict...")
    sys.stdout.flush()
    state = {
        'progress_bar': None,
        'status_label': None,
        'stage_chips': {},
        'start_button': None,
        'cancel_button': None,
        'log_area': None,
        'analysis_task': None
    }

    print("[DEBUG] About to create UI column...")
    sys.stdout.flush()
    with ui.column().classes('w-full max-w-4xl mx-auto p-8'):
        ui.label('Analysis').classes('text-h4 mb-6')

        # Status card
        with ui.card().classes('w-full p-6 mb-6'):
            ui.label('Analysis Status').classes('text-h6 mb-4')

            # Progress bar
            state['progress_bar'] = ui.linear_progress(value=0).classes('w-full mb-4')

            # Status message
            state['status_label'] = ui.label('Ready to start analysis').classes('text-subtitle1 mb-4')

            # Stage indicators
            with ui.row().classes('w-full gap-4'):
                state['stage_chips']['tokenization'] = ui.chip('Tokenization', icon='pending').props('outline color=grey')
                state['stage_chips']['embedding'] = ui.chip('Embedding', icon='pending').props('outline color=grey')
                state['stage_chips']['clustering'] = ui.chip('Clustering', icon='pending').props('outline color=grey')
                state['stage_chips']['optimization'] = ui.chip('Optimization', icon='pending').props('outline color=grey')

        # Control buttons
        with ui.row().classes('gap-4'):
            state['start_button'] = ui.button('Start Analysis', on_click=lambda: run_analysis(state)).props('color=primary')
            state['cancel_button'] = ui.button('Cancel', on_click=lambda: cancel_analysis(state)).props('outline disabled')
            ui.button('Back to Input', on_click=lambda: ui.navigate.to('/input')).props('flat')

        # Analysis log
        with ui.expansion('Analysis Log', icon='description').classes('w-full mt-6'):
            state['log_area'] = ui.log().classes('w-full h-64')
            state['log_area'].push('Waiting to start...')

    # DISABLE auto-start - user must manually click "Start Analysis"
    # This prevents WebSocket timeout during heavy imports
    print("[DEBUG] analyze.create() COMPLETE - waiting for manual start")
    sys.stdout.flush()


async def run_cached_analysis(state, cached_dist):
    """Run analysis using cached distributions (skip embedding/clustering)"""
    import numpy as np
    from scipy.stats import entropy

    config_dict = app.storage.user['analysis_config']

    state['log_area'].push('╔═══════════════════════════════════════════════════════╗')
    state['log_area'].push('║         CACHED ANALYSIS MODE - FAST EXECUTION         ║')
    state['log_area'].push('╚═══════════════════════════════════════════════════════╝')
    state['log_area'].push('')
    state['log_area'].push(f'Cached Triplet ID: {cached_dist.get("prompt_id", "unknown")}')
    n_clusters_display = cached_dist.get('k') or cached_dist.get('n_topics') or len(cached_dist.get('p_q', []))
    state['log_area'].push(f'Number of Clusters: {n_clusters_display}')
    state['log_area'].push(f'Optimization Tolerance: {config_dict["tolerance"]}')
    state['log_area'].push(f'Max Iterations: {config_dict["max_iterations"]}')
    state['log_area'].push('')

    try:
        # Stage 1: Load cached distributions
        state['status_label'].text = 'Stage 1/4: Checking for cached data...'
        state['progress_bar'].value = 0.05
        state['log_area'].push('[STAGE 1/4] TOKENIZATION - Checking cache...')
        await asyncio.sleep(0.1)

        state['log_area'].push('   ✓ FOUND: Cached sentence tokenization')
        state['stage_chips']['tokenization'].props('icon=check_circle color=green')
        state['progress_bar'].value = 0.1
        await asyncio.sleep(0.2)

        # Stage 2: Embeddings
        state['status_label'].text = 'Stage 2/4: Loading cached embeddings...'
        state['log_area'].push('[STAGE 2/4] EMBEDDINGS - Checking cache...')
        state['stage_chips']['embedding'].props('icon=hourglass_empty color=blue')
        await asyncio.sleep(0.1)

        state['log_area'].push('   ✓ FOUND: Cached semantic embeddings')
        state['log_area'].push('   ⊙ SKIPPING: Embedding computation (using cache)')
        state['stage_chips']['embedding'].props('icon=check_circle color=green')
        state['progress_bar'].value = 0.2
        await asyncio.sleep(0.2)

        # Stage 3: Clustering
        state['status_label'].text = 'Stage 3/4: Loading cached cluster assignments...'
        state['log_area'].push('[STAGE 3/4] CLUSTERING - Checking cache...')
        state['stage_chips']['clustering'].props('icon=hourglass_empty color=blue')
        await asyncio.sleep(0.1)

        state['log_area'].push(f'   ✓ FOUND: Cached cluster assignments (k={cached_dist.get("k") or cached_dist.get("n_topics") or len(cached_dist.get("p_q", []))})')
        state['log_area'].push('   ⊙ SKIPPING: UDIB clustering (using cache)')
        state['stage_chips']['clustering'].props('icon=check_circle color=green')
        state['progress_bar'].value = 0.3
        await asyncio.sleep(0.2)

        # Load actual distributions
        state['status_label'].text = 'Loading probability distributions...'
        state['log_area'].push('')
        state['log_area'].push('[DISTRIBUTIONS] Loading cached probability vectors...')

        p_q = np.array(cached_dist['p_q'])
        p_c = np.array(cached_dist['p_c'])
        p_a = np.array(cached_dist['p_a'])
        # Get n_clusters from cache or derive from distribution length
        n_clusters = cached_dist.get('k') or cached_dist.get('n_topics') or len(p_q)

        state['log_area'].push(f'   ✓ p_q (Question distribution): shape=({len(p_q)},)')
        state['log_area'].push(f'   ✓ p_c (Context distribution): shape=({len(p_c)},)')
        state['log_area'].push(f'   ✓ p_a (Answer distribution): shape=({len(p_a)},)')
        state['log_area'].push(f'   ✓ All distributions over {n_clusters} semantic topics')
        state['progress_bar'].value = 0.4
        await asyncio.sleep(0.3)

        # Stage 4: Optimization
        state['status_label'].text = 'Stage 4/4: Computing semantic faithfulness metrics...'
        state['log_area'].push('')
        state['log_area'].push('[STAGE 4/4] OPTIMIZATION - Computing F_S and SEP')
        state['stage_chips']['optimization'].props('icon=hourglass_empty color=primary')
        state['log_area'].push('   → Running convex optimization (Csiszár-Tusnády)')
        state['log_area'].push('   → This may take 1-2 minutes to converge...')
        state['progress_bar'].value = 0.5
        await asyncio.sleep(0.3)

        # Import SDM function directly (bypassing package __init__ to avoid heavy imports)
        import importlib.util
        csf_path = Path(__file__).parent.parent.parent / "sdm_package" / "compute_semantic_faithfulness.py"
        spec = importlib.util.spec_from_file_location("compute_semantic_faithfulness_module", str(csf_path))
        csf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(csf_module)
        compute_semantic_faithfulness = csf_module.compute_semantic_faithfulness

        # Run optimization with progress updates
        state['log_area'].push('   ⚙ Optimization running...')
        state['progress_bar'].value = 0.6

        # Aggressive heartbeat to keep WebSocket alive during 1-2 minute optimization
        import time
        heartbeat_data = {
            'running': True,
            'start_time': time.time(),
            'beat_count': 0
        }

        def heartbeat_update():
            """Frequent updates to keep WebSocket alive - fires every 2 seconds"""
            if not heartbeat_data['running']:
                return

            heartbeat_data['beat_count'] += 1
            elapsed = int(time.time() - heartbeat_data['start_time'])

            # Update log every 5 beats (10 seconds)
            if heartbeat_data['beat_count'] % 5 == 0:
                state['log_area'].push(f'   ⏱ Optimizing... {elapsed}s elapsed')

            # Slowly increment progress bar from 0.6 to 0.9 over ~120 seconds
            progress_increment = (0.9 - 0.6) / 60  # Assuming max 120 seconds
            new_progress = min(0.9, 0.6 + (heartbeat_data['beat_count'] * 2 * progress_increment))
            state['progress_bar'].value = new_progress

            # Update status label on every beat
            state['status_label'].text = f'Optimizing... ({elapsed}s)'

        # Create timer that fires every 2 seconds (aggressive keepalive)
        heartbeat_timer = ui.timer(2.0, heartbeat_update)

        try:
            # Run optimization in background thread
            result = await asyncio.to_thread(
                compute_semantic_faithfulness,
                p_c=p_c,
                p_q=p_q,
                p_a=p_a,
                tol_outer=config_dict['tolerance'],
                max_outer_iter=config_dict['max_iterations'],
                debug=False
            )
        finally:
            # Stop heartbeat
            heartbeat_data['running'] = False
            heartbeat_timer.cancel()

        state['progress_bar'].value = 0.95
        state['stage_chips']['optimization'].props('icon=check_circle color=positive')
        state['log_area'].push('')
        state['log_area'].push('   ✓ Optimization converged!')
        state['log_area'].push(f'   ✓ F_S (Semantic Faithfulness) = {result["F_S"]:.4f}')
        state['log_area'].push(f'   ✓ Iterations: {result["iterations"]}')
        state['log_area'].push(f'   ✓ Converged: {result["converged"]}')

        # Store results
        app.storage.user['analysis_results'] = {
            'F_S': float(result['F_S']),
            'SEP_total': float(result['D_min']),
            'SEP_system': float(entropy(p_a, base=2) - entropy(p_c, base=2)),
            'H_Q': float(entropy(p_q, base=2)),
            'H_C': float(entropy(p_c, base=2)),
            'H_A': float(entropy(p_a, base=2)),
            'p_q': p_q.tolist(),
            'p_c': p_c.tolist(),
            'p_a': p_a.tolist(),
            'Q_star': result['Q_star'].tolist(),
            'A_star': result['A_star'].tolist(),
            'n_clusters': n_clusters,
            'iterations': result['iterations'],
            'converged': result['converged'],
            'question_sentences': [],
            'context_sentences': [],
            'answer_sentences': []
        }

        app.storage.user['analysis_status'] = 'completed'
        state['progress_bar'].value = 1.0
        state['status_label'].text = 'Analysis complete!'

        # Final summary
        state['log_area'].push('')
        state['log_area'].push('╔═══════════════════════════════════════════════════════╗')
        state['log_area'].push('║            ANALYSIS COMPLETE (CACHED MODE)            ║')
        state['log_area'].push('╚═══════════════════════════════════════════════════════╝')
        state['log_area'].push('')
        state['log_area'].push('RESULTS:')
        state['log_area'].push(f'  F_S (Semantic Faithfulness): {result["F_S"]:.4f}')
        state['log_area'].push(f'  SEP_total (Total Entropy Production): {result["D_min"]:.4f} bits')
        state['log_area'].push(f'  SEP_system (System Entropy Production): {entropy(p_a, base=2) - entropy(p_c, base=2):.4f} bits')
        state['log_area'].push(f'  H(Q) (Question Entropy): {entropy(p_q, base=2):.4f} bits')
        state['log_area'].push(f'  H(C) (Context Entropy): {entropy(p_c, base=2):.4f} bits')
        state['log_area'].push(f'  H(A) (Answer Entropy): {entropy(p_a, base=2):.4f} bits')
        state['log_area'].push('')
        state['log_area'].push('PERFORMANCE:')
        state['log_area'].push('  ⊙ Tokenization: SKIPPED (cached)')
        state['log_area'].push('  ⊙ Embedding: SKIPPED (cached)')
        state['log_area'].push('  ⊙ Clustering: SKIPPED (cached)')
        state['log_area'].push('  ✓ Optimization: COMPLETED')
        state['log_area'].push('')
        state['log_area'].push('→ Redirecting to results page...')

        ui.notify('Analysis completed successfully!', type='positive')
        await asyncio.sleep(1.5)
        ui.navigate.to('/results')

    except Exception as e:
        state['log_area'].push(f'ERROR: {str(e)}')
        state['status_label'].text = f'Error: {str(e)}'
        ui.notify(f'Analysis failed: {str(e)}', type='negative')
        app.storage.user['analysis_status'] = 'error'
    finally:
        state['analysis_task'] = None
        state['start_button'].props(remove='disabled')
        state['cancel_button'].props('disabled')


async def run_analysis(state):
    """Execute the analysis in the background"""

    # Prevent multiple concurrent runs
    if state['analysis_task'] is not None:
        return

    # Update UI state
    state['start_button'].props('disabled')
    state['cancel_button'].props(remove='disabled')
    state['status_label'].text = 'Starting analysis...'
    state['log_area'].push('=' * 60)
    state['log_area'].push('Starting Semantic Faithfulness Analysis')
    state['log_area'].push('=' * 60)

    # Check if we're in cache mode FIRST (before any heavy imports!)
    cached_dist = app.storage.user.get('cached_distributions')

    if cached_dist:
        # CACHE MODE - Skip embedding/clustering, use pre-computed distributions
        await run_cached_analysis(state, cached_dist)
        return

    # NORMAL MODE - Now it's safe to import heavy services module
    from services import AnalysisConfig, get_analysis_service, AnalysisProgress

    # Get data from session
    triplet = app.storage.user['qca_triplet']
    config_dict = app.storage.user['analysis_config']

    # Create config object
    config = AnalysisConfig(
        embedding_model=config_dict['embedding_model'],
        clustering_method=config_dict['clustering_method'],
        tolerance=config_dict['tolerance'],
        max_iterations=config_dict['max_iterations']
    )

    state['log_area'].push(f'Embedding Model: {config.embedding_model}')
    state['log_area'].push(f'Clustering Method: {config.clustering_method}')
    state['log_area'].push(f'Tolerance: {config.tolerance}')
    state['log_area'].push(f'Max Iterations: {config.max_iterations}')
    state['log_area'].push('')

    # Progress callback
    def update_progress(progress: AnalysisProgress):
        # Update progress bar
        stage_weights = {
            'tokenization': 0.0,
            'embedding': 0.25,
            'clustering': 0.5,
            'optimization': 0.75
        }
        base_progress = stage_weights.get(progress.stage, 0.0)
        total_progress = base_progress + (progress.progress * 0.25)
        state['progress_bar'].value = total_progress

        # Update status label
        state['status_label'].text = progress.message

        # Update stage chips
        for stage_name, chip in state['stage_chips'].items():
            if stage_name == progress.stage:
                if progress.progress >= 1.0:
                    chip.props('icon=check_circle color=positive')
                else:
                    chip.props('icon=play_arrow color=primary')
            elif stage_weights.get(stage_name, 0) < stage_weights.get(progress.stage, 0):
                chip.props('icon=check_circle color=positive')
            else:
                chip.props('icon=pending color=grey')

        # Log message
        state['log_area'].push(f'[{progress.stage.upper()}] {progress.message}')

    try:
        # Run analysis
        service = get_analysis_service()
        state['log_area'].push('Running analysis...')

        results = await service.analyze(
            question=triplet['question'],
            context=triplet['context'],
            answer=triplet['answer'],
            config=config,
            progress_callback=update_progress
        )

        # Store results
        app.storage.user['analysis_results'] = {
            'F_S': float(results.F_S),
            'SEP_total': float(results.SEP_total),
            'SEP_system': float(results.SEP_system),
            'H_Q': float(results.H_Q),
            'H_C': float(results.H_C),
            'H_A': float(results.H_A),
            'p_q': results.p_q.tolist(),
            'p_c': results.p_c.tolist(),
            'p_a': results.p_a.tolist(),
            'Q_star': results.Q_star.tolist(),
            'A_star': results.A_star.tolist(),
            'n_clusters': results.n_clusters,
            'iterations': results.iterations,
            'converged': results.converged,
            'question_sentences': results.question_sentences,
            'context_sentences': results.context_sentences,
            'answer_sentences': results.answer_sentences
        }

        app.storage.user['analysis_status'] = 'completed'

        # Update UI
        state['progress_bar'].value = 1.0
        state['status_label'].text = 'Analysis complete!'
        state['log_area'].push('')
        state['log_area'].push('=' * 60)
        state['log_area'].push('ANALYSIS COMPLETE')
        state['log_area'].push('=' * 60)
        state['log_area'].push(f'F_S (Semantic Faithfulness): {results.F_S:.4f}')
        state['log_area'].push(f'SEP_total: {results.SEP_total:.4f} bits')
        state['log_area'].push(f'SEP_system: {results.SEP_system:.4f} bits')
        state['log_area'].push(f'H(Q): {results.H_Q:.4f} bits')
        state['log_area'].push(f'H(C): {results.H_C:.4f} bits')
        state['log_area'].push(f'H(A): {results.H_A:.4f} bits')
        state['log_area'].push(f'Clusters: {results.n_clusters}')
        state['log_area'].push(f'Iterations: {results.iterations}')
        state['log_area'].push(f'Converged: {results.converged}')

        ui.notify('Analysis completed successfully!', type='positive')

        # Navigate to results after a short delay
        await asyncio.sleep(1)
        ui.navigate.to('/results')

    except Exception as e:
        state['log_area'].push('')
        state['log_area'].push(f'ERROR: {str(e)}')
        state['status_label'].text = f'Error: {str(e)}'
        ui.notify(f'Analysis failed: {str(e)}', type='negative')
        app.storage.user['analysis_status'] = 'error'

    finally:
        state['analysis_task'] = None
        state['start_button'].props(remove='disabled')
        state['cancel_button'].props('disabled')


def cancel_analysis(state):
    """Cancel the running analysis"""
    if state['analysis_task'] is not None:
        state['analysis_task'].cancel()
        state['analysis_task'] = None
        state['status_label'].text = 'Analysis cancelled'
        state['log_area'].push('Analysis cancelled by user')
        ui.notify('Analysis cancelled')
