#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 21:01:16 2025

@author: igorhalperin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from tqdm.auto import tqdm
import warnings
from scipy.stats import mode
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.metrics import pairwise_distances_argmin_min


# Suppress runtime warnings for log(0)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class DIB_Simple:
    """
    Simplified Deterministic Information Bottleneck (DIB) Clustering.

    This implementation uses the rescaled objective function where the tradeoff
    is controlled by a single parameter, lambda_val, which combines the original
    s and beta parameters (lambda = 2*s^2 / beta).

    Parameters:
    ----------
    lambda_val : float
        The tradeoff parameter. Larger lambda values place more weight on the
        cluster size regularization term (-log q(c)), encouraging more balanced
        and larger clusters. Smaller values prioritize geometric cohesion.

    max_n_clusters : int
        The initial maximum number of clusters.

    max_iter : int, default=100
        Maximum number of iterations.
        
    non_local_merge_steps : int, default=5
        Number of passes to attempt merging clusters to escape local minima.
    """
    def __init__(self, tau_val, max_n_clusters, max_iter=100, non_local_merge_steps=5):
        self.tau_val = tau_val
        self.lambda_val = self.tau_val  # for backwad comatibility 
        self.max_n_clusters = max_n_clusters
        self.max_iter = max_iter
        self.non_local_merge_steps = non_local_merge_steps
        self.assignments_ = None
        self.n_clusters_ = None

    def _calculate_total_cost(self, X, assignments):
        """Calculates the total Lagrangian cost for the current assignments."""
        total_cost = 0
        N = X.shape[0]
        unique_clusters, n_c = np.unique(assignments, return_counts=True)
        q_c = n_c / N
        log_q_c = np.log(q_c)

        cluster_sq_norms = {c: np.sum(np.linalg.norm(X[assignments == c], axis=1)**2) for c in unique_clusters}
        cluster_sums = {c: np.sum(X[assignments == c], axis=0) for c in unique_clusters}
        
        for i in range(N):
            c_i = assignments[i]
            c_idx = np.where(unique_clusters == c_i)[0][0]
            
            # Efficiently compute average squared distance
            n_ci = n_c[c_idx]
            xi_sq_norm = np.linalg.norm(X[i])**2
            sum_xj = cluster_sums[c_i]
            sum_xj_sq_norm = cluster_sq_norms[c_i]
            sum_sq_dist = n_ci * xi_sq_norm - 2 * np.dot(X[i], sum_xj) + sum_xj_sq_norm
            avg_sq_dist = sum_sq_dist / n_ci
            
            cost_i = avg_sq_dist - self.lambda_val * log_q_c[c_idx]
            total_cost += cost_i
            
        return total_cost
    
    def fit(self, X, seed=None):
        N, D = X.shape

        # --- Formal Initialization (as per Strouse & Schwab, Alg. 1) ---
        # 1. Initialize cluster assignments c*(0)(i)
        # Use a dedicated random number generator for reproducibility
        rng = np.random.default_rng(seed)
        
        assignments = np.random.randint(0, self.max_n_clusters, size=N)
        
        # This loop now represents the main iterative part, starting from n=1
        for it in range(self.max_iter):
            # --- Batch Update Logic ---
            # 1. Store the state from the previous iteration (e.g., n-1)
            prev_assignments = assignments.copy()
            
            # 2. Compute cluster properties based on the previous state (n-1)
            unique_clusters, n_c = np.unique(prev_assignments, return_counts=True)
            
            if len(unique_clusters) == 1:
                break # Converged to a single cluster

            q_c = n_c / N
            log_q_c = np.log(q_c)
            
            cluster_sq_norms = {c: np.sum(np.linalg.norm(X[prev_assignments == c], axis=1)**2) for c in unique_clusters}
            cluster_sums = {c: np.sum(X[prev_assignments == c], axis=0) for c in unique_clusters}

            # 3. Compute new assignments for the current iteration (n)
            new_assignments = np.zeros_like(assignments)
            for i in range(N):
                costs = []
                for c_idx, c_label in enumerate(unique_clusters):
                    n_ci = n_c[c_idx]
                    xi_sq_norm = np.linalg.norm(X[i])**2
                    sum_xj = cluster_sums[c_label]
                    sum_xj_sq_norm = cluster_sq_norms[c_label]
                    sum_sq_dist = n_ci * xi_sq_norm - 2 * np.dot(X[i], sum_xj) + sum_xj_sq_norm
                    avg_sq_dist = sum_sq_dist / n_ci
                    cost = avg_sq_dist - self.lambda_val * log_q_c[c_idx]
                    costs.append(cost)
                new_assignments[i] = unique_clusters[np.argmin(costs)]
            
            # 4. Update the state
            assignments = new_assignments
            
            # Check for convergence
            if np.array_equal(prev_assignments, assignments):
                break
        
        # --- The non-local merge steps are a post-processing/refinement step ---
        # This is an addition to the base DIB algorithm to escape local minima
        for merge_step in range(self.non_local_merge_steps):
            current_cost = self._calculate_total_cost(X, assignments)
            best_merge, best_cost_reduction = None, 0
            
            unique_clusters_after_iter = np.unique(assignments)
            if len(unique_clusters_after_iter) <= 1: 
                break

            from itertools import combinations
            for c1, c2 in combinations(unique_clusters_after_iter, 2):
                temp_assignments = assignments.copy()
                temp_assignments[temp_assignments == c2] = c1
                merged_cost = self._calculate_total_cost(X, temp_assignments)
                cost_reduction = current_cost - merged_cost # Note: cost should decrease
                if cost_reduction > best_cost_reduction:
                    best_cost_reduction, best_merge = cost_reduction, (c1, c2)
            
            if best_merge:
                c1, c2 = best_merge
                assignments[assignments == c2] = c1
            else:
                # No merge improved the objective, so stop refining
                break

        unique_labels, self.assignments_ = np.unique(assignments, return_inverse=True)
        self.n_clusters_ = len(unique_labels)
        return self    
        
    def fit_old(self, X):
        N, D = X.shape
        # Initialize assignments
        assignments = np.random.randint(0, self.max_n_clusters, size=N)

        for merge_step in range(self.non_local_merge_steps + 1):
            for it in range(self.max_iter):
                # --- This is the Batch Update Fix ---
                # 1. Store the state from the previous iteration
                prev_assignments = assignments.copy()
                
                # 2. Compute all cluster properties based on the STABLE previous state
                unique_clusters, n_c = np.unique(prev_assignments, return_counts=True)
                
                if len(unique_clusters) == 1:
                    break

                q_c = n_c / N
                log_q_c = np.log(q_c)
                
                cluster_sq_norms = {c: np.sum(np.linalg.norm(X[prev_assignments == c], axis=1)**2) for c in unique_clusters}
                cluster_sums = {c: np.sum(X[prev_assignments == c], axis=0) for c in unique_clusters}

                # 3. Compute ALL new assignments and store them in a temporary array
                new_assignments = np.zeros_like(assignments)
                for i in range(N):
                    costs = []
                    # Calculate cost to join each non-empty cluster
                    for c_idx, c_label in enumerate(unique_clusters):
                        n_ci = n_c[c_idx]
                        xi_sq_norm = np.linalg.norm(X[i])**2
                        sum_xj = cluster_sums[c_label]
                        sum_xj_sq_norm = cluster_sq_norms[c_label]
                        
                        sum_sq_dist = n_ci * xi_sq_norm - 2 * np.dot(X[i], sum_xj) + sum_xj_sq_norm
                        avg_sq_dist = sum_sq_dist / n_ci
                        
                        cost = avg_sq_dist - self.lambda_val * log_q_c[c_idx]
                        costs.append(cost)
                    
                    new_assignments[i] = unique_clusters[np.argmin(costs)]
                
                # 4. Update the state with the new assignments in one go
                assignments = new_assignments
                # --- End of Fix ---

                # Check for convergence
                if np.array_equal(prev_assignments, assignments):
                    break
            
            # --- Non-local merge step (this part was already correct) ---
            if merge_step < self.non_local_merge_steps:
                current_cost = self._calculate_total_cost(X, assignments)
                best_merge, best_cost_reduction = None, 0
                
                unique_clusters_after_iter = np.unique(assignments)
                if len(unique_clusters_after_iter) <= 1: 
                    break

                from itertools import combinations
                for c1, c2 in combinations(unique_clusters_after_iter, 2):
                    temp_assignments = assignments.copy()
                    temp_assignments[temp_assignments == c2] = c1
                    merged_cost = self._calculate_total_cost(X, temp_assignments)
                    cost_reduction = current_cost - merged_cost
                    if cost_reduction > best_cost_reduction:
                        best_cost_reduction, best_merge = cost_reduction, (c1, c2)
                
                if best_merge:
                    c1, c2 = best_merge
                    assignments[assignments == c2] = c1
                else:
                    break

        unique_labels, self.assignments_ = np.unique(assignments, return_inverse=True)
        self.n_clusters_ = len(unique_labels)
        return self
    
class DIB_KL_upper_bound:
    """
    Deterministic Information Bottleneck (DIB) Clustering using an upper bound
    for the KL divergence term, as derived from Hershey & Olsen (2007).

    This implementation is based on Algorithm 1 from Strouse & Schwab (2017).
    The KL divergence KL[p(x|i) || q(x|c)] between a Gaussian and a GMM is
    replaced by its convexity upper bound, which simplifies to the average
    squared Euclidean distance between point i and all points in cluster c.

    Parameters:
    ----------
    s : float
        The smoothing scale (std dev) for the Gaussian p(x|i). This sets the
        length scale of the problem.

    beta : float
        The tradeoff parameter. Larger beta values encourage retaining more
        information, leading to more clusters.

    max_n_clusters : int
        The initial maximum number of clusters. The algorithm will prune empty clusters.

    max_iter : int, default=100
        Maximum number of iterations for the main assignment loop.

    tol : float, default=1e-6
        Tolerance for checking convergence. Not used in this implementation as we
        check for exact assignment stability.
        
    non_local_merge_steps : int, default=5
        Number of passes to attempt merging clusters to escape local minima.
    """
    def __init__(self, s, beta, max_n_clusters, max_iter=100, non_local_merge_steps=5):
        self.s = s
        self.beta = beta
        self.max_n_clusters = max_n_clusters
        self.max_iter = max_iter
        self.non_local_merge_steps = non_local_merge_steps

        # Results
        self.assignments_ = None
        self.n_clusters_ = None
        self.history_ = []

    def _objective_cost(self, X, assignments):
        """Calculates the total objective cost for the current assignments."""
        total_cost = 0
        N = X.shape[0]
        unique_clusters, n_c = np.unique(assignments, return_counts=True)
        q_c = n_c / N
        log_q_c = np.log(q_c)

        # Precompute cluster properties for efficiency
        cluster_sq_norms = {}
        cluster_sums = {}
        for c_idx, c_label in enumerate(unique_clusters):
            cluster_points = X[assignments == c_label]
            cluster_sq_norms[c_label] = np.sum(np.linalg.norm(cluster_points, axis=1)**2)
            cluster_sums[c_label] = np.sum(cluster_points, axis=0)

        for i in range(N):
            c_i = assignments[i]
            c_idx = np.where(unique_clusters == c_i)[0][0]
            
            # Efficiently compute KL upper bound
            avg_sq_dist = self._kl_upper_bound(X, i, c_i, assignments, 
                                              cluster_sq_norms, cluster_sums) * (2 * self.s**2)
            
            kl_bound = avg_sq_dist / (2 * self.s**2)
            cost_i = -log_q_c[c_idx] + self.beta * kl_bound
            total_cost += cost_i
            
        return total_cost
        
    def _kl_upper_bound(self, X, i, c, assignments, cluster_sq_norms, cluster_sums):
        """
        Calculates the upper bound for KL[p(x|i) || q(x|c)] efficiently.
        The bound is (1/(2*s^2)) * mean_j_in_c(||x_i - x_j||^2).
        """
        n_c = np.sum(assignments == c)
        if n_c == 0:
            return np.inf

        # Efficient calculation of sum of squared distances:
        # sum_j ||x_i - x_j||^2 = sum_j (||x_i||^2 - 2*x_i.T*x_j + ||x_j||^2)
        # = n_c*||x_i||^2 - 2*x_i.T*(sum_j x_j) + (sum_j ||x_j||^2)
        xi_sq_norm = np.linalg.norm(X[i])**2
        sum_xj = cluster_sums[c]
        sum_xj_sq_norm = cluster_sq_norms[c]
        
        sum_sq_dist = n_c * xi_sq_norm - 2 * np.dot(X[i], sum_xj) + sum_xj_sq_norm
        avg_sq_dist = sum_sq_dist / n_c
        
        return avg_sq_dist / (2 * self.s**2)

    def fit(self, X):
        """
        Fits the DIB clustering model to the data X.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        """
        N, D = X.shape
        
        # 1. Initial assignment
        assignments = np.random.randint(0, self.max_n_clusters, size=N)

        for merge_step in range(self.non_local_merge_steps + 1):
            # 2. Main iterative loop
            for it in range(self.max_iter):
                prev_assignments = assignments.copy()
                unique_clusters, n_c = np.unique(assignments, return_counts=True)
                
                if len(unique_clusters) == 1:
                    break # Already collapsed to one cluster

                q_c = n_c / N
                log_q_c = np.log(q_c)
                
                # Precompute cluster properties for efficiency
                cluster_sq_norms = {c: np.sum(np.linalg.norm(X[assignments == c], axis=1)**2) for c in unique_clusters}
                cluster_sums = {c: np.sum(X[assignments == c], axis=0) for c in unique_clusters}

                # Update assignments for each point
                for i in range(N):
                    costs = []
                    for c_idx, c_label in enumerate(unique_clusters):
                        kl_bound = self._kl_upper_bound(X, i, c_label, assignments, cluster_sq_norms, cluster_sums)
                        cost = log_q_c[c_idx] - self.beta * kl_bound
                        costs.append(cost)
                    
                    assignments[i] = unique_clusters[np.argmax(costs)]

                # Check for convergence
                if np.array_equal(prev_assignments, assignments):
                    break
            
            # --- Non-local merge step ---
            if merge_step < self.non_local_merge_steps:
                current_cost = self._objective_cost(X, assignments)
                best_merge = None
                best_cost_reduction = 0
                
                unique_clusters = np.unique(assignments)
                if len(unique_clusters) <= 1:
                    break # Nothing to merge

                # Iterate over pairs of clusters
                from itertools import combinations
                for c1, c2 in combinations(unique_clusters, 2):
                    temp_assignments = assignments.copy()
                    temp_assignments[temp_assignments == c2] = c1 # Merge c2 into c1
                    
                    # Recalculate cost
                    merged_cost = self._objective_cost(X, temp_assignments)
                    cost_reduction = current_cost - merged_cost
                    
                    if cost_reduction > best_cost_reduction:
                        best_cost_reduction = cost_reduction
                        best_merge = (c1, c2)
                
                if best_merge:
                    # Perform the best merge and continue iterating
                    c1, c2 = best_merge
                    assignments[assignments == c2] = c1
                else:
                    # No merge improved the objective, so we are done
                    break

        # Relabel clusters to be contiguous from 0
        unique_labels, self.assignments_ = np.unique(assignments, return_inverse=True)
        self.n_clusters_ = len(unique_labels)
        
        return self

class DIBAnalyzer:
    def __init__(self, X, sentences):
        self.X = X
        self.results_ = []
        self._stable_solutions = None # Cache for analysis results
        self.max_n_clusters_run_ = None # Stores the max_n_clusters for the last run
        self.sentences = np.array(sentences) # Correctly store the sentences

    def _compute_h_c(self, assignments):
        N = len(assignments)
        _, n_c = np.unique(assignments, return_counts=True)
        p_c = n_c / N
        return entropy(p_c, base=2)

    def run(self, tau_values, max_n_clusters=None, seed=None):
        # 1. Check for empty input data
        if self.X is None or self.X.shape[0] == 0:
            print("Error: Input data X is empty. Aborting run.")
            self.results_ = []
            return self
            
        # 2. Determine the max_n_clusters for THIS RUN
        if max_n_clusters is None:
            num_samples = self.X.shape[0]
            default_k = int(np.sqrt(num_samples))
            run_max_clusters = max(1, default_k)
        else:
            run_max_clusters = max_n_clusters
                
        if run_max_clusters <= 0:
            print(f"Error: max_n_clusters must be > 0, but got {run_max_clusters}. Setting to 1.")
            run_max_clusters = 1
        
        # Store the max clusters used for this specific run, for plotting later
        self.max_n_clusters_run_ = run_max_clusters
        
        self.results_ = []
        self._stable_solutions = None # Invalidate cache
        pbar = tqdm(sorted(tau_values), desc="Scanning tau values")
        for tau_val in pbar:
            # THIS IS THE CRITICAL FIX: Pass run_max_clusters, not a stale self attribute.
            model = DIB_Simple(
                tau_val=tau_val, 
                max_n_clusters=run_max_clusters 
            )
            model.fit(self.X, seed=seed)
            self.results_.append({
                'tau': tau_val,
                'n_clusters': model.n_clusters_,
                'assignments': model.assignments_,
            })
            pbar.set_postfix({'n_clusters': model.n_clusters_})
        return self

    def _analyze_stability(self, window_size=2):
        """
        Analyzes sweep results using a robust kink detection method.
        It uses linear regression on a window of points to estimate the slope
        on either side of a potential kink, as implied by Strouse & Schwab's figure.

        Parameters:
        ----------
        window_size : int, default=2
            The number of points to use on each side of a candidate point
            to perform linear regression for slope estimation. Must be >= 1.
        """
        # Simple caching check
        if self._stable_solutions is not None and self._cache_params == {'window_size': window_size}:
            return self._stable_solutions
        
        if not self.results_:
            return []

        from collections import defaultdict
        from sklearn.linear_model import LinearRegression

        groups = defaultdict(list)
        for res in self.results_:
            groups[res['n_clusters']].append(res)
        
        # We need the full tau range for each nc for the diagnostics
        tau_ranges_for_nc = {
            nc: (min(res['tau'] for res in results), max(res['tau'] for res in results))
            for nc, results in groups.items()
        }

        # Create a list of representative solutions, then sort by nc
        # This fixes the non-monotonic plot bug.
        rep_solutions_list = sorted([
            {
                'n_clusters': nc,
                'H(c)': self._compute_h_c(results[0]['assignments']),
                'assignments': results[0]['assignments'],
                'tau': max(res['tau'] for res in results)
            }
            for nc, results in groups.items()
        ], key=lambda x: x['n_clusters'])
        
        num_solutions = len(rep_solutions_list)
        if num_solutions < (2 * window_size + 1):
            print(f"Warning: Not enough stable solutions ({num_solutions}) to calculate kinks with window size {window_size}. "
                  "Try a smaller window_size or a denser tau sweep.")
            # Fallback to empty list
            self._stable_solutions = []
            self._cache_params = {'window_size': window_size}
            return self._stable_solutions

        stable_solutions = []

        # Iterate through the points on the curve where a kink can be calculated
        for i in range(num_solutions):
            current_sol = rep_solutions_list[i]
            nc = current_sol['n_clusters']
            
            # Kinks at the edges are undefined. We need a full window on both sides.
            if i < window_size or i >= num_solutions - window_size:
                kink_angle = 0.0
            else:
                # --- Left side for regression (points BEFORE current point) ---
                left_indices = range(i - window_size, i)
                X_left = np.array([[rep_solutions_list[j]['H(c)']] for j in left_indices])
                y_left = np.array([rep_solutions_list[j]['tau'] for j in left_indices])
                
                # --- Right side for regression (points INCLUDING current point) ---
                # The "slope after" is best represented by the trend starting from the current point.
                right_indices = range(i, i + window_size)
                X_right = np.array([[rep_solutions_list[j]['H(c)']] for j in right_indices])
                y_right = np.array([rep_solutions_list[j]['tau'] for j in right_indices])

                # Perform linear regression to find the slopes
                model_left = LinearRegression().fit(X_left, y_left)
                model_right = LinearRegression().fit(X_right, y_right)
                
                slope_before = model_left.coef_[0]
                slope_after = model_right.coef_[0]

                # The kink is the change in the angle. Slopes on this plot are expected to be negative.
                # Since tan is an increasing function, and slopes are negative (e.g., -5 vs -10),
                # a positive angle means slope_before > slope_after (e.g., -5 > -10).
                kink_angle = np.rad2deg(np.arctan(slope_before) - np.arctan(slope_after))

            tau_min, tau_max = tau_ranges_for_nc[nc]
            
            stable_solutions.append({
                'n_clusters': nc,
                'kink_angle': kink_angle,
                'assignments': current_sol['assignments'],
                'H(c)': current_sol['H(c)'],
                'tau_min': tau_min,
                'tau_max': tau_max,
            })
        
        # Cache the results
        self._stable_solutions = stable_solutions
        self._cache_params = {'window_size': window_size}
        return self._stable_solutions


    def get_recommendation(self, min_clusters=3, metric='kink_angle', window_size=2):
        """
        Provides recommendations based on two heuristics: the 'kink_angle' for
        local sharpness and the 'elbow' for global diminishing returns.
        """
        
        
        # --- Helper function to calculate AND STORE diagnostics ---
        def process_diagnostics(solution):
            # ... (all the calculation code is the same) ...
            tau_mid = np.sqrt(solution['tau_min'] * solution['tau_max']); assignments = solution['assignments']; unique_clusters, n_c = np.unique(assignments, return_counts=True); q_c = n_c / self.X.shape[0]; log_q_c = np.log(q_c); cluster_sq_norms = {c: np.sum(np.linalg.norm(self.X[assignments == c], axis=1)**2) for c in unique_clusters}; cluster_sums = {c: np.sum(self.X[assignments == c], axis=0) for c in unique_clusters}; dist_terms, reg_terms = [], []
            for i in range(self.X.shape[0]):
                c_i = assignments[i]; c_idx = np.where(unique_clusters == c_i)[0][0]; n_ci = n_c[c_idx]; xi_sq_norm = np.linalg.norm(self.X[i])**2; sum_xj = cluster_sums[c_i]; sum_xj_sq_norm = cluster_sq_norms[c_i]; sum_sq_dist = n_ci * xi_sq_norm - 2 * np.dot(self.X[i], sum_xj) + sum_xj_sq_norm; avg_sq_dist = sum_sq_dist / n_ci; dist_terms.append(avg_sq_dist); reg_terms.append(-tau_mid * log_q_c[c_idx])
            
            avg_dist_term = np.mean(dist_terms)
            avg_reg_term = np.mean(reg_terms)
        
            # --- THIS IS THE FIX ---
            # Store the diagnostics in the solution dictionary itself
            solution['diagnostics'] = {
                'avg_distance_term': avg_dist_term,
                'avg_regularization_term': avg_reg_term,
                'tau_for_diagnostics': tau_mid
            }
            # --- END OF FIX ---
            
            return solution, avg_dist_term, avg_reg_term, tau_mid
        
        # --- Helper function to print diagnostics ---
        def print_report(solution, title, metric):
            diag_data = solution['diagnostics']
            print(f"\n--- {title} ---")
            print(f"Recommended number of clusters (nc): {solution['n_clusters']}")
            print(f"Stable in tau range: [{solution['tau_min']:.4f}, {solution['tau_max']:.4f}]")
            print(f"Robustness ({metric}): {solution.get(metric, 0):.2f}")
            print("\n--- Cost Function Diagnostics ---")
            print(f"(Evaluated at geometric mean tau = {diag_data['tau_for_diagnostics']:.4f})")
            print(f"  Avg. Distance Term Contribution: {diag_data['avg_distance_term']:.4f}")
            print(f"  Avg. Regularization Term Contribution: {diag_data['avg_regularization_term']:.4f}")
            print("-------------------------------------")
        
        
        
        stable_solutions = self._analyze_stability(window_size)
        
        if not stable_solutions:
            print("Could not find any stable clustering solutions.")
            return None, None
        
        

        

        # --- Helper function to print diagnostics for any given solution ---
        def print_diagnostics(solution, title):
            # Calculate diagnostic values
            tau_mid = np.sqrt(solution['tau_min'] * solution['tau_max'])
            assignments = solution['assignments']
            unique_clusters, n_c = np.unique(assignments, return_counts=True)
            q_c = n_c / self.X.shape[0]
            log_q_c = np.log(q_c)
            cluster_sq_norms = {c: np.sum(np.linalg.norm(self.X[assignments == c], axis=1)**2) for c in unique_clusters}
            cluster_sums = {c: np.sum(self.X[assignments == c], axis=0) for c in unique_clusters}
            dist_terms, reg_terms = [], []
            for i in range(self.X.shape[0]):
                c_i = assignments[i]
                c_idx = np.where(unique_clusters == c_i)[0][0]
                n_ci = n_c[c_idx]
                xi_sq_norm = np.linalg.norm(self.X[i])**2
                sum_xj = cluster_sums[c_i]
                sum_xj_sq_norm = cluster_sq_norms[c_i]
                sum_sq_dist = n_ci * xi_sq_norm - 2 * np.dot(self.X[i], sum_xj) + sum_xj_sq_norm
                avg_sq_dist = sum_sq_dist / n_ci
                dist_terms.append(avg_sq_dist)
                reg_terms.append(-tau_mid * log_q_c[c_idx])
                
            avg_dist_term = np.mean(dist_terms)
            avg_reg_term = np.mean(reg_terms)
                
            # Print the results
            print(f"\n--- {title} ---")
            print(f"Recommended number of clusters (nc): {solution['n_clusters']}")
            print(f"Stable in tau range: [{solution['tau_min']:.4f}, {solution['tau_max']:.4f}]")
            print(f"Robustness ({metric}): {solution.get(metric, 0):.2f}")
            print("\n--- Cost Function Diagnostics ---")
            print(f"(Evaluated at geometric mean tau = {tau_mid:.4f})")
            print(f"  Avg. Distance Term Contribution: {avg_dist_term:.4f}")
            print(f"  Avg. Regularization Term Contribution: {avg_reg_term:.4f}")
            print("-------------------------------------")

        # --- Heuristic 1: Kink Angle (Local Sharpness) ---
        candidate_solutions = [s for s in stable_solutions if s['n_clusters'] >= min_clusters]
        if not candidate_solutions:
            best_kink_solution = max(stable_solutions, key=lambda x: x.get(metric, 0))
            print(f"Warning: No solutions found with nc >= {min_clusters}. Using best overall kink.")
        else:
            best_kink_solution = max(candidate_solutions, key=lambda x: x.get(metric, 0))
        
        print_diagnostics(best_kink_solution, f"Recommendation via '{metric}' Heuristic")
        
        # Process and print diagnostics
        best_kink_solution, _, _, _ = process_diagnostics(best_kink_solution)
        print_report(best_kink_solution, f"Recommendation via '{metric}' Heuristic", metric)
        
        # --- Heuristic 2: Elbow Method (Global Information Gain) ---
        points = np.array([[sol['n_clusters'], sol['H(c)']] for sol in stable_solutions])
        best_elbow_solution = None
        if len(points) > 2:
            points_normalized = points.copy()
            points_normalized[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min() + 1e-9)
            points_normalized[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min() + 1e-9)
            
            line_start, line_end = points_normalized[0], points_normalized[-1]
            line_vec = line_end - line_start
            line_vec_norm = np.linalg.norm(line_vec)
            
            point_vecs = points_normalized - line_start
            cross_product = np.cross(point_vecs, line_vec)
            distances = np.abs(cross_product) / (line_vec_norm + 1e-9)
            
            elbow_index = np.argmax(distances)
            best_elbow_nc = int(points[elbow_index, 0])
            best_elbow_solution = next(s for s in stable_solutions if s['n_clusters'] == best_elbow_nc)
            print_diagnostics(best_elbow_solution, "Recommendation via 'Elbow' Heuristic")
            
            # Process and print diagnostics
            best_elbow_solution, _, _, _ = process_diagnostics(best_elbow_solution)
            print_report(best_elbow_solution, "Recommendation via 'Elbow' Heuristic", metric)
            
        else:
            print("\nNote: Not enough points to calculate a meaningful elbow.")

        return best_kink_solution, best_elbow_solution


    def analyze_cluster_topics(self, recommendation, top_n_words=5, num_example_sentences=3):
        """
        Analyzes the text content of clusters, generates a topic name from
        keywords, and shows representative sentences from each cluster.

        Parameters:
        ----------
        recommendation : dict
            A single recommendation dictionary.
        top_n_words : int, default=5
            The number of top keywords to extract for the topic name.
        num_example_sentences : int, default=3
            The number of representative sentences to display for each cluster.
        """
        if recommendation is None:
            print("Cannot analyze topics: Recommendation is None.")
            return None

        print(f"\n--- Detailed Topic Analysis for Clustering (nc={recommendation['n_clusters']}) ---")
        
        
        assignments = recommendation['assignments']

        # --- THIS IS THE FIX ---
        # Add a verification check to ensure data consistency
        if len(assignments) != len(self.sentences):
            print("\n--- CRITICAL ERROR in analyze_cluster_topics ---")
            print("The number of cluster assignments does not match the number of sentences.")
            print(f"  - len(assignments): {len(assignments)}")
            print(f"  - len(self.sentences): {len(self.sentences)}")
            print("This can happen if results from different experimental runs are mixed.")
            print("Aborting topic analysis.")
            return None
        # --- END OF FIX ---

        
        unique_labels = sorted(np.unique(assignments))
        
        # --- Step 1: Get TF-IDF Keywords (as before) ---
        docs_by_cluster = defaultdict(list)
        for i, sentence in enumerate(self.sentences):
            docs_by_cluster[assignments[i]].append(sentence)
        corpus = [" ".join(docs_by_cluster[label]) for label in unique_labels]
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        topic_analysis_results = {}

        # --- Step 2: Loop through each cluster to analyze and print ---
        for i, label in enumerate(unique_labels):
            print(f"\n==================== Cluster {label} ====================")
            
            # --- Generate Topic Name ---
            cluster_tfidf_row = tfidf_matrix[i]
            top_word_indices = cluster_tfidf_row.toarray()[0].argsort()[-top_n_words:][::-1]
            top_words = feature_names[top_word_indices]
            topic_name = " | ".join(top_words)
            print(f"Topic Name: {topic_name}")

            # --- Find and Display Representative Sentences ---
            cluster_indices = np.where(assignments == label)[0]
            if len(cluster_indices) == 0:
                print("  (No sentences in this cluster)")
                continue

            # Get embeddings for sentences in this cluster
            cluster_embeddings = self.X[cluster_indices]
            
            # Calculate the centroid (geometric center) of the cluster
            centroid = cluster_embeddings.mean(axis=0)
            
            # Find the indices of the sentences closest to the centroid
            # We use pairwise_distances_argmin_min to efficiently find the closest points.
            # We need to reshape centroid to be a 2D array for the function.
            closest_indices_in_cluster, _ = pairwise_distances_argmin_min(
                cluster_embeddings, centroid.reshape(1, -1)
            )
            
            # If we need more than one, we find the argsort of the distances
            if num_example_sentences > 1 and len(cluster_embeddings) > 1:
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                # Get the indices of the N smallest distances within the cluster
                closest_indices_in_cluster = distances.argsort()[:num_example_sentences]

            # Map these intra-cluster indices back to the global sentence indices
            representative_global_indices = cluster_indices[closest_indices_in_cluster]
            
            print(f"\n  Representative Sentences (closest to cluster centroid):")
            for global_idx in representative_global_indices:
                print(f"    - \"{self.sentences[global_idx]}\"")

            topic_analysis_results[label] = {
                'name': topic_name,
                'keywords': list(top_words),
                'representative_sentences': list(self.sentences[representative_global_indices])
            }
        
        print("\n===================================================")
        return topic_analysis_results


    def get_final_recommendation(self, min_clusters=3):
        """
        Performs a robust model selection by analyzing kink angles with multiple
        window sizes and selecting the best result based on a set of criteria.

        Criteria for "Best":
        1.  Must have nc >= min_clusters.
        2.  Among candidates, prefer the one with the smallest recommended nc.
        3.  If there's a tie in nc, prefer the one with the largest kink angle (robustness).

        Parameters:
        ----------
        min_clusters : int, default=3
            The minimum number of clusters to consider for a valid recommendation.
        """
        print("\n--- Starting Robust Final Recommendation Analysis ---")
        
        window_sizes_to_test = [2, 3]
        all_recommendations = []

        # Step 1: Gather the best recommendation from the kink-angle heuristic for each window size
        for ws in window_sizes_to_test:
            print(f"\nAnalyzing with kink window_size = {ws}...")
            # We need to invalidate the cache to force recalculation with the new window size
            self._stable_solutions = None
            self._cache_params = None
            
            stable_solutions = self._analyze_stability(window_size=ws)
            
            if not stable_solutions:
                print("  No stable solutions found for this window size.")
                continue

            # Filter for valid candidates
            candidate_solutions = [s for s in stable_solutions if s['n_clusters'] >= min_clusters]
            
            if not candidate_solutions:
                print(f"  Warning: No solutions found with nc >= {min_clusters}. Checking non-trivial solutions.")
                candidate_solutions = [s for s in stable_solutions if s['n_clusters'] > 1]
                if not candidate_solutions:
                    print("  Only nc=1 solution found.")
                    continue
            
            # Find the best solution for this window size
            best_solution_for_ws = max(candidate_solutions, key=lambda x: x['kink_angle'])
            best_solution_for_ws['window_size'] = ws # Tag with the window size used
            all_recommendations.append(best_solution_for_ws)
            
            print(f"  Best candidate for ws={ws}: nc={best_solution_for_ws['n_clusters']}, kink={best_solution_for_ws['kink_angle']:.2f}")

        # Step 2: Apply the selection criteria to the collected recommendations
        if not all_recommendations:
            print("\n--- Final Recommendation: No suitable clustering found. ---")
            return None

        print("\n--- Comparing candidates from all window sizes ---")
        # Select the solution with the strongest clustering signal (largest kink angle)
        # This indicates the most robust and well-defined cluster structure
        final_recommendation = max(all_recommendations, key=lambda x: x['kink_angle'])

        # --- Print and return the final, definitive recommendation ---
        print("\n=======================================================")
        print("--- Final DIB Clustering Recommendation ---")
        print(f"Selected from candidates with window_size in {window_sizes_to_test}")
        print(f"Selection criteria: Largest kink angle (strongest clustering signal).")
        print("=======================================================")
        
        # Use the existing print helper for a full report
        self._print_final_report(final_recommendation, 'kink_angle')

        return final_recommendation

    def _print_final_report(self, solution, metric):
        """A helper to print the final recommendation and diagnostics."""
        # This reuses the diagnostic calculation and printing logic
        tau_mid = np.sqrt(solution['tau_min'] * solution['tau_max'])
        assignments = solution['assignments']
        unique_clusters, n_c = np.unique(assignments, return_counts=True)
        q_c = n_c / self.X.shape[0]
        log_q_c = np.log(q_c)
        cluster_sq_norms = {c: np.sum(np.linalg.norm(self.X[assignments == c], axis=1)**2) for c in unique_clusters}
        cluster_sums = {c: np.sum(self.X[assignments == c], axis=0) for c in unique_clusters}
        dist_terms, reg_terms = [], []
        for i in range(self.X.shape[0]):
            c_i = assignments[i]; c_idx = np.where(unique_clusters == c_i)[0][0]; n_ci = n_c[c_idx]; xi_sq_norm = np.linalg.norm(self.X[i])**2; sum_xj = cluster_sums[c_i]; sum_xj_sq_norm = cluster_sq_norms[c_i]; sum_sq_dist = n_ci * xi_sq_norm - 2 * np.dot(self.X[i], sum_xj) + sum_xj_sq_norm; avg_sq_dist = sum_sq_dist / n_ci; dist_terms.append(avg_sq_dist); reg_terms.append(-tau_mid * log_q_c[c_idx])
        avg_dist_term = np.mean(dist_terms)
        avg_reg_term = np.mean(reg_terms)
        
        print(f"Recommended number of clusters (nc): {solution['n_clusters']}")
        print(f"Stable in tau range: [{solution['tau_min']:.4f}, {solution['tau_max']:.4f}]")
        print(f"Robustness ({metric}): {solution.get(metric, 0):.2f} (found with window_size={solution.get('window_size', 'N/A')})")
        print("\n--- Cost Function Diagnostics ---")
        print(f"(Evaluated at geometric mean tau = {tau_mid:.4f})")
        print(f"  Avg. Distance Term Contribution: {avg_dist_term:.4f}")
        print(f"  Avg. Regularization Term Contribution: {avg_reg_term:.4f}")
        print("-------------------------------------")


    def plot(self, recommendation, metric='kink_angle',window_size=2):
        """
        Generates diagnostic plots, including visualizations for the
        recommended solution and the second-best alternative.
        
        Parameters:
        ----------
        recommendation : dict
            The dictionary returned by get_recommendation(). This is the primary
            solution to be plotted as "Best".
        metric : str, default='log_width'
            The stability metric used to find the second-best solution for comparison.
        """
        stable_solutions = self._analyze_stability(window_size)

        if not stable_solutions:
            print("Plotting aborted: No stable solutions found.")
            return
        
        # The primary solution to plot is the one passed in.
        best_sol = recommendation
        
        # --- Find the second-best solution for comparison ---
        # Filter out the best solution and any trivial ones
        candidates_for_second_best = [
            s for s in stable_solutions 
            if s['n_clusters'] != best_sol['n_clusters'] and s['n_clusters'] > 1
        ]
        
        if candidates_for_second_best:
            second_best_sol = max(candidates_for_second_best, key=lambda x: x[metric])
        else:
            second_best_sol = None

        # --- Create Figure ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"DIB Clustering Diagnostics (sweeping tau, ranked by {metric})", fontsize=16)
        
        # --- Top Row: Standard Diagnostic Plots ---
        n_clusters_vals = [sol['n_clusters'] for sol in stable_solutions]
        metric_vals = [sol[metric] for sol in stable_solutions]

        axes[0, 0].plot(n_clusters_vals, [s['H(c)'] for s in stable_solutions], 'o-')
        axes[0, 0].set_title('Information Profile')
        axes[0, 0].set_xlabel('# of Clusters (nc)')
        axes[0, 0].set_ylabel('Compression H(c)')
        axes[0, 0].grid(True, linestyle=':')

        axes[0, 1].bar(n_clusters_vals, metric_vals)
        axes[0, 1].set_title('Model Selection')
        axes[0, 1].set_xlabel('# of Clusters (nc)')
        axes[0, 1].set_ylabel(f'Stability ({metric})')
        axes[0, 1].grid(True, linestyle=':')
        
        # --- Prepare data for scatter plots ---
        if self.X.shape[1] > 2:
            X_2d = PCA(n_components=2).fit_transform(self.X)
        else:
            X_2d = self.X

        # --- Plot Best Solution ---
        if best_sol:
            assignments = best_sol['assignments']
            unique_labels = np.unique(assignments)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                cluster_points = X_2d[assignments == label]
                axes[0, 2].scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=str(label), alpha=0.7)
            axes[0, 2].legend(title="Clusters")
            axes[0, 2].set_title(f"Recommended Clustering (nc={best_sol['n_clusters']})")

        # --- Plot Second-Best Solution ---
        if second_best_sol:
            assignments = second_best_sol['assignments']
            unique_labels = np.unique(assignments)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                cluster_points = X_2d[assignments == label]
                axes[1, 2].scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=str(label), alpha=0.7)
            axes[1, 2].legend(title="Clusters")
            axes[1, 2].set_title(f"2nd Best Clustering (nc={second_best_sol['n_clusters']})")
        else:
            axes[1, 2].text(0.5, 0.5, "No second-best solution found", ha='center', va='center', transform=axes[1, 2].transAxes)

        # --- Clean up unused axes in the second row ---
        fig.delaxes(axes[1, 0])
        fig.delaxes(axes[1, 1])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.savefig(("Plots_of_DIB.png"))
        plt.show()
        
        
# High level class for repated experiments and reporting statistics
class DIBExperimentRunner:
    """
    Runs the DIBAnalyzer multiple times with different random seeds to assess
    the stability and variance of the clustering recommendations.
    """
    def __init__(self, X, tau_values, max_n_clusters=None):
        
        self.X = X
        self.tau_values = tau_values
        self.max_n_clusters = max_n_clusters
        self.kink_recommendations = []
        self.elbow_recommendations = []
        self.analyzers = []
        
        self.all_run_stable_solutions = [] # To store full results for plotting
        self.seeds = [] # To store seeds for reproducibility
                
        # self.sentences = np.array(sentences) # Store sentences as a numpy array for easier indexing
        

    def run_experiments(self, M=10, min_clusters=3, metric='kink_angle', sentences=None):
        
        if sentences is None:
            raise ValueError("You must provide the 'sentences' keyword argument to run_experiments.")
    
        self.kink_recommendations = []; self.elbow_recommendations = []
        self.all_run_stable_solutions = []; self.seeds = []
        master_rng = np.random.default_rng()
        self.analyzers = []
        
        
        for m in range(M):
            exp_seed = master_rng.integers(1e9); self.seeds.append(exp_seed)
            print(f"\n--- Running Experiment {m+1}/{M} (Seed: {exp_seed}) ---")
            analyzer = DIBAnalyzer(self.X, sentences)
            analyzer.run(self.tau_values, self.max_n_clusters, seed=exp_seed)
            self.all_run_stable_solutions.append(analyzer._analyze_stability())
            
            kink_reco, elbow_reco = analyzer.get_recommendation(min_clusters, metric)
            
            if kink_reco: self.kink_recommendations.append(kink_reco)
            if elbow_reco: self.elbow_recommendations.append(elbow_reco)
            
            # Store the analyzer object itself
            self.analyzers.append(analyzer) 
            
        print(f"\n--- {M} Experiments Complete ---"); print(f"Seeds used: {self.seeds}"); return self



    def plot_information_profiles(self, num_to_plot=5):
        """
        Plots the 'Information Profile' (H(c) vs nc) for a subset of the
        experiment runs to visualize variability.
        """
        if not self.all_run_stable_solutions:
            print("No experiment data to plot. Please run experiments first.")
            return

        plt.figure(figsize=(10, 7))
        
        num_to_plot = min(num_to_plot, len(self.all_run_stable_solutions))
        
        for i in range(num_to_plot):
            solutions = self.all_run_stable_solutions[i]
            seed = self.seeds[i]
            if not solutions:
                continue
            
            n_clusters_vals = [sol['n_clusters'] for sol in solutions]
            hc_vals = [sol['H(c)'] for sol in solutions]
            
            plt.plot(n_clusters_vals, hc_vals, 'o-', alpha=0.7, label=f'Seed: {seed}')

        plt.title('Information Profiles Across Multiple Runs')
        plt.xlabel('# of Clusters (nc)')
        plt.ylabel('Compression H(c)')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.savefig("Information_profiles_DIB_experiments.png")
        plt.show()
        
        

    def report_statistics(self):
        """Calculates and prints summary statistics from all experiment runs."""
        print("\n--- Overall Experiment Statistics ---")
        
        if self.kink_recommendations:
            self._report_single_heuristic(self.kink_recommendations, "Kink Angle")
        
        if self.elbow_recommendations:
            self._report_single_heuristic(self.elbow_recommendations, "Elbow Method")

    def _report_single_heuristic(self, recommendations, title):
        if not recommendations:
            print(f"\nNo successful recommendations found for '{title}' heuristic.")
            return

        print(f"\n--- Statistics for '{title}' Heuristic ({len(recommendations)} runs) ---")
        
        # Recommended nc statistics
        n_clusters = [rec['n_clusters'] for rec in recommendations]
        nc_mean = np.mean(n_clusters)
        nc_std = np.std(n_clusters)
        nc_mode_result = mode(n_clusters)
        nc_mode = nc_mode_result.mode
        
        print(f"Recommended nc: Mean={nc_mean:.2f}, Std={nc_std:.2f}, Mode={nc_mode}")

        # Diagnostic statistics
        dist_terms = [rec['diagnostics']['avg_distance_term'] for rec in recommendations]
        reg_terms = [rec['diagnostics']['avg_regularization_term'] for rec in recommendations]
        
        print(f"Avg. Distance Term: Mean={np.mean(dist_terms):.4f}, Std={np.std(dist_terms):.4f}")
        print(f"Avg. Regularization Term: Mean={np.mean(reg_terms):.4f}, Std={np.std(reg_terms):.4f}")
        print("--------------------------------------------------")        
        
    def report_latex_table(self):
        """
        Calculates and prints a comprehensive summary table of the experiment results
        in a format ready for a LaTeX document.
        """
        print("\n--- LaTeX Summary Table ---")
        print("Copy the text between the dashed lines into your LaTeX document.")
        print("You will need the 'booktabs' package: \\usepackage{booktabs}")
        print("----------------------------------------------------------")
        
        # --- LaTeX Table Header ---
        print("\\begin{table}[h!]")
        print("\\centering")
        print("\\caption{Statistical Summary of DIB Clustering Experiments (M=10 Runs)}")
        print("\\label{tab:dib_summary}")
        # Use 'lcr' for left, center, right alignment to make it look nicer
        print("\\begin{tabular}{lcr}")
        print("\\toprule")
        print("Metric & Kink Angle Heuristic & Elbow Heuristic \\\\")
        print("\\midrule")

        # --- Helper to format a row with specific precision ---
        def format_row(label, kink_data, elbow_data, precision=2):
            kink_mean = np.mean(kink_data) if kink_data else np.nan
            kink_std = np.std(kink_data) if kink_data else np.nan
            elbow_mean = np.mean(elbow_data) if elbow_data else np.nan
            elbow_std = np.std(elbow_data) if elbow_data else np.nan
            
            kink_str = f"{kink_mean:.{precision}f} $\\pm$ {kink_std:.{precision}f}" if kink_data else "N/A"
            elbow_str = f"{elbow_mean:.{precision}f} $\\pm$ {elbow_std:.{precision}f}" if elbow_data else "N/A"
            
            print(f"{label} & {kink_str} & {elbow_str} \\\\")

        # --- Data Extraction (including the new stats) ---
        kink_recs = self.kink_recommendations
        elbow_recs = self.elbow_recommendations if self.elbow_recommendations else []

        # Recommended number of clusters
        format_row("Recommended \\# Clusters (nc)", [r['n_clusters'] for r in kink_recs], [r['n_clusters'] for r in elbow_recs], precision=2)
        
        # Kink angle
        format_row("Robustness (kink angle)", [r['kink_angle'] for r in kink_recs], [r['kink_angle'] for r in elbow_recs], precision=2)

        # Add a small vertical space for readability
        print("\\addlinespace")

        # Stability plateau for tau
        format_row("Stability Lower Bound ($\\tau_{min}$)", [r['tau_min'] for r in kink_recs], [r['tau_min'] for r in elbow_recs], precision=4)
        format_row("Stability Upper Bound ($\\tau_{max}$)", [r['tau_max'] for r in kink_recs], [r['tau_max'] for r in elbow_recs], precision=4)
        
        print("\\addlinespace")

        # Cost function diagnostics
        format_row("Avg. Distance Term", [r['diagnostics']['avg_distance_term'] for r in kink_recs], [r['diagnostics']['avg_distance_term'] for r in elbow_recs], precision=4)
        format_row("Avg. Regularization Term", [r['diagnostics']['avg_regularization_term'] for r in kink_recs], [r['diagnostics']['avg_regularization_term'] for r in elbow_recs], precision=4)

        # --- LaTeX Table Footer ---
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        print("----------------------------------------------------------")

        # --- Helper to format a row ---
        def format_row(label, kink_data, elbow_data):
            # Calculate mean and std, handling potential empty lists
            kink_mean = np.mean(kink_data) if kink_data else np.nan
            kink_std = np.std(kink_data) if kink_data else np.nan
            elbow_mean = np.mean(elbow_data) if elbow_data else np.nan
            elbow_std = np.std(elbow_data) if elbow_data else np.nan
            
            # Format strings with mean ± std
            kink_str = f"{kink_mean:.2f} $\\pm$ {kink_std:.2f}" if kink_data else "N/A"
            elbow_str = f"{elbow_mean:.2f} $\\pm$ {elbow_std:.2f}" if elbow_data else "N/A"
            
            print(f"{label} & {kink_str} & {elbow_str} \\\\")

        # --- Data Extraction ---
        kink_ncs = [rec['n_clusters'] for rec in self.kink_recommendations]
        kink_dists = [rec['diagnostics']['avg_distance_term'] for rec in self.kink_recommendations]
        kink_regs = [rec['diagnostics']['avg_regularization_term'] for rec in self.kink_recommendations]
        
        elbow_ncs = [rec['n_clusters'] for rec in self.elbow_recommendations] if self.elbow_recommendations else []
        elbow_dists = [rec['diagnostics']['avg_distance_term'] for rec in self.elbow_recommendations] if self.elbow_recommendations else []
        elbow_regs = [rec['diagnostics']['avg_regularization_term'] for rec in self.elbow_recommendations] if self.elbow_recommendations else []

        # --- Populate Table Rows ---
        format_row("Recommended \# Clusters (nc)", kink_ncs, elbow_ncs)
        format_row("Avg. Distance Term", kink_dists, elbow_dists)
        format_row("Avg. Regularization Term", kink_regs, elbow_regs)

        # --- LaTeX Table Footer ---
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        print("----------------------------------------------------------")        
        
        
        
if __name__ == "__main__":
    
    from sklearn.datasets import make_blobs

    # 1. Generate synthetic data
    n_samples = 300
    centers = [[0, 0], [5, 5], [0, 5]]
    X, y_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.8, random_state=42)
    
    # 2. Instantiate and run the analyzer
    analyzer = DIBAnalyzer(X)
    
    # Define ranges for s and beta
    # s should be on the order of the data's standard deviation
    s_values = [0.5, 1.0, 2.0] 
    
    s_values = [1.0, 1.3, 1.5, 1.8, 2.0]
    
    # Beta values can span several orders of magnitude
    # beta_values = np.logspace(-1, 2, 20)
    
    beta_values = np.logspace(-1, 1.4, 12) # np.logspace(-1, 1.3, 10)
    
    # Run the full analysis
    analyzer.run(s_values=s_values, beta_values=beta_values, max_n_clusters=10)
    
    recommendation = analyzer.get_recommendation()
    
    # 3. Plot the results
    analyzer.plot()