#!/usr/bin/env python3
"""
Corrected Semantic Faithfulness (SF) computation with proper constraint enforcement.
Q-step enforces the marginal constraint: sum_i p_c[i] * Q[i,j] = p_q[j]
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy


def compute_semantic_faithfulness(p_c, p_q, p_a, Q_init=None,
                                  tol_outer=1e-6, tol_inner=1e-8,
                                  max_outer_iter=100, max_inner_iter=50,
                                  debug=False):
    """
    Compute Semantic Faithfulness (SF) with proper constraint enforcement.

    The Q-step enforces: sum_i p_c[i] * Q[i,j] = p_q[j] for all j

    Args:
        p_c: Context probability distribution
        p_q: Question probability distribution
        p_a: Answer probability distribution
        Q_init: Initial Q matrix (N x N)
        tol_outer: Outer loop tolerance
        tol_inner: Inner loop tolerance
        max_outer_iter: Max outer iterations
        max_inner_iter: Max inner iterations for xi-nu alternation
        debug: Print convergence information

    Returns:
        dict with F_S, D_min, A_star, Q_star, iterations, converged
    """
    N = len(p_c)
    epsilon = 1e-12

    # Normalize inputs
    p_c = p_c / (p_c.sum() + epsilon)
    p_q = p_q / (p_q.sum() + epsilon)
    p_a = p_a / (p_a.sum() + epsilon)

    # Initialize Q matrix
    if Q_init is None:
        np.random.seed(42)
        Q = np.random.dirichlet(np.ones(N), size=N)
    else:
        Q = Q_init.copy()

    # Initialize A matrix
    A = np.ones((N, N)) / N

    prev_objective = float('inf')
    convergence_history = []
    converged = False

    if debug:
        print("="*80)
        print("SEMANTIC FAITHFULNESS (WITH CONSTRAINT ENFORCEMENT)")
        print("="*80)
        print(f"N: {N}, tol_outer: {tol_outer}, tol_inner: {tol_inner}")
        print()
        print(f"{'Iter':<6} {'D':>18} {'|ΔD|':>15} {'Rel Δ':>15} {'u_iter':>8} {'ξν_iter':>8} {'Constr_err':>12}")
        print("-"*90)

    for outer_iter in range(max_outer_iter):
        # ======================================================================
        # A-STEP: Update A given Q
        # ======================================================================
        u = np.ones(N)
        u_converged = False

        for u_iter in range(100):
            u_old = u.copy()
            denominator = np.zeros(N)
            for i in range(N):
                Q_i_dot_u = np.dot(Q[i, :], u)
                if Q_i_dot_u > epsilon:
                    denominator += p_c[i] * Q[i, :] / Q_i_dot_u
            u = p_a / (denominator + epsilon)

            if np.linalg.norm(u - u_old) < tol_inner:
                u_converged = True
                break

        # Compute A from u
        for i in range(N):
            Q_i_dot_u = np.dot(Q[i, :], u)
            if Q_i_dot_u > epsilon:
                A[i, :] = Q[i, :] * u / Q_i_dot_u
            else:
                A[i, :] = Q[i, :] / N

        # ======================================================================
        # Q-STEP: Find Q that minimizes KL(A||Q) subject to constraints
        # ======================================================================
        # Constraints:
        #   1. sum_j Q[i,j] = 1 for all i (row-stochastic)
        #   2. sum_i p_c[i] * Q[i,j] = p_q[j] for all j (marginal constraint)

        # Use Lagrangian formulation: find (ξ, ν) by maximizing L_2
        # Then recover Q[i,j] = A[i,j] / (ν_i + ξ_j)

        # Initialize Lagrange multipliers
        xi = np.ones(N) * 0.1
        nu = np.ones(N) * 0.1

        xi_nu_iters = 0
        for xi_nu_iter in range(max_inner_iter):
            xi_old = xi.copy()
            nu_old = nu.copy()

            # Compute current Q from (ξ, ν)
            Q_temp = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    Q_temp[i, j] = A[i, j] / (nu[i] + xi[j] + epsilon)

            # Check constraint violations
            # Constraint 1: row sums should be 1
            row_sums = Q_temp.sum(axis=1)

            # Constraint 2: marginals should match p_q
            marginals = np.dot(p_c, Q_temp)  # shape (N,)

            # Adjust nu to satisfy row-stochastic constraint
            # From sum_j A[i,j]/(nu_i + xi_j) = 1, solve for nu_i
            for i in range(N):
                # We want: sum_j A[i,j]/(nu_i + xi_j) = 1
                # This is implicit equation for nu_i, solve numerically
                def row_sum_eq(nu_i_val):
                    return sum(A[i, j] / (nu_i_val + xi[j] + epsilon) for j in range(N)) - 1.0

                from scipy.optimize import fsolve
                nu_i_new = fsolve(row_sum_eq, nu[i], full_output=False)
                if nu_i_new[0] > epsilon:
                    nu[i] = nu_i_new[0]

            # Adjust xi to satisfy marginal constraint
            # From sum_i p_c[i] * A[i,j]/(nu_i + xi_j) = p_q[j], solve for xi_j
            for j in range(N):
                def marginal_eq(xi_j_val):
                    return sum(p_c[i] * A[i, j] / (nu[i] + xi_j_val + epsilon) for i in range(N)) - p_q[j]

                from scipy.optimize import fsolve
                xi_j_new = fsolve(marginal_eq, xi[j], full_output=False)
                if xi_j_new[0] > epsilon:
                    xi[j] = xi_j_new[0]

            # Check convergence
            xi_change = np.linalg.norm(xi - xi_old)
            nu_change = np.linalg.norm(nu - nu_old)
            total_change = xi_change + nu_change

            xi_nu_iters = xi_nu_iter + 1

            if total_change < tol_inner:
                break

        # Compute final Q from converged (ξ, ν)
        for i in range(N):
            for j in range(N):
                Q[i, j] = A[i, j] / (nu[i] + xi[j] + epsilon)

        # Verify constraints (for debugging)
        row_sums = Q.sum(axis=1)
        marginals = np.dot(p_c, Q)
        constraint_error_rows = np.max(np.abs(row_sums - 1.0))
        constraint_error_marginals = np.linalg.norm(marginals - p_q)
        constraint_error = max(constraint_error_rows, constraint_error_marginals)

        # ======================================================================
        # Compute objective
        # ======================================================================
        objective = 0.0
        for i in range(N):
            for j in range(N):
                if A[i, j] > epsilon and Q[i, j] > epsilon:
                    objective += p_c[i] * A[i, j] * np.log(A[i, j] / Q[i, j])

        convergence_history.append(objective)

        # Check convergence
        abs_change = abs(prev_objective - objective)
        if abs(prev_objective) > epsilon:
            relative_change = abs_change / abs(prev_objective)
        else:
            relative_change = abs_change

        if debug:
            print(f"{outer_iter:<6} {objective:>18.12f} {abs_change:>15.6e} {relative_change:>15.6e} {u_iter+1:>8} {xi_nu_iters:>8} {constraint_error:>12.6e}")

        if relative_change < tol_outer and outer_iter > 0:
            converged = True
            if debug:
                print(f"\n✓ Converged at iteration {outer_iter + 1}")
            break

        prev_objective = objective

    # Final results
    D_min = objective
    F_S = 1.0 / (1.0 + D_min)

    if debug:
        print()
        print("="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"D_min: {D_min:.15f}")
        print(f"F_S:   {F_S:.15f}")
        print(f"Iterations: {outer_iter + 1}/{max_outer_iter}")
        print(f"Converged: {converged}")
        print()
        print("Final constraint check:")
        print(f"  Max row sum error: {np.max(np.abs(Q.sum(axis=1) - 1.0)):.6e}")
        print(f"  Marginal error ||p_c^T Q - p_q||: {np.linalg.norm(np.dot(p_c, Q) - p_q):.6e}")
        print("="*80)

    return {
        'F_S': F_S,
        'D_min': D_min,
        'A_star': A,
        'Q_star': Q,
        'iterations': outer_iter + 1,
        'converged': converged,
        'convergence_history': convergence_history if debug else None
    }


if __name__ == "__main__":
    import json

    print("\nTesting SF with proper constraint enforcement...")

    with open('nvidia_rich_qca_results/cache/distributions.json', 'r') as f:
        cache = json.load(f)

    # Test with PROMPT_0
    triplet = cache['triplets'][0]
    print(f"\nTesting with: {triplet['prompt_id']}")

    p_q = np.array(triplet['p_q'])
    p_c = np.array(triplet['p_c'])
    p_a = np.array(triplet['p_a'])

    result = compute_semantic_faithfulness(p_c, p_q, p_a, debug=True)

    print(f"\nReturned result:")
    print(f"  F_S: {result['F_S']}")
    print(f"  D_min: {result['D_min']}")
    print(f"  Converged: {result['converged']}")
