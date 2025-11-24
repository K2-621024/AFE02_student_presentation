import numpy as np
import pandas as pd
from scipy.optimize import minimize

def solve_qp_subset(mu_sub, sigma_sub, target_return, penalty_m, min_weight, max_weight):
    """
    Solves the QP for a fixed subset of assets.
    Minimize x.T * Sigma * x + M * (mu.T * x - rho)^2
    Subject to sum(x) = 1, min_weight <= x <= max_weight
    """
    n = len(mu_sub)
    if n == 0:
        return None, np.inf

    # Initial guess: equal weights
    x0 = np.ones(n) / n

    # Objective function
    # We want to minimize Variance + Penalty
    # Var = x.T S x
    # Pen = M * (R - rho)^2
    def objective(x):
        port_return = np.dot(x, mu_sub)
        port_var = np.dot(x, np.dot(sigma_sub, x))
        penalty = penalty_m * (port_return - target_return)**2
        return port_var + penalty

    # Constraints
    # Sum of weights = 1
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]

    # Bounds
    # min_weight <= x_i <= max_weight
    # Note: If min_weight * n > 1, this is infeasible.
    # We assume the caller handles feasibility checks or we just try.
    bounds = [(min_weight, max_weight) for _ in range(n)]

    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-6)
        return result.x, result.fun
    except Exception:
        return None, np.inf

def solve_lam_stqp(mu, sigma, target_return, k_max, min_weight, max_weight, penalty_m):
    """
    Solves the portfolio optimization problem using a Forward Selection heuristic.
    
    Args:
        mu (pd.Series): Expected returns.
        sigma (pd.DataFrame): Covariance matrix.
        target_return (float): Target return (rho).
        k_max (int): Maximum cardinality (K).
        min_weight (float): Minimum weight for selected assets.
        max_weight (float): Maximum weight for selected assets.
        penalty_m (float): Penalty coefficient (M).

    Returns:
        dict: Result dictionary containing:
            - 'weights': pd.Series of optimal weights.
            - 'assets': List of selected assets.
            - 'return': Portfolio return.
            - 'risk': Portfolio risk (std dev).
            - 'objective': Objective function value.
    """
    assets = mu.index.tolist()
    n_total = len(assets)
    
    current_assets = []
    best_overall_fun = np.inf
    best_overall_weights = None
    best_overall_assets = []

    print(f"Starting optimization with K={k_max}, Target={target_return:.4f}, M={penalty_m}")

    # Forward Selection
    # We will build the set up to k_max assets.
    # In each step, we add the asset that minimizes the objective when combined with current_set.
    
    # Optimization: For the first asset, we can just calculate analytically or search.
    # f_i = sigma_ii + M(mu_i - rho)^2 (since x=1)
    # But we also have bounds. If min_weight > 1 or max_weight < 1, single asset might be infeasible.
    # Usually max_weight=1.0, min_weight=0.01 etc.
    
    for k in range(1, k_max + 1):
        print(f"  Step k={k}/{k_max}...")
        best_fun_k = np.inf
        best_asset_to_add = None
        best_weights_k = None
        
        # Candidates to add: all assets not in current_assets
        # To speed up, maybe we don't check ALL 500 assets every time if k is large.
        # But for K=10, N=500, it's manageable.
        
        candidates = [a for a in assets if a not in current_assets]
        
        # Heuristic speedup: Only consider top N candidates by some metric?
        # For now, let's try exhaustive search over candidates.
        
        for asset in candidates:
            trial_assets = current_assets + [asset]
            
            # Extract sub-matrices
            mu_sub = mu[trial_assets].values
            sigma_sub = sigma.loc[trial_assets, trial_assets].values
            
            # Solve QP
            w, fun = solve_qp_subset(mu_sub, sigma_sub, target_return, penalty_m, min_weight, max_weight)
            
            if fun < best_fun_k:
                best_fun_k = fun
                best_asset_to_add = asset
                best_weights_k = w
        
        if best_asset_to_add is None:
            print("    Could not find a feasible addition.")
            break
            
        current_assets.append(best_asset_to_add)
        
        # Update overall best if this k is better (or we just take the result at k_max? 
        # Usually we want to satisfy cardinality <= K, so any k <= K is allowed.
        # But usually larger k allows lower objective (better diversification).
        # However, with penalty M, maybe not.
        # We'll track the best across all k steps.
        
        if best_fun_k < best_overall_fun:
            best_overall_fun = best_fun_k
            best_overall_assets = list(current_assets)
            best_overall_weights = best_weights_k
            print(f"    New best found at k={k}: Obj={best_fun_k:.6f}")
        else:
            print(f"    k={k}: Obj={best_fun_k:.6f} (Not better)")

    # Construct result
    if best_overall_weights is not None:
        final_weights = pd.Series(0.0, index=assets)
        final_weights[best_overall_assets] = best_overall_weights
        
        port_ret = np.dot(final_weights, mu)
        port_var = np.dot(final_weights, np.dot(sigma, final_weights))
        port_risk = np.sqrt(port_var)
        
        return {
            'weights': final_weights[final_weights > 1e-6], # Filter zero weights
            'assets': best_overall_assets,
            'return': port_ret,
            'risk': port_risk,
            'objective': best_overall_fun
        }
    else:
        raise RuntimeError("Optimization failed to find any feasible solution.")
