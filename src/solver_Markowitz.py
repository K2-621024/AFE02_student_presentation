import numpy as np
import pandas as pd
from scipy.optimize import minimize

def solve_markowitz(mu, sigma, target_return, 
                    min_weight=0.0, max_weight=1.0):
    """
    Standard long-only Markowitz model:
        minimize x' Σ x
        s.t. x' μ = target_return
             1' x = 1
             min_weight ≤ x_i ≤ max_weight
    Returns in same format as solve_lam_stqp.
    """

    asset_names = mu.index.tolist()
    mu_vec = mu.values
    sigma_mat = sigma.values

    n = len(mu)

    x0 = np.ones(n) / n

    def variance(x):
        return x @ sigma_mat @ x

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.dot(x, mu_vec) - target_return}
    ]

    bounds = [(min_weight, max_weight) for _ in range(n)]

    result = minimize(
        variance, 
        x0, 
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-9
    )

    if not result.success:
        raise RuntimeError("Markowitz optimization failed: " + result.message)

    weights = result.x

    port_var = weights @ sigma_mat @ weights
    port_risk = np.sqrt(port_var)
    port_ret = weights @ mu_vec
    weights_series = pd.Series(weights, index=asset_names)
    weights_nonzero = weights_series[weights_series > 1e-6]

    return {
        "weights": weights_nonzero,
        "assets": list(weights_nonzero.index),
        "return": port_ret,
        "risk": port_risk,
        "objective": port_var
    }
