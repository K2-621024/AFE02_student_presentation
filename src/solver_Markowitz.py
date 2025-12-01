import numpy as np
import pandas as pd
from scipy.optimize import minimize

def solve_markowitz(mu, sigma, target_return, 
                           min_weight=0.0, max_weight=1.0,
                           penalty_m=1000.0):
    """
    Stable Markowitz optimization using penalty for target return.
    Returns same format as solve_lam_stqp.
    """

    asset_names = mu.index.tolist()
    mu_vec = mu.values
    sigma_mat = sigma.values
    n = len(mu)

    x0 = np.ones(n) / n

    def objective(x):
        var = x @ sigma_mat @ x
        penalty = penalty_m * (x @ mu_vec - target_return)**2
        return var + penalty

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    bounds = [(min_weight, max_weight) for _ in range(n)]

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-9
    )

    if not result.success:
        print("Warning: Optimization did not converge. Message:", result.message)

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
