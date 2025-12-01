import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from solver import solve_lam_stqp
from data_loader import load_data

def compute_efficient_frontier(mu, sigma, 
                               n_points=100, 
                               k_max=10, 
                               min_weight=0.01, max_weight=1.0, 
                               penalty_m=1000.0):
    """
    Computes efficient frontier points using solve_lam_stqp().
    """

    target_min = mu.min()
    target_max = mu.max() * 1.5
    target_returns = np.linspace(target_min, target_max, n_points)

    risks = []
    returns = []
    weights_list = []

    start_time = time()

    for tr in target_returns:
        try:
            res = solve_lam_stqp(mu, sigma, tr, 
                                 k_max=k_max,
                                 min_weight=min_weight, 
                                 max_weight=max_weight, 
                                 penalty_m=penalty_m)
            
            risks.append(res["risk"])
            returns.append(res["return"])
            weights_list.append(res["weights"])

        except Exception as e:
            print(f"Warning: Optimization failed at target={tr:.4f}: {e}")
            risks.append(np.nan)
            returns.append(np.nan)
            weights_list.append(None)

    elapsed = time() - start_time
    print(f"\nFinished efficient frontier in {elapsed:.2f} seconds.")

    return {
        "risks": np.array(risks),
        "returns": np.array(returns),
        "weights": weights_list,
        "targets": target_returns
    }

def plot_efficient_frontier(frontier):
    """
    Plots the efficient frontier.
    """
    risks = frontier["risks"]
    returns = frontier["returns"]

    plt.figure(figsize=(8,6))
    plt.plot(risks, returns, marker='o', markersize=3, linestyle='-', alpha=0.8)
    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Expected Return (Daily)")
    plt.title("Efficient Frontier (LAM-STQP)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    mu, sigma, asset_names, df = load_data("data/topix-large500.csv")

    frontier = compute_efficient_frontier(
        mu, sigma,
        target_min=0.00,
        target_max=0.05,
        n_points=100,
        k_max=10,
        min_weight=0.01,
        max_weight=1.0,
        penalty_m=1000.0
    )

    plot_efficient_frontier(frontier)
