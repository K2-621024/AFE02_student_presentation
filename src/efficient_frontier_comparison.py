import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict

from solver_LAM import solve_lam_stqp
from solver_Markowitz import solve_markowitz
from data_loader import load_data


def compute_efficient_frontier_lam(mu, sigma,
                                            n_points=50,
                                            k_max=10,
                                            min_weight=0.01, max_weight=1.0,
                                            penalty_m=1000.0,
                                            max_workers=None) -> Dict:
    """
    Compute efficient frontier points (LAM-STQP) in parallel.
    Returns dict with risks, returns, weights, targets.
    """
    target_min = mu.min()
    target_max = mu.max() * 1.5
    target_returns = np.linspace(target_min, target_max, n_points)

    risks = np.full(n_points, np.nan)
    returns = np.full(n_points, np.nan)
    weights_list = [None] * n_points

    start_time = time()
    print(f"Starting LAM-STQP parallel frontier ({n_points} pts)...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                solve_lam_stqp,
                mu, sigma, tr,
                k_max, min_weight, max_weight, penalty_m
            ): idx
            for idx, tr in enumerate(target_returns)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            tr = target_returns[idx]
            try:
                res = future.result()
                risks[idx] = res["risk"]
                returns[idx] = res["return"]
                weights_list[idx] = res["weights"]
            except Exception as e:
                print(f"\n[LAM] Warning: target={tr:.6f} failed: {e}")
                risks[idx] = np.nan
                returns[idx] = np.nan
                weights_list[idx] = None

            completed += 1
            print(f"\r[LAM] Progress: {completed}/{n_points} ({completed/n_points*100:.1f}%)", end="")

    elapsed = time() - start_time
    print(f"\nFinished LAM-STQP frontier in {elapsed:.2f}s.")
    return {"risks": risks, "returns": returns, "weights": weights_list, "targets": target_returns}


def compute_efficient_frontier_markowitz(mu, sigma,
                                                  n_points=50,
                                                  min_weight=0.01, max_weight=1.0,
                                                  penalty_m=1000.0,
                                                  max_workers=None) -> Dict:
    """
    Compute efficient frontier points (Markowitz stable penalty version) in parallel.
    Note: solve_markowitz signature expected:
          solve_markowitz(mu, sigma, target_return, min_weight, max_weight, penalty_m)
    """
    target_min = mu.min()
    target_max = mu.max() * 1.5
    target_returns = np.linspace(target_min, target_max, n_points)

    risks = np.full(n_points, np.nan)
    returns = np.full(n_points, np.nan)
    weights_list = [None] * n_points

    start_time = time()
    print(f"Starting Markowitz parallel frontier ({n_points} pts)...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                solve_markowitz,
                mu, sigma, tr,
                min_weight, max_weight, penalty_m
            ): idx
            for idx, tr in enumerate(target_returns)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            tr = target_returns[idx]
            try:
                res = future.result()
                risks[idx] = res["risk"]
                returns[idx] = res["return"]
                weights_list[idx] = res["weights"]
            except Exception as e:
                print(f"\n[MK] Warning: target={tr:.6f} failed: {e}")
                risks[idx] = np.nan
                returns[idx] = np.nan
                weights_list[idx] = None

            completed += 1
            print(f"\r[MK] Progress: {completed}/{n_points} ({completed/n_points*100:.1f}%)", end="")

    elapsed = time() - start_time
    print(f"\nFinished Markowitz frontier in {elapsed:.2f}s.")
    return {"risks": risks, "returns": returns, "weights": weights_list, "targets": target_returns}


def plot_efficient_frontier_comparison(frontier_lam: Dict, frontier_mkw: Dict,
                                       title="Efficient Frontier: LAM-STQP vs Markowitz",
                                       output_filename="efficient_frontier_comparison.png"):
    """
    Plot two frontiers on the same figure.
    frontier_* dicts must contain 'risks' and 'returns' arrays.
    """
    r_lam = frontier_lam["risks"]
    ret_lam = frontier_lam["returns"]
    r_mkw = frontier_mkw["risks"]
    ret_mkw = frontier_mkw["returns"]

    plt.figure(figsize=(9, 6))
    # Plot LAM (blue)
    plt.plot(r_lam, ret_lam, label="LAM-STQP", color="blue", marker='o', markersize=4, linewidth=1)
    # Plot Markowitz (red)
    plt.plot(r_mkw, ret_mkw, label="Markowitz", color="red", marker='o', markersize=4, linewidth=1)

    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Expected Return (Daily)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    output_dir = "output_final"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, bbox_inches='tight')
    print(f"Efficient frontier comparison saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    # Example usage
    # Load mu, sigma using your data loader (mu, sigma are Series/DataFrame of daily returns)
    mu, sigma, assets, df = load_data("data/topix-large500.csv", start_date=None, end_date=None)

    # Parameters
    N_POINTS = 50
    K_MAX = 10
    MIN_W = 0.01
    MAX_W = 1.0
    PENALTY = 1000.0
    MAX_WORKERS = None  # None -> uses default, you can set to os.cpu_count()

    # Compute both frontiers in parallel separately (two executor pools)
    frontier_lam = compute_efficient_frontier_lam_parallel(
        mu, sigma,
        n_points=N_POINTS,
        k_max=K_MAX,
        min_weight=MIN_W, max_weight=MAX_W,
        penalty_m=PENALTY,
        max_workers=MAX_WORKERS
    )

    frontier_mkw = compute_efficient_frontier_markowitz_parallel(
        mu, sigma,
        n_points=N_POINTS,
        min_weight=MIN_W, max_weight=MAX_W,
        penalty_m=PENALTY,
        max_workers=MAX_WORKERS
    )

    # Plot comparison
    plot_efficient_frontier_comparison(frontier_lam, frontier_mkw)
