import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from solver_LAM import solve_lam_stqp
from data_loader import load_data


def compute_efficient_frontier_parallel(
        mu, sigma,
        n_points=100,
        k_max=10,
        min_weight=0.01, max_weight=1.0,
        penalty_m=1000.0,
        max_workers=None
    ):

    target_min = mu.min()
    target_max = mu.max() * 1.5
    target_returns = np.linspace(target_min, target_max, n_points)

    risks = np.zeros(n_points)
    returns = np.zeros(n_points)
    weights_list = [None] * n_points

    start_time = time()

    print(f"Starting parallel efficient frontier ({n_points} points)...")

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
                print(f"Warning: Optimization failed at target={tr:.4f}: {e}")
                risks[idx] = np.nan
                returns[idx] = np.nan
                weights_list[idx] = None

            completed += 1
            progress = completed / n_points * 100
            print(f"\rProgress: {completed}/{n_points} ({progress:.1f}%)", end="")

    elapsed = time() - start_time
    print(f"\nFinished efficient frontier in {elapsed:.2f} seconds (parallel).")

    return {
        "risks": risks,
        "returns": returns,
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
