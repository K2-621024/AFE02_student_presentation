import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data
from solver import solve_lam_stqp

def main():
    # Parameters
    DATA_FILE = 'data/topix-large500.csv'
    TARGET_RETURN = 0.0005  # 0.05% daily return (approx 12% annual)
    CARDINALITY_K = 10      # Max 10 assets
    MIN_WEIGHT = 0.01       # Min 1% per asset
    MAX_WEIGHT = 1.0        # Max 100% per asset
    PENALTY_M = 1000.0      # Penalty for return deviation

    print("=== Portfolio Optimization (LAM/StQP) ===")
    print(f"Target Return: {TARGET_RETURN:.6f}")
    print(f"Cardinality K: {CARDINALITY_K}")
    print(f"Penalty M: {PENALTY_M}")
    
    # 1. Load Data
    try:
        mu, sigma, assets = load_data(DATA_FILE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Solve
    try:
        result = solve_lam_stqp(
            mu, sigma, 
            target_return=TARGET_RETURN, 
            k_max=CARDINALITY_K, 
            min_weight=MIN_WEIGHT, 
            max_weight=MAX_WEIGHT, 
            penalty_m=PENALTY_M
        )
        
        # 3. Output Results
        print("\n=== Optimization Results ===")
        print(f"Objective Value: {result['objective']:.6f}")
        print(f"Portfolio Return: {result['return']:.6f}")
        print(f"Portfolio Risk (Std): {result['risk']:.6f}")
        print(f"Number of Assets: {len(result['assets'])}")
        print("\nSelected Assets and Weights:")
        print(result['weights'].sort_values(ascending=False))
        
        # Save to CSV
        result['weights'].to_csv('optimization_result.csv', header=['Weight'])
        print("\nResults saved to optimization_result.csv")

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
