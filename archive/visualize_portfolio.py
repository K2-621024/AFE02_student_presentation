import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(__file__))

from data_loader import load_data
from solver import solve_lam_stqp

def visualize_portfolio_performance():
    # Parameters
    DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'topix-large500.csv')
    
    # Define Optimization Period (In-Sample)
    # Example: Use 1 year for optimization
    START_DATE = '2022-01-01'
    END_DATE = '2022-12-31'
    
    TARGET_RETURN = 0.0005  # 0.05% daily return
    CARDINALITY_K = 10
    MIN_WEIGHT = 0.01
    MAX_WEIGHT = 1.0
    PENALTY_M = 1000.0

    print("=== Portfolio Performance Visualization ===")
    print(f"Optimization Period: {START_DATE} to {END_DATE}")
    
    # 1. Load Data with Date Filtering for Optimization
    try:
        # mu and sigma are calculated based on the filtered period
        # df_full contains the entire dataset
        mu, sigma, assets, df_full = load_data(DATA_FILE, start_date=START_DATE, end_date=END_DATE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Solve Optimization Problem
    try:
        result = solve_lam_stqp(
            mu, sigma, 
            target_return=TARGET_RETURN, 
            k_max=CARDINALITY_K, 
            min_weight=MIN_WEIGHT, 
            max_weight=MAX_WEIGHT, 
            penalty_m=PENALTY_M
        )
        
        print("\nOptimization Successful!")
        print(f"Selected Assets: {result['assets']}")
        
        weights = result['weights']
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        return

    # 3. Calculate Portfolio Performance over the Entire Period
    # Calculate daily returns for the full dataset
    full_returns = df_full.pct_change().dropna()
    
    # Align weights with full_returns columns
    # Create a weight vector aligned with full_returns columns (assets)
    # Weights are 0 for assets not selected
    aligned_weights = pd.Series(0.0, index=full_returns.columns)
    aligned_weights[weights.index] = weights
    
    # Calculate portfolio daily returns: R_p = sum(w_i * r_i)
    portfolio_daily_returns = full_returns.dot(aligned_weights)
    
    # Calculate Cumulative Return
    # Cumulative Return = (1 + r_1) * (1 + r_2) * ... - 1
    # Or simpler: Cumulative Wealth = cumprod(1 + r)
    cumulative_wealth = (1 + portfolio_daily_returns).cumprod()
    
    # 4. Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot Cumulative Wealth
    plt.plot(cumulative_wealth.index, cumulative_wealth, label='Portfolio Cumulative Return', color='blue')
    
    # Highlight Optimization Period
    plt.axvspan(pd.to_datetime(START_DATE), pd.to_datetime(END_DATE), color='green', alpha=0.1, label='Optimization Period (In-Sample)')
    
    # Highlight Out-of-Sample Period (after END_DATE)
    last_date = cumulative_wealth.index[-1]
    if last_date > pd.to_datetime(END_DATE):
        plt.axvspan(pd.to_datetime(END_DATE), last_date, color='orange', alpha=0.1, label='Evaluation Period (Out-of-Sample)')

    plt.title('Portfolio Performance: In-Sample vs Out-of-Sample')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Wealth (Start=1.0)')
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'portfolio_performance.png')
    plt.savefig(output_path)
    print(f"\nVisualization saved to {output_path}")
    # plt.show() # Commented out for headless environment

if __name__ == "__main__":
    visualize_portfolio_performance()
