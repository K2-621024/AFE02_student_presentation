import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(__file__))

from data_loader import load_data


def visualize_portfolio_performance(
        start_date,
        end_date,
        price_csv_path,
        weight_csv_paths,
        output_dir="image",
        output_filename="portfolio_performance.png"
    ):
    """
    Visualize portfolio cumulative performance using:
    - Optimization period: start_date ~ end_date
    - Multiple portfolio weights loaded from CSV (list of paths)
    """

    # === 1. Directory setup ===
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        _, _, _, df_full = load_data(
            price_csv_path,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        print(f"Error loading market data: {e}")
        return

    # === 2. Compute daily returns for full data ===
    full_returns = df_full.pct_change().dropna()

    plt.figure(figsize=(12, 6))

    # === 3. Process each weight CSV ===
    for weight_csv_path in weight_csv_paths:
        try:
            weights = pd.read_csv(weight_csv_path, index_col=0)
            if "Weight" not in weights.columns:
                raise ValueError("CSV must contain a 'Weight' column.")
            weights = weights["Weight"]
        except Exception as e:
            print(f"Error loading weight CSV '{weight_csv_path}': {e}")
            continue

        # Align weights
        aligned_weights = pd.Series(0.0, index=full_returns.columns)
        for asset_code, w in weights.items():
            if str(asset_code) in aligned_weights.index:
                aligned_weights[str(asset_code)] = w
            else:
                print(f"Warning: Asset {asset_code} not found in data. Weight ignored.")

        # Portfolio returns
        portfolio_daily_returns = full_returns.dot(aligned_weights)
        cumulative_wealth = (1 + portfolio_daily_returns).cumprod()

        # ラベルはファイル名から生成
        label = os.path.splitext(os.path.basename(weight_csv_path))[0]
        plt.plot(cumulative_wealth.index, cumulative_wealth, label=label)

    # Highlight in-sample period
    plt.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date),
                color='green', alpha=0.1, label='In-Sample (Optimization)')

    # Highlight out-of-sample
    last_date = cumulative_wealth.index[-1]
    if last_date > pd.to_datetime(end_date):
        plt.axvspan(pd.to_datetime(end_date), last_date,
                    color='orange', alpha=0.1, label='Out-of-Sample')

    plt.title("Portfolio Performance: Multiple Portfolios")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Wealth (Start = 1)")
    plt.grid(True)
    plt.legend()

    # Save figure
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    print(f"Saved figure to: {output_path}")
    plt.show()


# Test run example
if __name__ == "__main__":
    visualize_portfolio_performance(
        start_date="2022-01-01",
        end_date="2022-12-31",
        price_csv_path="data/topix-large500.csv",
        weight_csv_paths=["weights_LAM_K10.csv", "weights_LAM_K05.csv", "weights_Markowitz.csv"],
        output_dir="image",
        output_filename="portfolio_performance.png"
    )
