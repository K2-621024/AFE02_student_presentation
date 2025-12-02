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
        market_csv_path="data/marketindex.csv",
        output_dir="image",
        output_filename="portfolio_performance.png"
    ):
    """
    Visualize portfolio cumulative performance and overlay TOPIX / Nikkei225.

    - start_date ~ end_date : optimization period
    - price_csv_path : asset price CSV
    - weight_csv_paths : list of weight files
    - market_csv_path : market index CSV containing 日付, TOPIX, 日経225, ...
    """

    # === 1. Directory setup ===
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # === 2. Load asset data ===
    try:
        _, _, _, df_full = load_data(
            price_csv_path,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        print(f"Error loading asset price data: {e}")
        return

    # === 3. Daily returns ===
    full_returns = df_full.pct_change().dropna()

    plt.figure(figsize=(12, 6))

    # --------------------------------------------------------
    # === 4. Load and process market index (TOPIX & Nikkei225) ===
    # --------------------------------------------------------
    try:
        df_market = pd.read_csv(market_csv_path, encoding="utf-8-sig")
        df_market.columns = [c.strip() for c in df_market.columns]  # remove BOM or spaces
        df_market["日付"] = pd.to_datetime(df_market["日付"], format="%Y/%m/%d")
        df_market = df_market.set_index("日付")
        # Extract needed columns
        df_market = df_market[["TOPIX", "日経225"]].dropna()
        # Convert to returns
        market_returns = df_market.pct_change().dropna()
        # Cumulative returns
        market_cum = (1 + market_returns).cumprod()
        # Align with portfolios by date
        market_cum = market_cum.loc[full_returns.index, :]
        # === plot TOPIX ===
        plt.plot(market_cum.index, market_cum["TOPIX"],
                 label="TOPIX", color="black", linestyle="-", linewidth=1.8)
        # === plot Nikkei225 ===
        plt.plot(market_cum.index, market_cum["日経225"],
                 label="Nikkei225", color="gray", linestyle="-", linewidth=1.8)

        print("Loaded market index and plotted TOPIX / Nikkei225.")

    except Exception as e:
        print(f"Error loading market index CSV: {e}")

    # --------------------------------------------------------
    # === 5. Process each portfolio weight CSV ===
    # --------------------------------------------------------
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
                print(f"Warning: Asset {asset_code} not found. Weight ignored.")

        # Portfolio returns
        portfolio_daily_returns = full_returns.dot(aligned_weights)
        cumulative_wealth = (1 + portfolio_daily_returns).cumprod()

        label = os.path.splitext(os.path.basename(weight_csv_path))[0]
        plt.plot(cumulative_wealth.index, cumulative_wealth, label=label)

    # --------------------------------------------------------
    # === 6. Highlight optimization periods ===
    # --------------------------------------------------------
    plt.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date),
                color='green', alpha=0.1, label='In-Sample')

    last_date = full_returns.index[-1]
    if last_date > pd.to_datetime(end_date):
        plt.axvspan(pd.to_datetime(end_date), last_date,
                    color='orange', alpha=0.1, label='Out-of-Sample')

    # --------------------------------------------------------
    # === 7. Finishing ===
    # --------------------------------------------------------
    plt.title("Portfolio Performance with Market Index")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Wealth (Start = 1)")
    plt.grid(True)
    plt.legend()

    # Save
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    print(f"Saved figure to: {output_path}")

    plt.show()

