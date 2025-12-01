import pandas as pd
import numpy as np
from data_loader import load_data

def calculate_portfolio_sharpe(
        price_csv_path,
        weight_csv_path,
        start_date,
        end_date,
        risk_free_rate=0
    ):
    """
    price_csv_path: 価格データ（行=日付, 列=銘柄コード）
    weight_csv_path: Weight列を持つCSV（index=銘柄コード）
    """

    # === 1. Load price data ===
    try:
        _, _, _, df_full = load_data(
            price_csv_path,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        print(f"Error loading market data: {e}")
        return

    price_df = df_full.copy()
    price_df.columns = price_df.columns.astype(str)

    # === 2. Load portfolio weights ===
    try:
        weights = pd.read_csv(weight_csv_path, index_col=0)
        if "Weight" not in weights.columns:
            raise ValueError("CSV must contain a 'Weight' column.")
        weights = weights["Weight"]
    except Exception as e:
        print(f"Error loading weight CSV: {e}")
        return

    weights.index = weights.index.astype(str)

    # === 3. Align weights with available data ===
    valid_assets = [a for a in weights.index if a in price_df.columns]

    if len(valid_assets) == 0:
        raise ValueError("No valid assets found: weight CSV assets do not match price CSV.")

    aligned_weights = weights.loc[valid_assets]
    aligned_weights = aligned_weights / aligned_weights.sum()

    # === 4. Daily returns ===
    returns = price_df[valid_assets].pct_change().dropna()

    # === 5. Portfolio daily return ===
    portfolio_daily = returns.dot(aligned_weights)

    # === 6. Annualized return & risk ===
    annual_return = portfolio_daily.mean() * 252
    annual_risk = portfolio_daily.std() * np.sqrt(252)

    # === 7. Sharpe ratio ===
    sharpe = (annual_return - risk_free_rate) / annual_risk if annual_risk > 0 else np.nan

    return {
        "annual_return": annual_return,
        "annual_risk": annual_risk,
        "sharpe_ratio": sharpe
    }