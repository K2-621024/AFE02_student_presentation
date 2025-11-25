import pandas as pd
import numpy as np
import os

def load_data(filepath, start_date=None, end_date=None):
    """
    Loads financial data from CSV, calculates expected returns and covariance matrix.

    Args:
        filepath (str): Path to the CSV file.
        start_date (str or pd.Timestamp, optional): Start date for optimization period.
        end_date (str or pd.Timestamp, optional): End date for optimization period.

    Returns:
        tuple: (mu, sigma, asset_names, df)
            - mu (pd.Series): Expected return vector (mean daily return) over the specified period.
            - sigma (pd.DataFrame): Covariance matrix of daily returns over the specified period.
            - asset_names (list): List of asset names.
            - df (pd.DataFrame): Full dataframe of prices (cleaned).
    """
    print(f"Loading data from {filepath}...")
    
    try:
        # Read just the header line
        header_df = pd.read_csv(filepath, encoding='shift_jis', header=None, skiprows=1, nrows=1)
        # The first column is 'コード', the rest are asset codes.
        asset_codes = header_df.iloc[0, 1:].tolist()
        
        # Read the data       
        df = pd.read_csv(filepath, encoding='shift_jis', header=None, skiprows=6)
        
        # Assign columns
        if df.shape[1] == len(asset_codes) + 1:
            df.columns = ['Date'] + asset_codes
        else:
            # Fallback if dimensions don't match exactly (e.g. trailing comma)
            print(f"Warning: Shape mismatch. Data cols: {df.shape[1]}, Codes: {len(asset_codes)}")
            # Try to match as many as possible
            cols = ['Date'] + asset_codes
            if len(cols) > df.shape[1]:
                df.columns = cols[:df.shape[1]]
            else:
                # If data has more columns, just name them generic
                extra_cols = [f"Unknown_{i}" for i in range(df.shape[1] - len(cols))]
                df.columns = cols + extra_cols

    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')
    
    # Sort by date just in case
    df = df.sort_index()

    # Drop columns that are not assets (if any)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop columns with too many NaNs
    df = df.dropna(axis=1, how='all')
    
    # Fill missing values
    df = df.ffill().bfill()

    # Calculate Daily Returns
    # r_t = (P_t - P_{t-1}) / P_{t-1}
    returns = df.pct_change().dropna()

    # Filter returns for optimization if dates are provided
    opt_returns = returns.copy()
    if start_date:
        opt_returns = opt_returns[opt_returns.index >= pd.to_datetime(start_date)]
    if end_date:
        opt_returns = opt_returns[opt_returns.index <= pd.to_datetime(end_date)]

    if opt_returns.empty:
        raise ValueError("No data found for the specified date range.")

    # Expected Return (Mean Daily Return)
    mu = opt_returns.mean()

    # Covariance Matrix
    sigma = opt_returns.cov()
    
    asset_names = df.columns.tolist()
    
    print(f"Data loaded successfully. {len(asset_names)} assets.")
    print(f"Optimization period: {opt_returns.index.min().date()} to {opt_returns.index.max().date()} ({len(opt_returns)} periods)")
    
    return mu, sigma, asset_names, df

if __name__ == "__main__":
    try:
        mu, sigma, assets, _ = load_data('data/topix-large500.csv')
        print("Top 5 Expected Returns:")
        print(mu.head())
        print("\nCovariance Matrix Shape:", sigma.shape)
    except Exception as e:
        print(f"Error: {e}")
