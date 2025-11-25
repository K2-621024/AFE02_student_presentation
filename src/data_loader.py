import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """
    Loads financial data from CSV, calculates expected returns and covariance matrix.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        tuple: (mu, sigma, asset_names)
            - mu (pd.Series): Expected return vector (mean daily return).
            - sigma (pd.DataFrame): Covariance matrix of daily returns.
            - asset_names (list): List of asset names.
    """
    print(f"Loading data from {filepath}...")
    
    # Load data with Shift-JIS encoding, skipping garbage lines
    # Based on inspection, the header seems to be around line 7 (0-indexed 6)
    # Load data with Shift-JIS encoding
    # Line 1 (0-indexed) has the asset codes: "コード,6098,4502,..."
    # Line 7 (0-indexed) seems to be the start of data: "2021/10/20,..."
    
    try:
        # Read just the header line
        header_df = pd.read_csv(filepath, encoding='shift_jis', header=None, skiprows=1, nrows=1)
        # The first column is 'コード', the rest are asset codes.
        asset_codes = header_df.iloc[0, 1:].tolist()
        
        # Read the data
        # Skip garbage lines. Based on inspection, data starts after some lines.
        # Let's try reading from line 7 (skiprows=6) but we need to handle the header manually.
        # Actually, if we use skiprows=6, the first line read will be the header.
        # But the line 6 (0-indexed) is "値,終値,終値..." which is useless.
        # Line 7 (0-indexed) is the first data row.
        # So we should read with header=None and skiprows=7.
        
        df = pd.read_csv(filepath, encoding='shift_jis', header=None, skiprows=6)
        
        # Assign columns
        # First column is Date, rest are asset codes
        # Check if dimensions match
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
    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop columns with too many NaNs
    df = df.dropna(axis=1, how='all')
    
    # Fill missing values
    df = df.ffill().bfill()

    # Calculate Daily Returns
    # r_t = (P_t - P_{t-1}) / P_{t-1}
    returns = df.pct_change().dropna()

    # Expected Return (Mean Daily Return)
    mu = returns.mean()

    # Covariance Matrix
    sigma = returns.cov()
    
    asset_names = df.columns.tolist()
    
    print(f"Data loaded successfully. {len(asset_names)} assets, {len(returns)} time periods.")
    
    return mu, sigma, asset_names, df

if __name__ == "__main__":
    try:
        mu, sigma, assets, _ = load_data('data/topix-large500.csv')
        print("Top 5 Expected Returns:")
        print(mu.head())
        print("\nCovariance Matrix Shape:", sigma.shape)
    except Exception as e:
        print(f"Error: {e}")
