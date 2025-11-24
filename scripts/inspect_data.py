import pandas as pd
import os

file_path = 'data/topix-large500.csv'

try:
    # Try reading with default settings
    df = pd.read_csv(file_path, encoding='shift_jis')
    print("--- Attempt 1: Default read_csv ---")
    print(df.head())
    print(df.columns)
except Exception as e:
    print(f"Attempt 1 failed: {e}")

try:
    # Try skipping first few lines if they are garbage
    # Based on previous `cat`, it seems line 8 is the header or data start. 
    # Let's try reading from line 7 (0-indexed 6) or 8.
    # The `cat` output showed "2021/11/1" on a line.
    
    # Let's try to find the header row dynamically or just guess.
    print("\n--- Attempt 2: Skip rows ---")
    df = pd.read_csv(file_path, skiprows=6, encoding='shift_jis') # Skip first 6 lines, so line 7 is header
    print(df.head())
    print(df.columns)
    print(f"Shape: {df.shape}")
    
    # Check if '日付' is in columns
    if '日付' in df.columns:
        print("Found '日付' column.")
    else:
        print("'日付' column not found in Attempt 2.")

except Exception as e:
    print(f"Attempt 2 failed: {e}")
