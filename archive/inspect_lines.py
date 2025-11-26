import pandas as pd

file_path = 'data/topix-large500.csv'
with open(file_path, 'r', encoding='shift_jis') as f:
    for i in range(10):
        print(f"Line {i}: {f.readline().strip()[:200]}") # Print first 200 chars of each line
