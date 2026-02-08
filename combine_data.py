"""
Consolidation Script: Combine per-symbol CSVs into single CSV per folder.

This script converts the structure:
    all_data/data_stocks_1d/AAPL.csv, MSFT.csv, ...
    
Into:
    all_data_combined/data_stocks_1d.csv (with Symbol column)
"""

import os
import pandas as pd
from tqdm import tqdm

SOURCE_DIR = "all_data"
TARGET_DIR = "all_data_combined"


def combine_folder(folder_name: str) -> None:
    """Combine all CSVs in a folder into one CSV with Symbol column."""
    folder_path = os.path.join(SOURCE_DIR, folder_name)
    
    if not os.path.isdir(folder_path):
        return
    
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    if not csv_files:
        print(f"[WARN] No CSV files in {folder_name}")
        return
    
    dfs = []
    for csv_file in tqdm(csv_files, desc=folder_name, leave=False):
        symbol = csv_file.replace(".csv", "")
        file_path = os.path.join(folder_path, csv_file)
        
        try:
            df = pd.read_csv(file_path)
            df.insert(0, "Symbol", symbol)  # Add Symbol as first column
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] {csv_file}: {e}")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(TARGET_DIR, f"{folder_name}.csv")
        combined.to_csv(output_path, index=False)
        print(f"[OK] {folder_name}: {len(csv_files)} files -> {len(combined):,} rows")


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Get all subfolders in all_data
    folders = [
        f for f in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, f))
    ]
    
    print(f"Found {len(folders)} folders to combine\n")
    
    for folder in sorted(folders):
        combine_folder(folder)
    
    print(f"\n[DONE] Combined files saved to {TARGET_DIR}/")
    
    # Show result summary
    output_files = os.listdir(TARGET_DIR)
    print(f"Created {len(output_files)} combined CSV files")


if __name__ == "__main__":
    main()
