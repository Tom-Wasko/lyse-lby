"""
Data Loading Utilities for Combined CSV Format

This module provides functions to load data from the combined CSV format
where each market/interval has a single CSV file with a Symbol column.

Usage in notebook:
    from data_loader import load_symbol_data, get_combined_data, DATA_DIR

The combined data format has columns:
    Symbol, Date, Adj Close, Close, High, Low, Open, Volume
"""

import os
import pandas as pd

# Directory containing combined CSV files
DATA_DIR = "all_data_combined"

# Cache for loaded combined dataframes to avoid repeated disk reads
_combined_data_cache = {}


def get_data_path(market: str, interval: str) -> str:
    """Get the path to the combined CSV for a market/interval."""
    return os.path.join(DATA_DIR, f"data_{market}_{interval}.csv")


def get_combined_data(market: str, interval: str, force_reload: bool = False) -> pd.DataFrame:
    """
    Load the combined CSV for a market/interval (with caching).
    
    Args:
        market: 'stocks', 'crypto', or 'commodities'
        interval: '4h', '1d', or '1week'
        force_reload: If True, reload from disk even if cached
    
    Returns:
        DataFrame with all symbols for that market/interval
    """
    cache_key = f"{market}_{interval}"
    
    if not force_reload and cache_key in _combined_data_cache:
        return _combined_data_cache[cache_key]
    
    path = get_data_path(market, interval)
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Combined data file not found: {path}")
    
    # Read with Symbol as first column, skip the second header row (if present)
    df = pd.read_csv(path)
    
    # Handle the extra header row that yfinance sometimes adds
    if df.iloc[0]["Date"] == df.columns[1]:  # Second row is just column names
        df = pd.read_csv(path, skiprows=[1])
    
    _combined_data_cache[cache_key] = df
    return df


def load_symbol_data(symbol: str, market: str, interval: str, window: int = None) -> pd.DataFrame:
    """
    Load data for a single symbol from the combined CSV.
    
    This replaces the old load_symbol_data that read individual files.
    
    Args:
        symbol: The ticker symbol (e.g., 'AAPL', 'BTC-USD')
        market: 'stocks', 'crypto', or 'commodities'
        interval: '4h', '1d', or '1week'
        window: Optional number of most recent rows to return
    
    Returns:
        DataFrame with OHLCV data for the symbol, or None if not found
    """
    try:
        combined = get_combined_data(market, interval)
    except FileNotFoundError:
        print(f"[ERROR] Data file not found for {market}/{interval}")
        return None
    
    # Filter for the requested symbol
    df = combined[combined["Symbol"] == symbol].copy()
    
    if df.empty:
        print(f"[WARN] Symbol not found: {symbol}")
        return None
    
    # Drop the Symbol column since we're returning single-symbol data
    df = df.drop(columns=["Symbol"])
    
    # Standard preprocessing
    REQUIRED = ["Date", "Open", "High", "Low", "Close", "Volume"]
    
    # Handle Datetime vs Date column name
    if "Datetime" in df.columns and "Date" not in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    
    if not all(col in df.columns for col in REQUIRED):
        print(f"[ERROR] Missing required columns for {symbol}")
        return None
    
    df = df[REQUIRED].copy()
    
    # Datetime handling
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    # Numeric conversion
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    
    # Apply window if specified
    if window and len(df) > window:
        df = df.tail(window)
    
    return df


def update_symbol_data(symbol: str, new_data: pd.DataFrame, market: str, interval: str) -> None:
    """
    Update/append data for a symbol in the combined CSV.
    
    Args:
        symbol: The ticker symbol
        new_data: DataFrame with new OHLCV data (must have Date column)
        market: 'stocks', 'crypto', or 'commodities'
        interval: '4h', '1d', or '1week'
    """
    path = get_data_path(market, interval)
    
    # Load existing combined data
    if os.path.isfile(path):
        combined = pd.read_csv(path)
    else:
        combined = pd.DataFrame()
    
    # Remove old data for this symbol
    if not combined.empty and "Symbol" in combined.columns:
        combined = combined[combined["Symbol"] != symbol]
    
    # Add Symbol column to new data
    new_data = new_data.copy()
    new_data.insert(0, "Symbol", symbol)
    
    # Combine and save
    updated = pd.concat([combined, new_data], ignore_index=True)
    updated.to_csv(path, index=False)
    
    # Invalidate cache
    cache_key = f"{market}_{interval}"
    if cache_key in _combined_data_cache:
        del _combined_data_cache[cache_key]


def clear_cache() -> None:
    """Clear the combined data cache."""
    _combined_data_cache.clear()


if __name__ == "__main__":
    # Quick test
    print("Testing data loader...")
    
    try:
        df = load_symbol_data("AAPL", "stocks", "1d", window=10)
        if df is not None:
            print(f"Loaded AAPL: {len(df)} rows")
            print(df.head())
        else:
            print("AAPL not found")
    except Exception as e:
        print(f"Error: {e}")
