import pandas as pd
from tqdm import tqdm
import os

from core.ladowanie_danych import add_indicators
from core.dodanie_formacji import add_candle_patterns

def mackowe_sygnaly(
    dfs: dict,
    settings: dict,
    require_vol_confirmation: bool = True,
    require_cmo_confirmation: bool = True,
    interval: str = '1w',
    entry_offset: int = 0,
    pattern_cols: list = None,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Generates trading signals based on candlestick patterns and indicator confirmations.
    
    Args:
        dfs: Dictionary of dataframes, e.g., output from create_stock_dfs
        settings: Settings dictionary for indicators
        require_vol_confirmation: Whether to require high volume for signal validity
        require_cmo_confirmation: Whether to require a recent short-term downtrend (CMO)
        interval: Interval string, mostly for logging semantics here
        entry_offset: Offset for entry time (e.g., 0 for same candle, 1 for next open)
        pattern_cols: List of candlestick pattern columns to look for
        debug: Print debug information regarding filtering
        
    Returns:
        DataFrame containing valid signals with 'symbol', 'entry_time', 'pattern', and 'signal_price'
    """
    
    if pattern_cols is None:
        pattern_cols = [
            'hammer',
            'inverted_hammer',
            'engulfing_bull',
            'piercing_line'
        ]
        
    entry_offset = max(entry_offset, 0)
    signals = []
    
    debug_counts = {
        "no_pattern_found": 0,
        "vol_reject": 0,
        "cmo_reject": 0,
        "entry_oob": 0,
    }

    for symbol, df in tqdm(dfs.items(), desc=f"Scanning {interval} signals"):
        if df is None or df.empty:
            continue
            
        # Ensure indicators and patterns are present
        df = df.copy()
        add_indicators(df, settings)
        df = add_candle_patterns(df, settings)
        
        for col in pattern_cols:
            if col not in df.columns:
                continue
                
            # Patterns returning > 0 are bullish hits
            series = pd.to_numeric(df[col], errors="coerce")
            hits = series[series > 0]
            
            for ts, val in hits.items():
                sig_row = df.loc[ts]
                
                # Check Volatility Confirmation
                if require_vol_confirmation:
                    v = sig_row.get("VOL_SIGNIFICANT", 0)
                    if isinstance(v, pd.Series): v = v.iloc[-1]
                    if v != 1:
                        debug_counts["vol_reject"] += 1
                        continue
                        
                # Check CMO (Downtrend) Confirmation
                if require_cmo_confirmation:
                    c = sig_row.get("DOWNTREND_SHORT", 0)
                    if isinstance(c, pd.Series): c = c.iloc[-1]
                    if c != 1:
                        debug_counts["cmo_reject"] += 1
                        continue
                        
                # Determine Entry
                try:
                    ts_idx = df.index.get_loc(ts)
                    entry_loc = ts_idx + entry_offset
                    if entry_loc >= len(df):
                        debug_counts["entry_oob"] += 1
                        continue
                        
                    entry_time = df.index[entry_loc]
                    signal_price = df.iloc[ts_idx]['Close'] # We might want to save the close price of the signal candle
                    
                    signals.append({
                        "symbol": symbol,
                        "signal_time": ts,
                        "entry_time": entry_time,
                        "pattern": col,
                        "signal_close": signal_price
                    })
                except Exception as e:
                    pass

    signals_df = pd.DataFrame(signals)
    
    if debug:
        if signals_df.empty:
            print("No signals found. Debug summary:")
        else:
            print(f"Found {len(signals_df)} signals. Debug summary:")
        for k, v in debug_counts.items():
            print(f"  {k}: {v}")

    return signals_df
