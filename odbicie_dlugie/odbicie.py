import pandas as pd
import numpy as np
from typing import Dict

def generate_odbicie_entries(
    signals_df: pd.DataFrame,
    market_data_daily: Dict[str, pd.DataFrame],
    threshold_pct: float = 0.05,
    max_setup_hold_bars: int = 15,
    buy_on_close: bool = False,
) -> pd.DataFrame:
    """
    Takes valid signals and scans daily market data for an entry 
    when the price drops to a certain threshold below the signal's close.
    
    Args:
        signals_df: DataFrame with 'symbol', 'signal_time', 'signal_close', 'pattern'.
        market_data_daily: Dict mapping symbol -> daily OHLCV DataFrame.
        threshold_pct: Percentage drop required to enter (e.g., 0.05 is 5%).
        max_setup_hold_bars: Maximum number of days to wait for the threshold drop before invalidating signal.
        buy_on_close: If True, enters at the close of the first bar after the signal
                      instead of waiting for the threshold drop.
        
    Returns:
        DataFrame containing executed entry trades.
    """
    
    entries = []
    
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()
        
    for _, sig in signals_df.iterrows():
        symbol = sig['symbol']
        signal_time = sig['signal_time']
        
        # Handle cases where signal_close isn't explicitly passed, though our mackowe_sygnaly generates it.
        # Fallback to computing it if missing.
        if 'signal_close' in sig:
            signal_close = sig['signal_close']
        else:
            if symbol in market_data_daily and signal_time in market_data_daily[symbol].index:
                signal_close = market_data_daily[symbol].loc[signal_time]['Close']
            else:
                continue
                
        if pd.isna(signal_close) or signal_close == 0:
            continue
            
        if symbol not in market_data_daily:
            continue
            
        df = market_data_daily[symbol]
        
        # Look at the days immediately following the signal
        future_df = df.loc[df.index > signal_time].head(max_setup_hold_bars)
        
        if future_df.empty:
            continue

        if buy_on_close:
            # Enter at the close of the first bar after the signal
            ts = future_df.index[0]
            row = future_df.iloc[0]
            entries.append({
                'symbol': symbol,
                'signal_time': signal_time,
                'pattern': sig['pattern'],
                'entry_time': ts,
                'entry_price': row['Close'],
                'signal_close': signal_close,
                'threshold_pct': threshold_pct,
                'entry_atr': row.get('ATR', np.nan)
            })
        else:
            target_entry_price = signal_close * (1 - threshold_pct)

            # Scan for entry trigger
            for ts, row in future_df.iterrows():
                if row['Low'] <= target_entry_price:
                    # Execution: we get filled at the target or lower if it gaps down
                    entry_price = min(row['Open'], target_entry_price)

                    # Fetch volatility if available for downstream use
                    atr = row.get('ATR', np.nan)

                    entries.append({
                        'symbol': symbol,
                        'signal_time': signal_time,
                        'pattern': sig['pattern'],
                        'entry_time': ts,
                        'entry_price': entry_price,
                        'signal_close': signal_close,
                        'threshold_pct': threshold_pct,
                        'entry_atr': atr
                    })
                    break  # Entry executed, stop scanning for this signal
                
    return pd.DataFrame(entries)
