import numpy as np
import pandas as pd
from typing import Tuple, Dict

def moving_triple_barrier_labels(
    entries_df: pd.DataFrame,
    market_data_daily: Dict[str, pd.DataFrame],
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    tp_trail_mult: float = 0.5,
    max_holding_bars: int = 15,
    time_decay_sl: bool = False,
    time_decay_mult: float = 1.0,
    active_trailing_sl: bool = False,
    sl_trail_mult: float = 2.0,
    exit_on_close: bool = False,
) -> pd.DataFrame:
    """
    Evaluates trades using a Moving Triple Barrier Method.
    The lower barrier (SL) trails upwards based on highs.
    The take profit trails once the take profit activation price is reached.
    
    Args:
        entries_df: DataFrame from generate_odbicie_entries.
        market_data_daily: Dict of daily OHLCV DataFrames.
        tp_mult: Take Profit multiplier (applied to atr or entry price).
        sl_mult: Stop Loss multiplier.
        tp_trail_mult: Distance variable for trailing take profit.
        max_holding_bars: Maximum duration before time exit.
        
    Returns:
        DataFrame containing finished trades with returns and exit reasons.
    """
    if entries_df is None or entries_df.empty:
        return pd.DataFrame()
        
    trades = []
    
    for _, entry in entries_df.iterrows():
        symbol = entry['symbol']
        entry_time = entry['entry_time']
        entry_price = entry['entry_price']
        atr = entry.get('entry_atr', np.nan)
        
        if symbol not in market_data_daily:
            continue
            
        df = market_data_daily[symbol]
        future = df.loc[df.index > entry_time].head(max_holding_bars)
        
        if future.empty:
            continue
            
        # If ATR isn't available, we'll use a standard percentage (e.g., 2% per mult).
        # It's better to ensure ATR is loaded in market_data.
        if pd.isna(atr) or atr == 0:
            vol_val = entry_price * 0.02
        else:
            vol_val = atr
            
        # Initial Barriers
        current_sl = entry_price - (vol_val * sl_mult)
        
        tp_activation = entry_price + (vol_val * tp_mult)
        tp_trail_distance = vol_val * tp_trail_mult
        
        is_tp_trailing = False
        current_tp_stop = np.nan
        initial_sl = current_sl
        max_sl_distance = entry_price - current_sl
        
        exit_price = future.iloc[-1]['Close']
        exit_time = future.index[-1]
        exit_reason = "TIME_EXIT"
        exit_loc = len(future)
        bars_available = len(future)  # track how many bars we actually got
        
        highs = future['High'].values
        lows = future['Low'].values
        closes = future['Close'].values
        dates = future.index
        
        for i in range(len(future)):
            h = highs[i]
            l = lows[i]
            c = closes[i]
            
            trigger_low = c if exit_on_close else l

            # 1. Check Trailing Take Profit execution
            if is_tp_trailing and trigger_low <= current_tp_stop:
                exit_price = c if exit_on_close else current_tp_stop
                exit_time = dates[i]
                exit_reason = "TRAILING_TP"
                exit_loc = i + 1
                break
                
            # 1.5 Update Stop Loss explicitly before checking (if time decay is active)
            if time_decay_sl:
                # Calculate decay amount based on remaining time and multiplier
                current_sl_distance = max_sl_distance * (1 - (i / max_holding_bars) * time_decay_mult)
                if current_sl_distance < 0:
                    current_sl_distance = 0
                new_decay_sl = entry_price - current_sl_distance
                if new_decay_sl > current_sl:
                    current_sl = new_decay_sl
                
            # 2. Check Stop Loss (Lower Barrier) execution
            if trigger_low <= current_sl:
                exit_price = c if exit_on_close else current_sl
                # If gap down, we get filled at Open loosely modeled
                exit_time = dates[i]
                exit_reason = "SL" if current_sl == initial_sl else "TRAILING_SL"
                exit_loc = i + 1
                break
                
            # 3. Check Take Profit Activation
            if not is_tp_trailing and h >= tp_activation:
                is_tp_trailing = True
                current_tp_stop = tp_activation - tp_trail_distance
                # Immediately update tp stop if high is higher than activation
                new_tp_stop = h - tp_trail_distance
                if new_tp_stop > current_tp_stop:
                    current_tp_stop = new_tp_stop
                    
            # 4. Update Trailing Stops based on current bar's high
            # Update SL
            if active_trailing_sl:
                new_sl = h - (vol_val * sl_trail_mult)
            else:
                new_sl = h - (vol_val * sl_mult)
                
            if new_sl > current_sl:
                current_sl = new_sl
                
            # Update TP (if active)
            if is_tp_trailing:
                new_tp_stop = h - tp_trail_distance
                if new_tp_stop > current_tp_stop:
                    current_tp_stop = new_tp_stop
                    
        # If we ran out of data before any barrier triggered OR before max_holding_bars,
        # the trade is still open — record it as OPEN with current barrier state.
        if exit_reason == "TIME_EXIT" and bars_available < max_holding_bars:
            trade_record = entry.to_dict()
            trade_record.update({
                'exit_time': pd.NaT,
                'exit_price': np.nan,
                'return_pct': np.nan,
                'exit_reason': 'OPEN',
                'hold_bars': bars_available,
                'current_sl': current_sl,
                'current_tp_stop': current_tp_stop,
                'is_tp_trailing': is_tp_trailing,
            })
            trades.append(trade_record)
            continue

        ret_pct = (exit_price - entry_price) / entry_price * 100
        
        trade_record = entry.to_dict()
        trade_record.update({
            'exit_time': exit_time,
            'exit_price': exit_price,
            'return_pct': ret_pct,
            'exit_reason': exit_reason,
            'hold_bars': exit_loc
        })
        trades.append(trade_record)
        
    return pd.DataFrame(trades)


