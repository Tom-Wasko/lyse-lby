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
    early_breakeven: bool = False,
    time_decay_sl: bool = False,
    active_trailing_sl: bool = False,
    sl_trail_mult: float = 2.0,
    max_loss_pct: float = 1.0, # 100% loss means disabled by default
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
        calculated_sl = entry_price - (vol_val * sl_mult)
        hard_cap_sl = entry_price * (1 - max_loss_pct)
        current_sl = max(calculated_sl, hard_cap_sl)
        
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
        
        highs = future['High'].values
        lows = future['Low'].values
        closes = future['Close'].values
        dates = future.index
        
        for i in range(len(future)):
            h = highs[i]
            l = lows[i]
            c = closes[i]
            
            # 0. Check Early Breakeven Bailout
            if early_breakeven and i >= max_holding_bars // 2:
                if c < entry_price:
                    exit_price = c
                    exit_time = dates[i]
                    exit_reason = "EARLY_TIME_BAILOUT"
                    exit_loc = i + 1
                    break
                    
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
                # Calculate decay amount based on remaining time
                current_sl_distance = max_sl_distance * (1 - (i / max_holding_bars))
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


