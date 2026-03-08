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

def simple_sl_tp_labels(
    entries_df: pd.DataFrame,
    market_data_daily: Dict[str, pd.DataFrame],
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding_bars: int = 15,
    exit_on_close: bool = False,
) -> pd.DataFrame:
    """
    Evaluates trades using a simple fixed Stop Loss and Take Profit.
    No trailing stops or time decay.
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
            
        if pd.isna(atr) or atr == 0:
            vol_val = entry_price * 0.02
        else:
            vol_val = atr
            
        # Fixed Barriers
        sl_price = entry_price - (vol_val * sl_mult)
        tp_price = entry_price + (vol_val * tp_mult)
        
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
            
            trigger_low = c if exit_on_close else l
            trigger_high = c if exit_on_close else h
            
            # 1. Check TP
            if trigger_high >= tp_price:
                exit_price = c if exit_on_close else tp_price
                exit_time = dates[i]
                exit_reason = "TP"
                exit_loc = i + 1
                break
                
            # 2. Check SL
            if trigger_low <= sl_price:
                exit_price = c if exit_on_close else sl_price
                exit_time = dates[i]
                exit_reason = "SL"
                exit_loc = i + 1
                break
                
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

def optimize_simple_sl_tp(
    entries_df: pd.DataFrame,
    market_data_daily: Dict[str, pd.DataFrame],
    tp_mults: list,
    sl_mults: list,
    max_holding_bars_list: list,
    exit_on_close: bool = False
) -> pd.DataFrame:
    """
    Grid search optimization for the simple SL/TP strategy.
    
    Args:
        entries_df: DataFrame with entry signals.
        market_data_daily: Dictionary of daily price DataFrames.
        tp_mults: List of Take Profit multipliers to test.
        sl_mults: List of Stop Loss multipliers to test.
        max_holding_bars_list: List of maximum holding periods to test.
        exit_on_close: Whether exits are evaluated on Close instead of High/Low.
        
    Returns:
        DataFrame containing optimization results sorted by average return.
    """
    import itertools
    try:
        from tqdm.notebook import tqdm
    except ImportError:
        from tqdm import tqdm
        
    results = []
    
    grid = list(itertools.product(tp_mults, sl_mults, max_holding_bars_list))
    
    for tp, sl, max_bars in tqdm(grid, desc="Optymalizacja Simple SL/TP"):
        trds = simple_sl_tp_labels(
            entries_df=entries_df,
            market_data_daily=market_data_daily,
            tp_mult=tp,
            sl_mult=sl,
            max_holding_bars=max_bars,
            exit_on_close=exit_on_close
        )
        
        if len(trds) > 0:
            win_rate = (trds['return_pct'] > 0).mean() * 100
            avg_return = trds['return_pct'].mean()
            avg_hold_bars = trds['hold_bars'].mean()
            
            results.append({
                'tp_mult': tp,
                'sl_mult': sl,
                'max_holding_bars': max_bars,
                'trades': len(trds),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'avg_hold_bars': avg_hold_bars,
                'return_per_bar': avg_return / avg_hold_bars if avg_hold_bars > 0 else 0
            })
            
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by='avg_return', ascending=False)
    return df_res
