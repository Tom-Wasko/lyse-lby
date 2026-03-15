import itertools
import numpy as np
import pandas as pd
from typing import Callable, Dict, List

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# Local imports — paths work regardless of cwd because Python resolves
# them relative to the package root (sys.path must include lyse-lby/).
from odbicie.tbm import moving_triple_barrier_labels
from odbicie.odbicie_atr import generate_odbicie_atr_entries
from odbicie.odbicie_bb import generate_odbicie_bb_entries


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_trade_metrics(trds: pd.DataFrame, entries_df: pd.DataFrame) -> dict:
    """Return a dict of standard performance metrics for a completed trades DF."""
    win_rate = (trds['return_pct'] > 0).mean() * 100
    avg_return = trds['return_pct'].mean()
    avg_hold_bars = trds['hold_bars'].mean()
    std_hold_bars = trds['hold_bars'].std()
    avg_setup_bars = entries_df['setup_bars'].mean() if 'setup_bars' in entries_df.columns else np.nan
    #std_setup_bars = entries_df['setup_bars'].std() if 'setup_bars' in entries_df.columns else np.nan
    return_per_bar = (trds['return_pct'] / trds['hold_bars']).mean() if not trds.empty else 0
    return {
        'trades': len(trds),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'avg_hold_bars': avg_hold_bars,
        #'std_hold_bars': std_hold_bars,
        'avg_setup_bars': avg_setup_bars,
        #'std_setup_bars': std_setup_bars,
        'return_per_bar': return_per_bar,
    }


def _run_optimization_loop(
    entry_grid: list,
    exit_grid: list,
    entry_fn: Callable,
    entry_fn_kwargs_builder: Callable,
    entry_param_names: List[str],
    market_data_daily: Dict[str, pd.DataFrame],
    tp_mults: list,
    sl_mults: list,
    tp_trail_mults: list,
    max_holding_bars_list: list,
    early_breakeven: bool = False,
    time_decay_sl: bool = False,
    active_trailing_sl: bool = False,
    sl_trail_mult: float = 3.0,
    max_loss_pct: float = 1.0,
    exit_on_close: bool = False,
    min_trades: int = 10,
    tqdm_desc: str = "Optymalizacja",
) -> pd.DataFrame:
    """
    Generic optimization loop shared by all entry-strategy optimizers.

    Parameters
    ----------
    entry_grid : list of tuples
        Each tuple contains one combination of entry parameters.
    exit_grid : list of tuples
        Each tuple is (tp, sl, tp_trail, max_bars).
    entry_fn : callable
        Function that generates an entries DataFrame given keyword arguments.
    entry_fn_kwargs_builder : callable
        Given one row of entry_grid, returns a dict of kwargs for entry_fn.
        This is what varies between ATR / BB / base strategies.
    entry_param_names : list[str]
        Names to use when recording entry param values in the results DataFrame.
    """
    results = []
    total_iters = len(entry_grid) * len(exit_grid)
    pbar = tqdm(total=total_iters, desc=tqdm_desc)

    for entry_combo in entry_grid:
        kwargs = entry_fn_kwargs_builder(entry_combo, market_data_daily)
        entries_df = entry_fn(**kwargs)

        if entries_df.empty or len(entries_df) < min_trades:
            pbar.update(len(exit_grid))
            continue

        for tp, sl, ttp, max_bars in exit_grid:
            trds = moving_triple_barrier_labels(
                entries_df=entries_df,
                market_data_daily=market_data_daily,
                tp_mult=tp,
                sl_mult=sl,
                tp_trail_mult=ttp,
                max_holding_bars=max_bars,
                early_breakeven=early_breakeven,
                time_decay_sl=time_decay_sl,
                active_trailing_sl=active_trailing_sl,
                sl_trail_mult=sl_trail_mult,
                max_loss_pct=max_loss_pct,
                exit_on_close=exit_on_close,
            )

            if len(trds) >= min_trades:
                row = dict(zip(entry_param_names, entry_combo))
                row.update({
                    'tp_mult': tp,
                    'sl_mult': sl,
                    'tp_trail_mult': ttp,
                    'max_holding_bars': max_bars,
                })
                row.update(_compute_trade_metrics(trds, entries_df))
                results.append(row)

            pbar.update(1)

    pbar.close()

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(
            by=['return_per_bar', 'avg_return'], ascending=[False, False]
        )
    return df_res


# ---------------------------------------------------------------------------
# Simple SL/TP (no TBM)
# ---------------------------------------------------------------------------

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

        vol_val = entry_price * 0.02 if (pd.isna(atr) or atr == 0) else atr

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

            if trigger_high >= tp_price:
                exit_price = c if exit_on_close else tp_price
                exit_time = dates[i]
                exit_reason = "TP"
                exit_loc = i + 1
                break

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
            'hold_bars': exit_loc,
        })
        trades.append(trade_record)

    return pd.DataFrame(trades)


def optimize_simple_sl_tp(
    entries_df: pd.DataFrame,
    market_data_daily: Dict[str, pd.DataFrame],
    tp_mults: list,
    sl_mults: list,
    max_holding_bars_list: list,
    exit_on_close: bool = False,
) -> pd.DataFrame:
    """
    Grid search optimization for the simple SL/TP strategy.

    Returns:
        DataFrame containing optimization results sorted by average return.
    """
    results = []
    grid = list(itertools.product(tp_mults, sl_mults, max_holding_bars_list))

    for tp, sl, max_bars in tqdm(grid, desc="Optymalizacja Simple SL/TP"):
        trds = simple_sl_tp_labels(
            entries_df=entries_df,
            market_data_daily=market_data_daily,
            tp_mult=tp,
            sl_mult=sl,
            max_holding_bars=max_bars,
            exit_on_close=exit_on_close,
        )

        if len(trds) > 0:
            avg_hold_bars = trds['hold_bars'].mean()
            results.append({
                'tp_mult': tp,
                'sl_mult': sl,
                'max_holding_bars': max_bars,
                'trades': len(trds),
                'win_rate': (trds['return_pct'] > 0).mean() * 100,
                'avg_return': trds['return_pct'].mean(),
                'avg_hold_bars': avg_hold_bars,
                'return_per_bar': (trds['return_pct'] / trds['hold_bars']).mean() if not trds.empty else 0,
            })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by='avg_return', ascending=False)
    return df_res


# ---------------------------------------------------------------------------
# ATR entry + TBM exit
# ---------------------------------------------------------------------------

def optimize_atr_tbm(
    signals_df: pd.DataFrame,
    market_data_daily: Dict[str, pd.DataFrame],
    atr_periods: list,
    atr_factors: list,
    max_setup_hold_bars_list: list,
    tp_mults: list,
    sl_mults: list,
    tp_trail_mults: list,
    max_holding_bars_list: list,
    buy_on_close_list: list = [False],
    early_breakeven: bool = False,
    time_decay_sl: bool = False,
    active_trailing_sl: bool = False,
    sl_trail_mult: float = 3.0,
    max_loss_pct: float = 1.0,
    exit_on_close: bool = False,
    min_trades: int = 10,
) -> pd.DataFrame:
    """
    Grid search optimization for ATR entry parameters combined with TBM exits.
    Returns results sorted by 'return_per_bar'.
    """
    entry_grid = list(itertools.product(
        atr_periods, atr_factors, max_setup_hold_bars_list, buy_on_close_list
    ))
    exit_grid = list(itertools.product(
        tp_mults, sl_mults, tp_trail_mults, max_holding_bars_list
    ))

    def build_kwargs(combo, mkt):
        atr_p, atr_f, max_setup, boc = combo
        return dict(
            signals_df=signals_df,
            market_data_daily=mkt,
            atr_period=atr_p,
            atr_factor=atr_f,
            max_setup_hold_bars=max_setup,
            buy_on_close=boc,
        )

    return _run_optimization_loop(
        entry_grid=entry_grid,
        exit_grid=exit_grid,
        entry_fn=generate_odbicie_atr_entries,
        entry_fn_kwargs_builder=build_kwargs,
        entry_param_names=['atr_period', 'atr_factor', 'max_setup_bars', 'buy_on_close'],
        market_data_daily=market_data_daily,
        tp_mults=tp_mults,
        sl_mults=sl_mults,
        tp_trail_mults=tp_trail_mults,
        max_holding_bars_list=max_holding_bars_list,
        early_breakeven=early_breakeven,
        time_decay_sl=time_decay_sl,
        active_trailing_sl=active_trailing_sl,
        sl_trail_mult=sl_trail_mult,
        max_loss_pct=max_loss_pct,
        exit_on_close=exit_on_close,
        min_trades=min_trades,
        tqdm_desc="Optymalizacja ATR + TBM",
    )


# ---------------------------------------------------------------------------
# BB entry + TBM exit
# ---------------------------------------------------------------------------

def optimize_bb_tbm(
    signals_df: pd.DataFrame,
    market_data_daily: Dict[str, pd.DataFrame],
    bb_periods: list,
    bb_stds: list,
    max_setup_hold_bars_list: list,
    tp_mults: list,
    sl_mults: list,
    tp_trail_mults: list,
    max_holding_bars_list: list,
    rsi_period: int = 14,
    buy_on_close_list: list = [False],
    early_breakeven: bool = False,
    time_decay_sl: bool = False,
    active_trailing_sl: bool = False,
    sl_trail_mult: float = 3.0,
    max_loss_pct: float = 1.0,
    exit_on_close: bool = False,
    min_trades: int = 10,
) -> pd.DataFrame:
    """
    Grid search optimization for BB entry parameters combined with TBM exits.
    Returns results sorted by 'return_per_bar'.
    """
    entry_grid = list(itertools.product(
        bb_periods, bb_stds, max_setup_hold_bars_list, buy_on_close_list
    ))
    exit_grid = list(itertools.product(
        tp_mults, sl_mults, tp_trail_mults, max_holding_bars_list
    ))

    def build_kwargs(combo, mkt):
        bb_p, bb_s, max_setup, boc = combo
        return dict(
            signals_df=signals_df,
            market_data_daily=mkt,
            bb_period=bb_p,
            bb_std=bb_s,
            rsi_period=rsi_period,
            max_setup_hold_bars=max_setup,
            buy_on_close=boc,
        )

    return _run_optimization_loop(
        entry_grid=entry_grid,
        exit_grid=exit_grid,
        entry_fn=generate_odbicie_bb_entries,
        entry_fn_kwargs_builder=build_kwargs,
        entry_param_names=['bb_period', 'bb_std', 'max_setup_bars', 'buy_on_close'],
        market_data_daily=market_data_daily,
        tp_mults=tp_mults,
        sl_mults=sl_mults,
        tp_trail_mults=tp_trail_mults,
        max_holding_bars_list=max_holding_bars_list,
        early_breakeven=early_breakeven,
        time_decay_sl=time_decay_sl,
        active_trailing_sl=active_trailing_sl,
        sl_trail_mult=sl_trail_mult,
        max_loss_pct=max_loss_pct,
        exit_on_close=exit_on_close,
        min_trades=min_trades,
        tqdm_desc="Optymalizacja BB + TBM",
    )
