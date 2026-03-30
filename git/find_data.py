import os
import glob
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas_ta as ta

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.poczatek_ustawienia import create_settings_ui
from core.kontrola_taliba import create_talib_control
from core.ladowanie_danych import create_stock_dfs
from core.dodanie_formacji import add_candle_patterns
from odbicie.mackowe_sygnaly import mackowe_sygnaly
from odbicie.odbicie import generate_odbicie_entries
from odbicie.strategie.odbicie_atr import generate_odbicie_atr_entries

# ---------------------------
# Indicator functions
# ---------------------------
def hma(close, length):
    return ta.hma(close, length=length)

def rsi(close, length):
    return ta.rsi(close, length=length)

def aroon(high, low, length):
    result = ta.aroon(high, low, length=length)
    aroon_cols = [c for c in result.columns if 'AROONU' in c]
    return result[aroon_cols[0]] if aroon_cols else pd.Series(0, index=high.index)

def williams_r(high, low, close, length):
    return ta.willr(high, low, close, length=length)

def supertrend(df):
    result = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
    supertrend_cols = [c for c in result.columns if 'SUPERT' in c]
    return result[supertrend_cols[0]] if supertrend_cols else pd.Series(0, index=df.index)


# ---------------------------
# CREATE LAST SIGNALS DATAFRAME
# ---------------------------
def create_last_signals_df(symbols, dfs_1d, dfs_1w, signal_col='bull_patterns'):
    """
    Create a dataframe storing info about the last signal for each ticker.
    """
    records = []

    def _index_pos(index, label):
        loc = index.get_loc(label)
        if isinstance(loc, slice):
            return loc.stop - 1
        if isinstance(loc, np.ndarray):
            if loc.dtype == bool:
                positions = np.flatnonzero(loc)
                return int(positions.max()) if positions.size else -1
            return int(loc.max()) if loc.size else -1
        return int(loc)

    for interval, dfs in [('1d', dfs_1d), ('1w', dfs_1w)]:
        for symbol in symbols:
            df = dfs.get(symbol)

            signal_str = None
            candles_from_last = None

            if df is not None and len(df) > 0:
                df_signals = df[df[signal_col].apply(
                    lambda x: len(x) > 0 if isinstance(x, list) else False
                )]

                if len(df_signals) > 0:
                    last_signal_row = df_signals.iloc[-1]
                    last_signal_idx = df_signals.index[-1]
                    last_signal_pos = _index_pos(df.index, last_signal_idx)

                    if last_signal_pos >= 0:
                        candles_from_last = len(df) - last_signal_pos
                        signals = last_signal_row[signal_col]
                        signal_str = ', '.join(signals) if isinstance(signals, list) else str(signals)

            records.append({
                'ticker': symbol,
                'interval': interval,
                'signal': signal_str,
                'candles_from_last': candles_from_last
            })

    return pd.DataFrame(records)


# ---------------------------
# Settings
# ---------------------------
signal_cols = ['hammer', 'inverted_hammer', 'engulfing', 'piercing_line']
bear_cols = ['shooting_star', 'hanging_man', 'dark_cloud_cover', 'evening_star']

bull_filters = {'CMO': -35, 'vol_ratio': 1.0}
bear_filters = {'CMO': 35, 'vol_ratio': 0.0}

ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
n_last_candles = 5

from pathlib import Path
output_dir = Path.home() / "lyse-lby" / "git"

# ---------------------------
# Main loop
# ---------------------------
all_area_signals = []

for area in ['sp500', 'crypto', 'europe']:
    settings = {
        "market": area,
        "interval": "1week",
        "week_start": "MON",
        "vol_enabled": True,
        "vol_ratio_window": 20,
        "vol_ratio_threshold": 1.2,
        "cmo_enabled": True,
        "cmo_len": 8,
        "cmo_thres": -35,
        "cmo_thres_prev": -50
    }

    dir_dir = os.path.join(output_dir, f'data_{area}')
    os.makedirs(dir_dir, exist_ok=True)

    # Initialize Talib / load dfs
    talib_panel, apply_params_fn, registry = create_talib_control(settings)
    apply_params_fn()
    dfs_1d, dfs_1w = create_stock_dfs(settings)
    print(f"Loaded data for {len(dfs_1d)} symbols in {area} market.")
    
    symbols = list(dfs_1d.keys())

    # Generate signals and entries
    signals_df_1w = mackowe_sygnaly(
        dfs=dfs_1w,
        settings=settings,
        require_vol_confirmation=True,
        require_cmo_confirmation=True,
        interval='1w',
        entry_offset=0,
        pattern_cols=['hammer', 'inverted_hammer', 'engulfing_bull', 'piercing_line'],
        debug=True
    )

    '''
    signals_df_1d = mackowe_sygnaly(
        dfs=dfs_1d,
        settings=settings,
        require_vol_confirmation=True,
        require_cmo_confirmation=True,
        interval='1d',
        entry_offset=0,
        pattern_cols=['hammer', 'inverted_hammer', 'engulfing_bull', 'piercing_line'],
        debug=True
    )
    '''


    entries_df = generate_odbicie_atr_entries(
        signals_df = signals_df_1w,
        market_data_daily = dfs_1d,
        atr_period = 20,
        atr_factor= 3.0,
        max_setup_hold_bars = 15,
        buy_on_close = False,
    )

    for interval in ['1d', '1w']:
        # Clean output directory
        files_to_remove = glob.glob(f"{dir_dir}/*_{interval}.csv")
        for f in files_to_remove:
            os.remove(f)

        for symbol in tqdm(symbols, desc=f"Processing {interval}"):
            df = dfs_1d.get(symbol) if interval == '1d' else dfs_1w.get(symbol)
            if df is None:
                continue

            # ---------------------------
            # ENTRY MARK (rebound)
            # ---------------------------
            df["rebound"] = 0
            sym_entries = entries_df[entries_df["symbol"] == symbol]
            entry_dates = pd.to_datetime(sym_entries["entry_time"])
            df.loc[df.index.isin(entry_dates), "rebound"] = 1

            # ---------------------------
            # Candle Patterns
            # ---------------------------
            df = add_candle_patterns(df, settings)

            # ---------------------------
            # SIGNAL (general) + add rebound
            # ---------------------------
            bull_mask = (
                (df['VOL_SIGNIFICANT'] == 1) &
                (df['DOWNTREND_SHORT'] == 1)
            )

            mask_any = df[signal_cols] != 0

            df['signal'] = [
                ([pattern for pattern, flag in zip(signal_cols, row) if flag] if bull_mask.iloc[i] else []) +
                (["rebound"] if df['rebound'].iloc[i] == 1 else [])
                for i, row in enumerate(mask_any.values)
            ]
            
            # ---------------------------
            # BULLISH PATTERNS
            # ---------------------------
            bull_condition = (df['CMO'] < bull_filters['CMO']) & (df['vol_ratio'] > bull_filters['vol_ratio'])
            df['bull_patterns'] = [
                ([pattern for pattern, val in zip(signal_cols, row) if val > 0] +
                 (["rebound"] if df['rebound'].iloc[i] == 1 else []))
                if bull_condition.iloc[i] else (["rebound"] if df['rebound'].iloc[i] == 1 else [])
                for i, row in enumerate(df[signal_cols].values)
            ]

            # ---------------------------
            # BEARISH PATTERNS
            # ---------------------------
            bear_condition = (df['CMO'] > bear_filters['CMO']) & (df['vol_ratio'] > bear_filters['vol_ratio'])
            df['bear_patterns'] = [
                [pattern for pattern, val in zip(bear_cols, row) if val < 0]
                if bear_condition.iloc[i] else []
                for i, row in enumerate(df[bear_cols].values)
            ]

            # ---------------------------
            # SAVE LAST N SIGNALS
            # ---------------------------
            last_n = df.tail(n_last_candles)
            has_recent_signal = any(len(row['bull_patterns']) > 0 for _, row in last_n.iterrows())
            
            if has_recent_signal or True:
                df_reset = df.reset_index()
                cols_to_save = ['Date'] + ohlcv_cols + ['signal', 'bull_patterns', 'bear_patterns']
                df_to_save = df_reset[cols_to_save].tail(200)
                df_to_save.to_csv(f"{dir_dir}/{symbol}_{interval}.csv", index=False)

            # Persist enriched dataframe back to the dictionary
            if interval == '1d':
                dfs_1d[symbol] = df
            else:
                dfs_1w[symbol] = df

    # ---------------------------
    # LAST SIGNALS SUMMARY (per area)
    # ---------------------------
    area_last_signals = create_last_signals_df(
        symbols=symbols,
        dfs_1d=dfs_1d,
        dfs_1w=dfs_1w,
        signal_col='bull_patterns'
    )
    if not area_last_signals.empty:
        area_last_signals.insert(0, 'area', area)
        all_area_signals.append(area_last_signals)

# ---------------------------
# FINAL SIGNALS SUMMARY (all areas + intervals)
# ---------------------------
if all_area_signals:
    final_signals_df = pd.concat(all_area_signals, ignore_index=True)
    final_signals_path = os.path.join(output_dir, 'signals.csv')
    final_signals_df.to_csv(final_signals_path, index=False)
