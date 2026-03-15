import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm
import numpy as np

# widgety do wyboru ustawień (interwal, filtry)
from core.poczatek_ustawienia import create_settings_ui

# kontrola taliba
from core.kontrola_taliba import create_talib_control

# zaladowanie df do pamieci
from core.ladowanie_danych import create_stock_dfs, add_indicators

# dodanie formacji
from core.dodanie_formacji import add_candle_patterns

# giga plot
from core.main_plot import create_chart_ui


# Indicator functions from pandas-ta
def hma(close, length):
    return ta.hma(close, length=length)

def rsi(close, length):
    return ta.rsi(close, length=length)

def aroon(high, low, length):
    result = ta.aroon(high, low, length=length)
    # Return just the Aroon Up indicator
    aroon_cols = [col for col in result.columns if 'AROONU' in col]
    return result[aroon_cols[0]] if aroon_cols else pd.Series(0, index=high.index)

def williams_r(high, low, close, length):
    return ta.willr(high, low, close, length=length)

def supertrend(df):
    result = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
    supertrend_cols = [col for col in result.columns if 'SUPERT' in col]
    return result[supertrend_cols[0]] if supertrend_cols else pd.Series(0, index=df.index)


settings_panel, settings = create_settings_ui()
talib_panel, apply_params_fn, registry = create_talib_control(settings)
apply_params_fn()
dfs_1d, dfs_1w = create_stock_dfs(settings)
symbols = list(dfs_1d.keys())

import os
import glob
import numpy as np


signal_cols = ['hammer', 'inverted_hammer', 'engulfing', 'piercing_line']

bear_cols = ['shooting_star', 'hanging_man', 'dark_cloud_cover', 'evening_star']

signal_filters = {'CMO': -20, 'vol_ratio': 1.0}

bull_filters = {'CMO': -35, 'vol_ratio': 1.0}
bear_filters = {'CMO': 35, 'vol_ratio': 0.0}

ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

n = 5  # liczba ostatnich świec do sprawdzenia
output_dir = "git/data"

for interval in ['1d', '1w']:

    # Clean directory for this interval
    files_to_remove = glob.glob(f"{output_dir}/*_{interval}.csv")
    for file in files_to_remove:
        os.remove(file)

    for symbol in tqdm(symbols, desc=f"Processing {interval}"):

        df = dfs_1d.get(symbol) if interval == '1d' else dfs_1w.get(symbol)

        if df is None:
            continue

        df = add_candle_patterns(df, settings)

        # ----------------------------------
        # SIGNAL (general)
        # ----------------------------------
        mask_any = df[signal_cols] != 0

        df['signal'] = [
            [pattern for pattern, flag in zip(signal_cols, row) if flag]
            for row in mask_any.values
        ]

        # ----------------------------------
        # BULLISH PATTERNS (value > 0)
        # ----------------------------------
        bull_condition = (
            (df['CMO'] < bull_filters['CMO']) &
            (df['vol_ratio'] > bull_filters['vol_ratio'])
        )

        df['bull_patterns'] = [
            [pattern for pattern, val in zip(signal_cols, row) if val > 0]
            if bull_condition.iloc[i] else []
            for i, row in enumerate(df[signal_cols].values)
        ]

        # ----------------------------------
        # BEARISH PATTERNS (value < 0)
        # ----------------------------------
        bear_condition = (
            (df['CMO'] > bear_filters['CMO']) &
            (df['vol_ratio'] > bear_filters['vol_ratio'])
        )

        df['bear_patterns'] = [
            [pattern for pattern, val in zip(bear_cols, row) if val < 0]
            if bear_condition.iloc[i] else []
            for i, row in enumerate(df[bear_cols].values)
        ]

        # ----------------------------------
        # CHECK LAST N CANDLES
        # ----------------------------------
        last_n = df.tail(n)

        has_recent_signal = any(
            (len(row['bull_patterns']) > 0) 
            #or
            #(len(row['bear_patterns']) > 0)
            for _, row in last_n.iterrows()
        )

        if has_recent_signal:
            df_reset = df.reset_index()

            cols_to_save = (
                ['Date'] +
                ohlcv_cols +
                ['signal', 'bull_patterns', 'bear_patterns']
            )

            df_to_save = df_reset[cols_to_save].tail(200)

            df_to_save.to_csv(
                f"{output_dir}/{symbol}_{interval}.csv",
                index=False
            )
