import pandas as pd
import numpy as np
from typing import Dict


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    """
    Oblicza Bollinger Bands na podanym okresie.

    Args:
        df: DataFrame z kolumną 'Close'.
        period: Okno SMA/odchylenia standardowego (domyślnie 20).
        std_mult: Mnożnik odchylenia standardowego (domyślnie 2.0).

    Returns:
        DataFrame z kolumnami: bb_middle, bb_upper, bb_lower, bb_bandwidth.
    """
    close = df['Close']
    bb_middle = close.rolling(window=period, min_periods=period).mean()
    bb_std = close.rolling(window=period, min_periods=period).std(ddof=0)

    bb_upper = bb_middle + (bb_std * std_mult)
    bb_lower = bb_middle - (bb_std * std_mult)
    bb_bandwidth = (bb_upper - bb_lower) / bb_middle

    return pd.DataFrame({
        'bb_middle': bb_middle,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_bandwidth': bb_bandwidth,
    }, index=df.index)


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Oblicza Relative Strength Index (RSI) na podanym okresie.

    Args:
        df: DataFrame z kolumną 'Close'.
        period: Okres RSI (domyślnie 14).

    Returns:
        Seria z wartościami RSI (0–100).
    """
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_odbicie_bb_entries(
    signals_df: pd.DataFrame,
    market_data_daily: Dict[str, pd.DataFrame],
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 14,
    max_setup_hold_bars: int = 15,
    buy_on_close: bool = False,
) -> pd.DataFrame:
    """
    Generuje wejścia na podstawie dotknięcia/przekroczenia dolnej wstęgi Bollingera
    po sygnale wzorca świecowego (mackowe_sygnaly).

    Logika wejścia:
        - Czekamy aż cena (Low) dotknie lub spadnie poniżej dolnej BB.
        - Wejście realizowane po cenie dolnej BB (lub niżej przy gap-down).

    Jeśli buy_on_close=True, wchodzimy natychmiast na zamknięciu pierwszego
    bara po sygnale bez czekania na BB.

    Kolumny w wyjściowym DataFrame:
        symbol, signal_time, pattern, entry_time, entry_price, signal_close,
        bb_period, bb_std, bb_lower, bb_middle, bb_upper, bb_bandwidth,
        rsi_at_entry, entry_atr, setup_bars.

    Args:
        signals_df: DataFrame z kolumnami 'symbol', 'signal_time', 'signal_close', 'pattern'.
        market_data_daily: Słownik symbol -> dzienny OHLCV DataFrame.
        bb_period: Okno Bollinger Bands (domyślnie 20).
        bb_std: Mnożnik odchylenia standardowego BB (domyślnie 2.0).
        rsi_period: Okres RSI do potwierdzenia momentum (domyślnie 14).
        max_setup_hold_bars: Maksymalna liczba dni oczekiwania na wejście.
        buy_on_close: Jeśli True, wchodzi natychmiast na zamknięciu pierwszego bara po sygnale.

    Returns:
        DataFrame z wykonanymi wejściami.
    """
    entries = []

    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    # Cache BB i RSI per symbol (obliczamy raz)
    bb_cache: Dict[str, pd.DataFrame] = {}
    rsi_cache: Dict[str, pd.Series] = {}

    for _, sig in signals_df.iterrows():
        symbol = sig['symbol']
        signal_time = sig['signal_time']

        # Pobierz signal_close
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

        # Oblicz i cache'uj BB oraz RSI
        if symbol not in bb_cache:
            bb_cache[symbol] = compute_bollinger_bands(df, period=bb_period, std_mult=bb_std)
        if symbol not in rsi_cache:
            rsi_cache[symbol] = compute_rsi(df, period=rsi_period)

        bb_df = bb_cache[symbol]
        rsi_series = rsi_cache[symbol]

        # Pobierz wartość BB na dzień sygnału (lub ostatnią dostępną przed nim)
        bb_at_signal = bb_df.loc[:signal_time]
        if bb_at_signal.empty or bb_at_signal.iloc[-1].isnull().any():
            # Niewystarczająca historia do obliczenia BB — pomijamy sygnał
            continue

        # Skanuj przyszłe świece
        future_df = df.loc[df.index > signal_time].head(max_setup_hold_bars)

        if future_df.empty:
            continue

        if buy_on_close:
            # Wejście natychmiast na zamknięciu pierwszego bara po sygnale
            ts = future_df.index[0]
            row = future_df.iloc[0]

            bb_row = bb_df.loc[ts] if ts in bb_df.index else bb_at_signal.iloc[-1]
            rsi_val = rsi_series.get(ts, np.nan)

            entries.append({
                'symbol': symbol,
                'signal_time': signal_time,
                'pattern': sig['pattern'],
                'entry_time': ts,
                'entry_price': row['Close'],
                'signal_close': signal_close,
                'bb_period': bb_period,
                'bb_std': bb_std,
                'bb_lower': bb_row['bb_lower'],
                'bb_middle': bb_row['bb_middle'],
                'bb_upper': bb_row['bb_upper'],
                'bb_bandwidth': bb_row['bb_bandwidth'],
                'rsi_at_entry': rsi_val,
                'entry_atr': row.get('ATR', np.nan),
                'setup_bars': 1,
            })
        else:
            for i, (ts, row) in enumerate(future_df.iterrows()):
                # Pobierz aktualną dolną BB dla tego bara
                if ts in bb_df.index:
                    bb_row = bb_df.loc[ts]
                else:
                    # Brak BB dla tego dnia (brak historii) — pomijamy
                    continue

                lower_band = bb_row['bb_lower']

                if pd.isna(lower_band):
                    continue

                # Trigger wejścia: Low <= dolna BB
                if row['Low'] <= lower_band:
                    # Fill po dolnej BB lub niżej przy gap-down
                    entry_price = min(row['Open'], lower_band)

                    rsi_val = rsi_series.get(ts, np.nan)

                    entries.append({
                        'symbol': symbol,
                        'signal_time': signal_time,
                        'pattern': sig['pattern'],
                        'entry_time': ts,
                        'entry_price': entry_price,
                        'signal_close': signal_close,
                        'bb_period': bb_period,
                        'bb_std': bb_std,
                        'bb_lower': lower_band,
                        'bb_middle': bb_row['bb_middle'],
                        'bb_upper': bb_row['bb_upper'],
                        'bb_bandwidth': bb_row['bb_bandwidth'],
                        'rsi_at_entry': rsi_val,
                        'entry_atr': row.get('ATR', np.nan),
                        'setup_bars': i + 1,
                    })
                    break  # Wejście zrealizowane — przerywamy skanowanie

    return pd.DataFrame(entries)
