import pandas as pd
import numpy as np
from typing import Dict


def compute_long_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Oblicza Average True Range na podanym okresie.
    
    Args:
        df: DataFrame z kolumnami High, Low, Close.
        period: Okres ATR (domyślnie 20).
        
    Returns:
        Seria z wartościami ATR.
    """
    high = df['High']
    low = df['Low']
    prev_close = df['Close'].shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    return tr.rolling(window=period, min_periods=period).mean()


def generate_odbicie_atr_entries(
    signals_df: pd.DataFrame,
    market_data_daily: Dict[str, pd.DataFrame],
    atr_period: int = 20,
    atr_factor: float = 3.0,
    max_setup_hold_bars: int = 15,
    buy_on_close: bool = False,
) -> pd.DataFrame:
    """
    Generuje wejścia na podstawie cofnięcia mierzonego wielokrotnością 
    długiego ATR zamiast sztywnego procentu.
    
    Cofnięcie = signal_close - (długi_ATR * atr_factor)
    
    Dzięki temu próg wejścia automatycznie dopasowuje się do zmienności
    konkretnego aktywa w danym momencie rynkowym.
    
    Args:
        signals_df: DataFrame z kolumnami 'symbol', 'signal_time', 'signal_close', 'pattern'.
        market_data_daily: Słownik symbol -> dzienny OHLCV DataFrame.
        atr_period: Okres ATR (np. 20 lub 30 świec) — im dłuższy, 
                     tym bardziej wygładzony i odporny na szpilki.
        atr_factor: Mnożnik ATR określający głębokość wymaganego cofnięcia
                     (np. 3.0 = czekamy na spadek o 3x dzienny ATR).
        max_setup_hold_bars: Maksymalna liczba dni oczekiwania na cofnięcie.
        buy_on_close: Jeśli True, wchodzi natychmiast po zamknięciu pierwszego bara
                      po sygnale (bez czekania na cofnięcie ATR).
        
    Returns:
        DataFrame z wykonanymi wejściami.
    """

    entries = []

    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    # Pre-compute długi ATR dla każdego symbolu (raz, zamiast w pętli)
    atr_cache: Dict[str, pd.Series] = {}

    for _, sig in signals_df.iterrows():
        symbol = sig['symbol']
        signal_time = sig['signal_time']

        # Fallback na signal_close
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

        # Oblicz i cache'uj ATR
        if symbol not in atr_cache:
            atr_cache[symbol] = compute_long_atr(df, period=atr_period)

        atr_series = atr_cache[symbol]

        # Pobierz wartość ATR na dzień sygnału (lub ostatnią dostępną przed nim)
        atr_at_signal = atr_series.loc[:signal_time]
        if atr_at_signal.empty or pd.isna(atr_at_signal.iloc[-1]):
            # Brak wystarczającej historii do obliczenia ATR — pomijamy sygnał
            continue

        long_atr = atr_at_signal.iloc[-1]

        # Oblicz cel wejścia: signal_close - (długi_ATR * factor)
        pullback_distance = long_atr * atr_factor
        target_entry_price = signal_close - pullback_distance

        # Zabezpieczenie: cel nie może być ujemny ani absurdalnie niski
        if target_entry_price <= 0:
            continue

        # Skanuj przyszłe świece w poszukiwaniu wejścia
        future_df = df.loc[df.index > signal_time].head(max_setup_hold_bars)

        if future_df.empty:
            continue

        if buy_on_close:
            # Wejście natychmiast na zamknięciu pierwszego bara po sygnale
            ts = future_df.index[0]
            row = future_df.iloc[0]
            entry_price = row['Close']
            entry_atr_val = atr_series.get(ts, np.nan)
            entries.append({
                'symbol': symbol,
                'signal_time': signal_time,
                'pattern': sig['pattern'],
                'entry_time': ts,
                'entry_price': entry_price,
                'signal_close': signal_close,
                'atr_period': atr_period,
                'atr_factor': atr_factor,
                'signal_atr': long_atr,
                'pullback_distance': pullback_distance,
                'pullback_pct': pullback_distance / signal_close * 100,
                'entry_atr': entry_atr_val if not pd.isna(entry_atr_val) else long_atr,
                'setup_bars': 1,
            })
        else:
            for i, (ts, row) in enumerate(future_df.iterrows()):
                if row['Low'] <= target_entry_price:
                    # Wykonanie: fill po cenie docelowej lub niżej przy gap down
                    entry_price = min(row['Open'], target_entry_price)

                    # ATR w dniu wejścia (do dalszego użycia w TBM)
                    entry_atr_val = atr_series.get(ts, np.nan)

                    entries.append({
                        'symbol': symbol,
                        'signal_time': signal_time,
                        'pattern': sig['pattern'],
                        'entry_time': ts,
                        'entry_price': entry_price,
                        'signal_close': signal_close,
                        'atr_period': atr_period,
                        'atr_factor': atr_factor,
                        'signal_atr': long_atr,
                        'pullback_distance': pullback_distance,
                        'pullback_pct': pullback_distance / signal_close * 100,
                        'entry_atr': entry_atr_val if not pd.isna(entry_atr_val) else long_atr,
                        'setup_bars': i + 1,
                    })
                    break  # Wejście zrealizowane — przerywamy skanowanie

    return pd.DataFrame(entries)
