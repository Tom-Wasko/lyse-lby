"""
Interactive chart with candlestick patterns and indicator controls.

Function: create_chart_ui(dfs_1d, dfs_1w, settings, symbols, 
                         add_indicators_fn, candle_patterns_fn, 
                         hma_fn, rsi_fn, aroon_fn, williams_r_fn, supertrend_fn)

Returns a widgets.VBox containing all controls, plot button, scan button,
and outputs ready to display in a Jupyter notebook.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import ipywidgets as widgets
import mplfinance as mpf
import requests
from bs4 import BeautifulSoup
from IPython.display import display
from tqdm.notebook import tqdm

# Helper: ensure OHLCV numeric and valid for mplfinance
def _ensure_numeric_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce OHLCV columns to numeric, ensure datetime index, drop invalid rows.

    This makes plotting with mplfinance robust against strings or missing values
    in input CSVs.
    """
    if df is None or df.empty:
        return df

    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if not cols:
        return df

    for col in cols:
        val = df[col]
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0]
        if not isinstance(val, pd.Series):
            val = pd.Series(val)
        df[col] = pd.to_numeric(val, errors="coerce")

    # drop rows where any required OHLCV value is missing
    df = df.dropna(subset=cols)

    return df

# ============== FETCH S&P 500 COMPANIES ==============
def fetch_sp500_symbols():
    """Fetch S&P 500 symbols from Wikipedia and detect exchange (NASDAQ/NYSE)."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        html = requests.get(url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"id": "constituents"})

        nasdaq = set()
        nyse = set()

        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if not cols:
                continue

            symbol = cols[0].text.strip()
            link = cols[0].find("a")

            if link and "nasdaq.com" in link.get("href", "").lower():
                nasdaq.add(symbol)
            else:
                nyse.add(symbol)

        return nasdaq, nyse
    except Exception as e:
        print(f"⚠️  Failed to fetch S&P 500 from Wikipedia: {e}")
        return set(), set()


NASDAQ_SYMBOLS, NYSE_SYMBOLS = fetch_sp500_symbols()


PATTERN_PLOT_CONFIG = {

    # Single Candles
    "hammer": dict(label="Młot", side="low", marker="^", color="green", size=30),
    "inverted_hammer": dict(label="Odwrócony młot", side="low", marker="^", color="#7CFC00", size=34),
    "shooting_star": dict(label="Shooting Star", side="high", marker="v", color="red", size=30),
    "hanging_man": dict(label="Hanging Man", side="high", marker="v", color="#FF4500", size=34),

    "gravestone_doji": dict(label="Gravestone Doji", side="high", marker="v", color="#B22222", size=32),
    "doji": dict(label="Doji", side="mid", marker="o", color="gray", size=24),
    "long_legged_doji": dict(label="Long-Legged Doji", side="mid", marker="o", color="darkgray", size=26),
    "spinning_top": dict(label="Spinning Top", side="mid", marker="o", color="silver", size=26),

    # Double Candles (separate bullish/bearish)
    "engulfing_bull": dict(label="Bullish Engulfing", side="low", marker="^", color="lime", size=40),
    "engulfing_bear": dict(label="Bearish Engulfing", side="high", marker="v", color="red", size=40),
    "piercing_line": dict(label="Piercing Line", side="low", marker="^", color="#32CD32", size=38),
    "dark_cloud": dict(label="Dark Cloud", side="high", marker="v", color="#B22222", size=38),

    # Triple Candles
    "evening_star": dict(label="Evening Star", side="high", marker="v", color="#B22222", size=42),
    "evening_doji_star": dict(label="Evening Doji Star", side="high", marker="v", color="#DC143C", size=40),

    "objecie_i_cofniecie": dict(label="objecie i cofniecie", side="low", marker="^", color="#00CED1",size=40,),
}


def create_chart_ui(
    dfs_1d: Dict[str, pd.DataFrame],
    dfs_1w: Dict[str, pd.DataFrame],
    settings: Dict,
    symbols: List[str],
    add_indicators_fn: Callable,
    candle_patterns_fn: Callable,
    hma_fn: Callable,
    rsi_fn: Callable,
    aroon_fn: Callable,
    williams_r_fn: Callable,
    supertrend_fn: Callable,
) -> widgets.VBox:
    """
    Create an interactive charting UI with pattern detection and indicators.

    Arguments:
        dfs_1d: dict mapping symbol -> daily DataFrame
        dfs_1w: dict mapping symbol -> weekly DataFrame
        settings: dict with 'interval', 'week_start', vol/cmo settings
        symbols: list of symbol strings
        add_indicators_fn: function to add indicators to DataFrame
        candle_patterns_fn: function to compute pattern columns
        hma_fn, rsi_fn, aroon_fn, williams_r_fn, supertrend_fn: indicator functions

    Returns:
        widgets.VBox containing the full interactive UI.
    """

    def get_df(symbol):
        """Select daily or weekly DF based on interval setting."""
        interval = settings.get("interval", "1d")
        return dfs_1w.get(symbol) if interval == "1week" else dfs_1d.get(symbol)




    def plot_chart(
        symbol,
        head=None,
        tail=None,
        plot_window=50,
        show_sma=False,
        sma1=10,
        sma2=50,
        sma3=100,
        show_ema=False,
        ema1=20,
        ema2=80,
        ema3=200,
        show_tema=False,
        tema1=10,
        tema2=20,
        tema3=50,
        show_aroon=True,
        aroon_len=14,
        show_rsi=False,
        rsi_len=14,
        show_cmo=False,
        cmo_len=9,
        show_hma=False,
        hma_len=55,
        show_supertrend=False,
        show_wr=False,
        wr_len=14,
        show_atr=False,
        atr_len=14,
        show_trade_levels=False,
        **kwargs,
    ):
        df_full = get_df(symbol)
        if df_full is None:
            print(f"❌ No data for {symbol}")
            return

        df_full = df_full.copy()
        tail_index = tail if tail is not None else len(df_full)
        head_index = head if head is not None else 0

        warmup = max(sma1, sma2, sma3, ema1, ema2, ema3, aroon_len, rsi_len, hma_len, wr_len, atr_len)
        if show_tema:
            warmup = max(warmup, tema1 * 3, tema2 * 3, tema3 * 3)

        start_index = max(head_index, tail_index - plot_window - warmup)
        df_plot = df_full.iloc[start_index:tail_index].copy()

        # Coerce OHLCV to numeric and drop invalid rows to avoid mplfinance errors
        df_plot = _ensure_numeric_ohlcv(df_plot)
        if df_plot is None or df_plot.empty:
            print(f"❌ No valid numeric OHLCV data for {symbol}")
            return

        sma_periods = (sma1, sma2, sma3)
        df_plot = add_indicators_fn(df_plot, settings, ema_periods=(ema1, ema2, ema3), sma_periods=sma_periods, tema_periods=(tema1, tema2, tema3), atr_len=atr_len)
        
        # Remove any duplicate columns that may have been added
        df_plot = df_plot.loc[:, ~df_plot.columns.duplicated(keep="first")]

        if show_hma:
            df_plot["HMA"] = hma_fn(df_plot["Close"], hma_len)
        if show_rsi:
            df_plot["RSI"] = rsi_fn(df_plot["Close"], rsi_len)
        if show_aroon:
            df_plot["Aroon"] = aroon_fn(df_plot["High"], df_plot["Low"], aroon_len)
        if show_wr:
            df_plot["WilliamsR"] = williams_r_fn(df_plot["High"], df_plot["Low"], df_plot["Close"], wr_len)
        if show_atr and "ATR" in df_plot:
            df_plot["ATR_PCT"] = df_plot["ATR"] / df_plot["Close"]

        if show_cmo and "CMO" not in df_plot and cmo_len != 9:
            delta = df_plot["Close"].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            df_plot["CMO"] = 100 * (up.rolling(cmo_len).sum() - down.rolling(cmo_len).sum()) / (up.rolling(cmo_len).sum() + down.rolling(cmo_len).sum())

        # add_candle_patterns returns the complete dataframe with patterns already merged
        df_plot = candle_patterns_fn(df_plot, settings)
        # Ensure DataFrame is properly structured
        df_plot = df_plot.copy()
        df_plot.columns = df_plot.columns.astype(str)  # Ensure column names are strings
        
        # Ensure OHLCV columns remain numeric after concat (in case patterns affected them)
        ohlcv_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df_plot.columns]
        if ohlcv_cols:
            for col in ohlcv_cols:
                if col in df_plot.columns:
                    val = df_plot[col]
                    # Handle DataFrames
                    if isinstance(val, pd.DataFrame):
                        val = val.iloc[:, 0]
                    # Ensure it's a Series before applying to_numeric
                    if not isinstance(val, pd.Series):
                        val = pd.Series(val)
                    df_plot[col] = pd.to_numeric(val, errors="coerce")



        nan_sensitive = []
        if show_rsi:
            nan_sensitive.append("RSI")
        if show_aroon:
            nan_sensitive.append("Aroon")
        if show_wr:
            nan_sensitive.append("WilliamsR")
        if show_atr and "ATR_PCT" in df_plot:
            nan_sensitive.append("ATR_PCT")
        if show_cmo:
            nan_sensitive.append("CMO")

        if nan_sensitive:
            df_plot = df_plot.loc[df_plot[nan_sensitive].notna().all(axis=1)]

        df_visible = df_plot.tail(plot_window).copy()

        addplots = []
        panel_ratios = [4, 1.2]
        panel = 2

        if show_ema:
            for p, c in zip([ema1, ema2, ema3], ["#FFA07A", "#87CEFA", "#9370DB"]):
                if f"EMA_{p}" in df_visible:
                    addplots.append(mpf.make_addplot(df_visible[f"EMA_{p}"], color=c, width=2.2))

        if show_sma:
            colors = ["#32CD32", "#FF8C00", "#FF1493"]
            for p, c in zip(sma_periods, colors[:len(sma_periods)]):
                if f"SMA_{p}" in df_visible:
                    addplots.append(mpf.make_addplot(df_visible[f"SMA_{p}"], color=c, width=2.2))

        if show_tema:
            tema_periods = [tema1, tema2, tema3]
            tema_colors = ["#00CED1", "#1E90FF", "#9400D3"]
            for p, c in zip(tema_periods, tema_colors):
                col = f"TEMA_{p}"
                if col in df_visible:
                    series = pd.to_numeric(df_visible[col], errors="coerce")
                    addplots.append(mpf.make_addplot(series, panel=0, color=c, width=2.4, linestyle="-"))

        if show_hma:
            addplots.append(mpf.make_addplot(df_visible["HMA"], color="#00CED1", width=2.6))

        if show_supertrend:
            df_visible["Supertrend"] = supertrend_fn(df_visible)
            addplots.append(mpf.make_addplot(df_visible["Supertrend"], color="#FF1493", width=2.0))

        def add_indicator_panel(series, color, panel_idx, ylabel, ylim=None):
            addplots.append(
                mpf.make_addplot(series, panel=panel_idx, color=color, width=2.0, secondary_y=False, ylabel=ylabel, ylim=ylim)
            )
            panel_ratios.append(1.4)

        if show_aroon:
            add_indicator_panel(df_visible["Aroon"], "#90EE90", panel, "Aroon", ylim=(-100, 100))
            panel += 1
        if show_rsi:
            add_indicator_panel(df_visible["RSI"], "#FFD700", panel, "RSI", ylim=(0, 100))
            panel += 1
        if show_wr:
            add_indicator_panel(df_visible["WilliamsR"], "#ADFF2F", panel, "%R", ylim=(-100, 0))
            panel += 1
        if show_atr and "ATR_PCT" in df_visible:
            add_indicator_panel(df_visible["ATR_PCT"], "#FF4500", panel, "ATR %")
            panel += 1

        if show_cmo:
            add_indicator_panel(df_visible["CMO"], color="#2962FF", panel_idx=panel, ylabel="CMO", ylim=(-100, 100))
            panel_idx = panel
            panel += 1

            zero = pd.Series(0, index=df_visible.index)
            n40 = pd.Series(-40, index=df_visible.index)
            p40 = pd.Series(40, index=df_visible.index)

            for level in (zero, n40, p40):
                addplots.append(
                    mpf.make_addplot(level, panel=panel_idx, color="#787B86", linestyle="--", width=1.0, secondary_y=False)
                )

        candle_range = df_visible["High"] - df_visible["Low"]
        offset = candle_range * 4.0

        pattern_toggles = kwargs

        for name, cfg in PATTERN_PLOT_CONFIG.items():
            if not pattern_toggles.get(name, False) or name not in df_visible.columns:
                continue

            signal = df_visible[name]

            if signal.abs().sum() == 0:
                continue

            y = pd.Series(np.nan, index=df_visible.index)
            y.loc[signal > 0] = df_visible["Low"] - offset
            y.loc[signal < 0] = df_visible["High"] + offset

            marker_sizes = pd.Series(cfg["size"], index=df_visible.index)
            marker_sizes *= (signal.abs() / 100.0).clip(0.2, None)

            addplots.append(
                mpf.make_addplot(y, type="scatter", panel=0, marker=cfg["marker"], markersize=marker_sizes, color=cfg["color"])
            )

        # Final safety: ensure OHLCV columns are numeric and index is datetime
        ohlcv_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df_visible.columns]
        
        # Aggressively coerce OHLCV columns to numeric
        if ohlcv_cols:
            for col in ohlcv_cols:
                val = df_visible[col]
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[:, 0]
                if not isinstance(val, pd.Series):
                    val = pd.Series(val)
                # Convert to numeric and then explicitly cast to float
                df_visible[col] = pd.to_numeric(val, errors="coerce").astype(float)
            
            # Drop rows with any NaN in OHLCV
            df_visible = df_visible.dropna(subset=ohlcv_cols, how='any')

        # Ensure datetime index (mplfinance requires a DatetimeIndex)
        if not isinstance(df_visible.index, pd.DatetimeIndex):
            try:
                df_visible.index = pd.to_datetime(df_visible.index)
            except Exception as e:
                print(f"⚠️  WARNING: Could not convert index to datetime: {e}")
                pass

        if df_visible.empty:
            print(f"❌ No valid numeric OHLCV data to plot for {symbol}")
            return

        # Debug: Check for duplicate columns and non-numeric OHLCV data
        if len(df_visible.columns) != len(df_visible.columns.unique()):
            print(f"⚠️  WARNING: Duplicate columns detected: {df_visible.columns.tolist()}")
            df_visible = df_visible.loc[:, ~df_visible.columns.duplicated(keep='first')]
        
        # Final validation: ensure ALL OHLCV values are numeric
        for col in ohlcv_cols:
            if col in df_visible.columns:
                non_numeric = df_visible[col].apply(lambda x: not isinstance(x, (int, float, np.integer, np.floating)))
                if non_numeric.any():
                    print(f"⚠️  Found {non_numeric.sum()} non-numeric values in {col}, converting...")
                    df_visible[col] = pd.to_numeric(df_visible[col], errors="coerce")
        
        #print(f"Data types before plot:\n{df_visible[ohlcv_cols].dtypes}")
        #print(f"Sample data:\n{df_visible[ohlcv_cols].head()}")
        
        mpf.plot(
            df_visible,
            type="candle",
            addplot=addplots,
            volume=True,
            volume_panel=1,
            style="yahoo",
            figsize=(18, 2.4 + 2.2 * len(panel_ratios)),
            panel_ratios=tuple(panel_ratios),
            ylabel="Price",
        )





    def plot_recent_pattern(
        symbol,
        show_ema=False,
        show_sma=False,
        show_tema=False,
        show_aroon=False,
        show_rsi=False,
        show_atr=False,
        ema1=20,
        ema2=80,
        ema3=200,
        sma1=10,
        sma2=50,
        sma3=100,
        tema1=10,
        tema2=20,
        tema3=50,
        aroon_len=14,
        atr_len=14,
        show_cmo=True,
        cmo_len=9,
        lookback=1,
        window=100,
        wybrane_formacje=None,
        vol_confirmation=True,
        cmo_confirmation=True,
    ):
        df = get_df(symbol)
        if df is None:
            return

        df = df.tail(window + 500).copy()

        patterns = candle_patterns_fn(df, settings)
        df = pd.concat([df, patterns], axis=1).tail(window)

        required_cols = ["VOL_SIGNIFICANT", "DOWNTREND_SHORT"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return

        if not wybrane_formacje:
            return

        pattern_cols = [k for k, v in wybrane_formacje.items() if v]
        if not pattern_cols:
            return

        recent = df.tail(lookback)

        confirmed_patterns = []

        for col in pattern_cols:
            for idx in recent.index:
                signal = recent.at[idx, col]
                if signal == 0:
                    continue

                direction = "Bullish" if signal > 0 else "Bearish"

                # Safe access: handle Series values that may come from duplicate columns
                vol_val = recent.loc[idx, "VOL_SIGNIFICANT"]
                trend_val = recent.loc[idx, "DOWNTREND_SHORT"]
                
                if isinstance(vol_val, pd.Series):
                    vol_val = vol_val.iloc[0]
                if isinstance(trend_val, pd.Series):
                    trend_val = trend_val.iloc[0]
                
                vol_ok = vol_val == 1
                trend_ok = trend_val == 1

                if vol_ok and trend_ok and vol_confirmation and cmo_confirmation:
                    confirmed_patterns.append((idx, col, direction, signal))

                if vol_ok and vol_confirmation and not cmo_confirmation:
                    confirmed_patterns.append((idx, col, direction, signal))

                if trend_ok and cmo_confirmation and not vol_confirmation:
                    confirmed_patterns.append((idx, col, direction, signal))

        if not confirmed_patterns:
            return

        exchange = "NASDAQ" if symbol in NASDAQ_SYMBOLS else "NYSE"
        interval_prefix = "W" if settings.get("interval") == "1week" or (settings.get("week_start") != "BASE" and settings.get("interval") == "1d") else "D"

        print(f"\n📌 Symbol: {symbol}")
        print("Detected patterns:")
        
        tv_link = (
            "https://pl.tradingview.com/chart/TscShlXF/"
            f"?symbol={exchange}%3A{symbol}&interval=1{interval_prefix}"
        )
        print(f"🔗 TradingView: {tv_link}")

        for col in pattern_cols:
            mask = recent[col] != 0
            if mask.sum() == 0:
                continue

            for idx in recent.index[mask]:
                direction = "Bullish" if recent.at[idx, col] > 0 else "Bearish"
                strength = recent.at[idx, col]
                print(f"  • {col} ({direction}, strength={strength}) at {idx}")

        plot_chart(
            symbol=symbol,
            plot_window=window,
            show_ema=show_ema,
            ema1=ema1,
            ema2=ema2,
            ema3=ema3,
            show_sma=show_sma,
            sma1=sma1,
            sma2=sma2,
            sma3=sma3,
            show_aroon=show_aroon,
            aroon_len=aroon_len,
            show_rsi=show_rsi,
            show_atr=show_atr,
            atr_len=atr_len,
            show_cmo=show_cmo,
            cmo_len=cmo_len,
            **wybrane_formacje,
        )


    # ============== BUTTON CALLBACKS ==============

    plot_button = widgets.Button(description="📈 Pokaż wykres", button_style="info")
    plot_output = widgets.Output()

    def plot_selected_symbol(b):
        plot_output.clear_output()
        with plot_output:
            plot_chart(
                symbol=symbol_widget.value,
                plot_window=window_widget.value,
                show_ema=show_ema_widget.value,
                ema1=ema1_widget.value,
                ema2=ema2_widget.value,
                ema3=ema3_widget.value,
                show_sma=show_sma_widget.value,
                sma1=sma1_widget.value,
                sma2=sma2_widget.value,
                sma3=sma3_widget.value,
                show_tema=show_tema_widget.value,
                tema1=tema1_widget.value,
                tema2=tema2_widget.value,
                tema3=tema3_widget.value,
                show_aroon=show_aroon_widget.value,
                aroon_len=aroon_len_widget.value,
                show_rsi=show_rsi_widget.value,
                show_atr=show_atr_widget.value,
                show_cmo=show_cmo_widget.value,
                cmo_len=cmo_len_widget.value,
                **{k: w.value for k, w in pattern_widgets.items()},
            )

    plot_button.on_click(plot_selected_symbol)

    scan_button = widgets.Button(description="🔍 Skanuj rynek", button_style="primary")
    scan_output = widgets.Output()

    def run_scan(b):
        scan_output.clear_output()

        patterns = {}
        for group in [bullish_widgets, bearish_widgets, neutral_widgets]:
            patterns.update({k: w.value for k, w in group.items()})

        if not any(patterns.values()):
            with scan_output:
                print("❗ Wybierz przynajmniej jedną formację")
            return

        with scan_output:
            for symbol in tqdm(symbols, desc="🔍 Skanowanie rynku", bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}"):
                plot_recent_pattern(
                    symbol,
                    window=window_widget.value,
                    lookback=lookback_widget.value,
                    show_ema=show_ema_widget.value,
                    show_sma=show_sma_widget.value,
                    show_rsi=show_rsi_widget.value,
                    show_atr=show_atr_widget.value,
                    show_aroon=show_aroon_widget.value,
                    ema1=ema1_widget.value,
                    ema2=ema2_widget.value,
                    ema3=ema3_widget.value,
                    sma1=sma1_widget.value,
                    sma2=sma2_widget.value,
                    sma3=sma3_widget.value,
                    show_tema=show_tema_widget.value,
                    tema1=tema1_widget.value,
                    tema2=tema2_widget.value,
                    tema3=tema3_widget.value,
                    aroon_len=aroon_len_widget.value,
                    show_cmo=show_cmo_widget.value,
                    cmo_len=cmo_len_widget.value,
                    wybrane_formacje=patterns,
                )

    scan_button.on_click(run_scan)



    # ============== WIDGET DEFINITIONS ==============

    symbol_widget = widgets.Combobox(placeholder="Type to search...", options=symbols, description="Aktywo:", ensure_option=True, continuous_update=True)

    lookback_widget = widgets.IntSlider(value=1, min=1, max=15, step=1, description="Lookback")
    window_widget = widgets.IntSlider(value=100, min=20, max=500, step=5, description="Okres (plot)")

    show_ema_widget = widgets.Checkbox(False, description="EMA")
    ema1_widget = widgets.IntSlider(20, 2, 50, 1, description="EMA 1")
    ema2_widget = widgets.IntSlider(80, 2, 150, 1, description="EMA 2")
    ema3_widget = widgets.IntSlider(200, 2, 300, 1, description="EMA 3")

    show_sma_widget = widgets.Checkbox(False, description="SMA")
    sma1_widget = widgets.IntSlider(10, 2, 100, 1, description="SMA 1")
    sma2_widget = widgets.IntSlider(50, 2, 200, 1, description="SMA 2")
    sma3_widget = widgets.IntSlider(100, 2, 300, 1, description="SMA 3")

    show_rsi_widget = widgets.Checkbox(False, description="RSI")
    show_atr_widget = widgets.Checkbox(False, description="ATR")

    show_aroon_widget = widgets.Checkbox(False, description="Aroon")
    aroon_len_widget = widgets.IntSlider(14, 5, 50, 1, description="Aroon len")

    show_cmo_widget = widgets.Checkbox(False, description="CMO")
    cmo_len_widget = widgets.IntSlider(9, 1, 50, 1, description="CMO len")

    show_tema_widget = widgets.Checkbox(False, description="TEMA")
    tema1_widget = widgets.IntSlider(10, 2, 50, 1, description="TEMA 1")
    tema2_widget = widgets.IntSlider(20, 2, 100, 1, description="TEMA 2")
    tema3_widget = widgets.IntSlider(50, 2, 200, 1, description="TEMA 3")

    # ============== PATTERN WIDGETS ==============

    bullish_patterns = {k: v for k, v in PATTERN_PLOT_CONFIG.items() if v.get("side") == "low"}
    bearish_patterns = {k: v for k, v in PATTERN_PLOT_CONFIG.items() if v.get("side") == "high"}
    neutral_patterns = {k: v for k, v in PATTERN_PLOT_CONFIG.items() if v.get("side") == "mid"}

    bullish_widgets = {name: widgets.Checkbox(value=False, description=cfg["label"]) for name, cfg in bullish_patterns.items()}
    bearish_widgets = {name: widgets.Checkbox(value=False, description=cfg["label"]) for name, cfg in bearish_patterns.items()}
    neutral_widgets = {name: widgets.Checkbox(value=False, description=cfg["label"]) for name, cfg in neutral_patterns.items()}

    pattern_widgets = {
        **bullish_widgets,
        **bearish_widgets,
        **neutral_widgets,
    }

    def select_all(group_widgets):
        for w in group_widgets.values():
            w.value = True

    def select_none(group_widgets):
        for w in group_widgets.values():
            w.value = False

    bullish_all_btn = widgets.Button(description="Wszystkie")
    bullish_none_btn = widgets.Button(description="Żaden")
    bearish_all_btn = widgets.Button(description="Wszystkie")
    bearish_none_btn = widgets.Button(description="Żaden")
    neutral_all_btn = widgets.Button(description="Wszystkie")
    neutral_none_btn = widgets.Button(description="Żaden")

    bullish_all_btn.on_click(lambda b: select_all(bullish_widgets))
    bullish_none_btn.on_click(lambda b: select_none(bullish_widgets))
    bearish_all_btn.on_click(lambda b: select_all(bearish_widgets))
    bearish_none_btn.on_click(lambda b: select_none(bearish_widgets))
    neutral_all_btn.on_click(lambda b: select_all(neutral_widgets))
    neutral_none_btn.on_click(lambda b: select_none(neutral_widgets))

    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    ui_bullish = widgets.VBox(
        [widgets.Label("🟢 Bullish Patterns")]
        + [widgets.HBox(row) for row in chunk(list(bullish_widgets.values()), 4)]
        + [widgets.HBox([bullish_all_btn, bullish_none_btn])]
    )

    ui_bearish = widgets.VBox(
        [widgets.Label("🔴 Bearish Patterns")]
        + [widgets.HBox(row) for row in chunk(list(bearish_widgets.values()), 4)]
        + [widgets.HBox([bearish_all_btn, bearish_none_btn])]
    )

    ui_neutral = widgets.VBox(
        [widgets.Label("⚪ Neutral Patterns")]
        + [widgets.HBox(row) for row in chunk(list(neutral_widgets.values()), 4)]
        + [widgets.HBox([neutral_all_btn, neutral_none_btn])]
    )

    ui_patterns = widgets.VBox([ui_bullish, ui_bearish, ui_neutral])

    # ============== MAIN UI PANEL ==============

    ui = widgets.VBox(
        [
            widgets.Label("🎯 Symbol"),
            symbol_widget,
            widgets.Label("⏱ Zakres"),
            window_widget,
            lookback_widget,
            widgets.Label("📈 Wykres"),
            widgets.HBox([show_ema_widget, ema1_widget, ema2_widget, ema3_widget]),
            widgets.HBox([show_sma_widget, sma1_widget, sma2_widget, sma3_widget]),
            widgets.HBox([show_tema_widget, tema1_widget, tema2_widget, tema3_widget]),
            widgets.HBox([show_rsi_widget, show_atr_widget]),
            widgets.HBox([show_aroon_widget, aroon_len_widget]),
            widgets.HBox([show_cmo_widget, cmo_len_widget]),
            ui_patterns,
            plot_button,
            plot_output,
            widgets.Label("🔍 Skaner"),
            scan_button,
            scan_output,
        ]
    )

    return ui
