"""
Triple Barrier Method (TBM) Module for Financial Machine Learning

This module implements the Triple Barrier Method for labeling trading signals
with dynamic volatility-based barriers (Take Profit, Stop Loss, Time).

Features:
- Dynamic volatility calculation (rolling std or ATR)
- Asymmetric TP/SL barriers
- Numba-optimized barrier scanning (optional, falls back to pure Python)
- Full vectorization support for large datasets

Reference: "Advances in Financial Machine Learning" by Marcos López de Prado
"""

import numpy as np
import pandas as pd
from typing import Tuple, Literal, Optional, Union

# Optional numba import - falls back to pure Python if not available
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(func):
        return func


# =============================================================================
# VOLATILITY FUNCTIONS
# =============================================================================

def compute_daily_volatility(
    close: Union[pd.Series, np.ndarray],
    span: int = 20,
    min_periods: int = 5,
) -> pd.Series:
    """
    Compute rolling exponential volatility from log returns.
    
    This is the standard volatility measure used in TBM - it represents
    the expected daily price movement as a percentage.
    
    Args:
        close: Close prices
        span: EWM span for volatility (default 20 days)
        min_periods: Minimum periods before outputting values
        
    Returns:
        Series of volatility values (as decimals, not percentages)
        
    Example:
        >>> vol = compute_daily_volatility(df['Close'], span=20)
        >>> vol.iloc[-1]  # e.g., 0.025 means 2.5% daily volatility
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    
    # Log returns
    log_ret = np.log(close / close.shift(1))
    
    # Exponentially weighted standard deviation
    volatility = log_ret.ewm(span=span, min_periods=min_periods).std()
    
    return volatility


def compute_atr(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    period: int = 14,
) -> pd.Series:
    """
    Compute Average True Range (ATR) as volatility measure.
    
    ATR measures market volatility by decomposing the entire range of an asset 
    price for that period. Often preferred for intraday trading.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR lookback period (default 14)
        
    Returns:
        Series of ATR values (in price units, not percentage)
        
    Note:
        To use ATR as barrier multiplier, normalize by price:
        atr_pct = compute_atr(...) / close
    """
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    
    prev_close = close.shift(1)
    
    # True Range components
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    # True Range = max of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR = EMA of True Range
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return atr


def compute_atr_pct(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    period: int = 14,
) -> pd.Series:
    """
    Compute ATR as percentage of price.
    
    This normalized version can be directly used as barrier multiplier.
    
    Args:
        high, low, close: Price data
        period: ATR lookback period
        
    Returns:
        Series of ATR values as percentage (decimals)
    """
    atr = compute_atr(high, low, close, period)
    return atr / close


# =============================================================================
# NUMBA-OPTIMIZED BARRIER SCANNING WITH MFE/MAE
# =============================================================================

@njit
def _scan_barriers_with_mfe_mae(
    prices: np.ndarray,
    entry_idx: int,
    entry_price: float,
    direction: int,
    upper_barrier: float,
    lower_barrier: float,
    max_bars: int,
) -> Tuple[float, int, int, float, float]:
    """
    Numba-optimized barrier scanning with MFE/MAE tracking.
    
    Scans future prices to find which barrier is touched first,
    while tracking maximum favorable and adverse excursions.
    
    Args:
        prices: Array of close prices
        entry_idx: Index of signal entry
        entry_price: Price at entry
        direction: 1 for Long, -1 for Short
        upper_barrier: Take profit price level
        lower_barrier: Stop loss price level
        max_bars: Maximum bars to hold
        
    Returns:
        Tuple of (return, label, bars_held, mfe, mae)
        - return: percentage return achieved
        - label: 1 (TP), -1 (SL), 0 (time exit)
        - bars_held: number of bars until exit
        - mfe: Maximum Favorable Excursion (best unrealized gain)
        - mae: Maximum Adverse Excursion (worst unrealized loss)
    """
    n = len(prices)
    end_idx = min(entry_idx + max_bars, n - 1)
    
    mfe = 0.0  # Max favorable (positive = good)
    mae = 0.0  # Max adverse (positive = bad, represents drawdown)
    
    for i in range(entry_idx + 1, end_idx + 1):
        price = prices[i]
        
        # Calculate current return based on direction
        if direction == 1:  # Long
            current_ret = (price - entry_price) / entry_price
        else:  # Short
            current_ret = (entry_price - price) / entry_price
        
        # Track MFE (max gain) and MAE (max drawdown)
        if current_ret > mfe:
            mfe = current_ret
        if current_ret < mae:
            mae = current_ret
        
        # Check barriers
        if direction == 1:  # Long position
            if price >= upper_barrier:
                ret = (price - entry_price) / entry_price
                return ret, 1, i - entry_idx, mfe, -mae
            elif price <= lower_barrier:
                ret = (price - entry_price) / entry_price
                return ret, -1, i - entry_idx, mfe, -mae
        else:  # Short position
            if price <= lower_barrier:
                ret = (entry_price - price) / entry_price
                return ret, 1, i - entry_idx, mfe, -mae
            elif price >= upper_barrier:
                ret = (entry_price - price) / entry_price
                return ret, -1, i - entry_idx, mfe, -mae
    
    # Time barrier hit
    exit_price = prices[end_idx]
    if direction == 1:
        ret = (exit_price - entry_price) / entry_price
    else:
        ret = (entry_price - exit_price) / entry_price
    
    # MAE is returned as positive (drawdown magnitude)
    return ret, 0, end_idx - entry_idx, mfe, -mae


# =============================================================================
# MAIN TBM LABELING FUNCTION
# =============================================================================

def get_triple_barrier_labels(
    signals: pd.DataFrame,
    market_data: pd.DataFrame,
    volatility_type: Literal['daily', 'atr'] = 'daily',
    volatility_span: int = 20,
    pt_sl_ratio: Tuple[float, float] = (2.0, 1.0),
    max_holding_bars: int = 10,
    price_column: str = 'Close',
) -> pd.DataFrame:
    """
    Apply Triple Barrier Method to label trading signals.
    
    For each signal, this function:
    1. Computes volatility-scaled barriers
    2. Scans future prices (look-ahead)
    3. Determines which barrier was hit first
    4. Returns labeled outcomes
    
    Args:
        signals: DataFrame with columns:
            - 'timestamp' or index as datetime
            - 'signal_type': e.g., 'hammer', 'ma_cross', 'rsi_oversold'
            - 'direction': 1 (Long), -1 (Short)
            
        market_data: OHLCV DataFrame with columns:
            - 'Open', 'High', 'Low', 'Close', 'Volume'
            - DatetimeIndex
            
        volatility_type: 'daily' (log return std) or 'atr' (Average True Range)
        
        volatility_span: Lookback period for volatility calculation
        
        pt_sl_ratio: Tuple of (take_profit_mult, stop_loss_mult)
            - Example: (2.0, 1.0) means TP = 2x volatility, SL = 1x volatility
            
        max_holding_bars: Vertical barrier - maximum bars before forced exit
        
        price_column: Column name for close prices (default 'Close')
        
    Returns:
        DataFrame with columns:
            - 'signal_type': Original signal type
            - 'direction': Original direction
            - 'entry_price': Price at signal
            - 'entry_time': Timestamp of entry
            - 'volatility': Volatility at entry
            - 'upper_barrier': Take profit price level
            - 'lower_barrier': Stop loss price level
            - 'ret': Actual return achieved
            - 'label': 1 (TP hit), -1 (SL hit), 0 (time exit)
            - 'touch_time': Exit timestamp
            - 'bars_held': Number of bars held
            - 'barrier_hit': 'take_profit', 'stop_loss', or 'time'
            
    Example:
        >>> signals = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2024-01-10', '2024-01-15']),
        ...     'signal_type': ['hammer', 'ma_cross'],
        ...     'direction': [1, -1]
        ... }).set_index('timestamp')
        >>> labels = get_triple_barrier_labels(
        ...     signals, market_data,
        ...     pt_sl_ratio=(2.0, 1.0),
        ...     max_holding_bars=10
        ... )
    """
    # Validate inputs
    if signals.empty:
        return pd.DataFrame()
    
    market_data = market_data.copy()
    
    # Ensure market_data has required columns
    required_cols = ['High', 'Low', price_column]
    for col in required_cols:
        if col not in market_data.columns:
            raise ValueError(f"market_data must contain '{col}' column")
    
    # Compute volatility
    if volatility_type == 'daily':
        volatility = compute_daily_volatility(
            market_data[price_column], 
            span=volatility_span
        )
    elif volatility_type == 'atr':
        volatility = compute_atr_pct(
            market_data['High'],
            market_data['Low'],
            market_data[price_column],
            period=volatility_span
        )
    else:
        raise ValueError(f"volatility_type must be 'daily' or 'atr', got {volatility_type}")
    
    market_data['_volatility'] = volatility
    
    # Get prices as numpy array for numba
    prices = market_data[price_column].values
    
    # Prepare results storage
    results = []
    
    tp_mult, sl_mult = pt_sl_ratio
    
    # Handle timestamp column vs index
    if 'timestamp' in signals.columns:
        signal_times = signals['timestamp']
    else:
        signal_times = signals.index
    
    # Process each signal
    for i, (sig_idx, sig_row) in enumerate(signals.iterrows()):
        sig_time = signal_times.iloc[i] if hasattr(signal_times, 'iloc') else sig_time
        
        # Handle both index-based and column-based timestamps
        if 'timestamp' in signals.columns:
            sig_time = sig_row['timestamp']
        else:
            sig_time = sig_idx
        
        # Find signal in market data
        try:
            if sig_time in market_data.index:
                entry_idx = market_data.index.get_loc(sig_time)
            else:
                # Find nearest index
                entry_idx = market_data.index.searchsorted(sig_time)
                if entry_idx >= len(market_data):
                    continue
        except Exception:
            continue
        
        # Get entry values
        entry_price = prices[entry_idx]
        vol = market_data['_volatility'].iloc[entry_idx]
        
        if pd.isna(vol) or vol <= 0:
            continue
        
        direction = sig_row['direction']
        signal_type = sig_row.get('signal_type', 'unknown')
        
        # Calculate barrier levels
        if direction == 1:  # Long
            upper_barrier = entry_price * (1 + tp_mult * vol)
            lower_barrier = entry_price * (1 - sl_mult * vol)
        else:  # Short
            upper_barrier = entry_price * (1 + sl_mult * vol)  # SL for short
            lower_barrier = entry_price * (1 - tp_mult * vol)  # TP for short
        
        # Scan barriers with MFE/MAE tracking
        ret, label, bars_held, mfe, mae = _scan_barriers_with_mfe_mae(
            prices,
            entry_idx,
            entry_price,
            direction,
            upper_barrier,
            lower_barrier,
            max_holding_bars
        )
        
        # Determine exit time
        exit_idx = min(entry_idx + bars_held, len(market_data) - 1)
        touch_time = market_data.index[exit_idx]
        
        # Map label to barrier name
        barrier_names = {1: 'take_profit', -1: 'stop_loss', 0: 'time'}
        
        # Normalize MFE/MAE by volatility
        mfe_normalized = mfe / vol if vol > 0 else 0.0
        mae_normalized = mae / vol if vol > 0 else 0.0
        
        results.append({
            'signal_type': signal_type,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': sig_time,
            'volatility': vol,
            'upper_barrier': upper_barrier,
            'lower_barrier': lower_barrier,
            'ret': ret,
            'label': label,
            'touch_time': touch_time,
            'bars_held': bars_held,
            'barrier_hit': barrier_names[label],
            'mfe': mfe,
            'mae': mae,
            'mfe_normalized': mfe_normalized,
            'mae_normalized': mae_normalized,
        })
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        result_df.set_index('entry_time', inplace=True)
    
    return result_df


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def summarize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for TBM labels.
    
    Args:
        labels: Output from get_triple_barrier_labels()
        
    Returns:
        DataFrame with hit rates, avg returns by barrier type
    """
    if labels.empty:
        return pd.DataFrame()
    
    summary = labels.groupby('barrier_hit').agg({
        'ret': ['count', 'mean', 'std'],
        'bars_held': 'mean',
    })
    summary.columns = ['count', 'avg_return', 'std_return', 'avg_bars']
    summary['hit_rate'] = summary['count'] / summary['count'].sum()
    
    return summary


def summarize_by_signal_type(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics grouped by signal type.
    
    Args:
        labels: Output from get_triple_barrier_labels()
        
    Returns:
        DataFrame with win rate, avg return per signal type
    """
    if labels.empty:
        return pd.DataFrame()
    
    def calc_stats(group):
        return pd.Series({
            'total_signals': len(group),
            'win_rate': (group['label'] == 1).mean(),
            'loss_rate': (group['label'] == -1).mean(),
            'avg_return': group['ret'].mean(),
            'avg_bars_held': group['bars_held'].mean(),
        })
    
    return labels.groupby('signal_type').apply(calc_stats)


# =============================================================================
# MFE/MAE MICROSTRUCTURE ANALYSIS
# =============================================================================

def compute_edge_ratio(
    labels: pd.DataFrame,
    by_signal: bool = False,
) -> Union[float, pd.DataFrame]:
    """
    Compute Edge Ratio = Mean(MFE) / Mean(MAE).
    
    Edge Ratio measures the quality of trade paths - a higher ratio indicates
    signals that reach favorable prices before adverse prices, suggesting
    better entry timing.
    
    Args:
        labels: Output from get_triple_barrier_labels() with MFE/MAE columns
        by_signal: If True, compute Edge Ratio per signal type
        
    Returns:
        Single float if by_signal=False, otherwise DataFrame with per-signal stats
        
    Reference:
        Edge Ratio > 1.0 indicates favorable path quality
        Edge Ratio < 1.0 indicates poor path quality (hit MAE before MFE)
    """
    if labels.empty:
        return np.nan if not by_signal else pd.DataFrame()
    
    required_cols = ['mfe_normalized', 'mae_normalized']
    for col in required_cols:
        if col not in labels.columns:
            raise ValueError(f"Labels must contain '{col}' column. Run get_triple_barrier_labels() first.")
    
    if not by_signal:
        mean_mfe = labels['mfe_normalized'].mean()
        mean_mae = labels['mae_normalized'].mean()
        return mean_mfe / mean_mae if mean_mae > 0 else np.inf
    
    def calc_edge(group):
        mean_mfe = group['mfe_normalized'].mean()
        mean_mae = group['mae_normalized'].mean()
        edge = mean_mfe / mean_mae if mean_mae > 0 else np.inf
        return pd.Series({
            'count': len(group),
            'mean_mfe': mean_mfe,
            'mean_mae': mean_mae,
            'edge_ratio': edge,
            'win_rate': (group['label'] == 1).mean(),
        })
    
    return labels.groupby('signal_type').apply(calc_edge)


def plot_mfe_mae_scatter(
    labels: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    percentile_lines: list = [50, 75, 90],
    save_path: Optional[str] = None,
) -> None:
    """
    Create scatter plot of Normalized MFE vs Normalized MAE.
    
    This visualization helps identify optimal stop-loss levels by showing
    the distribution of adverse excursions relative to favorable excursions.
    
    Args:
        labels: Output from get_triple_barrier_labels() with MFE/MAE columns
        figsize: Figure size (width, height)
        percentile_lines: Percentiles for MAE decision lines
        save_path: If provided, save figure to this path
        
    Note:
        Points below the diagonal achieved more MFE than MAE (good entries).
        MAE percentile lines help determine "safe" stop-loss levels.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[ERROR] matplotlib required for visualization. Install with: pip install matplotlib")
        return
    
    if labels.empty:
        print("[WARNING] No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by outcome
    colors = {'take_profit': '#2ECC71', 'stop_loss': '#E74C3C', 'time': '#95A5A6'}
    markers = {'take_profit': 'o', 'stop_loss': 'x', 'time': 's'}
    
    for barrier_type in ['take_profit', 'stop_loss', 'time']:
        subset = labels[labels['barrier_hit'] == barrier_type]
        if not subset.empty:
            ax.scatter(
                subset['mae_normalized'],
                subset['mfe_normalized'],
                c=colors.get(barrier_type, 'gray'),
                marker=markers.get(barrier_type, 'o'),
                alpha=0.6,
                s=50,
                label=f"{barrier_type} (n={len(subset)})"
            )
    
    # Add diagonal (MFE = MAE line)
    max_val = max(labels['mfe_normalized'].max(), labels['mae_normalized'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='MFE = MAE')
    
    # Add MAE percentile lines
    for pct in percentile_lines:
        mae_pct = labels['mae_normalized'].quantile(pct / 100)
        ax.axvline(x=mae_pct, linestyle=':', alpha=0.5, 
                   label=f'{pct}% MAE = {mae_pct:.2f}x vol')
    
    ax.set_xlabel('Normalized MAE (multiples of volatility)', fontsize=12)
    ax.set_ylabel('Normalized MFE (multiples of volatility)', fontsize=12)
    ax.set_title('MFE vs MAE - Trade Path Quality Analysis', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Figure saved to: {save_path}")
    
    plt.show()


def suggest_optimal_stoploss(
    labels: pd.DataFrame,
    percentile: float = 90,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Suggest tightest safe stop-loss for each signal type based on MAE distribution.
    
    Analyzes the Maximum Adverse Excursion distribution for winning trades
    and suggests a stop-loss level that would have preserved X% of them.
    
    Args:
        labels: Output from get_triple_barrier_labels() with MFE/MAE columns
        percentile: Percentile of winning trades to preserve (e.g., 90 = 90%)
        min_samples: Minimum samples required per signal type
        
    Returns:
        DataFrame with columns:
        - signal_type: Signal name
        - suggested_sl: Suggested SL in volatility units
        - coverage: Percentage of winners preserved at this SL
        - total_winners: Total winning trades analyzed
        - avg_mfe: Average MFE for context
        - edge_ratio: MFE/MAE ratio for signal quality
        
    Example:
        >>> suggestions = suggest_optimal_stoploss(labels, percentile=90)
        >>> print(suggestions)
        # For 'hammer' signals, SL = 1.2x volatility preserves 90% of winners
    """
    if labels.empty:
        return pd.DataFrame()
    
    required_cols = ['mfe_normalized', 'mae_normalized', 'label', 'signal_type']
    for col in required_cols:
        if col not in labels.columns:
            raise ValueError(f"Labels must contain '{col}' column")
    
    results = []
    
    for signal_type in labels['signal_type'].unique():
        subset = labels[labels['signal_type'] == signal_type]
        
        # Focus on winning trades (label == 1) to find their MAE distribution
        winners = subset[subset['label'] == 1]
        
        if len(winners) < min_samples:
            continue
        
        # The percentile of MAE tells us the "tightest safe SL"
        suggested_sl = winners['mae_normalized'].quantile(percentile / 100)
        
        # Also compute for all trades in the signal
        mean_mfe = subset['mfe_normalized'].mean()
        mean_mae = subset['mae_normalized'].mean()
        edge = mean_mfe / mean_mae if mean_mae > 0 else np.inf
        
        results.append({
            'signal_type': signal_type,
            'suggested_sl': suggested_sl,
            'coverage': percentile,
            'total_winners': len(winners),
            'total_signals': len(subset),
            'win_rate': len(winners) / len(subset),
            'avg_mfe': mean_mfe,
            'avg_mae': mean_mae,
            'edge_ratio': edge,
        })
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('edge_ratio', ascending=False)
    
    return result_df


def analyze_mfe_mae_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detailed MFE/MAE distribution statistics.
    
    Args:
        labels: Output from get_triple_barrier_labels()
        
    Returns:
        DataFrame with percentiles and statistics for MFE and MAE
    """
    if labels.empty:
        return pd.DataFrame()
    
    stats = {}
    for col in ['mfe_normalized', 'mae_normalized']:
        if col not in labels.columns:
            continue
        stats[col] = {
            'mean': labels[col].mean(),
            'std': labels[col].std(),
            'min': labels[col].min(),
            '25%': labels[col].quantile(0.25),
            '50%': labels[col].quantile(0.50),
            '75%': labels[col].quantile(0.75),
            '90%': labels[col].quantile(0.90),
            '95%': labels[col].quantile(0.95),
            'max': labels[col].max(),
        }
    
    return pd.DataFrame(stats)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import datetime
    
    print("=" * 60)
    print("Triple Barrier Method - Example Usage with MFE/MAE")
    print("=" * 60)
    
    if not NUMBA_AVAILABLE:
        print("[INFO] Numba not installed - using pure Python fallback")
    else:
        print("[INFO] Numba available - using JIT acceleration")
    
    # Generate synthetic market data
    np.random.seed(42)
    n_bars = 500
    
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='D')
    
    # Random walk price
    returns = np.random.randn(n_bars) * 0.02  # 2% daily volatility
    price = 100 * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'Open': price * (1 + np.random.randn(n_bars) * 0.005),
        'High': price * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
        'Low': price * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
        'Close': price,
        'Volume': np.random.randint(1000000, 5000000, n_bars),
    }, index=dates)
    
    # Ensure High >= Open, Close and Low <= Open, Close
    market_data['High'] = market_data[['Open', 'High', 'Close']].max(axis=1)
    market_data['Low'] = market_data[['Open', 'Low', 'Close']].min(axis=1)
    
    print(f"\n[DATA] Market Data: {len(market_data)} bars")
    
    # Generate sample signals
    signal_dates = np.random.choice(dates[50:450], size=50, replace=False)
    signal_dates = pd.to_datetime(sorted(signal_dates))
    
    signals = pd.DataFrame({
        'timestamp': signal_dates,
        'signal_type': np.random.choice(['hammer', 'ma_cross', 'rsi_oversold'], size=50),
        'direction': np.random.choice([1, -1], size=50),
    })
    
    print(f"[SIGNALS] Signals: {len(signals)} total")
    
    # Apply Triple Barrier Method with MFE/MAE
    print("\n[TBM] Applying Triple Barrier Method...")
    labels = get_triple_barrier_labels(
        signals,
        market_data,
        volatility_type='daily',
        volatility_span=20,
        pt_sl_ratio=(2.0, 1.0),
        max_holding_bars=10,
    )
    
    print(f"[LABELS] Labels: {len(labels)} labeled signals")
    
    # Show MFE/MAE columns
    print("\n" + "=" * 60)
    print("[MFE/MAE] Sample Data with Microstructure Metrics")
    print("=" * 60)
    print(labels[['signal_type', 'ret', 'label', 'mfe', 'mae', 'mfe_normalized', 'mae_normalized']].head(10))
    
    # Edge Ratio Analysis
    print("\n" + "=" * 60)
    print("[EDGE RATIO] Overall and By Signal Type")
    print("=" * 60)
    overall_edge = compute_edge_ratio(labels, by_signal=False)
    print(f"Overall Edge Ratio: {overall_edge:.3f}")
    print("\nBy Signal Type:")
    print(compute_edge_ratio(labels, by_signal=True))
    
    # MFE/MAE Distribution
    print("\n" + "=" * 60)
    print("[DISTRIBUTION] MFE/MAE Statistics")
    print("=" * 60)
    print(analyze_mfe_mae_distribution(labels))
    
    # Optimal Stop-Loss Suggestions
    print("\n" + "=" * 60)
    print("[SUGGESTIONS] Optimal Stop-Loss (90th percentile)")
    print("=" * 60)
    suggestions = suggest_optimal_stoploss(labels, percentile=90)
    if not suggestions.empty:
        print(suggestions[['signal_type', 'suggested_sl', 'win_rate', 'edge_ratio']])
    else:
        print("Not enough data for suggestions")
    
    print("\n[OK] Module ready for integration with kulturkaaa.ipynb!")
    print("\nTo visualize MFE/MAE scatter plot, call:")
    print("  plot_mfe_mae_scatter(labels, save_path='mfe_mae_analysis.png')")


