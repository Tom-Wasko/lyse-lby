"""
Monte Carlo Permutation Test for Strategy Robustness Verification

This module implements a permutation test following Timothy Masters' methodology
to verify that trading strategy results are statistically significant and not
the result of overfitting or random chance.

Algorithm:
1. Take returns from the market data
2. Shuffle returns randomly (destroy serial correlation, preserve distribution)
3. Reconstruct price series from shuffled returns
4. Run strategy on shuffled data N times (default 1000)
5. Calculate p-value: proportion of random runs >= actual result
6. If p-value > 0.05, the strategy is likely overfitted

Reference: Timothy Masters - "Testing and Tuning Market Trading Systems"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Callable, Dict, List
from dataclasses import dataclass

# Import from our modules
from triple_barrier import get_triple_barrier_labels


# =============================================================================
# DATA SHUFFLING
# =============================================================================

def shuffle_returns(market_data: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Shuffle market returns to create synthetic price series.
    
    This destroys serial correlation and technical patterns while
    preserving the statistical distribution of returns.
    
    Args:
        market_data: Original OHLCV DataFrame
        seed: Random seed for reproducibility
        
    Returns:
        New DataFrame with shuffled price series
    """
    if seed is not None:
        np.random.seed(seed)
    
    df = market_data.copy()
    
    # Calculate log returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    
    # Shuffle the returns
    shuffled_returns = log_returns.sample(frac=1.0).values
    
    # Reconstruct prices from shuffled returns
    initial_price = df['Close'].iloc[0]
    new_close = initial_price * np.exp(np.concatenate([[0], np.cumsum(shuffled_returns)]))
    
    # Create new OHLCV maintaining relative relationships
    close_ratio = new_close / df['Close'].values
    
    new_df = pd.DataFrame({
        'Open': df['Open'].values * close_ratio,
        'High': df['High'].values * close_ratio,
        'Low': df['Low'].values * close_ratio,
        'Close': new_close,
        'Volume': df['Volume'].values,  # Keep volume unchanged
    }, index=df.index)
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    new_df['High'] = new_df[['Open', 'High', 'Close']].max(axis=1)
    new_df['Low'] = new_df[['Open', 'Low', 'Close']].min(axis=1)
    
    return new_df


# =============================================================================
# PERMUTATION TEST RESULT
# =============================================================================

@dataclass
class PermutationTestResult:
    """Container for permutation test results."""
    actual_score: float
    random_scores: np.ndarray
    p_value: float
    n_permutations: int
    objective: str
    is_significant: bool
    interpretation: str
    
    def __repr__(self):
        return (f"PermutationTest(actual={self.actual_score:.4f}, "
                f"p_value={self.p_value:.4f}, significant={self.is_significant})")


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def calc_sharpe(labels: pd.DataFrame) -> float:
    """Calculate Sharpe Ratio from labels."""
    if labels.empty or len(labels) < 2:
        return 0.0
    returns = labels['ret'].values
    if np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)


def calc_total_return(labels: pd.DataFrame) -> float:
    """Calculate total return from labels."""
    if labels.empty:
        return 0.0
    return labels['ret'].sum()


def calc_profit_factor(labels: pd.DataFrame) -> float:
    """Calculate profit factor from labels."""
    if labels.empty:
        return 0.0
    returns = labels['ret'].values
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calc_edge_ratio(labels: pd.DataFrame) -> float:
    """Calculate edge ratio from labels."""
    if labels.empty or 'mfe_normalized' not in labels.columns:
        return 0.0
    mean_mfe = labels['mfe_normalized'].mean()
    mean_mae = labels['mae_normalized'].mean()
    if mean_mae <= 0:
        return np.inf if mean_mfe > 0 else 0.0
    return mean_mfe / mean_mae


OBJECTIVE_FUNCTIONS = {
    'sharpe': calc_sharpe,
    'total_return': calc_total_return,
    'profit_factor': calc_profit_factor,
    'edge_ratio': calc_edge_ratio,
}


# =============================================================================
# MAIN PERMUTATION TEST
# =============================================================================

def run_permutation_test(
    signals: pd.DataFrame,
    market_data: pd.DataFrame,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_bars: int = 10,
    objective: str = 'sharpe',
    n_permutations: int = 1000,
    alpha: float = 0.05,
    volatility_type: str = 'daily',
    volatility_span: int = 20,
    verbose: bool = True,
) -> PermutationTestResult:
    """
    Run Monte Carlo permutation test on a trading strategy.
    
    This test answers: "Could a random shuffle of returns produce
    the same or better results than my strategy?"
    
    Args:
        signals: Trading signals DataFrame
        market_data: OHLCV DataFrame
        tp_mult: Take profit multiplier
        sl_mult: Stop loss multiplier
        max_bars: Maximum holding period
        objective: Metric to evaluate ('sharpe', 'total_return', etc.)
        n_permutations: Number of Monte Carlo iterations (default 1000)
        alpha: Significance level (default 0.05)
        volatility_type: Volatility calculation method
        volatility_span: Volatility lookback period
        verbose: Print progress
        
    Returns:
        PermutationTestResult with p-value and interpretation
        
    Example:
        >>> result = run_permutation_test(signals, market_data, n_permutations=1000)
        >>> print(f"p-value: {result.p_value:.4f}")
        >>> if result.is_significant:
        ...     print("Strategy is statistically significant!")
    """
    if objective not in OBJECTIVE_FUNCTIONS:
        raise ValueError(f"Unknown objective: {objective}")
    
    obj_func = OBJECTIVE_FUNCTIONS[objective]
    
    if verbose:
        print(f"[PERMUTATION TEST] Starting with {n_permutations} iterations...")
        print(f"  Objective: {objective}")
        print(f"  Parameters: TP={tp_mult}x, SL={sl_mult}x, max_bars={max_bars}")
    
    # Step 1: Calculate actual strategy performance
    if verbose:
        print("\n[STEP 1] Evaluating actual strategy...")
    
    actual_labels = get_triple_barrier_labels(
        signals,
        market_data,
        volatility_type=volatility_type,
        volatility_span=volatility_span,
        pt_sl_ratio=(tp_mult, sl_mult),
        max_holding_bars=max_bars,
    )
    
    actual_score = obj_func(actual_labels)
    
    if verbose:
        print(f"  Actual {objective}: {actual_score:.4f}")
    
    # Step 2: Run permutations
    if verbose:
        print(f"\n[STEP 2] Running {n_permutations} permutations...")
    
    random_scores = []
    
    for i in range(n_permutations):
        # Shuffle returns
        shuffled_data = shuffle_returns(market_data, seed=i)
        
        # Run strategy on shuffled data
        shuffled_labels = get_triple_barrier_labels(
            signals,
            shuffled_data,
            volatility_type=volatility_type,
            volatility_span=volatility_span,
            pt_sl_ratio=(tp_mult, sl_mult),
            max_holding_bars=max_bars,
        )
        
        score = obj_func(shuffled_labels)
        random_scores.append(score)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{n_permutations}")
    
    random_scores = np.array(random_scores)
    
    # Step 3: Calculate p-value
    # p-value = proportion of random results >= actual result
    p_value = (random_scores >= actual_score).mean()
    
    is_significant = p_value < alpha
    
    # Generate interpretation
    if p_value < 0.01:
        interpretation = (
            f"HIGHLY SIGNIFICANT (p={p_value:.4f}). "
            f"Less than 1% of random shuffles matched your strategy. "
            f"Strategy appears to have genuine predictive power."
        )
    elif p_value < 0.05:
        interpretation = (
            f"SIGNIFICANT (p={p_value:.4f}). "
            f"Less than 5% of random shuffles matched your strategy. "
            f"Strategy shows statistical significance."
        )
    elif p_value < 0.10:
        interpretation = (
            f"MARGINALLY SIGNIFICANT (p={p_value:.4f}). "
            f"Result is borderline. Consider more data or stricter parameters."
        )
    else:
        interpretation = (
            f"NOT SIGNIFICANT (p={p_value:.4f}). "
            f"{p_value*100:.1f}% of random shuffles matched or exceeded your result. "
            f"Strategy may be OVERFITTED. Consider rejecting these parameters."
        )
    
    if verbose:
        print(f"\n[STEP 3] Calculating p-value...")
        print(f"  Random scores: mean={random_scores.mean():.4f}, std={random_scores.std():.4f}")
        print(f"  Actual score: {actual_score:.4f}")
        print(f"  P-VALUE: {p_value:.4f}")
        print(f"\n[RESULT] {interpretation}")
    
    return PermutationTestResult(
        actual_score=actual_score,
        random_scores=random_scores,
        p_value=p_value,
        n_permutations=n_permutations,
        objective=objective,
        is_significant=is_significant,
        interpretation=interpretation,
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_permutation_results(
    result: PermutationTestResult,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot histogram of random results with actual strategy marked.
    
    Args:
        result: PermutationTestResult from run_permutation_test()
        figsize: Figure size
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib required. Install with: pip install matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram of random scores
    n, bins, patches = ax.hist(
        result.random_scores,
        bins=50,
        color='#3498DB',
        alpha=0.7,
        edgecolor='black',
        label=f'Random shuffles (n={result.n_permutations})'
    )
    
    # Mark actual score
    ax.axvline(
        result.actual_score,
        color='#E74C3C',
        linewidth=3,
        linestyle='--',
        label=f'Actual strategy ({result.objective}={result.actual_score:.3f})'
    )
    
    # Add p-value annotation
    significance_color = '#27AE60' if result.is_significant else '#E74C3C'
    significance_text = 'SIGNIFICANT' if result.is_significant else 'NOT SIGNIFICANT'
    
    ax.axvline(
        result.random_scores.mean(),
        color='gray',
        linewidth=2,
        linestyle=':',
        label=f'Random mean ({result.random_scores.mean():.3f})'
    )
    
    # Text box with results
    textstr = (
        f'p-value: {result.p_value:.4f}\n'
        f'Status: {significance_text}\n'
        f'Random >= Actual: {(result.random_scores >= result.actual_score).sum()}/{result.n_permutations}'
    )
    
    props = dict(boxstyle='round', facecolor=significance_color, alpha=0.3)
    ax.text(
        0.95, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
    )
    
    ax.set_xlabel(f'{result.objective.upper()} Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(
        f'Permutation Test: Strategy vs Random (n={result.n_permutations})',
        fontsize=14
    )
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Histogram saved to: {save_path}")
    
    plt.show()


# =============================================================================
# BATCH TESTING
# =============================================================================

def test_multiple_objectives(
    signals: pd.DataFrame,
    market_data: pd.DataFrame,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_bars: int = 10,
    n_permutations: int = 500,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run permutation test for multiple objective functions.
    
    Args:
        See run_permutation_test for parameter descriptions
        
    Returns:
        DataFrame with p-values for each objective
    """
    results = []
    
    for objective in OBJECTIVE_FUNCTIONS.keys():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing: {objective}")
            print('='*60)
        
        result = run_permutation_test(
            signals, market_data,
            tp_mult=tp_mult, sl_mult=sl_mult, max_bars=max_bars,
            objective=objective, n_permutations=n_permutations,
            verbose=verbose
        )
        
        results.append({
            'objective': objective,
            'actual_score': result.actual_score,
            'random_mean': result.random_scores.mean(),
            'random_std': result.random_scores.std(),
            'p_value': result.p_value,
            'significant': result.is_significant,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Monte Carlo Permutation Test - Example")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_bars = 500
    
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='D')
    returns = np.random.randn(n_bars) * 0.02
    price = 100 * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'Open': price * (1 + np.random.randn(n_bars) * 0.005),
        'High': price * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
        'Low': price * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
        'Close': price,
        'Volume': np.random.randint(1000000, 5000000, n_bars),
    }, index=dates)
    
    market_data['High'] = market_data[['Open', 'High', 'Close']].max(axis=1)
    market_data['Low'] = market_data[['Open', 'Low', 'Close']].min(axis=1)
    
    # Generate signals
    signal_dates = np.random.choice(dates[50:450], size=30, replace=False)
    signal_dates = pd.to_datetime(sorted(signal_dates))
    
    signals = pd.DataFrame({
        'timestamp': signal_dates,
        'signal_type': np.random.choice(['hammer', 'ma_cross'], size=30),
        'direction': np.random.choice([1, -1], size=30),
    })
    
    print(f"\n[DATA] {len(market_data)} bars, {len(signals)} signals")
    
    # Run permutation test (reduced for demo)
    result = run_permutation_test(
        signals, market_data,
        tp_mult=2.0, sl_mult=1.0, max_bars=10,
        objective='sharpe',
        n_permutations=100,  # Use 1000 in production
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("[FINAL RESULT]")
    print("=" * 60)
    print(f"Actual Sharpe: {result.actual_score:.4f}")
    print(f"P-Value: {result.p_value:.4f}")
    print(f"Significant: {result.is_significant}")
    print(f"\n{result.interpretation}")
    
    print("\n[OK] To visualize, call:")
    print("  plot_permutation_results(result, save_path='permutation_test.png')")
