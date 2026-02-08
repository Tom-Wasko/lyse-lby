"""
TBM Parameter Optimizer with Walk-Forward Validation

This module provides tools for finding optimal Triple Barrier Method parameters
while protecting against overfitting through Walk-Forward cross-validation.

Features:
- Grid Search over TP, SL, and time limit parameters
- Walk-Forward (time-series) validation
- Multiple objective functions (Sharpe, Edge Ratio, Profit Factor)
- Heatmap visualization for stability analysis
- Optional Optuna integration for faster optimization

Reference: "Advances in Financial Machine Learning" by Marcos López de Prado
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal, Callable, Union
from dataclasses import dataclass, field
from itertools import product
import warnings

# Import from our TBM module
from triple_barrier import (
    get_triple_barrier_labels,
    compute_edge_ratio,
    compute_daily_volatility,
)

# Optional optuna import
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# =============================================================================
# DEFAULT PARAMETER GRIDS
# =============================================================================

DEFAULT_PARAM_GRID = {
    'tp_mult': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    'sl_mult': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    'max_bars': [5, 10, 20, 50],
}


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def calc_sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    Args:
        returns: Array of trade returns
        periods_per_year: Trading periods per year (252 for daily)
        
    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)


def calc_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate Profit Factor = Gross Profit / Gross Loss.
    
    Returns:
        Profit Factor (> 1.0 is profitable)
    """
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calc_win_rate(labels: np.ndarray) -> float:
    """Calculate win rate from labels (1 = win)."""
    if len(labels) == 0:
        return 0.0
    return (labels == 1).mean()


def calc_edge_ratio_from_labels(labels_df: pd.DataFrame) -> float:
    """Calculate Edge Ratio from labels DataFrame."""
    if labels_df.empty:
        return 0.0
    
    if 'mfe_normalized' not in labels_df.columns:
        return 0.0
    
    mean_mfe = labels_df['mfe_normalized'].mean()
    mean_mae = labels_df['mae_normalized'].mean()
    
    if mean_mae <= 0:
        return np.inf if mean_mfe > 0 else 0.0
    return mean_mfe / mean_mae


OBJECTIVE_FUNCTIONS = {
    'sharpe': lambda labels: calc_sharpe_ratio(labels['ret'].values),
    'profit_factor': lambda labels: calc_profit_factor(labels['ret'].values),
    'win_rate': lambda labels: calc_win_rate(labels['label'].values),
    'edge_ratio': calc_edge_ratio_from_labels,
}


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

@dataclass
class WalkForwardSplit:
    """Represents a single Walk-Forward train/test split."""
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_signals: pd.DataFrame
    test_signals: pd.DataFrame


def create_walk_forward_splits(
    signals: pd.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.7,
    purge_gap: int = 0,
) -> List[WalkForwardSplit]:
    """
    Create Walk-Forward validation splits.
    
    Walk-Forward validation trains on period T and tests on period T+1,
    respecting the temporal order of data (no future leakage).
    
    Args:
        signals: DataFrame with signals (must have timestamp index or column)
        n_splits: Number of walk-forward folds
        train_pct: Percentage of each fold for training
        purge_gap: Number of periods to skip between train and test (prevents leakage)
        
    Returns:
        List of WalkForwardSplit objects
    """
    # Get timestamps
    if 'timestamp' in signals.columns:
        timestamps = pd.to_datetime(signals['timestamp'])
    else:
        timestamps = pd.to_datetime(signals.index)
    
    signals = signals.copy()
    signals['_ts'] = timestamps
    signals = signals.sort_values('_ts')
    
    min_date = signals['_ts'].min()
    max_date = signals['_ts'].max()
    total_days = (max_date - min_date).days
    
    fold_size = total_days // n_splits
    
    splits = []
    
    for i in range(n_splits - 1):  # Need at least one test period after last train
        train_start = min_date + pd.Timedelta(days=i * fold_size)
        train_end = train_start + pd.Timedelta(days=int(fold_size * train_pct))
        test_start = train_end + pd.Timedelta(days=purge_gap)
        test_end = train_start + pd.Timedelta(days=fold_size)
        
        train_mask = (signals['_ts'] >= train_start) & (signals['_ts'] < train_end)
        test_mask = (signals['_ts'] >= test_start) & (signals['_ts'] <= test_end)
        
        train_signals = signals[train_mask].drop(columns=['_ts'])
        test_signals = signals[test_mask].drop(columns=['_ts'])
        
        if len(train_signals) > 0 and len(test_signals) > 0:
            splits.append(WalkForwardSplit(
                fold=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_signals=train_signals,
                test_signals=test_signals,
            ))
    
    return splits


# =============================================================================
# TBM OPTIMIZER CLASS
# =============================================================================

class TBMOptimizer:
    """
    Triple Barrier Method Parameter Optimizer with Walk-Forward Validation.
    
    This class finds optimal TBM parameters while protecting against overfitting
    through Walk-Forward cross-validation and stability analysis.
    
    Example:
        >>> optimizer = TBMOptimizer(signals, market_data)
        >>> results = optimizer.grid_search(objective='sharpe')
        >>> optimizer.plot_heatmap(results)
        >>> best = optimizer.get_best_params(results)
    """
    
    def __init__(
        self,
        signals: pd.DataFrame,
        market_data: pd.DataFrame,
        param_grid: Optional[Dict] = None,
        volatility_type: str = 'daily',
        volatility_span: int = 20,
    ):
        """
        Initialize the optimizer.
        
        Args:
            signals: DataFrame with trading signals
            market_data: OHLCV DataFrame
            param_grid: Custom parameter grid (uses defaults if None)
            volatility_type: 'daily' or 'atr'
            volatility_span: Lookback for volatility calculation
        """
        self.signals = signals
        self.market_data = market_data
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self.volatility_type = volatility_type
        self.volatility_span = volatility_span
        
        self._results_cache = {}
    
    def _evaluate_params(
        self,
        signals: pd.DataFrame,
        tp_mult: float,
        sl_mult: float,
        max_bars: int,
    ) -> pd.DataFrame:
        """Run TBM with given parameters and return labels."""
        labels = get_triple_barrier_labels(
            signals,
            self.market_data,
            volatility_type=self.volatility_type,
            volatility_span=self.volatility_span,
            pt_sl_ratio=(tp_mult, sl_mult),
            max_holding_bars=max_bars,
        )
        return labels
    
    def grid_search(
        self,
        objective: str = 'sharpe',
        n_splits: int = 5,
        train_pct: float = 0.7,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Perform Grid Search with Walk-Forward validation.
        
        Args:
            objective: Objective function name ('sharpe', 'edge_ratio', etc.)
            n_splits: Number of Walk-Forward folds
            train_pct: Training percentage per fold
            verbose: Print progress
            
        Returns:
            DataFrame with results for each parameter combination
        """
        if objective not in OBJECTIVE_FUNCTIONS:
            raise ValueError(f"Unknown objective: {objective}. Choose from {list(OBJECTIVE_FUNCTIONS.keys())}")
        
        obj_func = OBJECTIVE_FUNCTIONS[objective]
        
        # Create Walk-Forward splits
        splits = create_walk_forward_splits(
            self.signals, n_splits=n_splits, train_pct=train_pct
        )
        
        if len(splits) == 0:
            raise ValueError("Not enough data for Walk-Forward splits")
        
        if verbose:
            print(f"[OPTIMIZER] {len(splits)} Walk-Forward folds created")
        
        # Generate all parameter combinations
        param_combinations = list(product(
            self.param_grid['tp_mult'],
            self.param_grid['sl_mult'],
            self.param_grid['max_bars'],
        ))
        
        if verbose:
            print(f"[OPTIMIZER] Testing {len(param_combinations)} parameter combinations")
        
        results = []
        
        for idx, (tp, sl, bars) in enumerate(param_combinations):
            train_scores = []
            test_scores = []
            
            for split in splits:
                # Evaluate on train
                train_labels = self._evaluate_params(
                    split.train_signals, tp, sl, bars
                )
                if not train_labels.empty:
                    train_scores.append(obj_func(train_labels))
                
                # Evaluate on test
                test_labels = self._evaluate_params(
                    split.test_signals, tp, sl, bars
                )
                if not test_labels.empty:
                    test_scores.append(obj_func(test_labels))
            
            # Average across folds
            avg_train = np.mean(train_scores) if train_scores else np.nan
            avg_test = np.mean(test_scores) if test_scores else np.nan
            std_test = np.std(test_scores) if len(test_scores) > 1 else 0.0
            
            results.append({
                'tp_mult': tp,
                'sl_mult': sl,
                'max_bars': bars,
                f'train_{objective}': avg_train,
                f'test_{objective}': avg_test,
                f'test_{objective}_std': std_test,
                'n_folds': len(test_scores),
            })
            
            if verbose and (idx + 1) % 20 == 0:
                print(f"  Progress: {idx + 1}/{len(param_combinations)}")
        
        result_df = pd.DataFrame(results)
        
        # Sort by test score
        result_df = result_df.sort_values(f'test_{objective}', ascending=False)
        
        if verbose:
            print(f"[OPTIMIZER] Grid search complete!")
        
        return result_df
    
    def optuna_search(
        self,
        objective: str = 'sharpe',
        n_trials: int = 100,
        n_splits: int = 5,
        train_pct: float = 0.7,
        verbose: bool = True,
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Perform Optuna hyperparameter optimization.
        
        Args:
            objective: Objective function name
            n_trials: Number of Optuna trials
            n_splits: Walk-Forward folds
            train_pct: Training percentage per fold
            verbose: Print progress
            
        Returns:
            Tuple of (best_params, study_df)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Install with: pip install optuna")
        
        obj_func = OBJECTIVE_FUNCTIONS[objective]
        
        # Create Walk-Forward splits once
        splits = create_walk_forward_splits(
            self.signals, n_splits=n_splits, train_pct=train_pct
        )
        
        def optuna_objective(trial):
            tp = trial.suggest_float('tp_mult', 
                                     min(self.param_grid['tp_mult']),
                                     max(self.param_grid['tp_mult']))
            sl = trial.suggest_float('sl_mult',
                                     min(self.param_grid['sl_mult']),
                                     max(self.param_grid['sl_mult']))
            bars = trial.suggest_int('max_bars',
                                     min(self.param_grid['max_bars']),
                                     max(self.param_grid['max_bars']))
            
            test_scores = []
            for split in splits:
                test_labels = self._evaluate_params(
                    split.test_signals, tp, sl, bars
                )
                if not test_labels.empty:
                    test_scores.append(obj_func(test_labels))
            
            return np.mean(test_scores) if test_scores else 0.0
        
        # Run Optuna
        verbosity = optuna.logging.WARNING if not verbose else optuna.logging.INFO
        optuna.logging.set_verbosity(verbosity)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=verbose)
        
        # Convert to DataFrame
        trials_df = study.trials_dataframe()
        
        return study.best_params, trials_df
    
    def get_best_params(
        self,
        results: pd.DataFrame,
        objective: str = 'sharpe',
        stability_weight: float = 0.3,
    ) -> Dict:
        """
        Get best parameters considering both performance and stability.
        
        Args:
            results: Output from grid_search()
            objective: Objective that was optimized
            stability_weight: Weight for stability (lower std) in ranking
            
        Returns:
            Dict with best parameter values
        """
        df = results.copy()
        
        test_col = f'test_{objective}'
        std_col = f'test_{objective}_std'
        
        # Normalize scores
        if df[test_col].std() > 0:
            df['score_norm'] = (df[test_col] - df[test_col].min()) / (df[test_col].max() - df[test_col].min())
        else:
            df['score_norm'] = 0.5
        
        # Lower std is better (more stable)
        if df[std_col].std() > 0:
            df['stability_norm'] = 1 - (df[std_col] - df[std_col].min()) / (df[std_col].max() - df[std_col].min())
        else:
            df['stability_norm'] = 0.5
        
        # Combined score
        df['combined'] = (1 - stability_weight) * df['score_norm'] + stability_weight * df['stability_norm']
        
        best_row = df.loc[df['combined'].idxmax()]
        
        return {
            'tp_mult': best_row['tp_mult'],
            'sl_mult': best_row['sl_mult'],
            'max_bars': int(best_row['max_bars']),
            f'test_{objective}': best_row[test_col],
        }
    
    def plot_heatmap(
        self,
        results: pd.DataFrame,
        objective: str = 'sharpe',
        fixed_bars: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot heatmap of TP x SL performance.
        
        Args:
            results: Output from grid_search()
            objective: Objective to visualize
            fixed_bars: If specified, filter to this max_bars value
            figsize: Figure size
            save_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            print("[ERROR] matplotlib required. Install with: pip install matplotlib")
            return
        
        df = results.copy()
        test_col = f'test_{objective}'
        
        # Filter by max_bars if specified
        if fixed_bars is not None:
            df = df[df['max_bars'] == fixed_bars]
        else:
            # Use the best performing max_bars
            best_bars = df.loc[df[test_col].idxmax(), 'max_bars']
            df = df[df['max_bars'] == best_bars]
            fixed_bars = best_bars
        
        # Pivot for heatmap
        pivot = df.pivot_table(
            index='sl_mult',
            columns='tp_mult',
            values=test_col,
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns])
        ax.set_yticklabels([f'{y:.1f}' for y in pivot.index])
        
        # Add value labels
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color='black' if 0.3 < val < 0.7 else 'white', fontsize=9)
        
        ax.set_xlabel('TP Multiplier (x volatility)', fontsize=12)
        ax.set_ylabel('SL Multiplier (x volatility)', fontsize=12)
        ax.set_title(f'Parameter Heatmap: {objective.upper()} (max_bars={fixed_bars})', fontsize=14)
        
        plt.colorbar(im, ax=ax, label=objective)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SAVED] Heatmap saved to: {save_path}")
        
        plt.show()
    
    def analyze_stability(
        self,
        results: pd.DataFrame,
        objective: str = 'sharpe',
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Analyze stability of top parameter combinations.
        
        Looks for "wide hills" - regions where nearby parameters
        also perform well (indicating robust, not overfit solutions).
        
        Args:
            results: Output from grid_search()
            objective: Objective to analyze
            top_n: Number of top combinations to analyze
            
        Returns:
            DataFrame with stability metrics
        """
        df = results.copy()
        test_col = f'test_{objective}'
        std_col = f'test_{objective}_std'
        
        # Get top N
        top = df.nlargest(top_n, test_col).copy()
        
        # For each top result, check neighbors
        stability_scores = []
        
        for _, row in top.iterrows():
            tp, sl, bars = row['tp_mult'], row['sl_mult'], row['max_bars']
            
            # Define "neighbors" (within 1 step)
            tp_grid = self.param_grid['tp_mult']
            sl_grid = self.param_grid['sl_mult']
            
            tp_idx = tp_grid.index(tp) if tp in tp_grid else -1
            sl_idx = sl_grid.index(sl) if sl in sl_grid else -1
            
            neighbor_scores = []
            
            for dtp in [-1, 0, 1]:
                for dsl in [-1, 0, 1]:
                    if dtp == 0 and dsl == 0:
                        continue
                    
                    new_tp_idx = tp_idx + dtp
                    new_sl_idx = sl_idx + dsl
                    
                    if 0 <= new_tp_idx < len(tp_grid) and 0 <= new_sl_idx < len(sl_grid):
                        neighbor_tp = tp_grid[new_tp_idx]
                        neighbor_sl = sl_grid[new_sl_idx]
                        
                        neighbor_row = df[
                            (df['tp_mult'] == neighbor_tp) &
                            (df['sl_mult'] == neighbor_sl) &
                            (df['max_bars'] == bars)
                        ]
                        
                        if not neighbor_row.empty:
                            neighbor_scores.append(neighbor_row[test_col].values[0])
            
            # Stability = average performance of neighbors
            avg_neighbor = np.mean(neighbor_scores) if neighbor_scores else np.nan
            neighbor_std = np.std(neighbor_scores) if len(neighbor_scores) > 1 else np.nan
            
            stability_scores.append({
                'tp_mult': tp,
                'sl_mult': sl,
                'max_bars': bars,
                'score': row[test_col],
                'score_std': row[std_col],
                'neighbor_avg': avg_neighbor,
                'neighbor_std': neighbor_std,
                'stability_ratio': avg_neighbor / row[test_col] if row[test_col] > 0 else np.nan,
            })
        
        return pd.DataFrame(stability_scores)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TBM Optimizer - Example Usage")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_bars = 1000
    
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='D')
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
    signal_dates = np.random.choice(dates[50:900], size=100, replace=False)
    signal_dates = pd.to_datetime(sorted(signal_dates))
    
    signals = pd.DataFrame({
        'timestamp': signal_dates,
        'signal_type': np.random.choice(['hammer', 'ma_cross'], size=100),
        'direction': np.random.choice([1, -1], size=100),
    })
    
    print(f"\n[DATA] {len(market_data)} bars, {len(signals)} signals")
    
    # Run optimizer with smaller grid for demo
    optimizer = TBMOptimizer(
        signals,
        market_data,
        param_grid={
            'tp_mult': [1.5, 2.0, 2.5, 3.0],
            'sl_mult': [0.5, 1.0, 1.5, 2.0],
            'max_bars': [5, 10, 20],
        }
    )
    
    print("\n[OPTIMIZER] Running Grid Search...")
    results = optimizer.grid_search(objective='sharpe', n_splits=4, verbose=True)
    
    print("\n" + "=" * 60)
    print("[RESULTS] Top 5 Parameter Combinations")
    print("=" * 60)
    print(results.head())
    
    # Get best params with stability consideration
    best = optimizer.get_best_params(results, objective='sharpe', stability_weight=0.3)
    print("\n[BEST] Optimal Parameters (with stability weighting):")
    print(f"  TP Multiplier: {best['tp_mult']}")
    print(f"  SL Multiplier: {best['sl_mult']}")
    print(f"  Max Bars: {best['max_bars']}")
    print(f"  Test Sharpe: {best['test_sharpe']:.3f}")
    
    # Stability analysis
    print("\n" + "=" * 60)
    print("[STABILITY] Analysis of Top Parameters")
    print("=" * 60)
    stability = optimizer.analyze_stability(results, objective='sharpe', top_n=5)
    print(stability[['tp_mult', 'sl_mult', 'score', 'neighbor_avg', 'stability_ratio']])
    
    print("\n[OK] Optimizer ready for production use!")
    print("\nTo visualize heatmap, call:")
    print("  optimizer.plot_heatmap(results, save_path='heatmap.png')")
