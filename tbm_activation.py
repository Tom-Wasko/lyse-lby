# =============================================================================
# TRIPLE BARRIER METHOD - ACTIVATION CODE FOR KULTURKAAA.IPYNB
# =============================================================================
# Add this code to the end of kulturkaaa.ipynb in separate cells
# Copy each section between "# --- CELL X ---" into a new cell

# --- CELL 1: Import TBM Modules ---
from triple_barrier import (
    get_triple_barrier_labels,
    compute_edge_ratio,
    plot_mfe_mae_scatter,
    suggest_optimal_stoploss,
    summarize_labels,
)
from tbm_optimizer import TBMOptimizer
from permutation_test import (
    run_permutation_test,
    plot_permutation_results,
)
print("TBM modules loaded successfully!")


# --- CELL 2: Prepare Signals from patterns_df ---
# Convert patterns_df to signals format required by TBM

def prepare_signals_from_patterns(patterns_df):
    """
    Convert patterns_df to signals format for Triple Barrier Method.
    
    patterns_df columns: symbol, Date, pattern, CMO, vol_ratio
    signals columns needed: timestamp, signal_type, direction
    """
    # Map pattern types to directions (1=Long, -1=Short)
    bullish_patterns = [
        'hammer', 'inverted_hammer', 'engulfing_bull', 'piercing', 
        'morning_star', 'three_white_soldiers', 'doji', 'dragonfly_doji',
        'morning_doji_star', 'abandoned_baby_bull'
    ]
    bearish_patterns = [
        'hanging_man', 'shooting_star', 'engulfing_bear', 'dark_cloud',
        'evening_star', 'three_black_crows', 'gravestone_doji',
        'evening_doji_star', 'abandoned_baby_bear'
    ]
    
    def get_direction(pattern):
        pattern_lower = pattern.lower()
        if any(bp in pattern_lower for bp in ['bull', 'hammer', 'piercing', 'morning', 'three_white', 'dragonfly']):
            return 1
        elif any(bp in pattern_lower for bp in ['bear', 'hanging', 'shooting', 'dark', 'evening', 'three_black', 'gravestone']):
            return -1
        else:
            return 1  # Default to long for neutral patterns like doji
    
    signals = pd.DataFrame({
        'timestamp': pd.to_datetime(patterns_df['Date']),
        'signal_type': patterns_df['pattern'],
        'direction': patterns_df['pattern'].apply(get_direction),
        'symbol': patterns_df['symbol'],
    })
    
    return signals

print("Signal preparation function ready!")


# --- CELL 3: Apply Triple Barrier Method to ALL patterns ---
# This requires iterating through each symbol since TBM needs price data

def run_tbm_for_all_patterns(
    patterns_df, 
    symbols_list, 
    tp_mult=2.0, 
    sl_mult=1.0, 
    max_bars=10,
    volatility_type='daily',
    volatility_span=20
):
    """
    Run Triple Barrier Method for all patterns across all symbols.
    
    Returns a combined DataFrame with TBM labels for all signals.
    """
    all_labels = []
    
    for symbol in tqdm(symbols_list, desc="Processing symbols"):
        # Get patterns for this symbol
        symbol_patterns = patterns_df[patterns_df['symbol'] == symbol].copy()
        
        if symbol_patterns.empty:
            continue
        
        # Get market data for this symbol
        try:
            market_data = globals()[f'{symbol}_df'].copy()
        except KeyError:
            print(f"Warning: No data for {symbol}, skipping...")
            continue
        
        # Prepare signals for this symbol
        signals = pd.DataFrame({
            'timestamp': pd.to_datetime(symbol_patterns['Date']),
            'signal_type': symbol_patterns['pattern'],
            'direction': symbol_patterns['pattern'].apply(
                lambda p: 1 if any(x in p.lower() for x in ['bull', 'hammer', 'piercing', 'morning', 'white', 'dragonfly']) else -1
            ),
        })
        
        # Apply TBM
        try:
            labels = get_triple_barrier_labels(
                signals,
                market_data,
                volatility_type=volatility_type,
                volatility_span=volatility_span,
                pt_sl_ratio=(tp_mult, sl_mult),
                max_holding_bars=max_bars,
            )
            
            if not labels.empty:
                labels['symbol'] = symbol
                all_labels.append(labels)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    if all_labels:
        combined_labels = pd.concat(all_labels, ignore_index=True)
        return combined_labels
    else:
        return pd.DataFrame()

print("TBM processing function ready!")


# --- CELL 4: Run TBM and Analyze Results ---
# Execute TBM on all patterns

# Get unique symbols that have patterns
symbols_with_patterns = patterns_df['symbol'].unique().tolist()

# Run TBM
print("Applying Triple Barrier Method to all patterns...")
tbm_labels = run_tbm_for_all_patterns(
    patterns_df,
    symbols_with_patterns,
    tp_mult=2.0,    # Take Profit = 2x volatility
    sl_mult=1.0,    # Stop Loss = 1x volatility
    max_bars=10,    # Max holding period
)

print(f"\nLabeled: {len(tbm_labels)} signals")

if not tbm_labels.empty:
    print("\n" + "="*60)
    print("TBM SUMMARY")
    print("="*60)
    print(summarize_labels(tbm_labels))


# --- CELL 5: MFE/MAE Analysis ---
# Analyze path quality of trades

if not tbm_labels.empty:
    # Edge Ratio by signal type
    print("\n" + "="*60)
    print("EDGE RATIO BY PATTERN")
    print("="*60)
    edge_by_signal = compute_edge_ratio(tbm_labels, by_signal=True)
    print(edge_by_signal)
    
    # Optimal stop-loss suggestions
    print("\n" + "="*60)
    print("OPTIMAL STOP-LOSS SUGGESTIONS (90th percentile)")
    print("="*60)
    suggestions = suggest_optimal_stoploss(tbm_labels, percentile=90)
    print(suggestions[['signal_type', 'suggested_sl', 'win_rate', 'edge_ratio']])
    
    # Visualization
    plot_mfe_mae_scatter(tbm_labels, save_path='mfe_mae_analysis.png')
    print("\n[OK] MFE/MAE scatter plot saved!")


# --- CELL 6: Parameter Optimization (Walk-Forward) ---
# Find optimal TP/SL parameters using Walk-Forward validation

# To run optimization, pick one symbol with enough signals
# Or use all signals if your data is large enough

# Example: optimize on first symbol with enough data
sample_symbol = symbols_with_patterns[0]
sample_patterns = patterns_df[patterns_df['symbol'] == sample_symbol]
sample_market = globals()[f'{sample_symbol}_df']

sample_signals = pd.DataFrame({
    'timestamp': pd.to_datetime(sample_patterns['Date']),
    'signal_type': sample_patterns['pattern'],
    'direction': sample_patterns['pattern'].apply(
        lambda p: 1 if any(x in p.lower() for x in ['bull', 'hammer', 'piercing', 'morning', 'white', 'dragonfly']) else -1
    ),
})

if len(sample_signals) >= 10:
    print(f"\nOptimizing on {sample_symbol} ({len(sample_signals)} signals)...")
    
    optimizer = TBMOptimizer(
        sample_signals,
        sample_market,
        param_grid={
            'tp_mult': [1.5, 2.0, 2.5, 3.0],
            'sl_mult': [0.5, 1.0, 1.5, 2.0],
            'max_bars': [5, 10, 20],
        }
    )
    
    results = optimizer.grid_search(objective='sharpe', n_splits=3, verbose=True)
    
    # Best params
    best = optimizer.get_best_params(results, stability_weight=0.3)
    print(f"\nBest Parameters:")
    print(f"  TP = {best['tp_mult']}x volatility")
    print(f"  SL = {best['sl_mult']}x volatility") 
    print(f"  Max bars = {best['max_bars']}")
    
    # Heatmap
    optimizer.plot_heatmap(results, save_path='param_heatmap.png')
    print("\n[OK] Parameter heatmap saved!")
else:
    print(f"Not enough signals ({len(sample_signals)}) for optimization")
    best = {'tp_mult': 2.0, 'sl_mult': 1.0, 'max_bars': 10}


# --- CELL 7: Robustness Verification (Permutation Test) ---
# Verify that strategy performance is not due to random chance

if len(sample_signals) >= 10:
    print("\n" + "="*60)
    print("PERMUTATION TEST (Monte Carlo)")
    print("="*60)
    
    result = run_permutation_test(
        sample_signals,
        sample_market,
        tp_mult=best['tp_mult'],
        sl_mult=best['sl_mult'],
        max_bars=best['max_bars'],
        objective='sharpe',
        n_permutations=100,  # Use 1000 for production
        verbose=True
    )
    
    # Histogram
    plot_permutation_results(result, save_path='permutation_test.png')
    
    # Final verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    if result.is_significant:
        print("[PASS] Strategy is statistically significant!")
        print("These parameters can be used with confidence.")
    else:
        print("[FAIL] Strategy may be OVERFITTED!")
        print("Consider different parameters or more data.")
