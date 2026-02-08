"""
Script to add TBM (Triple Barrier Method) activation cells to main.ipynb.
This adds cells that fully utilize functions from:
- tbm_optimizer.py (TBMOptimizer, grid_search, optuna_search, get_best_params, etc.)
- triple_barrier.py (get_triple_barrier_labels, compute_edge_ratio, summarize_labels, etc.)
- permutation_test.py (run_permutation_test, plot_permutation_results, test_multiple_objectives)

Run this script once to add the cells, then delete it.
"""

import json

import random

def create_markdown_cell(text):
    """Create a markdown cell."""
    cell_id = f"tbm_md_{random.randint(1000, 9999)}"
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": text.split("\n") if isinstance(text, str) else text
    }

def create_code_cell(source_lines):
    """Create a code cell."""
    cell_id = f"tbm_code_{random.randint(10000, 99999)}"
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def main():
    notebook_path = "main.ipynb"
    
    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # Define the new cells to add
    new_cells = []
    
    # --- Cell 1: Header markdown ---
    new_cells.append(create_markdown_cell([
        "# Triple Barrier Method (TBM) - Full Analysis\n",
        "\n",
        "This section provides a complete implementation utilizing all functions from:\n",
        "- `triple_barrier.py` - TBM labeling, MFE/MAE analysis, edge ratio computation\n",
        "- `tbm_optimizer.py` - Walk-Forward parameter optimization\n",
        "- `permutation_test.py` - Monte Carlo robustness verification"
    ]))
    
    # --- Cell 2: Import TBM modules ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 1: Import TBM Modules\n",
        "# =============================================================================\n",
        "\n",
        "from triple_barrier import (\n",
        "    get_triple_barrier_labels,\n",
        "    compute_daily_volatility,\n",
        "    compute_atr,\n",
        "    compute_atr_pct,\n",
        "    summarize_labels,\n",
        "    summarize_by_signal_type,\n",
        "    compute_edge_ratio,\n",
        "    plot_mfe_mae_scatter,\n",
        "    suggest_optimal_stoploss,\n",
        ")\n",
        "\n",
        "from tbm_optimizer import (\n",
        "    TBMOptimizer,\n",
        "    calc_sharpe_ratio,\n",
        "    calc_profit_factor,\n",
        "    calc_win_rate,\n",
        "    calc_edge_ratio_from_labels,\n",
        "    create_walk_forward_splits,\n",
        "    OBJECTIVE_FUNCTIONS,\n",
        "    DEFAULT_PARAM_GRID,\n",
        ")\n",
        "\n",
        "from permutation_test import (\n",
        "    run_permutation_test,\n",
        "    plot_permutation_results,\n",
        "    test_multiple_objectives,\n",
        "    shuffle_returns,\n",
        "    PermutationTestResult,\n",
        ")\n",
        "\n",
        "print(\"[OK] TBM modules loaded successfully!\")\n",
        "print(f\"   - triple_barrier: TBM labeling and MFE/MAE analysis\")\n",
        "print(f\"   - tbm_optimizer: Walk-Forward parameter optimization\")\n",
        "print(f\"   - permutation_test: Monte Carlo robustness verification\")"
    ]))
    
    # --- Cell 3: Helper function to prepare signals ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 2: Signal Preparation Helper Function\n",
        "# =============================================================================\n",
        "\n",
        "def prepare_signals_from_patterns(patterns_df):\n",
        "    \"\"\"\n",
        "    Convert patterns_df to signals format for Triple Barrier Method.\n",
        "    \n",
        "    patterns_df columns: symbol, Date, pattern, CMO, vol_ratio\n",
        "    signals columns needed: timestamp, signal_type, direction\n",
        "    \"\"\"\n",
        "    def get_direction(pattern):\n",
        "        pattern_lower = pattern.lower()\n",
        "        bullish_keywords = ['bull', 'hammer', 'piercing', 'morning', 'white', 'dragonfly']\n",
        "        bearish_keywords = ['bear', 'hanging', 'shooting', 'dark', 'evening', 'black', 'gravestone']\n",
        "        \n",
        "        if any(kw in pattern_lower for kw in bullish_keywords):\n",
        "            return 1  # Long\n",
        "        elif any(kw in pattern_lower for kw in bearish_keywords):\n",
        "            return -1  # Short\n",
        "        else:\n",
        "            return 1  # Default to long for neutral patterns\n",
        "    \n",
        "    signals = pd.DataFrame({\n",
        "        'timestamp': pd.to_datetime(patterns_df['Date']),\n",
        "        'signal_type': patterns_df['pattern'],\n",
        "        'direction': patterns_df['pattern'].apply(get_direction),\n",
        "        'symbol': patterns_df['symbol'],\n",
        "    })\n",
        "    \n",
        "    return signals\n",
        "\n",
        "print(\"[OK] Signal preparation function ready!\")"
    ]))
    
    # --- Cell 4: Run TBM for all patterns ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 3: Run Triple Barrier Method for All Patterns\n",
        "# =============================================================================\n",
        "\n",
        "def run_tbm_for_all_patterns(\n",
        "    patterns_df, \n",
        "    symbols_list, \n",
        "    tp_mult=2.0, \n",
        "    sl_mult=1.0, \n",
        "    max_bars=10,\n",
        "    volatility_type='daily',\n",
        "    volatility_span=20\n",
        "):\n",
        "    \"\"\"\n",
        "    Run Triple Barrier Method for all patterns across all symbols.\n",
        "    \n",
        "    Uses get_triple_barrier_labels to label each signal with:\n",
        "    - label: 1 (TP hit), -1 (SL hit), 0 (timeout)\n",
        "    - ret: return from entry to exit\n",
        "    - mfe: Maximum Favorable Excursion\n",
        "    - mae: Maximum Adverse Excursion\n",
        "    - barrier_type: which barrier was touched ('take_profit', 'stop_loss', 'time')\n",
        "    \n",
        "    Returns a combined DataFrame with TBM labels for all signals.\n",
        "    \"\"\"\n",
        "    all_labels = []\n",
        "    \n",
        "    for symbol in tqdm(symbols_list, desc=\"Processing symbols with TBM\"):\n",
        "        # Get patterns for this symbol\n",
        "        symbol_patterns = patterns_df[patterns_df['symbol'] == symbol].copy()\n",
        "        \n",
        "        if symbol_patterns.empty:\n",
        "            continue\n",
        "        \n",
        "        # Get market data for this symbol\n",
        "        try:\n",
        "            market_data = globals()[f'{symbol}_df'].copy()\n",
        "        except KeyError:\n",
        "            continue\n",
        "        \n",
        "        # Prepare signals for this symbol\n",
        "        signals = pd.DataFrame({\n",
        "            'timestamp': pd.to_datetime(symbol_patterns['Date']),\n",
        "            'signal_type': symbol_patterns['pattern'],\n",
        "            'direction': symbol_patterns['pattern'].apply(\n",
        "                lambda p: 1 if any(x in p.lower() for x in ['bull', 'hammer', 'piercing', 'morning', 'white', 'dragonfly']) else -1\n",
        "            ),\n",
        "        })\n",
        "        \n",
        "        # Apply TBM using get_triple_barrier_labels\n",
        "        try:\n",
        "            labels = get_triple_barrier_labels(\n",
        "                signals,\n",
        "                market_data,\n",
        "                volatility_type=volatility_type,\n",
        "                volatility_span=volatility_span,\n",
        "                pt_sl_ratio=(tp_mult, sl_mult),\n",
        "                max_holding_bars=max_bars,\n",
        "            )\n",
        "            \n",
        "            if not labels.empty:\n",
        "                labels['symbol'] = symbol\n",
        "                all_labels.append(labels)\n",
        "        except Exception as e:\n",
        "            continue\n",
        "    \n",
        "    if all_labels:\n",
        "        combined_labels = pd.concat(all_labels, ignore_index=True)\n",
        "        return combined_labels\n",
        "    else:\n",
        "        return pd.DataFrame()\n",
        "\n",
        "print(\"[OK] TBM processing function ready!\")"
    ]))
    
    # --- Cell 5: Execute TBM ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 4: Execute TBM on All Patterns\n",
        "# =============================================================================\n",
        "\n",
        "# Get unique symbols that have patterns\n",
        "symbols_with_patterns = patterns_df['symbol'].unique().tolist()\n",
        "print(f\"Symbols with patterns: {len(symbols_with_patterns)}\")\n",
        "\n",
        "# Run TBM with default parameters\n",
        "print(\"\\nApplying Triple Barrier Method to all patterns...\")\n",
        "tbm_labels = run_tbm_for_all_patterns(\n",
        "    patterns_df,\n",
        "    symbols_with_patterns,\n",
        "    tp_mult=2.0,    # Take Profit = 2x volatility\n",
        "    sl_mult=1.0,    # Stop Loss = 1x volatility\n",
        "    max_bars=10,    # Max holding period = 10 bars\n",
        ")\n",
        "\n",
        "print(f\"\\n[OK] Labeled: {len(tbm_labels)} signals\")\n",
        "\n",
        "# Display TBM labels\n",
        "if not tbm_labels.empty:\n",
        "    print(\"\\nTBM Labels Sample:\")\n",
        "    display(tbm_labels.head(10))"
    ]))
    
    # --- Cell 6: Summarize labels ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 5: TBM Summary Statistics (using summarize_labels & summarize_by_signal_type)\n",
        "# =============================================================================\n",
        "\n",
        "if not tbm_labels.empty:\n",
        "    print(\"=\" * 60)\n",
        "    print(\"TBM OVERALL SUMMARY - summarize_labels()\")\n",
        "    print(\"=\" * 60)\n",
        "    summary = summarize_labels(tbm_labels)\n",
        "    display(summary)\n",
        "    \n",
        "    print(\"\\n\" + \"=\" * 60)\n",
        "    print(\"SUMMARY BY SIGNAL TYPE - summarize_by_signal_type()\")\n",
        "    print(\"=\" * 60)\n",
        "    signal_summary = summarize_by_signal_type(tbm_labels)\n",
        "    display(signal_summary)\n",
        "else:\n",
        "    print(\"No TBM labels available for summary.\")"
    ]))
    
    # --- Cell 7: Edge ratio analysis ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 6: MFE/MAE Analysis (using compute_edge_ratio & suggest_optimal_stoploss)\n",
        "# =============================================================================\n",
        "\n",
        "if not tbm_labels.empty:\n",
        "    # Overall Edge Ratio\n",
        "    print(\"=\" * 60)\n",
        "    print(\"OVERALL EDGE RATIO - compute_edge_ratio()\")\n",
        "    print(\"=\" * 60)\n",
        "    overall_edge = compute_edge_ratio(tbm_labels, by_signal=False)\n",
        "    print(f\"Edge Ratio: {overall_edge:.3f}\")\n",
        "    print(\"(Edge Ratio > 1.0 indicates favorable path quality)\")\n",
        "    \n",
        "    # Edge Ratio by signal type\n",
        "    print(\"\\n\" + \"=\" * 60)\n",
        "    print(\"EDGE RATIO BY PATTERN - compute_edge_ratio(by_signal=True)\")\n",
        "    print(\"=\" * 60)\n",
        "    edge_by_signal = compute_edge_ratio(tbm_labels, by_signal=True)\n",
        "    display(edge_by_signal)\n",
        "    \n",
        "    # Optimal stop-loss suggestions\n",
        "    print(\"\\n\" + \"=\" * 60)\n",
        "    print(\"OPTIMAL STOP-LOSS SUGGESTIONS - suggest_optimal_stoploss()\")\n",
        "    print(\"=\" * 60)\n",
        "    suggestions = suggest_optimal_stoploss(tbm_labels, percentile=90)\n",
        "    display(suggestions[['signal_type', 'suggested_sl', 'win_rate', 'edge_ratio']])\n",
        "else:\n",
        "    print(\"No TBM labels available for analysis.\")"
    ]))
    
    # --- Cell 8: MFE/MAE visualization ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 7: MFE/MAE Visualization (using plot_mfe_mae_scatter)\n",
        "# =============================================================================\n",
        "\n",
        "if not tbm_labels.empty:\n",
        "    print(\"Generating MFE/MAE scatter plot...\")\n",
        "    plot_mfe_mae_scatter(tbm_labels, save_path='mfe_mae_analysis.png')\n",
        "    print(\"\\n[OK] MFE/MAE scatter plot saved to 'mfe_mae_analysis.png'\")\n",
        "else:\n",
        "    print(\"No TBM labels available for visualization.\")"
    ]))
    
    # --- Cell 9: TBMOptimizer setup ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 8: TBM Parameter Optimization Setup (using TBMOptimizer)\n",
        "# =============================================================================\n",
        "\n",
        "# Select a symbol with enough signals for optimization\n",
        "# We need at least 30 signals for meaningful Walk-Forward cross-validation\n",
        "\n",
        "min_signals_for_optimization = 30\n",
        "optimization_symbol = None\n",
        "\n",
        "for symbol in symbols_with_patterns:\n",
        "    symbol_patterns = patterns_df[patterns_df['symbol'] == symbol]\n",
        "    if len(symbol_patterns) >= min_signals_for_optimization:\n",
        "        optimization_symbol = symbol\n",
        "        break\n",
        "\n",
        "if optimization_symbol:\n",
        "    print(f\"Selected {optimization_symbol} for optimization\")\n",
        "    \n",
        "    # Prepare signals\n",
        "    opt_patterns = patterns_df[patterns_df['symbol'] == optimization_symbol]\n",
        "    opt_market = globals()[f'{optimization_symbol}_df']\n",
        "    \n",
        "    opt_signals = pd.DataFrame({\n",
        "        'timestamp': pd.to_datetime(opt_patterns['Date']),\n",
        "        'signal_type': opt_patterns['pattern'],\n",
        "        'direction': opt_patterns['pattern'].apply(\n",
        "            lambda p: 1 if any(x in p.lower() for x in ['bull', 'hammer', 'piercing', 'morning', 'white', 'dragonfly']) else -1\n",
        "        ),\n",
        "    })\n",
        "    \n",
        "    print(f\"Number of signals: {len(opt_signals)}\")\n",
        "    print(f\"Date range: {opt_signals['timestamp'].min()} to {opt_signals['timestamp'].max()}\")\n",
        "    \n",
        "    # Show default parameter grid\n",
        "    print(\"\\nDefault Parameter Grid:\")\n",
        "    for key, values in DEFAULT_PARAM_GRID.items():\n",
        "        print(f\"  {key}: {values}\")\n",
        "else:\n",
        "    print(f\"No symbol found with >= {min_signals_for_optimization} signals.\")\n",
        "    print(\"Consider using a smaller threshold or aggregating signals.\")"
    ]))
    
    # --- Cell 10: Grid search optimization ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 9: Grid Search Optimization (using TBMOptimizer.grid_search)\n",
        "# =============================================================================\n",
        "\n",
        "if optimization_symbol:\n",
        "    print(f\"\\nRunning Grid Search optimization on {optimization_symbol}...\")\n",
        "    print(\"=\" * 60)\n",
        "    \n",
        "    # Initialize optimizer with custom parameter grid\n",
        "    optimizer = TBMOptimizer(\n",
        "        opt_signals,\n",
        "        opt_market,\n",
        "        param_grid={\n",
        "            'tp_mult': [1.5, 2.0, 2.5, 3.0],  # Take Profit multipliers\n",
        "            'sl_mult': [0.5, 1.0, 1.5, 2.0],  # Stop Loss multipliers\n",
        "            'max_bars': [5, 10, 20],          # Max holding periods\n",
        "        },\n",
        "        volatility_type='daily',\n",
        "        volatility_span=20,\n",
        "    )\n",
        "    \n",
        "    # Run grid search with Walk-Forward validation\n",
        "    grid_results = optimizer.grid_search(\n",
        "        objective='sharpe',  # Use Sharpe Ratio as objective\n",
        "        n_splits=3,          # Number of Walk-Forward folds\n",
        "        train_pct=0.7,       # 70% training, 30% testing per fold\n",
        "        verbose=True,\n",
        "    )\n",
        "    \n",
        "    print(\"\\n[OK] Grid Search completed!\")\n",
        "    print(f\"Tested {len(grid_results)} parameter combinations.\")\n",
        "    display(grid_results.head(10))"
    ]))
    
    # --- Cell 11: Get best params ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 10: Get Best Parameters (using get_best_params & analyze_stability)\n",
        "# =============================================================================\n",
        "\n",
        "if optimization_symbol and 'grid_results' in dir() and not grid_results.empty:\n",
        "    print(\"=\" * 60)\n",
        "    print(\"BEST PARAMETERS - get_best_params()\")\n",
        "    print(\"=\" * 60)\n",
        "    \n",
        "    # Get best params considering both performance and stability\n",
        "    best_params = optimizer.get_best_params(\n",
        "        grid_results, \n",
        "        objective='sharpe',\n",
        "        stability_weight=0.3,  # Weight for stability in ranking\n",
        "    )\n",
        "    \n",
        "    print(f\"Best Parameters:\")\n",
        "    print(f\"  Take Profit = {best_params['tp_mult']}x volatility\")\n",
        "    print(f\"  Stop Loss   = {best_params['sl_mult']}x volatility\")\n",
        "    print(f\"  Max Bars    = {best_params['max_bars']}\")\n",
        "    \n",
        "    # Analyze stability of top parameters\n",
        "    print(\"\\n\" + \"=\" * 60)\n",
        "    print(\"PARAMETER STABILITY ANALYSIS - analyze_stability()\")\n",
        "    print(\"=\" * 60)\n",
        "    stability = optimizer.analyze_stability(grid_results, objective='sharpe', top_n=10)\n",
        "    display(stability)\n",
        "else:\n",
        "    print(\"No optimization results available.\")\n",
        "    best_params = {'tp_mult': 2.0, 'sl_mult': 1.0, 'max_bars': 10}"
    ]))
    
    # --- Cell 12: Heatmap visualization ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 11: Parameter Heatmap Visualization (using plot_heatmap)\n",
        "# =============================================================================\n",
        "\n",
        "if optimization_symbol and 'optimizer' in dir():\n",
        "    print(\"Generating parameter heatmap...\")\n",
        "    optimizer.plot_heatmap(grid_results, save_path='param_heatmap.png')\n",
        "    print(\"\\n[OK] Parameter heatmap saved to 'param_heatmap.png'\")"
    ]))
    
    # --- Cell 13: Walk-forward splits visualization ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 12: Walk-Forward Splits Visualization (using create_walk_forward_splits)\n",
        "# =============================================================================\n",
        "\n",
        "if optimization_symbol:\n",
        "    print(\"=\" * 60)\n",
        "    print(\"WALK-FORWARD SPLITS - create_walk_forward_splits()\")\n",
        "    print(\"=\" * 60)\n",
        "    \n",
        "    # Create and visualize Walk-Forward splits\n",
        "    wf_splits = create_walk_forward_splits(\n",
        "        opt_signals,\n",
        "        n_splits=3,\n",
        "        train_pct=0.7,\n",
        "        purge_gap=0,\n",
        "    )\n",
        "    \n",
        "    for i, split in enumerate(wf_splits):\n",
        "        print(f\"\\nFold {i+1}:\")\n",
        "        print(f\"  Train: {split.train_start.date()} to {split.train_end.date()}\")\n",
        "        print(f\"  Test:  {split.test_start.date()} to {split.test_end.date()}\")\n",
        "        print(f\"  Train signals: {len(split.train_signals)}, Test signals: {len(split.test_signals)}\")"
    ]))
    
    # --- Cell 14: Permutation test ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 13: Permutation Test (using run_permutation_test)\n",
        "# =============================================================================\n",
        "\n",
        "if optimization_symbol and 'best_params' in dir():\n",
        "    print(\"=\" * 60)\n",
        "    print(\"PERMUTATION TEST - run_permutation_test()\")\n",
        "    print(\"=\" * 60)\n",
        "    print(\"Testing if strategy performance is statistically significant...\")\n",
        "    print(\"(This may take a minute)\\n\")\n",
        "    \n",
        "    perm_result = run_permutation_test(\n",
        "        opt_signals,\n",
        "        opt_market,\n",
        "        tp_mult=best_params['tp_mult'],\n",
        "        sl_mult=best_params['sl_mult'],\n",
        "        max_bars=best_params['max_bars'],\n",
        "        objective='sharpe',\n",
        "        n_permutations=100,  # Use 1000 for production\n",
        "        alpha=0.05,\n",
        "        verbose=True,\n",
        "    )\n",
        "    \n",
        "    print(\"\\n[OK] Permutation test completed!\")\n",
        "    print(perm_result)"
    ]))
    
    # --- Cell 15: Permutation visualization ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 14: Permutation Test Visualization (using plot_permutation_results)\n",
        "# =============================================================================\n",
        "\n",
        "if optimization_symbol and 'perm_result' in dir():\n",
        "    print(\"Generating permutation test histogram...\")\n",
        "    plot_permutation_results(perm_result, save_path='permutation_test.png')\n",
        "    print(\"\\n[OK] Permutation test plot saved to 'permutation_test.png'\")"
    ]))
    
    # --- Cell 16: Multiple objectives test ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 15: Test Multiple Objectives (using test_multiple_objectives)\n",
        "# =============================================================================\n",
        "\n",
        "if optimization_symbol and 'best_params' in dir():\n",
        "    print(\"=\" * 60)\n",
        "    print(\"MULTI-OBJECTIVE PERMUTATION TEST - test_multiple_objectives()\")\n",
        "    print(\"=\" * 60)\n",
        "    print(\"Testing robustness across multiple metrics...\\n\")\n",
        "    \n",
        "    multi_results = test_multiple_objectives(\n",
        "        opt_signals,\n",
        "        opt_market,\n",
        "        tp_mult=best_params['tp_mult'],\n",
        "        sl_mult=best_params['sl_mult'],\n",
        "        max_bars=best_params['max_bars'],\n",
        "        n_permutations=50,  # Use 500 for production\n",
        "        verbose=True,\n",
        "    )\n",
        "    \n",
        "    print(\"\\n\" + \"=\" * 60)\n",
        "    print(\"MULTI-OBJECTIVE RESULTS\")\n",
        "    print(\"=\" * 60)\n",
        "    display(multi_results)"
    ]))
    
    # --- Cell 17: Final verdict ---
    new_cells.append(create_code_cell([
        "# =============================================================================\n",
        "# CELL 16: Final Verdict and Recommendations\n",
        "# =============================================================================\n",
        "\n",
        "print(\"=\" * 60)\n",
        "print(\"FINAL VERDICT\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "if 'perm_result' in dir() and perm_result.is_significant:\n",
        "    print(\"\\n[OK] [PASS] Strategy is STATISTICALLY SIGNIFICANT!\")\n",
        "    print(f\"\\n   P-value: {perm_result.p_value:.4f}\")\n",
        "    print(f\"   Actual {perm_result.objective}: {perm_result.actual_score:.4f}\")\n",
        "    print(f\"   Random mean: {perm_result.random_scores.mean():.4f}\")\n",
        "    print(f\"\\n   Interpretation: {perm_result.interpretation}\")\n",
        "    \n",
        "    print(\"\\n\" + \"-\" * 60)\n",
        "    print(\"RECOMMENDED PARAMETERS:\")\n",
        "    print(\"-\" * 60)\n",
        "    print(f\"   Take Profit: {best_params['tp_mult']}x volatility\")\n",
        "    print(f\"   Stop Loss:   {best_params['sl_mult']}x volatility\")\n",
        "    print(f\"   Max Holding: {best_params['max_bars']} bars\")\n",
        "    print(\"\\n   These parameters can be used with confidence.\")\n",
        "    \n",
        "elif 'perm_result' in dir():\n",
        "    print(\"\\n[ERROR] [FAIL] Strategy may be OVERFITTED!\")\n",
        "    print(f\"\\n   P-value: {perm_result.p_value:.4f}\")\n",
        "    print(f\"\\n   Interpretation: {perm_result.interpretation}\")\n",
        "    print(\"\\n   Recommendations:\")\n",
        "    print(\"   - Use more conservative parameters\")\n",
        "    print(\"   - Gather more data\")\n",
        "    print(\"   - Consider different pattern filters\")\n",
        "else:\n",
        "    print(\"\\nNo permutation test results available.\")\n",
        "    print(\"Run the permutation test cells above for robustness verification.\")\n",
        "\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"TBM Analysis Complete!\")\n",
        "print(\"=\" * 60)"
    ]))
    
    # Add all new cells to the notebook
    notebook["cells"].extend(new_cells)
    
    # Write the updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"[OK] Added {len(new_cells)} TBM activation cells to {notebook_path}")
    print("\nCells added:")
    print("  1.  Header markdown")
    print("  2.  Import TBM modules")
    print("  3.  Signal preparation helper")
    print("  4.  Run TBM for all patterns")
    print("  5.  Execute TBM")
    print("  6.  Summary statistics")
    print("  7.  Edge ratio analysis")
    print("  8.  MFE/MAE visualization")
    print("  9.  TBMOptimizer setup")
    print("  10. Grid search optimization")
    print("  11. Get best parameters")
    print("  12. Parameter heatmap")
    print("  13. Walk-Forward splits")
    print("  14. Permutation test")
    print("  15. Permutation visualization")
    print("  16. Multi-objective test")
    print("  17. Final verdict")
    print("\nYou can now delete this script (add_tbm_activation.py)")

if __name__ == "__main__":
    main()
