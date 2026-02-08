# Triple Barrier Method - Complete Walkthrough

This document summarizes all TBM modules created for signal labeling and analysis.

---

## Module Overview

| File | Purpose |
|------|---------|
| `triple_barrier.py` | TBM labeling with MFE/MAE |
| `tbm_optimizer.py` | Parameter optimization |
| `permutation_test.py` | Robustness verification |

---

## 1. Triple Barrier Method (`triple_barrier.py`)

### Core Functions
- `get_triple_barrier_labels()` - Main labeling function
- `compute_edge_ratio()` - MFE/MAE ratio analysis
- `plot_mfe_mae_scatter()` - Visualization
- `suggest_optimal_stoploss()` - SL recommendations

### Usage
```python
from triple_barrier import get_triple_barrier_labels, compute_edge_ratio

labels = get_triple_barrier_labels(
    signals, market_data,
    pt_sl_ratio=(2.0, 1.0),  # TP=2x, SL=1x volatility
    max_holding_bars=10
)

# Analyze path quality
print(compute_edge_ratio(labels, by_signal=True))
```

---

## 2. TBM Optimizer (`tbm_optimizer.py`)

### Features
- Walk-Forward validation (no look-ahead bias)
- Multiple objectives: Sharpe, Edge Ratio, Profit Factor
- Heatmap visualization
- Stability analysis ("wide hills")

### Usage
```python
from tbm_optimizer import TBMOptimizer

optimizer = TBMOptimizer(signals, market_data)
results = optimizer.grid_search(objective='sharpe', n_splits=5)

best = optimizer.get_best_params(results, stability_weight=0.3)
optimizer.plot_heatmap(results)
```

---

## 3. Permutation Test (`permutation_test.py`)

### Algorithm (Timothy Masters)
1. Shuffle returns (destroy patterns, keep distribution)
2. Run strategy on shuffled data 1000x
3. Calculate p-value = % random >= actual
4. p-value > 0.05 → likely overfitted

### Usage
```python
from permutation_test import run_permutation_test, plot_permutation_results

result = run_permutation_test(
    signals, market_data,
    tp_mult=2.0, sl_mult=1.0,
    n_permutations=1000
)

if result.is_significant:
    print("Strategy has genuine edge!")
else:
    print("Reject - may be overfitted")

plot_permutation_results(result)
```

---

## Recommended Workflow

1. **Label signals** with `get_triple_barrier_labels()`
2. **Analyze quality** with `compute_edge_ratio()`
3. **Optimize params** with `TBMOptimizer.grid_search()`
4. **Verify robustness** with `run_permutation_test()`
5. Only use params if p-value < 0.05
