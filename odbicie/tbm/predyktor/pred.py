"""
pred.py
================
Meta-learning module that predicts optimal TBM parameters for any
(strategy, entry_params) combination using the historical lookup/tbm_lookup.json
dataset as training data.

Two-stage prediction pipeline
------------------------------
1. ``predict(entry_params)``
   Returns a dict of all 9 TBM params predicted by the trained model.
   Requires >= MIN_RECORDS records for the strategy in the lookup file.

2. ``knn_neighbors(entry_params, k=3)``
   Returns the k closest historical records by euclidean distance
   (same normalisation as tbm_lookup.py's _normalized_distance).
   Used to warm-start Optuna via enqueue_trial().

Model architecture
------------------
- 6 continuous / integer targets (tp_mult, sl_mult, tp_trail_mult,
  sl_trail_mult, time_decay_mult, max_holding_bars):
    sklearn.multioutput.RegressorChain wrapping XGBRegressor.
    Chain order is by descending feature importance proxy so each head
    can condition on the previous predictions.

- 2 boolean targets (active_trail_sl, time_decay_sl):
    sklearn.multioutput.ClassifierChain wrapping XGBClassifier.

Features are scaled with RobustScaler (handles fat-tailed financial data).

Model persistence
-----------------
Models are saved to  ``<cache_dir>/tbm_predictor_<strategy>.pkl``.
They auto-retrain whenever the number of records in the JSON has grown
since the last save (tracked by a record_count field in the pkl).

Usage
-----
    from odbicie.tbm.predyktor.pred import TbmPredictor

    predictor = TbmPredictor(
        lookup_path='odbicie/cache/lookup/tbm_lookup.json',
        strategy='base',
        cache_dir='odbicie/cache',
    )

    predicted_tbm = predictor.predict({'threshold_pct': 0.05, 'max_setup_hold_bars': 7})
    neighbors     = predictor.knn_neighbors({'threshold_pct': 0.05, 'max_setup_hold_bars': 7}, k=3)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.multioutput import RegressorChain, ClassifierChain
    from sklearn.preprocessing import RobustScaler
    from xgboost import XGBRegressor, XGBClassifier
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Harshness
# ─────────────────────────────────────────────────────────────────────────────

def _compute_harshness(rec: Dict[str, Any], strat: str) -> float:
    ep = rec.get('numeric_entry_params', {})
    hold = ep.get('max_setup_hold_bars', 10.0)
    
    if strat == 'base':
        h_main = (ep.get('threshold_pct', 0.02) - 0.02) / (0.40 - 0.02)
        h_hold = 1.0 - (hold - 1.0) / (35.0 - 1.0)
    elif strat == 'atr':
        h_main = (ep.get('atr_factor', 2.0) - 2.0) / (5.5 - 2.0)
        h_hold = 1.0 - (hold - 3.0) / (20.0 - 3.0)
    else:  # bb
        h_main = (ep.get('bb_std', 1.5) - 1.5) / (3.5 - 1.5)
        h_hold = 1.0 - (hold - 3.0) / (20.0 - 3.0)
        
    return float(np.clip(0.6 * h_main + 0.4 * h_hold, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_RECORDS = 15

_REG_TARGETS: List[str] = [
    'tp_mult', 'sl_mult', 'tp_trail_mult', 'sl_trail_mult',
    'time_decay_mult', 'max_holding_bars',
]

_CLS_TARGETS: List[str] = [
    'active_trail_sl',
    'time_decay_sl',
]

# Context features shared across all strategies (computed from entries_df)
_CONTEXT_KEYS_SHARED: List[str] = [
    'avg_rsi_at_entry', 'avg_atr_pct', 'avg_setup_bars', 'entry_count_log',
]

# Feature keys per strategy: numeric entry params + shared context + strategy-specific context
_FEATURE_KEYS: Dict[str, List[str]] = {
    'base': ['threshold_pct', 'max_setup_hold_bars'] + _CONTEXT_KEYS_SHARED,
    'atr':  ['atr_period', 'atr_factor', 'max_setup_hold_bars'] + _CONTEXT_KEYS_SHARED + ['avg_pullback_pct'],
    'bb':   ['bb_period', 'bb_std', 'max_setup_hold_bars'] + _CONTEXT_KEYS_SHARED + ['avg_bb_bandwidth'],
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalized_distance(
    params_a: Dict[str, float],
    params_b: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]],
) -> float:
    all_keys = set(params_a) | set(params_b)
    sq_sum = 0.0
    for k in all_keys:
        va = params_a.get(k)
        vb = params_b.get(k)
        lo, hi = ranges.get(k, (0.0, 1.0))
        span = hi - lo if hi != lo else 1.0
        if va is None or vb is None:
            sq_sum += 1.0
        else:
            sq_sum += ((va - vb) / span) ** 2
    return math.sqrt(sq_sum)


def _compute_ranges(records: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, list] = {}
    for rec in records:
        for k, v in rec.get('numeric_entry_params', {}).items():
            ranges.setdefault(k, []).append(float(v))
    return {k: (min(vs), max(vs)) for k, vs in ranges.items()}


def _numeric_entry_params(entry_params: Dict[str, Any]) -> Dict[str, float]:
    excluded = {'enter_on_close', 'exit_on_close'}
    return {
        k: float(v)
        for k, v in entry_params.items()
        if k not in excluded and isinstance(v, (int, float)) and not isinstance(v, bool)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TbmPredictor:
    """
    Predicts 9 optimal TBM parameters for a given (strategy, entry_params)
    pair using a trained XGBoost RegressorChain + ClassifierChain.

    Parameters
    ----------
    lookup_path : str
        Path to lookup/tbm_lookup.json.
    strategy : str
        One of 'base', 'atr', 'bb' (or any strategy with >= MIN_RECORDS entries).
    cache_dir : str
        Directory where trained model pickle files are stored.
        Defaults to the same directory as lookup_path.
    """

    def __init__(
        self,
        lookup_path: str,
        strategy: str,
        cache_dir: Optional[str] = None,
        max_harshness: float = 1.0,
    ) -> None:
        self.lookup_path = lookup_path
        self.strategy = strategy
        self.cache_dir = cache_dir if cache_dir else os.path.dirname(os.path.abspath(lookup_path))
        self.max_harshness = max_harshness
        self._pkl_path = os.path.join(self.cache_dir, f'tbm_predictor_{strategy}_{max_harshness}.pkl')

        self._records: List[Dict[str, Any]] = []
        self._scaler = None
        self._reg_chain = None
        self._cls_chain = None
        self._trained_record_count = 0
        self._feature_keys: List[str] = _FEATURE_KEYS.get(strategy, [])

        self._load_lookup()
        self._ensure_trained()

    # ── public API ─────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return (
            _SKLEARN_AVAILABLE
            and len(self._records) >= MIN_RECORDS
            and self._reg_chain is not None
            and self._cls_chain is not None
        )

    def predict(
        self,
        entry_params: Dict[str, Any],
        context_features: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.is_ready:
            return None

        X = self._make_feature_vector(entry_params, context_features)
        if X is None:
            return None

        X_scaled = self._scaler.transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reg_preds = self._reg_chain.predict(X_scaled)[0]
            cls_preds = self._cls_chain.predict(X_scaled)[0]

        result = {}

        for i, key in enumerate(_REG_TARGETS):
            val = float(reg_preds[i])
            if key == 'max_holding_bars':
                val = max(1, round(val))
            result[key] = val

        for i, key in enumerate(_CLS_TARGETS):
            result[key] = bool(cls_preds[i])

        result['exit_on_close'] = True
        return result

    def knn_neighbors(
        self,
        entry_params: Dict[str, Any],
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        if not self._records:
            return []

        query = _numeric_entry_params(entry_params)
        ranges = _compute_ranges(self._records)

        distances = []
        for rec in self._records:
            stored = rec.get('numeric_entry_params', {})
            dist = _normalized_distance(query, stored, ranges)
            distances.append((dist, rec))

        distances.sort(key=lambda x: x[0])

        return [
            {**rec, 'distance': dist}
            for dist, rec in distances[:k]
        ]

    def retrain(self) -> None:
        self._load_lookup()
        self._train()
        self._save_pkl()

    # ── private helpers ─────────────────────────────────────────────────────

    def _load_lookup(self) -> None:
        if not os.path.exists(self.lookup_path):
            self._records = []
            return
        with open(self.lookup_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        all_records = data.get(self.strategy, [])
        
        filtered_records = []
        for rec in all_records:
            harshness = _compute_harshness(rec, self.strategy)
            
            # Only keep setups/trades with harshness <= max_harshness
            if harshness <= self.max_harshness:
                filtered_records.append(rec)
                
        self._records = filtered_records

    def _ensure_trained(self) -> None:
        if not _SKLEARN_AVAILABLE:
            return
        if len(self._records) < MIN_RECORDS:
            return

        if os.path.exists(self._pkl_path):
            try:
                with open(self._pkl_path, 'rb') as f:
                    cached = pickle.load(f)
                cached_count = cached.get('record_count', 0)
                if cached_count == len(self._records):
                    self._scaler = cached['scaler']
                    self._reg_chain = cached['reg_chain']
                    self._cls_chain = cached['cls_chain']
                    self._trained_record_count = cached_count
                    self._feature_keys = cached.get('feature_keys', self._feature_keys)
                    return
            except Exception:
                pass  # corrupt cache — retrain

        self._train()
        self._save_pkl()

    def _make_feature_vector(
        self,
        entry_params: Dict[str, Any],
        context_features: Optional[Dict[str, float]] = None,
    ) -> Optional[np.ndarray]:
        numeric = _numeric_entry_params(entry_params)
        ctx = context_features if context_features else {}
        row = []
        for key in self._feature_keys:
            if key in numeric:
                row.append(float(numeric[key]))
            elif key in ctx:
                row.append(float(ctx[key]))
            else:
                row.append(0.0)  # backward compat — old records have no context

        return np.array(row, dtype=float).reshape(1, -1)

    def _build_dataset(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        X_rows: List[List[float]] = []
        Y_reg_rows: List[List[float]] = []
        Y_cls_rows: List[List[int]] = []

        for rec in self._records:
            numeric = rec.get('numeric_entry_params', {})
            context = rec.get('context_features', {})
            tbm = rec.get('tbm_params', {})

            feat_row: List[float] = []
            for key in self._feature_keys:
                if key in numeric:
                    feat_row.append(float(numeric[key]))
                elif key in context:
                    feat_row.append(float(context[key]))
                else:
                    feat_row.append(0.0)

            reg_row: List[float] = []
            missing_reg = False
            for key in _REG_TARGETS:
                val = tbm.get(key)
                if val is None:
                    missing_reg = True
                    break
                reg_row.append(float(val))
            if missing_reg:
                continue

            cls_row: List[int] = []
            missing_cls = False
            for key in _CLS_TARGETS:
                val = tbm.get(key)
                if val is None:
                    missing_cls = True
                    break
                cls_row.append(int(bool(val)))
            if missing_cls:
                continue

            X_rows.append(feat_row)
            Y_reg_rows.append(reg_row)
            Y_cls_rows.append(cls_row)

        if len(X_rows) < MIN_RECORDS:
            return None

        return (
            np.array(X_rows, dtype=float),
            np.array(Y_reg_rows, dtype=float),
            np.array(Y_cls_rows, dtype=int),
        )

    def _train(self) -> None:
        if not _SKLEARN_AVAILABLE:
            return
        dataset = self._build_dataset()
        if dataset is None:
            return

        X, Y_reg, Y_cls = dataset

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        xgb_reg = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        reg_chain = RegressorChain(xgb_reg, order='random', random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reg_chain.fit(X_scaled, Y_reg)

        xgb_cls = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            eval_metric='logloss',
        )
        cls_chain = ClassifierChain(xgb_cls, order='random', random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls_chain.fit(X_scaled, Y_cls)

        self._scaler = scaler
        self._reg_chain = reg_chain
        self._cls_chain = cls_chain
        self._trained_record_count = len(self._records)

        '''print(
            f'[TbmPredictor] Trained on {len(X)} records for strategy=\'{self.strategy}\' '
            f'({len(self._feature_keys)} features -> {len(_REG_TARGETS)} reg + {len(_CLS_TARGETS)} cls targets)'
        )'''

    def _save_pkl(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self._pkl_path, 'wb') as f:
            pickle.dump({
                'record_count': self._trained_record_count,
                'scaler':       self._scaler,
                'reg_chain':    self._reg_chain,
                'cls_chain':    self._cls_chain,
                'feature_keys': self._feature_keys,
            }, f)

    def leave_one_out_errors(self) -> Optional[Dict[str, Any]]:
        """
        Leave-one-out cross-validation.

        Returns a dict with keys: reg_targets, cls_targets, reg_errors,
        cls_errors, Y_reg, Y_cls, X, feature_keys, n.
        Same structure as TbmPredictorMTL so notebook plots work with both.
        """
        if not _SKLEARN_AVAILABLE:
            return None
        dataset = self._build_dataset()
        if dataset is None:
            return None

        X, Y_reg, Y_cls = dataset
        n = len(X)
        if n < MIN_RECORDS:
            return None

        reg_errors = np.zeros((n, len(_REG_TARGETS)))
        cls_errors = np.zeros((n, len(_CLS_TARGETS)), dtype=int)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_train     = X[mask]
            Y_reg_train = Y_reg[mask]
            Y_cls_train = Y_cls[mask]

            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s   = scaler.transform(X[i:i+1])

            xgb_r = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
            )
            rc = RegressorChain(xgb_r, order='random', random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rc.fit(X_train_s, Y_reg_train)
            pred_reg = rc.predict(X_val_s)[0]
            reg_errors[i] = np.abs(pred_reg - Y_reg[i])

            xgb_c = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbosity=0, eval_metric='logloss',
            )
            cc = ClassifierChain(xgb_c, order='random', random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cc.fit(X_train_s, Y_cls_train)
            pred_cls = cc.predict(X_val_s)[0]
            cls_errors[i] = np.abs(pred_cls.astype(int) - Y_cls[i])

        return {
            'reg_targets':  _REG_TARGETS,
            'cls_targets':  _CLS_TARGETS,
            'reg_errors':   reg_errors,
            'cls_errors':   cls_errors,
            'Y_reg':        Y_reg,
            'Y_cls':        Y_cls,
            'X':            X,
            'feature_keys': self._feature_keys,
            'n':            n,
        }

    def __repr__(self) -> str:
        return (
            f"TbmPredictor(strategy={self.strategy!r}, "
            f"records={len(self._records)}, "
            f"ready={self.is_ready})"
        )
