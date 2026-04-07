"""
pred_mtl.py
====================
Multi-Task Learning (MTL) neural network predictor for optimal TBM parameters.

Architecture
------------
Input  (n_features)
  -> Shared Trunk:  Linear(n, 64) -> LayerNorm -> GELU -> Dropout(0.35)
                   Linear(64, 64) -> LayerNorm -> GELU -> Dropout(0.35)
                   Linear(64, 32) -> LayerNorm -> GELU
  -> 6 Regression heads   : Linear(32, 1) each  — continuous / integer targets
  -> 2 Classification heads: Linear(32, 1) each  — logits -> Sigmoid -> boolean

Why MTL over chained XGBoost?
------------------------------
The TBM parameters are correlated (wider TP -> wider SL works best together).
A shared trunk forces the model to learn a *common latent representation* of the
market regime, then each head specialises.  With weight-sharing this is
more data-efficient than 8 independent trees — helpful for n = 50–400 records.

Loss
----
  L = mean(MSELoss per reg head) + 0.5 × mean(BCEWithLogitsLoss per cls head)
Regression targets are z-score normalised during training (μ=0, σ=1).

Public API  (drop-in compatible with TbmPredictor)
----------
  predict(entry_params, context_features=None) -> dict | None
  knn_neighbors(entry_params, k=3)             -> list
  leave_one_out_errors()                       -> dict | None
  is_ready                                     -> bool

Model persistence
-----------------
Saved to <cache_dir>/tbm_predictor_mtl_<strategy>.pt
Auto-retrains when the JSON has grown since the last save.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import RobustScaler
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


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



# Reuse the same constants as TbmPredictor for full compatibility
MIN_RECORDS = 15

_REG_TARGETS: List[str] = [
    'tp_mult', 'sl_mult', 'tp_trail_mult', 'sl_trail_mult',
    'time_decay_mult', 'max_holding_bars',
]

_CLS_TARGETS: List[str] = [
    'active_trail_sl',
    'time_decay_sl',
]

_CONTEXT_KEYS_SHARED: List[str] = [
    'avg_rsi_at_entry', 'avg_atr_pct', 'avg_setup_bars', 'entry_count_log',
]

_FEATURE_KEYS: Dict[str, List[str]] = {
    'base': ['threshold_pct', 'max_setup_hold_bars'] + _CONTEXT_KEYS_SHARED,
    'atr':  ['atr_period', 'atr_factor', 'max_setup_hold_bars'] + _CONTEXT_KEYS_SHARED + ['avg_pullback_pct'],
    'bb':   ['bb_period', 'bb_std', 'max_setup_hold_bars'] + _CONTEXT_KEYS_SHARED + ['avg_bb_bandwidth'],
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers  (same as pred.py to stay independent)
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
# Neural network
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class _TbmMTLNet(nn.Module):
        """
        Shared-trunk MTL network.

        LayerNorm is used instead of BatchNorm because small datasets (n<500)
        make per-batch statistics unreliable during full-batch training.
        """

        def __init__(
            self,
            n_features: int,
            n_reg: int,
            n_cls: int,
            hidden: int = 64,
            dropout: float = 0.35,
        ) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 32),
                nn.LayerNorm(32),
                nn.GELU(),
            )
            self.reg_heads  = nn.ModuleList([nn.Linear(32, 1) for _ in range(n_reg)])
            self.cls_heads  = nn.ModuleList([nn.Linear(32, 1) for _ in range(n_cls)])

        def forward(self, x: 'torch.Tensor') -> Tuple['torch.Tensor', 'torch.Tensor']:
            h = self.trunk(x)
            reg_out = torch.cat([head(h) for head in self.reg_heads], dim=1)
            cls_out = torch.cat([head(h) for head in self.cls_heads], dim=1)
            return reg_out, cls_out


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TbmPredictorMTL:
    """
    MTL neural-network predictor for 9 optimal TBM parameters.

    Drop-in compatible with ``TbmPredictor`` (XGBoost).  Both can be swapped
    via ``tbm_predictor_factory.get_predictor()``.

    Parameters
    ----------
    lookup_path : str
        Path to lookup/tbm_lookup.json.
    strategy : str
        One of 'base', 'atr', 'bb'.
    cache_dir : str, optional
        Directory for the .pt model cache file.
    n_epochs : int
        Maximum training epochs (default 500, with early stopping).
    patience : int
        Early-stopping patience in epochs.
    hidden : int
        Hidden layer size (shared trunk width).
    dropout : float
        Dropout probability in the trunk.
    """

    def __init__(
        self,
        lookup_path: str,
        strategy: str,
        cache_dir: Optional[str] = None,
        max_harshness: float = 1.0,
        n_epochs: int = 500,
        patience: int = 40,
        hidden: int = 64,
        dropout: float = 0.35,
    ) -> None:
        self.lookup_path = lookup_path
        self.strategy    = strategy
        self.cache_dir   = cache_dir if cache_dir else os.path.dirname(os.path.abspath(lookup_path))
        self.max_harshness = max_harshness
        self.n_epochs    = n_epochs
        self.patience    = patience
        self.hidden      = hidden
        self.dropout     = dropout

        self._pt_path    = os.path.join(self.cache_dir, f'tbm_predictor_mtl_{strategy}_{max_harshness}.pt')

        self._records: List[Dict[str, Any]] = []
        self._scaler     = None
        self._net        = None
        self._Y_reg_mean = None
        self._Y_reg_std  = None
        self._feature_keys: List[str] = _FEATURE_KEYS.get(strategy, [])
        self._trained_record_count = 0

        self._load_lookup()
        self._ensure_trained()

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return (
            _TORCH_AVAILABLE
            and len(self._records) >= MIN_RECORDS
            and self._net is not None
        )

    def predict(
        self,
        entry_params: Dict[str, Any],
        context_features: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Predict all 9 TBM parameters.

        Parameters
        ----------
        entry_params     : Numeric entry settings dict.
        context_features : Optional aggregate market context statistics
                           (avg_rsi_at_entry, avg_atr_pct, avg_setup_bars,
                           entry_count_log, avg_bb_bandwidth, avg_pullback_pct).
                           Defaults to 0 for missing keys (backward compatible).
        """
        if not self.is_ready:
            return None

        X = self._make_feature_vector(entry_params, context_features)
        if X is None:
            return None

        X_scaled = self._scaler.transform(X)
        x_t = torch.tensor(X_scaled, dtype=torch.float32)

        self._net.eval()
        with torch.no_grad():
            reg_out, cls_out = self._net(x_t)

        reg_np = reg_out.cpu().numpy()[0]       # shape (n_reg,)
        cls_np = cls_out.cpu().numpy()[0]       # shape (n_cls,)

        # De-normalise regression targets
        reg_vals = reg_np * (self._Y_reg_std + 1e-6) + self._Y_reg_mean

        result = {}
        for i, key in enumerate(_REG_TARGETS):
            val = float(reg_vals[i])
            if key == 'max_holding_bars':
                val = max(1, round(val))
            result[key] = val

        for i, key in enumerate(_CLS_TARGETS):
            result[key] = bool(torch.sigmoid(torch.tensor(cls_np[i])).item() >= 0.5)

        result['exit_on_close'] = True
        return result

    def knn_neighbors(
        self,
        entry_params: Dict[str, Any],
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return up to k nearest historical records by normalised Euclidean distance."""
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
        return [{**rec, 'distance': dist} for dist, rec in distances[:k]]

    def retrain(self) -> None:
        """Force full retrain from the current JSON data."""
        self._load_lookup()
        dataset = self._build_dataset()
        if dataset is None:
            return
        X, Y_reg, Y_cls = dataset
        self._scaler, self._Y_reg_mean, self._Y_reg_std, self._net = self._fit(X, Y_reg, Y_cls)
        self._trained_record_count = len(self._records)
        self._save_pt()

    # ── private helpers ───────────────────────────────────────────────────────

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
        if not _TORCH_AVAILABLE:
            return
        if len(self._records) < MIN_RECORDS:
            return

        if os.path.exists(self._pt_path):
            try:
                cached = torch.load(self._pt_path, map_location='cpu', weights_only=False)
                cached_count = cached.get('record_count', 0)
                if cached_count == len(self._records):
                    self._scaler       = cached['scaler']
                    self._Y_reg_mean   = cached['Y_reg_mean']
                    self._Y_reg_std    = cached['Y_reg_std']
                    self._feature_keys = cached.get('feature_keys', self._feature_keys)
                    cfg = cached['net_config']
                    self._net = _TbmMTLNet(**cfg)
                    self._net.load_state_dict(cached['net_state'])
                    self._net.eval()
                    self._trained_record_count = cached_count
                    return
            except Exception:
                pass  # corrupt cache — retrain

        dataset = self._build_dataset()
        if dataset is None:
            return
        X, Y_reg, Y_cls = dataset
        self._scaler, self._Y_reg_mean, self._Y_reg_std, self._net = self._fit(X, Y_reg, Y_cls)
        self._trained_record_count = len(self._records)
        self._save_pt()

    def _make_feature_vector(
        self,
        entry_params: Dict[str, Any],
        context_features: Optional[Dict[str, float]] = None,
    ) -> Optional[np.ndarray]:
        """Build (1, n_features) numpy array, same lookup order as TbmPredictor."""
        numeric = _numeric_entry_params(entry_params)
        ctx = context_features if context_features else {}
        row: List[float] = []
        for key in self._feature_keys:
            if key in numeric:
                row.append(float(numeric[key]))
            elif key in ctx:
                row.append(float(ctx[key]))
            else:
                row.append(0.0)
        return np.array(row, dtype=float).reshape(1, -1)

    def _build_dataset(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        X_rows:     List[List[float]] = []
        Y_reg_rows: List[List[float]] = []
        Y_cls_rows: List[List[int]]   = []

        for rec in self._records:
            numeric = rec.get('numeric_entry_params', {})
            context = rec.get('context_features', {})
            tbm     = rec.get('tbm_params', {})

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

    def _fit(
        self,
        X: np.ndarray,
        Y_reg: np.ndarray,
        Y_cls: np.ndarray,
        n_epochs: Optional[int] = None,
    ) -> Tuple[Any, np.ndarray, np.ndarray, '_TbmMTLNet']:
        """
        Fit scaler + net on (X, Y_reg, Y_cls).

        Returns (scaler, Y_reg_mean, Y_reg_std, net).
        Uses an 80/20 split for early stopping when n >= 25, otherwise
        trains on the full set without early stopping.
        """
        epochs = n_epochs if n_epochs is not None else self.n_epochs
        n = len(X)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        Y_reg_mean = Y_reg.mean(axis=0)
        Y_reg_std  = Y_reg.std(axis=0)
        Y_reg_n    = (Y_reg - Y_reg_mean) / (Y_reg_std + 1e-6)

        use_val = n >= 25
        if use_val:
            n_val   = max(1, int(n * 0.2))
            idx_val = np.random.default_rng(42).choice(n, n_val, replace=False)
            mask_val = np.zeros(n, dtype=bool)
            mask_val[idx_val] = True

            X_train_s  = X_scaled[~mask_val]
            Y_reg_train = Y_reg_n[~mask_val]
            Y_cls_train = Y_cls[~mask_val]

            X_val_s    = X_scaled[mask_val]
            Y_reg_val  = Y_reg_n[mask_val]
            Y_cls_val  = Y_cls[mask_val]
        else:
            X_train_s  = X_scaled
            Y_reg_train = Y_reg_n
            Y_cls_train = Y_cls

        n_features = X.shape[1]
        n_reg = len(_REG_TARGETS)
        n_cls = len(_CLS_TARGETS)

        net = _TbmMTLNet(n_features, n_reg, n_cls, hidden=self.hidden, dropout=self.dropout)
        optimiser = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        X_t     = torch.tensor(X_train_s,  dtype=torch.float32)
        Yr_t    = torch.tensor(Y_reg_train, dtype=torch.float32)
        Yc_t    = torch.tensor(Y_cls_train, dtype=torch.float32)

        if use_val:
            X_v_t   = torch.tensor(X_val_s,   dtype=torch.float32)
            Yr_v_t  = torch.tensor(Y_reg_val,  dtype=torch.float32)
            Yc_v_t  = torch.tensor(Y_cls_val,  dtype=torch.float32)

        best_val_loss  = float('inf')
        best_state     = None
        patience_count = 0

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for epoch in range(epochs):
                net.train()
                optimiser.zero_grad()
                reg_out, cls_out = net(X_t)
                loss_reg = mse_loss(reg_out, Yr_t)
                loss_cls = bce_loss(cls_out, Yc_t)
                loss = loss_reg + 0.5 * loss_cls
                loss.backward()
                optimiser.step()
                scheduler.step()

                if use_val:
                    net.eval()
                    with torch.no_grad():
                        reg_v, cls_v = net(X_v_t)
                        val_loss_reg = mse_loss(reg_v, Yr_v_t).item()
                        val_loss_cls = bce_loss(cls_v, Yc_v_t).item()
                        val_loss     = val_loss_reg + 0.5 * val_loss_cls

                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss  = val_loss
                        best_state     = {k: v.clone() for k, v in net.state_dict().items()}
                        patience_count = 0
                    else:
                        patience_count += 1
                        if patience_count >= self.patience:
                            break

        if use_val and best_state is not None:
            net.load_state_dict(best_state)

        net.eval()

        '''print(
            f'[TbmPredictorMTL] Trained on {n} records for strategy=\'{self.strategy}\' '
            f'({n_features} features -> {n_reg} reg + {n_cls} cls heads)'
        )'''

        return scaler, Y_reg_mean, Y_reg_std, net

    def _save_pt(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        cfg = {
            'n_features': len(self._feature_keys),
            'n_reg':      len(_REG_TARGETS),
            'n_cls':      len(_CLS_TARGETS),
            'hidden':     self.hidden,
            'dropout':    self.dropout,
        }
        torch.save({
            'record_count': self._trained_record_count,
            'feature_keys': self._feature_keys,
            'scaler':       self._scaler,
            'Y_reg_mean':   self._Y_reg_mean,
            'Y_reg_std':    self._Y_reg_std,
            'net_config':   cfg,
            'net_state':    self._net.state_dict(),
        }, self._pt_path)

    def leave_one_out_errors(self, n_epochs_loo: int = 250) -> Optional[Dict[str, Any]]:
        """
        Leave-one-out cross-validation.  Returns the same dict structure as
        ``TbmPredictor.leave_one_out_errors()`` so all notebook plots work
        unchanged with both predictors.

        Parameters
        ----------
        n_epochs_loo : Training epochs per LOO fold (default 250, less than the
                       full 500 to keep wall-clock time reasonable).
        """
        if not _TORCH_AVAILABLE:
            return None
        dataset = self._build_dataset()
        if dataset is None:
            return None

        X, Y_reg, Y_cls = dataset
        n = len(X)
        if n < MIN_RECORDS:
            return None

        n_reg = len(_REG_TARGETS)
        n_cls = len(_CLS_TARGETS)
        reg_errors = np.zeros((n, n_reg))
        cls_errors = np.zeros((n, n_cls), dtype=int)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_train    = X[mask]
            Yr_train   = Y_reg[mask]
            Yc_train   = Y_cls[mask]

            scaler, Y_reg_mean, Y_reg_std, net = self._fit(
                X_train, Yr_train, Yc_train, n_epochs=n_epochs_loo
            )

            X_val_s = scaler.transform(X[i:i+1])
            x_t     = torch.tensor(X_val_s, dtype=torch.float32)

            net.eval()
            with torch.no_grad():
                reg_out, cls_out = net(x_t)

            reg_pred = reg_out.cpu().numpy()[0] * (Y_reg_std + 1e-6) + Y_reg_mean
            cls_pred = (torch.sigmoid(cls_out).cpu().numpy()[0] >= 0.5).astype(int)

            reg_errors[i] = np.abs(reg_pred - Y_reg[i])
            cls_errors[i] = np.abs(cls_pred - Y_cls[i])

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
            f"TbmPredictorMTL(strategy={self.strategy!r}, "
            f"records={len(self._records)}, "
            f"ready={self.is_ready})"
        )
