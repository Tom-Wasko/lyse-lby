"""
tbm_settings_generator.py
==========================
Automatically sweeps all entry parameter combinations for all strategies,
finds optimal TBM settings via Optuna on TRAIN data, validates on TEST data,
and saves only non-overfitted results to lookup/tbm_lookup.json.

IMPORTANT: Run download_history.py first to get data back to 2015.

Usage:
    python odbicie/tbm/tbm_settings_generator.py

Set DRY_RUN = True below to test without saving anything.
To run only specific strategies, edit STRATEGIES list.
To run only specific strategies, edit STRATEGIES list.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- Add project root to sys.path ----------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from odbicie.tbm.lookup.tbm_lookup import TbmLookup
from odbicie.tbm.predyktor.pred import TbmPredictor
from odbicie.tbm.tbm import moving_triple_barrier_labels
from odbicie.mackowe_sygnaly import mackowe_sygnaly
from odbicie.strategie.odbicie import generate_odbicie_entries
from odbicie.strategie.odbicie_atr import generate_odbicie_atr_entries
from odbicie.strategie.odbicie_bb import generate_odbicie_bb_entries
from core.ladowanie_danych import create_stock_dfs

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x  # type: ignore

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these before running
# ══════════════════════════════════════════════════════════════════════════════

# Time-based train/test split
TRAIN_START = "2015-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2026-12-31"   # effectively 'today' if data is current

# Optuna trials per entry-param combo (warm-start fallback only; zero-shot uses 0)
N_TRIALS = 20

# Overfit gate thresholds
MIN_TEST_RPB      = 0.02    # test return/bar must be at least this (positive)
MAX_OVERFIT_RATIO = 4.0    # train_rpb must not be more than this × test_rpb
MIN_TEST_TRADES   = 8      # minimum closed trades on test set

# Which strategies to run ('base', 'atr', 'bb')
STRATEGIES = ['bb']

# Dry run: compute everything but do NOT save
DRY_RUN = False

# Market to use for data loading
# 'all' = sp500 + europe + crypto (includes NASDAQ-100 and MidCap-400 which
# are stored in data_sp500_1d/ after running download_history.py)
MARKET = "all"

# Paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "cache"))
LOOKUP_PATH = os.path.join(_CACHE_DIR, "lookup/tbm_lookup.json")
# NOTE: Delete this cache file after running download_history.py to pick up new symbols
DATA_CACHE  = os.path.join(_CACHE_DIR, f"dfs_cache_{MARKET}.pkl")

# -- Entry param grids ---------------------------------------------------------
# Only numeric params; enter_on_close is always True.

GRID_BASE = {
    "threshold_pct":       [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40],
    "max_setup_hold_bars": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 28, 35],
}

GRID_ATR = {
    "atr_period":          [7, 10, 14, 17, 20, 25],
    "atr_factor":          [2.0, 2.5, 3.0, 3.5, 3.8, 4.0, 4.25, 4.5, 5.0, 5.5],
    "max_setup_hold_bars": [3, 5, 7, 10, 15, 20],
}

GRID_BB = {
    "bb_period":           [5, 7, 10, 14, 20, 30],
    "bb_std":              [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5],
    "max_setup_hold_bars": [3, 5, 7, 10, 15, 20],
}

# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

DATA_SETTINGS = {
    "market":             MARKET,
    "interval":           "1week",
    "vol_enabled":        True,
    "vol_ratio_window":   20,
    "vol_ratio_threshold": 1.2,
    "cmo_enabled":        True,
    "cmo_len":            6,
    "cmo_thres":          -35,
    "cmo_thres_prev":     -50,
    "week_start":         "MON",
}


def _split_dfs(dfs_dict: Dict[str, pd.DataFrame], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Return a new dict with each DataFrame sliced to [start, end] (inclusive)."""
    result = {}
    for sym, df in dfs_dict.items():
        sliced = df.loc[(df.index >= start) & (df.index <= end)]
        if not sliced.empty:
            result[sym] = sliced
    return result


def _split_signals(signals_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter signals_df to those whose signal_time is within [start, end]."""
    mask = (signals_df["signal_time"] >= start) & (signals_df["signal_time"] <= end)
    return signals_df.loc[mask].copy()


def _compute_rpb(trds: pd.DataFrame) -> float:
    """Return average return-per-bar; 0.0 if empty."""
    if trds.empty or "return_pct" not in trds.columns or "hold_bars" not in trds.columns:
        return 0.0
    closed = trds[trds["hold_bars"] > 0].copy()
    if closed.empty:
        return 0.0
    return (closed["return_pct"] / closed["hold_bars"]).mean()


def _n_closed_trades(trds: pd.DataFrame) -> int:
    if trds.empty:
        return 0
    return int((trds.get("exit_reason", pd.Series()) != "OPEN").sum())


# ══════════════════════════════════════════════════════════════════════════════
# Entry generation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_entries(
    strategy: str,
    entry_params: Dict[str, Any],
    signals_df: pd.DataFrame,
    dfs_1d: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Generate entries for a given strategy and entry_params dict."""
    eoc = entry_params.get("enter_on_close", True)

    if strategy == "base":
        return generate_odbicie_entries(
            signals_df=signals_df,
            market_data_daily=dfs_1d,
            threshold_pct=entry_params["threshold_pct"],
            max_setup_hold_bars=entry_params["max_setup_hold_bars"],
            enter_on_close=eoc,
        )
    elif strategy == "atr":
        return generate_odbicie_atr_entries(
            signals_df=signals_df,
            market_data_daily=dfs_1d,
            atr_period=entry_params["atr_period"],
            atr_factor=entry_params["atr_factor"],
            max_setup_hold_bars=entry_params["max_setup_hold_bars"],
            enter_on_close=eoc,
        )
    elif strategy == "bb":
        return generate_odbicie_bb_entries(
            signals_df=signals_df,
            market_data_daily=dfs_1d,
            bb_period=entry_params["bb_period"],
            bb_std=entry_params["bb_std"],
            max_setup_hold_bars=entry_params["max_setup_hold_bars"],
            enter_on_close=eoc,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════

def _make_objective(
    entries_df: pd.DataFrame,
    dfs_1d_full: Dict[str, pd.DataFrame],  # full daily data for TBM execution
):
    """Factory returning an Optuna objective that optimises TBM params."""

    def objective(trial: optuna.Trial) -> float:
        tp_mult        = trial.suggest_float("tp_mult",        0.1, 5.0, step=0.1)
        sl_mult        = trial.suggest_float("sl_mult",        0.2, 8.0, step=0.1)
        tp_trail_mult  = trial.suggest_float("tp_trail_mult",  0.01, 1.0, step=0.01)
        sl_trail_mult  = trial.suggest_float("sl_trail_mult",  0.2, 10.0, step=0.1)
        max_holding_bars = trial.suggest_int("max_holding_bars", 3, 25)
        active_trail_sl  = trial.suggest_categorical("active_trail_sl", [True, False])
        time_decay_sl    = trial.suggest_categorical("time_decay_sl",   [True, False])
        time_decay_mult  = trial.suggest_float("time_decay_mult", 0.1, 8.0, step=0.1)

        trds = moving_triple_barrier_labels(
            entries_df=entries_df,
            market_data_daily=dfs_1d_full,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            tp_trail_mult=tp_trail_mult,
            sl_trail_mult=sl_trail_mult,
            max_holding_bars=max_holding_bars,
            active_trailing_sl=active_trail_sl,
            time_decay_sl=time_decay_sl,
            time_decay_mult=time_decay_mult,
            exit_on_close=True,
        )
        if _n_closed_trades(trds) < 5:
            return 0.0
        rpb = _compute_rpb(trds)
        return rpb if not np.isnan(rpb) else 0.0

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# Per-combo pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_tbm_params(
    tbm_params: Dict[str, Any],
    train_entries: pd.DataFrame,
    test_entries: pd.DataFrame,
    dfs_1d_full: Dict[str, pd.DataFrame],
) -> Tuple[float, float, int]:
    """
    Run TBM with the given params on train and test entries.
    Returns (train_rpb, test_rpb, test_n).
    """
    train_trds = moving_triple_barrier_labels(
        entries_df=train_entries,
        market_data_daily=dfs_1d_full,
        tp_mult=tbm_params["tp_mult"],
        sl_mult=tbm_params["sl_mult"],
        tp_trail_mult=tbm_params["tp_trail_mult"],
        sl_trail_mult=tbm_params["sl_trail_mult"],
        max_holding_bars=int(tbm_params["max_holding_bars"]),
        active_trailing_sl=bool(tbm_params["active_trail_sl"]),
        time_decay_sl=bool(tbm_params["time_decay_sl"]),
        time_decay_mult=tbm_params["time_decay_mult"],
        exit_on_close=True,
    )
    train_rpb = _compute_rpb(train_trds)

    test_trds = moving_triple_barrier_labels(
        entries_df=test_entries,
        market_data_daily=dfs_1d_full,
        tp_mult=tbm_params["tp_mult"],
        sl_mult=tbm_params["sl_mult"],
        tp_trail_mult=tbm_params["tp_trail_mult"],
        sl_trail_mult=tbm_params["sl_trail_mult"],
        max_holding_bars=int(tbm_params["max_holding_bars"]),
        active_trailing_sl=bool(tbm_params["active_trail_sl"]),
        time_decay_sl=bool(tbm_params["time_decay_sl"]),
        time_decay_mult=tbm_params["time_decay_mult"],
        exit_on_close=True,
    )
    test_rpb = _compute_rpb(test_trds)
    test_n = _n_closed_trades(test_trds)
    return train_rpb, test_rpb, test_n


def _passes_overfit_gate(train_rpb: float, test_rpb: float, test_n: int) -> Tuple[bool, str]:
    """Check overfit gate. Returns (passed, status_string)."""
    if test_n < MIN_TEST_TRADES:
        return False, f"reject_too_few_test_trades({test_n})"
    if test_rpb < MIN_TEST_RPB:
        return False, f"reject_test_rpb_too_low({test_rpb:.4f})"
    if train_rpb > 0 and test_rpb > 0:
        ratio = train_rpb / test_rpb
    else:
        ratio = float("inf")
    if ratio > MAX_OVERFIT_RATIO:
        return False, f"reject_overfit(ratio={ratio:.2f})"
    return True, f"ratio={ratio:.2f}"


def _run_combo(
    strategy: str,
    entry_params: Dict[str, Any],
    train_signals: pd.DataFrame,
    test_signals:  pd.DataFrame,
    dfs_1d_full:   Dict[str, pd.DataFrame],
    lookup: TbmLookup,
    predictor: Optional["TbmPredictor"] = None,
) -> Dict[str, Any]:
    """
    Full pipeline for one (strategy, entry_params) combination.

    Stage 0 — Zero-shot prediction (if predictor is ready):
        The ML model predicts 9 TBM params. If they pass the overfit gate,
        save and return immediately (0 Optuna trials).

    Stage 1 — Warm-start Optuna fallback:
        Enqueue the ML prediction + up to 3 KNN historical neighbors into
        Optuna, then run N_TRIALS trials. Optuna evaluates the enqueued
        candidates first, then Bayesian-searches nearby regions.

    Returns a result dict with 'status' key.
    """
    result: Dict[str, Any] = {"strategy": strategy, "entry_params": entry_params, "status": "?"}

    # -- 1. Generate entries ---------------------------------------------------
    train_entries = _make_entries(strategy, entry_params, train_signals, dfs_1d_full)
    if train_entries.empty or len(train_entries) < 10:
        result["status"] = "skip_too_few_train_entries"
        return result

    test_entries = _make_entries(strategy, entry_params, test_signals, dfs_1d_full)
    if test_entries.empty:
        result["status"] = "skip_no_test_entries"
        return result

    entry_key_params = {k: v for k, v in entry_params.items() if k != "enter_on_close"}

    # ── Stage 0: Zero-shot ML prediction ──────────────────────────────────────
    if predictor is not None and predictor.is_ready:
        predicted_tbm = predictor.predict(entry_params)
        if predicted_tbm is not None:
            try:
                train_rpb, test_rpb, test_n = _evaluate_tbm_params(
                    predicted_tbm, train_entries, test_entries, dfs_1d_full
                )
                passed, gate_str = _passes_overfit_gate(train_rpb, test_rpb, test_n)
                if passed:
                    ratio = round(train_rpb / test_rpb, 3) if test_rpb > 0 else float("inf")
                    metrics = {
                        "return_per_bar": test_rpb,
                        "train_return_per_bar": train_rpb,
                        "test_trade_count": test_n,
                        "overfit_ratio": ratio,
                    }
                    if not DRY_RUN:
                        lookup.save(strategy, entry_key_params, predicted_tbm, metrics)
                    result["status"] = "SAVED_ZERO_SHOT" if not DRY_RUN else "DRY_RUN_ZERO_SHOT"
                    result["train_rpb"] = train_rpb
                    result["test_rpb"] = test_rpb
                    result["test_trades"] = test_n
                    result["best_tbm"] = predicted_tbm
                    result["metrics"] = metrics
                    return result
            except Exception:
                pass  # prediction or backtest failed — fall through to Optuna

    # ── Stage 1: Warm-start Optuna ────────────────────────────────────────────
    study = optuna.create_study(direction="maximize")

    # Collect warm-start candidates to enqueue
    warm_candidates: List[Dict[str, Any]] = []

    # a) ML zero-shot prediction (if available but didn't pass gate)
    if predictor is not None and predictor.is_ready:
        predicted_tbm = predictor.predict(entry_params)
        if predicted_tbm is not None:
            warm_candidates.append(predicted_tbm)

    # b) Up to 3 KNN historical neighbors
    if predictor is not None:
        neighbors = predictor.knn_neighbors(entry_params, k=3)
        for nb in neighbors:
            warm_candidates.append(nb["tbm_params"])

    # Enqueue all candidates (Optuna evaluates these before Bayesian sampling)
    _OPTUNA_PARAM_MAP = {
        "tp_mult":         lambda t: t.suggest_float("tp_mult",        0.1, 5.0, step=0.1),
        "sl_mult":         lambda t: t.suggest_float("sl_mult",        0.2, 8.0, step=0.1),
        "tp_trail_mult":   lambda t: t.suggest_float("tp_trail_mult",  0.01, 1.0, step=0.01),
        "sl_trail_mult":   lambda t: t.suggest_float("sl_trail_mult",  0.2, 10.0, step=0.1),
        "time_decay_mult": lambda t: t.suggest_float("time_decay_mult", 0.1, 8.0, step=0.1),
        "max_holding_bars":lambda t: t.suggest_int("max_holding_bars", 3, 25),
        "active_trail_sl": lambda t: t.suggest_categorical("active_trail_sl", [True, False]),
        "time_decay_sl":   lambda t: t.suggest_categorical("time_decay_sl",   [True, False]),
    }

    for candidate in warm_candidates:
        try:
            enqueue_dict = {
                "tp_mult":          float(round(max(0.1, min(5.0, candidate["tp_mult"])), 1)),
                "sl_mult":          float(round(max(0.2, min(8.0, candidate["sl_mult"])), 1)),
                "tp_trail_mult":    float(round(max(0.01, min(1.0, candidate["tp_trail_mult"])), 2)),
                "sl_trail_mult":    float(round(max(0.2, min(10.0, candidate["sl_trail_mult"])), 1)),
                "time_decay_mult":  float(round(max(0.1, min(8.0, candidate["time_decay_mult"])), 1)),
                "max_holding_bars": int(max(3, min(25, round(candidate["max_holding_bars"])))),
                "active_trail_sl":  bool(candidate["active_trail_sl"]),
                "time_decay_sl":    bool(candidate["time_decay_sl"]),
            }
            study.enqueue_trial(enqueue_dict)
        except Exception:
            pass  # skip malformed candidate

    # Run Optuna (enqueued trials execute first, then Bayesian sampling)
    obj = _make_objective(train_entries, dfs_1d_full)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_trial
    train_rpb = best.value
    p = best.params

    best_tbm = {
        "tp_mult":          p["tp_mult"],
        "sl_mult":          p["sl_mult"],
        "tp_trail_mult":    p["tp_trail_mult"],
        "sl_trail_mult":    p["sl_trail_mult"],
        "active_trail_sl":  p["active_trail_sl"],
        "time_decay_sl":    p["time_decay_sl"],
        "time_decay_mult":  p["time_decay_mult"],
        "max_holding_bars": p["max_holding_bars"],
        "exit_on_close":    True,
    }

    result["train_rpb"] = train_rpb
    result["best_tbm"]  = best_tbm

    # -- Validate on test ------------------------------------------------------
    _, test_rpb, test_n = _evaluate_tbm_params(
        best_tbm, train_entries, test_entries, dfs_1d_full
    )

    result["test_rpb"]    = test_rpb
    result["test_trades"] = test_n

    # -- Overfit gate ----------------------------------------------------------
    passed, gate_str = _passes_overfit_gate(train_rpb, test_rpb, test_n)
    if not passed:
        result["status"] = gate_str
        return result

    ratio = round(train_rpb / test_rpb, 3) if test_rpb > 0 else float("inf")

    # -- Save ------------------------------------------------------------------
    metrics = {
        "return_per_bar": test_rpb,
        "train_return_per_bar": train_rpb,
        "test_trade_count": test_n,
        "overfit_ratio": ratio,
    }

    if not DRY_RUN:
        lookup.save(strategy, entry_key_params, best_tbm, metrics)

    result["status"] = "SAVED_WARM_START" if not DRY_RUN else "DRY_RUN_WARM_START"
    result["metrics"] = metrics
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  TBM Settings Generator")
    print(f"  Strategies : {STRATEGIES}")
    print(f"  Train      : {TRAIN_START} -> {TRAIN_END}")
    print(f"  Test       : {TEST_START} -> {TEST_END}")
    print(f"  Optuna trials per combo : {N_TRIALS}")
    print(f"  Overfit gate : test_rpb>={MIN_TEST_RPB}, ratio<={MAX_OVERFIT_RATIO}, trades>={MIN_TEST_TRADES}")
    print(f"  DRY_RUN    : {DRY_RUN}")
    print("=" * 65)

    # -- Load data -------------------------------------------------------------
    import pickle
    print("\nLoading market data ...")
    if os.path.exists(DATA_CACHE):
        print(f"  Using cache: {DATA_CACHE}")
        with open(DATA_CACHE, "rb") as f:
            dfs_1d, dfs_1w = pickle.load(f)
    else:
        print("  No cache found, loading from CSVs (this may take a while)...")
        dfs_1d, dfs_1w = create_stock_dfs(DATA_SETTINGS)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(DATA_CACHE, "wb") as f:
            pickle.dump((dfs_1d, dfs_1w), f)
        print(f"  Saved cache to {DATA_CACHE}")

    print(f"  Loaded {len(dfs_1d)} daily symbol DataFrames.")

    # -- Generate signals (on the full weekly dataset) -------------------------
    print("\nGenerating candlestick signals (full history) ...")
    signals_df = mackowe_sygnaly(
        dfs=dfs_1w,
        settings=DATA_SETTINGS,
        require_vol_confirmation=True,
        require_cmo_confirmation=True,
        interval="1w",
        entry_offset=0,
        pattern_cols=["hammer", "inverted_hammer", "engulfing_bull", "piercing_line"],
        debug=True,
    )
    print(f"  Total signals: {len(signals_df)}")

    # -- Split signals ---------------------------------------------------------
    train_signals = _split_signals(signals_df, TRAIN_START, TRAIN_END)
    test_signals  = _split_signals(signals_df, TEST_START,  TEST_END)
    print(f"  Train signals: {len(train_signals)} | Test signals: {len(test_signals)}")

    if train_signals.empty:
        print("ERROR: No train signals found. Check date range and data.")
        sys.exit(1)

    # -- Lookup store ----------------------------------------------------------
    lookup = TbmLookup(LOOKUP_PATH)
    print(f"\nLookup store: {lookup}")

    # -- Meta-learning predictors (one per strategy) ---------------------------
    predictors: Dict[str, Optional[TbmPredictor]] = {}
    for strat in STRATEGIES:
        try:
            pred = TbmPredictor(LOOKUP_PATH, strat, cache_dir=_CACHE_DIR)
            predictors[strat] = pred
            print(f"  Predictor [{strat}]: {pred}")
        except Exception as exc:
            print(f"  Predictor [{strat}]: unavailable ({exc})")
            predictors[strat] = None

    # -- Build strategy grids --------------------------------------------------
    strategy_grids: Dict[str, List[Dict]] = {}

    if "base" in STRATEGIES:
        g = GRID_BASE
        combos = [
            {"threshold_pct": t, "max_setup_hold_bars": m, "enter_on_close": True}
            for t, m in product(g["threshold_pct"], g["max_setup_hold_bars"])
        ]
        strategy_grids["base"] = combos

    if "atr" in STRATEGIES:
        g = GRID_ATR
        combos = [
            {"atr_period": p, "atr_factor": f, "max_setup_hold_bars": m, "enter_on_close": True}
            for p, f, m in product(g["atr_period"], g["atr_factor"], g["max_setup_hold_bars"])
        ]
        strategy_grids["atr"] = combos

    if "bb" in STRATEGIES:
        g = GRID_BB
        combos = [
            {"bb_period": p, "bb_std": s, "max_setup_hold_bars": m, "enter_on_close": True}
            for p, s, m in product(g["bb_period"], g["bb_std"], g["max_setup_hold_bars"])
        ]
        strategy_grids["bb"] = combos

    total_combos = sum(len(v) for v in strategy_grids.values())
    print(f"\nTotal entry-param combos to evaluate: {total_combos}")

    # -- Run -------------------------------------------------------------------
    results: List[Dict] = []
    combo_idx = 0

    for strategy, combos in strategy_grids.items():
        print(f"\n{'-'*65}")
        print(f"  Strategy: {strategy.upper()}  ({len(combos)} combos)")
        print(f"{'-'*65}")

        for entry_params in tqdm(combos, desc=f"  {strategy}", unit="combo"):
            combo_idx += 1

            # --- SKIP if already exists in lookup ---
            entry_key_params = {k: v for k, v in entry_params.items() if k != "enter_on_close"}
            existing = lookup.lookup(strategy, entry_key_params, max_distance=0)
            if existing:
                # Still show in tqdm output but skip processing
                # tqdm.write(f"  [{combo_idx}/{total_combos}] {strategy} {_short_params(entry_params)} -> SKIPPED (already exists)")
                continue

            res = _run_combo(
                strategy=strategy,
                entry_params=entry_params,
                train_signals=train_signals,
                test_signals=test_signals,
                dfs_1d_full=dfs_1d,
                lookup=lookup,
                predictor=predictors.get(strategy),
            )
            results.append(res)
            status = res["status"]
            train_rpb = res.get("train_rpb", float("nan"))
            test_rpb  = res.get("test_rpb",  float("nan"))
            tqdm.write(
                f"  [{combo_idx}/{total_combos}] {strategy} {_short_params(entry_params)} "
                f"-> {status} "
                f"(train={train_rpb:.3f}, test={test_rpb:.3f})"
                if not np.isnan(train_rpb)
                else f"  [{combo_idx}/{total_combos}] {strategy} {_short_params(entry_params)} -> {status}"
            )

    # -- Summary ---------------------------------------------------------------
    _print_summary(results, lookup)


def _short_params(ep: Dict) -> str:
    parts = [f"{k}={v}" for k, v in ep.items() if k != "enter_on_close"]
    return "{" + ", ".join(parts) + "}"


def _print_summary(results: List[Dict], lookup: TbmLookup):
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    zero_shot  = [r for r in results if "ZERO_SHOT" in r["status"]]
    warm_start = [r for r in results if "WARM_START" in r["status"]]
    saved      = zero_shot + warm_start
    rejected   = [r for r in results if r["status"].startswith("reject")]
    skipped    = [r for r in results if r["status"].startswith("skip")]

    print(f"  Total combos   : {len(results)}")
    print(f"  Saved (zero-shot)  : {len(zero_shot)}")
    print(f"  Saved (warm-start) : {len(warm_start)}")
    print(f"  Rejected (overfit/low perf) : {len(rejected)}")
    print(f"  Skipped (not enough data)   : {len(skipped)}")

    if saved:
        print("\n  Saved combos:")
        for r in saved:
            m = r.get("metrics", {})
            print(
                f"    [{r['strategy']}] {_short_params(r['entry_params'])} "
                f"| test_rpb={m.get('return_per_bar', 0):.4f} "
                f"| train_rpb={m.get('train_return_per_bar', 0):.4f} "
                f"| trades={m.get('test_trade_count', 0)} "
                f"| ratio={m.get('overfit_ratio', 0):.2f}"
            )

    print(f"\n  Lookup store: {lookup}")
    if not DRY_RUN:
        print(f"  Results saved to: {LOOKUP_PATH}")
    else:
        print("  DRY_RUN mode — nothing was written to disk.")


if __name__ == "__main__":
    main()
