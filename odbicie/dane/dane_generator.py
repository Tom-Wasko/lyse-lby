"""
dane_generator.py
=================
Generates three model-ready CSV datasets for TBM parameter prediction,
optimising solely on the 'base' entry strategy using Optuna.

Output files (in odbicie/dane/):
  - simple_tbm.csv        : Targets: tp_mult, sl_mult, max_holding_bars
  - trailing_tp_tbm.csv   : Targets: above + tp_trail_mult
  - full_active_tbm.csv   : All TBM params; active_trail_sl=True, time_decay_sl=True

Usage:
    python odbicie/dane/dane_generator.py                  # all three CSVs
    python odbicie/dane/dane_generator.py --mode simple
    python odbicie/dane/dane_generator.py --mode trailing
    python odbicie/dane/dane_generator.py --mode full
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── project root on sys.path ──────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from odbicie.tbm.tbm import moving_triple_barrier_labels
from odbicie.mackowe_sygnaly import mackowe_sygnaly
from odbicie.strategie.odbicie import generate_odbicie_entries
from core.ladowanie_danych import create_stock_dfs, supertrend

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):  # type: ignore
        return x

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_START = "2015-01-01"
TRAIN_END   = "2022-12-31"

N_TRIALS_SIMPLE   = 30
N_TRIALS_TRAILING = 50
N_TRIALS_FULL     = 80

MIN_TRADES = 5
MARKET     = "all"
OVERWRITE  = False   # set True to regenerate all rows from scratch

# Paths
_CACHE_DIR   = os.path.abspath(os.path.join(_ROOT, "odbicie", "cache"))
DATA_CACHE   = os.path.join(_CACHE_DIR, f"dfs_cache_{MARKET}.pkl")
OUTPUT_DIR   = os.path.abspath(os.path.join(_ROOT, "odbicie", "dane"))
SIMPLE_CSV   = os.path.join(OUTPUT_DIR, "simple_tbm.csv")
TRAILING_CSV = os.path.join(OUTPUT_DIR, "trailing_tp_tbm.csv")
FULL_CSV     = os.path.join(OUTPUT_DIR, "full_active_tbm.csv")
OPT_ACTIVE_CSV = os.path.join(OUTPUT_DIR, "opt_active_tbm.csv")

DATA_SETTINGS: dict = {
    "market":              MARKET,
    "interval":            "1week",
    "vol_enabled":         True,
    "vol_ratio_window":    20,
    "vol_ratio_threshold": 1.2,
    "cmo_enabled":         True,
    "cmo_len":             6,
    "cmo_thres":           -35,
    "cmo_thres_prev":      -50,
    "week_start":          "MON",
}

# Entry parameter grid
GRID_BASE = {
    "threshold_pct": [
        0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        0.11, 0.12, 0.13, 0.14, 0.16, 0.17, 0.18, 0.19, 0.20,
        0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40,
    ],
    "max_setup_hold_bars": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        12, 14, 16, 18, 20, 22, 25, 28,
        30, 35, 40, 45, 50,
    ],
}

GRID_OPT_ACTIVE = {
    "threshold_pct": GRID_BASE["threshold_pct"],
    "max_setup_hold_bars": list(range(1, 16)),
}

# ══════════════════════════════════════════════════════════════════════════════
# CSV column schemas
# ══════════════════════════════════════════════════════════════════════════════

_COMMON_COLS: List[str] = [
    # entry params
    "threshold_pct", "max_setup_hold_bars",
    # context — trend
    "ctx_dist_ema200", "ctx_ema50_slope", "ctx_supertrend",
    # context — volatility
    "ctx_vol_z", "ctx_bb_width_ratio", "ctx_atr_pct",
    # context — momentum
    "ctx_rsi", "ctx_cmo", "ctx_williams_r",
    # context — volume
    "ctx_vol_ratio", "ctx_vol_sig",
    # context — setup
    "ctx_setup_bars", "entry_threshold_pct",
    # performance of the optimal TBM on train data
    "trades", "win_rate", "avg_return", "avg_hold_bars", "return_per_bar",
]

SIMPLE_COLS   = _COMMON_COLS + ["tp_mult", "sl_mult", "max_holding_bars"]
TRAILING_COLS = _COMMON_COLS + ["tp_mult", "sl_mult", "tp_trail_mult", "max_holding_bars"]
FULL_COLS     = _COMMON_COLS + [
    "tp_mult", "sl_mult", "tp_trail_mult", "sl_trail_mult",
    "max_holding_bars", "time_decay_mult",
    "active_trail_sl", "time_decay_sl",
]

OPT_ACTIVE_COLS = _COMMON_COLS + [
    "tp_mult", "sl_mult", "tp_trail_mult", "sl_trail_mult",
    "max_holding_bars",
    "active_trail_sl", "time_decay_sl",
]

# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _split_signals(signals_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (signals_df["signal_time"] >= start) & (signals_df["signal_time"] <= end)
    return signals_df.loc[mask].copy()


def _enrich_dfs(dfs_1d: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Add extra derived columns that are NOT produced by create_stock_dfs:
      - EMA_50 slope (5-bar % change relative to Close)
      - SUPERTREND_DIR: +1 if Close > supertrend line, -1 otherwise
    """
    out = {}
    for sym, df in tqdm(dfs_1d.items(), desc="  Enriching dfs", leave=False):
        df = df.copy()

        # EMA_50 slope — 5-bar percentage change of EMA_50
        if "EMA_50" in df.columns:
            slope = df["EMA_50"].diff(5)
            close_safe = df["Close"].replace(0, np.nan)
            df["_EMA50_SLOPE"] = (slope / close_safe).fillna(0.0)
        else:
            df["_EMA50_SLOPE"] = 0.0

        # Supertrend direction
        try:
            st_line = supertrend(df)
            df["_SUPERTREND_DIR"] = np.where(df["Close"] > st_line, 1.0, -1.0)
        except Exception:
            df["_SUPERTREND_DIR"] = 0.0

        out[sym] = df
    return out

# ══════════════════════════════════════════════════════════════════════════════
# Context feature extraction
# ══════════════════════════════════════════════════════════════════════════════

_CTX_DEFAULTS: Dict[str, float] = {
    "ctx_dist_ema200": 0.0, "ctx_ema50_slope": 0.0, "ctx_supertrend": 0.0,
    "ctx_vol_z": 0.0, "ctx_bb_width_ratio": 0.0, "ctx_atr_pct": 0.0,
    "ctx_rsi": 0.0, "ctx_cmo": 0.0, "ctx_williams_r": 0.0,
    "ctx_vol_ratio": 0.0, "ctx_vol_sig": 0.0, "ctx_setup_bars": 0.0,
}

def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if np.isnan(v) or np.isinf(v) else v
    except Exception:
        return default


def _compute_context_features(
    entries_df: pd.DataFrame,
    dfs_1d: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """
    For every entry in entries_df, look up the daily bar at entry_time and
    extract indicator values. Aggregate (mean) across all entries.
    """
    records: List[Dict[str, float]] = []

    for _, entry in entries_df.iterrows():
        symbol      = entry["symbol"]
        entry_time  = entry["entry_time"]
        signal_time = entry["signal_time"]
        entry_price = _safe_float(entry.get("entry_price", 0.0))

        if symbol not in dfs_1d:
            continue
        df = dfs_1d[symbol]
        if df.empty:
            continue

        # Locate entry bar (exact or nearest)
        if entry_time in df.index:
            bar = df.loc[entry_time]
        else:
            idx_pos = df.index.searchsorted(entry_time, side="left")
            if idx_pos >= len(df):
                continue
            bar = df.iloc[idx_pos]

        def g(col: str) -> float:
            return _safe_float(bar.get(col, 0.0) if hasattr(bar, "get") else getattr(bar, col, 0.0))

        # ctx_dist_ema200: (entry_price - EMA_200) / entry_price
        ema200 = g("EMA_200")
        if entry_price != 0 and ema200 != 0:
            dist_ema200 = (entry_price - ema200) / entry_price
        else:
            dist_ema200 = 0.0

        # setup_bars: count daily bars strictly after signal_time up to entry_time
        try:
            sig_idx = df.index.searchsorted(signal_time, side="right")
            ent_idx = df.index.searchsorted(entry_time,  side="right")
            setup_bars = float(max(0, ent_idx - sig_idx))
        except Exception:
            setup_bars = 0.0

        records.append({
            "ctx_dist_ema200":    _safe_float(dist_ema200),
            "ctx_ema50_slope":    g("_EMA50_SLOPE"),
            "ctx_supertrend":     g("_SUPERTREND_DIR"),
            "ctx_vol_z":          g("VOL_EST_Z"),
            "ctx_bb_width_ratio": g("BB_WIDTH"),
            "ctx_atr_pct":        g("ATR_PCT"),
            "ctx_rsi":            g("RSI"),
            "ctx_cmo":            g("CMO"),
            "ctx_williams_r":     g("WILLR"),
            "ctx_vol_ratio":      g("vol_ratio"),
            "ctx_vol_sig":        g("VOL_SIGNIFICANT"),
            "ctx_setup_bars":     setup_bars,
        })

    if not records:
        return dict(_CTX_DEFAULTS)

    ctx_df = pd.DataFrame(records)
    return {col: round(float(ctx_df[col].mean()), 6) for col in ctx_df.columns}

# ══════════════════════════════════════════════════════════════════════════════
# Trade metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_rpb(trds: pd.DataFrame) -> float:
    if trds.empty:
        return 0.0
    closed = trds[trds.get("exit_reason", pd.Series([""] * len(trds))) != "OPEN"] \
        if "exit_reason" in trds.columns else trds
    if closed.empty:
        return 0.0
    valid = closed.dropna(subset=["return_pct", "hold_bars"])
    valid = valid[valid["hold_bars"] > 0]
    if valid.empty:
        return 0.0
    return float((valid["return_pct"] / valid["hold_bars"]).mean())


def _n_closed(trds: pd.DataFrame) -> int:
    if trds.empty:
        return 0
    if "exit_reason" in trds.columns:
        return int((trds["exit_reason"] != "OPEN").sum())
    return len(trds)


def _trade_metrics(trds: pd.DataFrame) -> Dict[str, float]:
    if trds.empty:
        return {"trades": 0, "win_rate": 0.0, "avg_return": 0.0,
                "avg_hold_bars": 0.0, "return_per_bar": 0.0}
    closed = trds[trds["exit_reason"] != "OPEN"] if "exit_reason" in trds.columns else trds
    closed = closed.dropna(subset=["return_pct", "hold_bars"])
    closed = closed[closed["hold_bars"] > 0]
    if closed.empty:
        return {"trades": 0, "win_rate": 0.0, "avg_return": 0.0,
                "avg_hold_bars": 0.0, "return_per_bar": 0.0}
    return {
        "trades":        len(closed),
        "win_rate":      round(float((closed["return_pct"] > 0).mean() * 100), 2),
        "avg_return":    round(float(closed["return_pct"].mean()), 4),
        "avg_hold_bars": round(float(closed["hold_bars"].mean()), 2),
        "return_per_bar": round(float((closed["return_pct"] / closed["hold_bars"]).mean()), 6),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Optuna objectives (one per CSV variant)
# ══════════════════════════════════════════════════════════════════════════════

def _make_simple_obj(entries_df: pd.DataFrame, dfs_1d: Dict[str, pd.DataFrame]):
    def objective(trial: optuna.Trial) -> float:
        tp   = trial.suggest_float("tp_mult",          0.5,  2.25, step=0.05)
        sl   = trial.suggest_float("sl_mult",          0.5,  8.0,  step=0.25)
        mhb  = trial.suggest_int(  "max_holding_bars", 3,    50)
        trds = moving_triple_barrier_labels(
            entries_df=entries_df, market_data_daily=dfs_1d,
            tp_mult=tp, sl_mult=sl, tp_trail_mult=0.01, sl_trail_mult=1.0,
            max_holding_bars=mhb, active_trailing_sl=False,
            time_decay_sl=False, time_decay_mult=1.0, exit_on_close=True,
        )
        if _n_closed(trds) < MIN_TRADES:
            return 0.0
        rpb = _compute_rpb(trds)
        return rpb if not np.isnan(rpb) else 0.0
    return objective


def _make_trailing_obj(entries_df: pd.DataFrame, dfs_1d: Dict[str, pd.DataFrame]):
    def objective(trial: optuna.Trial) -> float:
        tp   = trial.suggest_float("tp_mult",          0.5,  2.25, step=0.05)
        sl   = trial.suggest_float("sl_mult",          0.5,  8.0,  step=0.25)
        mhb  = trial.suggest_int(  "max_holding_bars", 3,    50)
        ttp  = trial.suggest_float("tp_trail_mult",    0.01, 1.0,  step=0.01)
        trds = moving_triple_barrier_labels(
            entries_df=entries_df, market_data_daily=dfs_1d,
            tp_mult=tp, sl_mult=sl, tp_trail_mult=ttp, sl_trail_mult=1.0,
            max_holding_bars=mhb, active_trailing_sl=False,
            time_decay_sl=False, time_decay_mult=1.0, exit_on_close=True,
        )
        if _n_closed(trds) < MIN_TRADES:
            return 0.0
        rpb = _compute_rpb(trds)
        return rpb if not np.isnan(rpb) else 0.0
    return objective


def _make_full_obj(entries_df: pd.DataFrame, dfs_1d: Dict[str, pd.DataFrame]):
    def objective(trial: optuna.Trial) -> float:
        tp   = trial.suggest_float("tp_mult",          0.5,  5.0,  step=0.1)
        sl   = trial.suggest_float("sl_mult",          0.5,  8.0,  step=0.25)
        mhb  = trial.suggest_int(  "max_holding_bars", 3,    50)
        ttp  = trial.suggest_float("tp_trail_mult",    0.01, 1.0,  step=0.01)
        sltr = trial.suggest_float("sl_trail_mult",    0.2,  10.0, step=0.1)
        tdm  = trial.suggest_float("time_decay_mult",  0.1,  8.0,  step=0.1)
        trds = moving_triple_barrier_labels(
            entries_df=entries_df, market_data_daily=dfs_1d,
            tp_mult=tp, sl_mult=sl, tp_trail_mult=ttp, sl_trail_mult=sltr,
            max_holding_bars=mhb, active_trailing_sl=True,
            time_decay_sl=True, time_decay_mult=tdm, exit_on_close=True,
        )
        if _n_closed(trds) < MIN_TRADES:
            return 0.0
        rpb = _compute_rpb(trds)
        return rpb if not np.isnan(rpb) else 0.0
    return objective


def _make_opt_active_obj(entries_df: pd.DataFrame, dfs_1d: Dict[str, pd.DataFrame]):
    def objective(trial: optuna.Trial) -> float:
        tp   = trial.suggest_float("tp_mult",          0.5,  3.0,  step=0.1)
        sl   = trial.suggest_float("sl_mult",          0.5,  5.0,  step=0.25)
        mhb  = trial.suggest_int(  "max_holding_bars", 3,    20)
        ttp  = trial.suggest_float("tp_trail_mult",    0.01, 0.5,  step=0.01)
        sltr = trial.suggest_float("sl_trail_mult",    0.0,  6.0,  step=0.25)
        trds = moving_triple_barrier_labels(
            entries_df=entries_df, market_data_daily=dfs_1d,
            tp_mult=tp, sl_mult=sl, tp_trail_mult=ttp, sl_trail_mult=sltr,
            max_holding_bars=mhb, active_trailing_sl=True,
            time_decay_sl=False, time_decay_mult=1.0, exit_on_close=True,
        )
        if _n_closed(trds) < MIN_TRADES:
            return 0.0
        rpb = _compute_rpb(trds)
        return rpb if not np.isnan(rpb) else 0.0
    return objective


def process_combo(
    threshold_pct: float,
    max_setup_hold_bars: int,
    train_signals: pd.DataFrame,
    dfs_1d: Dict[str, pd.DataFrame],
    mode: str,
    lock
):
    """Worker function for parallel processing of entry combos."""
    entries_df = generate_odbicie_entries(
        signals_df=train_signals,
        market_data_daily=dfs_1d,
        threshold_pct=threshold_pct,
        max_setup_hold_bars=int(max_setup_hold_bars),
        enter_on_close=True,
    )

    if entries_df.empty or len(entries_df) < MIN_TRADES:
        return f"SKIP (only {len(entries_df)} entries)"

    entry_symbols = set(entries_df["symbol"].unique())
    dfs_combo = {sym: dfs_1d[sym] for sym in entry_symbols if sym in dfs_1d}
    ctx = _compute_context_features(entries_df, dfs_combo)

    common_row: Dict = {
        "threshold_pct":       threshold_pct,
        "max_setup_hold_bars": int(max_setup_hold_bars),
        **ctx,
        "entry_threshold_pct": threshold_pct,
    }

    if mode == "opt_active":
        study = optuna.create_study(direction="maximize")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(
                _make_opt_active_obj(entries_df, dfs_combo),
                n_trials=80,
                show_progress_bar=False,
            )

        if study.best_value > 0:
            p = study.best_trial.params
            trds = moving_triple_barrier_labels(
                entries_df=entries_df, market_data_daily=dfs_combo,
                tp_mult=p["tp_mult"], sl_mult=p["sl_mult"],
                tp_trail_mult=p["tp_trail_mult"], sl_trail_mult=p["sl_trail_mult"],
                max_holding_bars=p["max_holding_bars"],
                active_trailing_sl=True, time_decay_sl=False,
                time_decay_mult=1.0, exit_on_close=True,
            )
            row = {**common_row, **_trade_metrics(trds),
                   "tp_mult":          p["tp_mult"],
                   "sl_mult":          p["sl_mult"],
                   "tp_trail_mult":    p["tp_trail_mult"],
                   "sl_trail_mult":    p["sl_trail_mult"],
                   "max_holding_bars": p["max_holding_bars"],
                   "active_trail_sl":  True,
                   "time_decay_sl":    False}
            
            with lock:
                _append_row(OPT_ACTIVE_CSV, row, OPT_ACTIVE_COLS)
            return f"OPT_ACTIVE rpb={study.best_value:.4f} n={row['trades']}"
        else:
            return "OPT_ACTIVE no valid trial"
            
    return "UNKNOWN MODE"

# ══════════════════════════════════════════════════════════════════════════════
# CSV I/O
# ══════════════════════════════════════════════════════════════════════════════

def _load_existing_keys(csv_path: str) -> set:
    """Return set of (threshold_pct_rounded, max_setup_hold_bars) already written."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(zip(df["threshold_pct"].round(4), df["max_setup_hold_bars"].astype(int)))
    except Exception:
        return set()


def _append_row(csv_path: str, row: dict, columns: List[str]) -> None:
    """Append a single row to CSV; write header only if file is new/empty."""
    # Ensure all columns are present (fill missing with NaN)
    for col in columns:
        if col not in row:
            row[col] = np.nan
    row_df = pd.DataFrame([row], columns=columns)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    row_df.to_csv(csv_path, mode="a", header=write_header, index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(mode: str = "all") -> None:
    run_simple   = mode in ("simple",   "all")
    run_trailing = mode in ("trailing", "all")
    run_full     = mode in ("full",     "all")
    run_opt      = mode == "opt_active"

    print("=" * 65)
    print("  TBM Dane Generator — 'base' strategy")
    print(f"  Mode       : {mode.upper()}")
    print(f"  Train      : {TRAIN_START} -> {TRAIN_END}")
    if run_opt:
        print(f"  N_Trials   : {N_TRIALS_FULL} (parallel)")
    else:
        print(f"  N_Trials   : simple={N_TRIALS_SIMPLE}  trailing={N_TRIALS_TRAILING}  full={N_TRIALS_FULL}")
    print(f"  Min trades : {MIN_TRADES}   |   Overwrite: {OVERWRITE}")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(_CACHE_DIR,  exist_ok=True)

    # ── Load (or build) market data ──────────────────────────────────────────
    print("\n[1/4] Loading market data ...")
    if os.path.exists(DATA_CACHE):
        print(f"  Cache found: {DATA_CACHE}")
        with open(DATA_CACHE, "rb") as f:
            dfs_1d, dfs_1w = pickle.load(f)
    else:
        print("  No cache — loading from CSVs (this will take a few minutes) ...")
        dfs_1d, dfs_1w = create_stock_dfs(DATA_SETTINGS)
        with open(DATA_CACHE, "wb") as f:
            pickle.dump((dfs_1d, dfs_1w), f)
        print(f"  Saved data cache -> {DATA_CACHE}")
    print(f"  Loaded {len(dfs_1d)} daily DataFrames.")

    # ── Enrich daily dfs with Supertrend + EMA50 slope ───────────────────────
    print("\n[2/4] Adding Supertrend + EMA50 slope columns ...")
    dfs_1d = _enrich_dfs(dfs_1d)

    # ── Generate candlestick signals ──────────────────────────────────────────
    print("\n[3/4] Generating candlestick signals ...")
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
    train_signals = _split_signals(signals_df, TRAIN_START, TRAIN_END)
    print(f"  Total signals: {len(signals_df)}  |  Train signals: {len(train_signals)}")

    if train_signals.empty:
        print("ERROR: No train signals found. Aborting.")
        sys.exit(1)

    # ── OPT_ACTIVE mode (Parallel) ──────────────────────────────────────────
    if run_opt:
        print("\n[4/4] Parallel optimized generation (opt_active) ...")
        manager = Manager()
        lock = manager.Lock()
        
        # Filter symbols to only those that could have signals to reduce memory transfer
        # but for simplicity we'll pass dfs_1d. In a very large dataset, we'd prune first.
        
        combos = list(product(GRID_OPT_ACTIVE["threshold_pct"], GRID_OPT_ACTIVE["max_setup_hold_bars"]))
        total = len(combos)
        
        existing = _load_existing_keys(OPT_ACTIVE_CSV) if not OVERWRITE else set()
        to_process = [c for c in combos if (round(c[0], 4), int(c[1])) not in existing]
        print(f"  Grid: {total} combos | To process: {len(to_process)}")

        if not to_process:
            print("  All combos already exist. Done.")
            return

        # We use a limited number of workers to avoid Optuna/Memory overhead
        # DataFrames being shared across processes might consume significant RAM
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(process_combo, thr, mhb, train_signals, dfs_1d, mode, lock): (thr, mhb)
                for thr, mhb in to_process
            }
            
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Processing")
            saved_count = 0
            for future in pbar:
                res = future.result()
                if "OPT_ACTIVE rpb=" in res:
                    saved_count += 1
                if "SKIP" not in res:
                    pbar.set_postfix_str(res[:40])

        print(f"\n  DONE. Saved {saved_count} new rows to {OPT_ACTIVE_CSV}")
        return

    # ── Legacy modes (Single-threaded) ───────────────────────────────────────
    # Determine which combos still need processing
    existing_simple   = _load_existing_keys(SIMPLE_CSV)   if run_simple   and not OVERWRITE else set()
    existing_trailing = _load_existing_keys(TRAILING_CSV) if run_trailing and not OVERWRITE else set()
    existing_full     = _load_existing_keys(FULL_CSV)     if run_full     and not OVERWRITE else set()

    combos = list(product(GRID_BASE["threshold_pct"], GRID_BASE["max_setup_hold_bars"]))
    total  = len(combos)
    print(f"\n[4/4] Entry grid: {total} combos  ({len(GRID_BASE['threshold_pct'])} thresholds × {len(GRID_BASE['max_setup_hold_bars'])} mhb)")

    saved = {"simple": 0, "trailing": 0, "full": 0}

    for idx, (threshold_pct, max_setup_hold_bars) in enumerate(
        tqdm(combos, desc="Entry combos")
    ):
        key = (round(threshold_pct, 4), int(max_setup_hold_bars))

        need_simple   = run_simple   and key not in existing_simple
        need_trailing = run_trailing and key not in existing_trailing
        need_full     = run_full     and key not in existing_full

        if not need_simple and not need_trailing and not need_full:
            continue

        # ── Generate entries ─────────────────────────────────────────────────
        entries_df = generate_odbicie_entries(
            signals_df=train_signals,
            market_data_daily=dfs_1d,
            threshold_pct=threshold_pct,
            max_setup_hold_bars=int(max_setup_hold_bars),
            enter_on_close=True,
        )

        if entries_df.empty or len(entries_df) < MIN_TRADES:
            continue

        entry_symbols = set(entries_df["symbol"].unique())
        dfs_combo = {sym: dfs_1d[sym] for sym in entry_symbols if sym in dfs_1d}
        ctx = _compute_context_features(entries_df, dfs_combo)

        common_row: Dict = {
            "threshold_pct":       threshold_pct,
            "max_setup_hold_bars": int(max_setup_hold_bars),
            **ctx,
            "entry_threshold_pct": threshold_pct,
        }

        if need_simple:
            study = optuna.create_study(direction="maximize")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                study.optimize(_make_simple_obj(entries_df, dfs_combo), n_trials=N_TRIALS_SIMPLE)
            if study.best_value > 0:
                p = study.best_trial.params
                trds = moving_triple_barrier_labels(
                    entries_df=entries_df, market_data_daily=dfs_combo,
                    tp_mult=p["tp_mult"], sl_mult=p["sl_mult"],
                    tp_trail_mult=0.01, sl_trail_mult=1.0,
                    max_holding_bars=p["max_holding_bars"],
                    active_trailing_sl=False, time_decay_sl=False,
                    time_decay_mult=1.0, exit_on_close=True,
                )
                row = {**common_row, **_trade_metrics(trds),
                       "tp_mult": p["tp_mult"], "sl_mult": p["sl_mult"],
                       "max_holding_bars": p["max_holding_bars"]}
                _append_row(SIMPLE_CSV, row, SIMPLE_COLS)
                saved["simple"] += 1

        if need_trailing:
            study = optuna.create_study(direction="maximize")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                study.optimize(_make_trailing_obj(entries_df, dfs_combo), n_trials=N_TRIALS_TRAILING)
            if study.best_value > 0:
                p = study.best_trial.params
                trds = moving_triple_barrier_labels(
                    entries_df=entries_df, market_data_daily=dfs_combo,
                    tp_mult=p["tp_mult"], sl_mult=p["sl_mult"],
                    tp_trail_mult=p["tp_trail_mult"], sl_trail_mult=1.0,
                    max_holding_bars=p["max_holding_bars"],
                    active_trailing_sl=False, time_decay_sl=False,
                    time_decay_mult=1.0, exit_on_close=True,
                )
                row = {**common_row, **_trade_metrics(trds),
                       "tp_mult": p["tp_mult"], "sl_mult": p["sl_mult"],
                       "tp_trail_mult": p["tp_trail_mult"],
                       "max_holding_bars": p["max_holding_bars"]}
                _append_row(TRAILING_CSV, row, TRAILING_COLS)
                saved["trailing"] += 1

        if need_full:
            study = optuna.create_study(direction="maximize")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                study.optimize(_make_full_obj(entries_df, dfs_combo), n_trials=N_TRIALS_FULL)
            if study.best_value > 0:
                p = study.best_trial.params
                trds = moving_triple_barrier_labels(
                    entries_df=entries_df, market_data_daily=dfs_combo,
                    tp_mult=p["tp_mult"], sl_mult=p["sl_mult"],
                    tp_trail_mult=p["tp_trail_mult"], sl_trail_mult=p["sl_trail_mult"],
                    max_holding_bars=p["max_holding_bars"],
                    active_trailing_sl=True, time_decay_sl=True,
                    time_decay_mult=p["time_decay_mult"], exit_on_close=True,
                )
                row = {**common_row, **_trade_metrics(trds),
                       "tp_mult": p["tp_mult"], "sl_mult": p["sl_mult"],
                       "tp_trail_mult": p["tp_trail_mult"], "sl_trail_mult": p["sl_trail_mult"],
                       "max_holding_bars": p["max_holding_bars"], "time_decay_mult": p["time_decay_mult"],
                       "active_trail_sl": True, "time_decay_sl": True}
                _append_row(FULL_CSV, row, FULL_COLS)
                saved["full"] += 1

    print("\n" + "=" * 65)
    print("  DONE")
    if run_simple: print(f"  simple_tbm.csv       : {saved['simple']} new rows")
    if run_trailing: print(f"  trailing_tp_tbm.csv  : {saved['trailing']} new rows")
    if run_full: print(f"  full_active_tbm.csv  : {saved['full']} new rows")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TBM Dane Generator — base strategy")
    parser.add_argument(
        "--mode",
        choices=["simple", "trailing", "full", "opt_active", "all"],
        default="all",
        help="Which CSV variant(s) to generate (default: all)",
    )
    args = parser.parse_args()
    main(mode=args.mode)
