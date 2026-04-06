"""
tbm_lookup.py
=============
Lightweight JSON-backed lookup table that maps (strategy, entry_params) → best TBM settings.

Workflow
--------
1. After Optuna finds best TBM params for a given entry config, call ``save()``.
2. Next time you want TBM params for a (possibly different) entry config, call ``suggest()``.
   It tries:  exact match → nearest neighbour → global best for strategy → None.

Excluded from keys: enter_on_close, exit_on_close (always True).

Usage
-----
    from odbicie.tbm.lookup.tbm_lookup import TbmLookup
    lookup = TbmLookup('odbicie/cache/lookup/tbm_lookup.json')

    # Save after Optuna
    lookup.save('bb', strat_settings, best_tbm, {'return_per_bar': study.best_value})

    # Suggest before / instead of Optuna
    suggestion = lookup.suggest('bb', strat_settings)
    if suggestion:
        print(suggestion['source'], suggestion['tbm_params'], suggestion['metrics'])
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Keys from entry_settings that are NOT used in the distance calculation
_EXCLUDED_ENTRY_KEYS = {"enter_on_close", "exit_on_close"}


def _numeric_entry_params(entry_params: Dict[str, Any]) -> Dict[str, float]:
    """Return only the numeric entry params (excluding boolean flags)."""
    return {
        k: float(v)
        for k, v in entry_params.items()
        if k not in _EXCLUDED_ENTRY_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _entry_key(entry_params: Dict[str, Any]) -> str:
    """Stable string key for a set of entry params (sorted, numerics only)."""
    numeric = _numeric_entry_params(entry_params)
    parts = [f"{k}={v}" for k, v in sorted(numeric.items())]
    return "|".join(parts)


def _normalized_distance(
    params_a: Dict[str, float],
    params_b: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]],
) -> float:
    """
    Euclidean distance between two param dicts, normalized by range.
    Keys present in A but not in B (or vice-versa) contribute full distance (1.0 each).
    """
    all_keys = set(params_a) | set(params_b)
    sq_sum = 0.0
    for k in all_keys:
        va = params_a.get(k)
        vb = params_b.get(k)
        lo, hi = ranges.get(k, (0.0, 1.0))
        span = hi - lo if hi != lo else 1.0
        if va is None or vb is None:
            sq_sum += 1.0  # maximum distance contribution
        else:
            sq_sum += ((va - vb) / span) ** 2
    return math.sqrt(sq_sum)


def _compute_ranges(records: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Compute per-key (min, max) across all stored records."""
    ranges: Dict[str, list] = {}
    for rec in records:
        for k, v in rec.get("numeric_entry_params", {}).items():
            ranges.setdefault(k, []).append(v)
    return {k: (min(vs), max(vs)) for k, vs in ranges.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TbmLookup:
    """
    JSON-backed store that records the best TBM settings found for each
    combination of (strategy, entry_params).

    Parameters
    ----------
    path : str
        Absolute or relative path to the JSON file.  Created if it doesn't exist.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def _save_to_disk(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ── public API ────────────────────────────────────────────────────────────

    def save(
        self,
        strategy: str,
        entry_params: Dict[str, Any],
        tbm_params: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> None:
        """
        Save (or update) the best TBM settings for a given strategy + entry_params.

        A record is updated only if the new ``return_per_bar`` is strictly better
        than the previously stored value.

        Parameters
        ----------
        strategy    : 'base', 'atr', or 'bb' (extensible to any string).
        entry_params: Full entry settings dict (boolean flags are stripped internally).
        tbm_params  : Dict of TBM settings to store.
        metrics     : Must contain at least ``return_per_bar`` for comparison.
        """
        key = _entry_key(entry_params)
        numeric = _numeric_entry_params(entry_params)
        new_rpb = metrics.get("return_per_bar", float("-inf"))

        records: List[Dict[str, Any]] = self._data.setdefault(strategy, [])

        # Try to update an existing record with the same key
        for rec in records:
            if rec.get("entry_key") == key:
                old_rpb = rec.get("metrics", {}).get("return_per_bar", float("-inf"))
                if new_rpb > old_rpb:
                    rec["tbm_params"] = tbm_params
                    rec["metrics"] = metrics
                    self._save_to_disk()
                    print(f"[TbmLookup] Updated record for strategy='{strategy}', key={key!r}")
                else:
                    print(
                        f"[TbmLookup] Existing record is better "
                        f"(stored rpb={old_rpb:.4f} vs new={new_rpb:.4f}) — not updated."
                    )
                return

        # New record
        records.append({
            "entry_key": key,
            "numeric_entry_params": numeric,
            "tbm_params": tbm_params,
            "metrics": metrics,
        })
        self._save_to_disk()
        print(f"[TbmLookup] Saved new record for strategy='{strategy}', key={key!r}")

    def lookup(
        self,
        strategy: str,
        entry_params: Dict[str, Any],
        max_distance: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the closest saved entry_params for the given strategy.

        Returns the best matching record dict (with keys ``tbm_params``, ``metrics``,
        ``entry_key``, ``distance``) or ``None`` if no records exist or all are too far.

        Parameters
        ----------
        max_distance : Normalised Euclidean distance threshold (0 = exact only).
                       Set to ``float('inf')`` to always return the nearest.
        """
        records = self._data.get(strategy, [])
        if not records:
            return None

        query_numeric = _numeric_entry_params(entry_params)
        query_key = _entry_key(entry_params)

        # Exact match fast path
        for rec in records:
            if rec.get("entry_key") == query_key:
                return {**rec, "distance": 0.0, "source": "exact"}

        # Nearest-neighbour search
        ranges = _compute_ranges(records)
        best_rec = None
        best_dist = float("inf")

        for rec in records:
            stored_numeric = rec.get("numeric_entry_params", {})
            dist = _normalized_distance(query_numeric, stored_numeric, ranges)
            if dist < best_dist:
                best_dist = dist
                best_rec = rec

        if best_rec is not None and len(records) == 1:
            # Only one record — always return it; can't do meaningful distance comparison
            return {**best_rec, "distance": best_dist, "source": "nearest"}

        if best_rec is not None and best_dist <= max_distance:
            return {**best_rec, "distance": best_dist, "source": "nearest"}

        return None

    def best_for(self, strategy: str) -> Optional[Dict[str, Any]]:
        """
        Return the globally best TBM settings for a strategy (highest return_per_bar).

        Falls back to this when no nearby entry params are found.
        """
        records = self._data.get(strategy, [])
        if not records:
            return None

        best = max(
            records,
            key=lambda r: r.get("metrics", {}).get("return_per_bar", float("-inf")),
        )
        return {**best, "source": "best_global"}

    def suggest(
        self,
        strategy: str,
        entry_params: Dict[str, Any],
        max_distance: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        """
        High-level suggestion method.

        Priority
        --------
        1. Exact match (distance == 0)
        2. Nearest neighbour within ``max_distance``
        3. Global best for strategy
        4. ``None`` (no data at all for this strategy)

        Returns a dict with keys: ``tbm_params``, ``metrics``, ``source``, ``distance``.
        Source is one of ``'exact'``, ``'nearest'``, ``'best_global'``.
        """
        result = self.lookup(strategy, entry_params, max_distance=max_distance)
        if result is not None:
            return result

        # Fallback: return global best (ignoring distance)
        global_best = self.best_for(strategy)
        if global_best is not None:
            return global_best

        return None

    def list_records(self, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return all stored records, optionally filtered by strategy.

        Useful for inspecting what's been saved.
        """
        if strategy is not None:
            return list(self._data.get(strategy, []))
        return [
            {**rec, "strategy": strat}
            for strat, recs in self._data.items()
            for rec in recs
        ]

    def __repr__(self) -> str:
        counts = {s: len(r) for s, r in self._data.items()}
        return f"TbmLookup(path={self.path!r}, records={counts})"
