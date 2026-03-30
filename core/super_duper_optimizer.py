"""
super_duper_optimizer.py
========================
A highly reusable, universal Multi-Start Optimization + DBSCAN Clustering pipeline.

Drop into any project.  The only requirement is that the objective function:
  - accepts a 1-D array-like of parameters as its **first** positional argument.
  - returns a **scalar** fitness score.

Features
--------
* Gradient-free local search via scipy's Nelder-Mead or L-BFGS-B (configurable).
* Simulated-Annealing (SA) via scipy's dual_annealing as an alternative engine.
* Support for mixed continuous/integer parameter spaces through optional
  ``integer_mask`` — integer parameters are rounded before each function call.
* Multi-start: ``n_runs`` independent random initialisations.
* DBSCAN clustering of the final parameter coordinates to detect distinct
  optima (both global and local).
* Noise points (cluster == -1) are each treated as their own distinct optimum.
* Passes *args/**kwargs through to the objective function transparently.
* Returns a clean, sorted list of dicts: [{params, score, cluster, n_runs_in_cluster}, ...]

Typical usage
-------------
>>> from core.super_duper_optimizer import MultimodalOptimizer
>>>
>>> def my_objective(params, my_data):
...     x, y = params
...     return -(x**2 + y**2)          # maximise -> return positive value
>>>
>>> opt = MultimodalOptimizer(
...     objective_fn=my_objective,
...     bounds=[(-5, 5), (-5, 5)],
...     n_runs=30,
...     eps=0.5,
...     min_samples=2,
...     maximize=True,
... )
>>> results = opt.optimize(my_data)
>>> for r in results[:3]:
...     print(r['score'], r['params'])
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, dual_annealing
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MultimodalOptimizer:
    """
    Multi-start optimization pipeline with DBSCAN-based optimum discovery.

    Parameters
    ----------
    objective_fn : callable
        The scalar objective function.  Signature::

            score = objective_fn(params: np.ndarray, *args, **kwargs) -> float

    bounds : sequence of (float, float)
        ``[(lo, hi), ...]`` — one pair per parameter dimension.
    n_runs : int
        Number of independent random starting points (higher → more thorough search).
    eps : float
        DBSCAN neighbourhood radius in **normalised** parameter space
        (the optimizer StandardScales the collected endpoints before clustering).
        A good starting value is ``0.5``; increase if too many clusters appear.
    min_samples : int
        DBSCAN minimum cluster size.  Set to ``1`` so that every isolated point
        is treated as a valid optimum rather than noise.
    maximize : bool
        ``True``  → maximise ``objective_fn``  (internally negated for scipy).
        ``False`` → minimise ``objective_fn``.
    method : str
        Local-search engine.  One of:

        * ``'nelder-mead'`` — derivative-free simplex; robust for noisy/discrete
          functions and the **recommended default** for this project.
        * ``'l-bfgs-b'``   — gradient-based (estimated numerically); faster for
          smooth continuous functions.
        * ``'dual-annealing'`` — global stochastic search; ignores ``n_runs``
          (each call is its own multi-start); most thorough but slowest.

    integer_mask : sequence of bool, optional
        ``True`` at position ``i`` means parameter ``i`` should be rounded to the
        nearest integer before every objective function evaluation.  Pass ``None``
        (default) when all parameters are continuous.
    scale_before_cluster : bool
        If ``True`` (default), StandardScale the collected endpoints before DBSCAN
        so that parameters with very different magnitudes are treated equally.
    random_seed : int or None
        Seed for NumPy's RNG.  ``None`` → fully random.
    verbose : bool
        Print a one-line progress update for each completed run.
    nelder_mead_options : dict, optional
        Extra options forwarded to ``scipy.optimize.minimize`` when
        ``method='nelder-mead'``.  Example: ``{'maxiter': 5000, 'xatol': 1e-4}``.
    """

    def __init__(
        self,
        objective_fn: Callable,
        bounds: Sequence[Tuple[float, ...]],
        n_runs: int = 20,
        eps: float = 0.5,
        min_samples: int = 1,
        maximize: bool = False,
        method: str = "nelder-mead",
        integer_mask: Optional[Sequence[bool]] = None,
        scale_before_cluster: bool = True,
        random_seed: Optional[int] = None,
        verbose: bool = False,
        nelder_mead_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.objective_fn = objective_fn
        self.bounds = list(bounds)
        self.scipy_bounds = [(b[0], b[1]) for b in self.bounds]
        
        steps = []
        for b in self.bounds:
            if len(b) >= 3 and b[2] is not None:
                steps.append(float(b[2]))
            else:
                steps.append(0.0)
        self.steps = np.array(steps)
        self.has_steps = np.any(self.steps > 0)
        
        self.n_runs = n_runs
        self.eps = eps
        self.min_samples = min_samples
        self.maximize = maximize
        self.method = method.lower()
        self.integer_mask = (
            np.array(integer_mask, dtype=bool)
            if integer_mask is not None
            else None
        )
        self.scale_before_cluster = scale_before_cluster
        self.random_seed = random_seed
        self.verbose = verbose
        self.nelder_mead_options = nelder_mead_options or {}
        self._eval_cache: Dict[Tuple[float, ...], float] = {}

        if self.method not in ("nelder-mead", "l-bfgs-b", "dual-annealing"):
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose from 'nelder-mead', 'l-bfgs-b', or 'dual-annealing'."
            )

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def optimize(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Execute the multi-start search and cluster the results.

        Any positional or keyword arguments beyond ``params`` that the
        objective function requires should be passed here; they are forwarded
        transparently on every call.

        Returns
        -------
        list of dict
            Sorted by fitness (best first).  Each entry contains:

            ``params``
                ``np.ndarray`` of optimised parameter values.
            ``score``
                Objective function value at those parameters
                (actual, not negated — honours ``maximize``).
            ``cluster``
                DBSCAN cluster label (``-1`` = noise / isolated point).
            ``n_runs_in_cluster``
                How many of the ``n_runs`` runs converged to this cluster.
        """
        self._eval_cache.clear()
        
        rng = np.random.default_rng(self.random_seed)

        if self.method == "dual-annealing":
            return self._run_dual_annealing(rng, *args, **kwargs)

        endpoints: List[np.ndarray] = []
        raw_scores: List[float] = []        # always "minimisation" oriented

        # Pre-compute bound arrays once for speed
        _lows  = np.array([b[0] for b in self.scipy_bounds])
        _highs = np.array([b[1] for b in self.scipy_bounds])

        def _wrapped(x: np.ndarray) -> float:
            # Clip first so Nelder-Mead never evaluates outside the declared bounds
            x_clipped = np.clip(x, _lows, _highs)
            x_eval = self._apply_discrete_snapping(x_clipped, _lows, _highs)
            
            # Use tuple of exact mapped parameters as cache key
            cache_key = tuple(x_eval.tolist())
            if cache_key in self._eval_cache:
                score = self._eval_cache[cache_key]
            else:
                score = self.objective_fn(x_eval, *args, **kwargs)
                self._eval_cache[cache_key] = score
                
            return -float(score) if self.maximize else float(score)

        iterator_range = range(self.n_runs)
        if self.verbose:
            try:
                from tqdm.notebook import tqdm
            except ImportError:
                from tqdm import tqdm
            iterator_range = tqdm(iterator_range, desc="Multi-start Runs")

        for run_idx in iterator_range:
            x0 = np.array(
                [rng.uniform(lo, hi) for lo, hi in self.scipy_bounds]
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if self.method == "l-bfgs-b":
                    res = minimize(
                        _wrapped,
                        x0,
                        bounds=self.scipy_bounds,
                        method="L-BFGS-B",
                    )
                else:  # nelder-mead (default)
                    opts = {"maxiter": 10_000, "xatol": 1e-5, "fatol": 1e-8}
                    opts.update(self.nelder_mead_options)
                    res = minimize(
                        _wrapped,
                        x0,
                        method="Nelder-Mead",
                        options=opts,
                    )

            x_final = np.clip(res.x, _lows, _highs)
            x_final = self._apply_discrete_snapping(x_final, _lows, _highs)

            # Get the true score at the final clipped+masked point
            true_raw_score = _wrapped(x_final)

            endpoints.append(x_final)
            raw_scores.append(true_raw_score)

            if self.verbose:
                actual = -true_raw_score if self.maximize else true_raw_score
                if hasattr(iterator_range, "set_postfix"):
                    iterator_range.set_postfix(score=f"{actual:.4g}")

        return self._cluster_and_rank(
            np.array(endpoints), np.array(raw_scores)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_discrete_snapping(self, x: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
        """Apply step sizes and optionally integer masking, ensuring we stay within bounds."""
        x_mapped = x.copy()
        
        if self.has_steps:
            for i, step in enumerate(self.steps):
                if step > 0:
                    steps_from_low = np.round((x_mapped[i] - lows[i]) / step)
                    x_mapped[i] = lows[i] + steps_from_low * step
                    
        if self.integer_mask is not None:
            x_mapped[self.integer_mask] = np.round(x_mapped[self.integer_mask])
            
        return np.clip(x_mapped, lows, highs)

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip each dimension to its declared bound."""
        lows = np.array([b[0] for b in self.scipy_bounds])
        highs = np.array([b[1] for b in self.scipy_bounds])
        return np.clip(x, lows, highs)

    def _run_dual_annealing(
        self, rng: np.random.Generator, *args: Any, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Single dual-annealing run (its own internal multi-start)."""

        _lows  = np.array([b[0] for b in self.scipy_bounds])
        _highs = np.array([b[1] for b in self.scipy_bounds])

        def _wrapped(x: np.ndarray) -> float:
            x_eval = self._apply_discrete_snapping(x, _lows, _highs)
            cache_key = tuple(x_eval.tolist())
            if cache_key in self._eval_cache:
                score = self._eval_cache[cache_key]
            else:
                score = self.objective_fn(x_eval, *args, **kwargs)
                self._eval_cache[cache_key] = score
            return -float(score) if self.maximize else float(score)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = dual_annealing(
                _wrapped,
                bounds=self.scipy_bounds,
                seed=int(rng.integers(0, 2**31)),
            )

        x_final = np.clip(res.x, _lows, _highs)
        x_final = self._apply_discrete_snapping(x_final, _lows, _highs)
        actual_score = -res.fun if self.maximize else res.fun

        return [{
            "params": x_final,
            "score": actual_score,
            "cluster": 0,
            "n_runs_in_cluster": 1,
        }]

    def _cluster_and_rank(
        self,
        endpoints: np.ndarray,   # shape (n_runs, n_dims)
        raw_scores: np.ndarray,  # shape (n_runs,)  — minimisation oriented
    ) -> List[Dict[str, Any]]:
        """DBSCAN-cluster the parameter endpoints then pick the best per cluster."""

        # Optionally scale so that parameter magnitudes don't distort distances
        if self.scale_before_cluster and endpoints.shape[0] > 1:
            scaler = StandardScaler()
            pts_for_cluster = scaler.fit_transform(endpoints)
        else:
            pts_for_cluster = endpoints

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(pts_for_cluster)
        labels = db.labels_

        cluster_optima: List[Dict[str, Any]] = []

        for label in set(labels):
            mask = labels == label
            indices = np.where(mask)[0]

            if label == -1:
                # Noise points — each is its own distinct optimum
                for idx in indices:
                    actual = -raw_scores[idx] if self.maximize else raw_scores[idx]
                    cluster_optima.append({
                        "params": endpoints[idx].copy(),
                        "score": actual,
                        "cluster": -1,
                        "n_runs_in_cluster": 1,
                    })
            else:
                # Find the single best point inside this cluster
                best_idx = indices[np.argmin(raw_scores[indices])]
                actual = (
                    -raw_scores[best_idx] if self.maximize else raw_scores[best_idx]
                )
                cluster_optima.append({
                    "params": endpoints[best_idx].copy(),
                    "score": actual,
                    "cluster": int(label),
                    "n_runs_in_cluster": int(mask.sum()),
                })

        # Sort: best score first
        cluster_optima.sort(key=lambda d: d["score"], reverse=self.maximize)
        return cluster_optima


# ---------------------------------------------------------------------------
# Convenience wrapper for pretty-printing results
# ---------------------------------------------------------------------------

def print_optima(
    optima: List[Dict[str, Any]],
    param_names: Optional[List[str]] = None,
    max_show: int = 10,
) -> None:
    """
    Pretty-print the list returned by ``MultimodalOptimizer.optimize()``.

    Parameters
    ----------
    optima : list of dict
        Return value of ``MultimodalOptimizer.optimize()``.
    param_names : list of str, optional
        Human-readable names for each parameter dimension.
    max_show : int
        Maximum number of optima to display.
    """
    print(f"{'='*60}")
    print(f"  Found {len(optima)} distinct optimum/optima")
    print(f"{'='*60}")
    for rank, opt in enumerate(optima[:max_show], 1):
        print(f"\n  Rank #{rank}  |  score = {opt['score']:.6g}"
              f"  |  cluster = {opt['cluster']}"
              f"  |  converged {opt['n_runs_in_cluster']}x")
        params = opt["params"]
        for i, val in enumerate(params):
            name = param_names[i] if param_names and i < len(param_names) else f"p[{i}]"
            print(f"    {name}: {val:.6g}")
    if len(optima) > max_show:
        print(f"\n  ... (+{len(optima) - max_show} more, not shown)")
    print(f"\n{'='*60}")
