"""
pred_factory.py
========================
Convenience factory so notebook cells only need one variable change to swap
between the XGBoost predictor and the MTL neural-network predictor.

Usage in main.ipynb
-------------------
    from odbicie.tbm.predyktor.pred_factory import get_predictor

    PREDICTOR_TYPE = 'xgb'   # <- change to 'mtl' for the neural network

    predictor = get_predictor(PREDICTOR_TYPE, LOOKUP_PATH, STRATEGY, cache_dir=DANE_DIR)
    print(predictor)

Both predictors share the same public API:
    predictor.predict(entry_params, context_features=None) -> dict | None
    predictor.knn_neighbors(entry_params, k=3)             -> list
    predictor.leave_one_out_errors()                       -> dict | None
    predictor.is_ready                                     -> bool
"""

from __future__ import annotations

from typing import Optional, Union


def get_predictor(
    predictor_type: str,
    lookup_path: str,
    strategy: str,
    cache_dir: Optional[str] = None,
) -> Union['TbmPredictor', 'TbmPredictorMTL']:
    """
    Return a ready-to-use predictor instance.

    Parameters
    ----------
    predictor_type : 'xgb' for XGBoost RegressorChain, 'mtl' for PyTorch MTL net.
    lookup_path    : Path to lookup/tbm_lookup.json.
    strategy       : One of 'base', 'atr', 'bb'.
    cache_dir      : Directory for cached model files (pkl / pt).
                     Defaults to the same directory as ``lookup_path``.

    Returns
    -------
    TbmPredictor or TbmPredictorMTL — both share the same public API.

    Raises
    ------
    ValueError : If ``predictor_type`` is not 'xgb' or 'mtl'.
    """
    ptype = predictor_type.strip().lower()

    if ptype == 'xgb':
        from tbm.predyktor.pred import TbmPredictor
        return TbmPredictor(lookup_path, strategy, cache_dir=cache_dir)

    if ptype == 'mtl':
        from tbm.predyktor.pred_mtl import TbmPredictorMTL
        return TbmPredictorMTL(lookup_path, strategy, cache_dir=cache_dir)

    raise ValueError(
        f"Unknown predictor_type={predictor_type!r}. Valid options: 'xgb', 'mtl'."
    )
