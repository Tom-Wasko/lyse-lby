"""Add candle pattern columns to a DataFrame using TA-Lib settings from JSON files.

Function: add_candle_patterns(df, settings, settings_dir=None)

- Loads `candle_settings_1d.json` or `candle_settings_1week.json` depending on
  `settings['interval']` (defaults to 1d if missing).
- Applies TA-Lib internal candle params before computing CDL pattern functions.
- Restores TA-Lib defaults after computation.

Returns the original DataFrame with new pattern columns (float, 0/1 or -100/100
depending on TA-Lib output), and ensures missing pattern columns are present.
"""

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import talib

from talib._ta_lib import (
    _ta_set_candle_settings as set_candle_settings,
    _ta_restore_candle_default_settings as restore_candle_default_settings,
    CandleSettingType as CST,
    RangeType as RT,
)


def add_candle_patterns(df: pd.DataFrame, settings: Dict, settings_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Compute candle pattern columns for `df` using TA-Lib.

    Args:
        df: DataFrame with columns `Open, High, Low, Close` and a DatetimeIndex.
        settings: dict containing at least `interval` ("1d" or "1week") and
                  optionally other keys.
        settings_dir: directory where candle_settings JSON files live. If None,
                      looks in repository root next to this module.

    Returns:
        DataFrame with added pattern columns (float dtype).
    """

    if settings_dir is None:
        settings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    interval = settings.get("interval", "1d")
    fname = "candle_settings_1week.json" if interval == "1week" else "candle_settings_1d.json"
    settings_path = os.path.join(settings_dir, fname)

    # load params if available
    params = {}
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r") as f:
                params = json.load(f)
        except Exception:
            params = {}

    def get_param(key, default=None):
        return params.get(key, default)

    def apply_setting(setting_type, range_type, lb_key, val_key):
        lb = get_param(lb_key)
        val = get_param(val_key)
        if lb is None and val is None:
            return
        # TA-Lib expects numeric values; if missing, pass 0
        lb_v = int(lb) if lb is not None else 0
        val_v = float(val) if val is not None else 0.0
        set_candle_settings(setting_type, range_type, lb_v, val_v)

    # Get OHLC data once
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    patterns = pd.DataFrame(index=df.index)

    # HAMMER: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyShort, RT.RealBody, "HAMMER_BODY_SHORT_LB", "HAMMER_BODY_SHORT")
    apply_setting(CST.ShadowLong, RT.RealBody, "HAMMER_SHADOW_LONG_LB", "HAMMER_SHADOW_LONG")
    apply_setting(CST.ShadowVeryShort, RT.HighLow, "HAMMER_SHADOW_VERY_SHORT_LB", "HAMMER_SHADOW_VERY_SHORT")
    apply_setting(CST.Near, RT.HighLow, "HAMMER_NEAR_LB", "HAMMER_NEAR")
    patterns["hammer"] = talib.CDLHAMMER(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # INVERTED_HAMMER: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyShort, RT.RealBody, "INVERTED_HAMMER_BODY_SHORT_LB", "INVERTED_HAMMER_BODY_SHORT")
    apply_setting(CST.ShadowLong, RT.RealBody, "INVERTED_HAMMER_SHADOW_LONG_LB", "INVERTED_HAMMER_SHADOW_LONG")
    apply_setting(CST.ShadowVeryShort, RT.HighLow, "INVERTED_HAMMER_SHADOW_VERY_SHORT_LB", "INVERTED_HAMMER_SHADOW_VERY_SHORT")
    patterns["inverted_hammer"] = talib.CDLINVERTEDHAMMER(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # SHOOTING_STAR: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyShort, RT.RealBody, "SHOOTING_STAR_BODY_SHORT_LB", "SHOOTING_STAR_BODY_SHORT")
    apply_setting(CST.ShadowLong, RT.RealBody, "SHOOTING_STAR_SHADOW_LONG_LB", "SHOOTING_STAR_SHADOW_LONG")
    apply_setting(CST.ShadowVeryShort, RT.HighLow, "SHOOTING_STAR_SHADOW_VERY_SHORT_LB", "SHOOTING_STAR_SHADOW_VERY_SHORT")
    patterns["shooting_star"] = talib.CDLSHOOTINGSTAR(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # HANGING_MAN: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyShort, RT.RealBody, "HANGING_MAN_BODY_SHORT_LB", "HANGING_MAN_BODY_SHORT")
    apply_setting(CST.ShadowLong, RT.RealBody, "HANGING_MAN_SHADOW_LONG_LB", "HANGING_MAN_SHADOW_LONG")
    apply_setting(CST.ShadowVeryShort, RT.HighLow, "HANGING_MAN_SHADOW_VERY_SHORT_LB", "HANGING_MAN_SHADOW_VERY_SHORT")
    apply_setting(CST.Near, RT.HighLow, "HANGING_MAN_NEAR_LB", "HANGING_MAN_NEAR")
    patterns["hanging_man"] = talib.CDLHANGINGMAN(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # DOJI: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyDoji, RT.HighLow, "DOJI_BODY_LB", "DOJI_BODY")
    patterns["doji"] = talib.CDLDOJI(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # LONG_LEGGED_DOJI: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyDoji, RT.HighLow, "DOJI_BODY_LB", "DOJI_BODY")
    apply_setting(CST.ShadowLong, RT.RealBody, "LONG_LEGGED_DOJI_SHADOW_LONG_LB", "LONG_LEGGED_DOJI_SHADOW_LONG")
    patterns["long_legged_doji"] = talib.CDLLONGLINE(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # DRAGONFLY_DOJI: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyDoji, RT.HighLow, "DOJI_BODY_LB", "DOJI_BODY")
    apply_setting(CST.ShadowLong, RT.RealBody, "DRAGONFLY_DOJI_SHADOW_LONG_LB", "DRAGONFLY_DOJI_SHADOW_LONG")
    apply_setting(CST.ShadowVeryShort, RT.HighLow, "DRAGONFLY_DOJI_SHADOW_VERY_SHORT_LB", "DRAGONFLY_DOJI_SHADOW_VERY_SHORT")
    patterns["dragonfly_doji"] = talib.CDLDRAGONFLYDOJI(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # GRAVESTONE_DOJI: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyDoji, RT.HighLow, "DOJI_BODY_LB", "DOJI_BODY")
    apply_setting(CST.ShadowLong, RT.RealBody, "GRAVESTONE_DOJI_SHADOW_LONG_LB", "GRAVESTONE_DOJI_SHADOW_LONG")
    apply_setting(CST.ShadowVeryShort, RT.HighLow, "GRAVESTONE_DOJI_SHADOW_VERY_SHORT_LB", "GRAVESTONE_DOJI_SHADOW_VERY_SHORT")
    patterns["gravestone_doji"] = talib.CDLGRAVESTONEDOJI(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # SPINNING_TOP: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyShort, RT.RealBody, "SPINNING_TOP_BODY_SHORT_LB", "SPINNING_TOP_BODY_SHORT")
    patterns["spinning_top"] = talib.CDLSPINNINGTOP(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # MARUBOZU: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyLong, RT.RealBody, "MARUBOZU_BODY_LONG_LB", "MARUBOZU_BODY_LONG")
    apply_setting(CST.ShadowVeryShort, RT.HighLow, "MARUBOZU_SHADOW_VERY_SHORT_LB", "MARUBOZU_SHADOW_VERY_SHORT")
    patterns["marubozu"] = talib.CDLMARUBOZU(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # ENGULFING: apply settings → compute pattern and variants → restore defaults
    engulfing = talib.CDLENGULFING(o, h, l, c)
    patterns["engulfing"] = engulfing
    patterns["engulfing_bull"] = (engulfing > 0).astype(float)
    patterns["engulfing_bear"] = (engulfing < 0).astype(float)
    restore_candle_default_settings(CST.AllCandleSettings)

    # PIERCING_LINE: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyLong, RT.RealBody, "PIERCING_BODY_LONG_LB", "PIERCING_BODY_LONG")
    patterns["piercing_line"] = talib.CDLPIERCING(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # DARK_CLOUD_COVER: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyLong, RT.RealBody, "DARK_CLOUD_BODY_LONG_LB", "DARK_CLOUD_BODY_LONG")
    patterns["dark_cloud_cover"] = talib.CDLDARKCLOUDCOVER(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # ABANDONED_BABY: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyLong, RT.RealBody, "ABANDONED_BABY_BODY_LONG_LB", "ABANDONED_BABY_BODY_LONG")
    apply_setting(CST.BodyDoji, RT.HighLow, "ABANDONED_BABY_DOJI_LB", "ABANDONED_BABY_DOJI")
    patterns["abandoned_baby"] = talib.CDLABANDONEDBABY(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # MORNING_STAR: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyLong, RT.RealBody, "MORNING_STAR_BODY_LONG_LB", "MORNING_STAR_BODY_LONG")
    apply_setting(CST.BodyShort, RT.RealBody, "MORNING_STAR_BODY_SHORT_LB", "MORNING_STAR_BODY_SHORT")
    patterns["morning_star"] = talib.CDLMORNINGSTAR(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # MORNING_DOJI_STAR: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyLong, RT.RealBody, "MORNING_DOJI_STAR_BODY_LONG_LB", "MORNING_DOJI_STAR_BODY_LONG")
    apply_setting(CST.BodyDoji, RT.HighLow, "MORNING_DOJI_STAR_DOJI_LB", "MORNING_DOJI_STAR_DOJI")
    patterns["morning_doji_star"] = talib.CDLMORNINGDOJISTAR(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # EVENING_STAR: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyLong, RT.RealBody, "EVENING_STAR_BODY_LONG_LB", "EVENING_STAR_BODY_LONG")
    apply_setting(CST.BodyShort, RT.RealBody, "EVENING_STAR_BODY_SHORT_LB", "EVENING_STAR_BODY_SHORT")
    patterns["evening_star"] = talib.CDLEVENINGSTAR(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # EVENING_DOJI_STAR: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyLong, RT.RealBody, "EVENING_DOJI_STAR_BODY_LONG_LB", "EVENING_DOJI_STAR_BODY_LONG")
    apply_setting(CST.BodyDoji, RT.HighLow, "EVENING_DOJI_STAR_DOJI_LB", "EVENING_DOJI_STAR_DOJI")
    patterns["evening_doji_star"] = talib.CDLEVENINGDOJISTAR(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # THREE_WHITE_SOLDIERS: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyLong, RT.RealBody, "THREE_WHITE_SOLDIERS_BODY_LONG_LB", "THREE_WHITE_SOLDIERS_BODY_LONG")
    patterns["three_white_soldiers"] = talib.CDL3WHITESOLDIERS(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # THREE_BLACK_CROWS: apply settings → compute pattern → restore defaults
    apply_setting(CST.BodyLong, RT.RealBody, "THREE_BLACK_CROWS_BODY_LONG_LB", "THREE_BLACK_CROWS_BODY_LONG")
    patterns["three_black_crows"] = talib.CDL3BLACKCROWS(o, h, l, c)
    restore_candle_default_settings(CST.AllCandleSettings)

    # Custom signal: engulfing + shallow pullback
    try:
        engulf_bull = patterns["engulfing_bull"] > 0
        body_low = np.minimum(df["Open"], df["Close"])
        body_high = np.maximum(df["Open"], df["Close"])
        body_range = body_high - body_low
        min_level = body_high - body_range * 0.25
        max_level = body_high - body_range * 0.80
        pullback_ok = (
            (df["Low"].shift(-1) <= min_level) &
            (df["Low"].shift(-1) >= max_level)
        )
        signal = engulf_bull & pullback_ok
        patterns["objecie_i_cofniecie"] = signal.shift(1).fillna(0).astype(float)
    except Exception:
        patterns["objecie_i_cofniecie"] = 0.0

    # Add any remaining TA-Lib CDL functions not explicitly added
    import inspect

    existing = set(patterns.columns)
    for name, func in inspect.getmembers(talib, inspect.isfunction):
        if not name.startswith("CDL"):
            continue
        col = name.lower().replace("cdl", "")
        if col in existing:
            continue
        try:
            patterns[col] = func(o, h, l, c)
        except Exception:
            patterns[col] = 0.0

    # Ensure all configured plot keys exist
    for key in [
        "hammer",
        "inverted_hammer",
        "engulfing",
        "engulfing_bull",
        "engulfing_bear",
        "piercing_line",
        "evening_star",
        "evening_doji_star",
        "objecie_i_cofniecie",
    ]:
        if key not in patterns.columns:
            patterns[key] = 0.0

    # Merge into original df (avoid overwriting existing columns)
    out = df.copy()
    for col in patterns.columns:
        out[col] = patterns[col].astype(float)

    return out
