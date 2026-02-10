import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

import ipywidgets as widgets

from talib._ta_lib import (
    _ta_set_candle_settings as set_candle_settings,
    _ta_restore_candle_default_settings as restore_candle_default_settings,
    CandleSettingType as CST,
    RangeType as RT,
)


def create_talib_control(
    settings: Dict[str, Any],
    default_params: Optional[Dict[str, Any]] = None,
    settings_file: Optional[str] = None,
) -> Tuple[widgets.VBox, Callable[[], None], Dict[str, Any]]:
    """
    Build UI for controlling TA-Lib internal candle settings.

    Args:
        settings: dict containing runtime settings (expects at least 'interval')
        default_params: dictionary with default parameter values used to populate widgets
        settings_file: optional path to save/load custom settings; if None it will be
            inferred from `settings['interval']` (uses existing behaviour in repo).

    Returns:
        (settings_panel, apply_all_settings_fn, widget_registry)

    The returned `apply_all_settings_fn` when called will apply current widget values
    to TA-Lib internal candle settings (via `_ta_set_candle_settings`).
    """

    # Derived settings file if not provided
    if settings_file is None:
        settings_file = (
            "candle_settings_1week.json"
            if settings.get("interval") == "1week"
            else "candle_settings_1d.json"
        )

    # If default_params not provided, try to load from top-level `candle_settings.json`
    if default_params is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candle_path = os.path.join(project_root, "candle_settings.json")
        if os.path.exists(candle_path):
            try:
                with open(candle_path, "r") as f:
                    default_params = json.load(f)
            except Exception:
                default_params = {}
        else:
            default_params = {}

    WIDGET_REGISTRY: Dict[str, Any] = {}
    SLIDERS: Dict[str, widgets.Widget] = {}

    # --- helper functions that interact with TA-Lib ---
    def apply_setting(setting_type, range_type, lb_key, val_key):
        set_candle_settings(
            setting_type,
            range_type,
            get_value(lb_key),
            get_value(val_key),
        )

    def reset_all():
        restore_candle_default_settings(CST.AllCandleSettings)

    def reset_all_files(button=None):
        # 1. Reset TA-Lib internal defaults
        restore_candle_default_settings(CST.AllCandleSettings)

        # 2. Move sliders back to default positions
        for key, default_value in default_params.items():
            if key in SLIDERS:
                SLIDERS[key].value = default_value

        # 3. Save all default parameters
        with open(settings_file, "w") as f:
            json.dump(default_params, f, indent=4)

    def load_saved_settings():
        if os.path.exists(settings_file):
            with open(settings_file, "r") as f:
                data = json.load(f)
            for key, value in data.items():
                WIDGET_REGISTRY[key] = value
        else:
            # populate registry from defaults
            for k, v in default_params.items():
                WIDGET_REGISTRY.setdefault(k, v)

    def save_settings_to_file(_=None):
        apply_all_settings()
        data = {key: get_value(key) for key in WIDGET_REGISTRY}
        with open(settings_file, "w") as f:
            json.dump(data, f, indent=4)

    def get_value(key):
        v = WIDGET_REGISTRY.get(key, default_params.get(key))
        if isinstance(v, (widgets.IntSlider, widgets.FloatSlider)):
            return v.value
        return v

    # --- build widgets ---
    # Slider factories place widgets into registry and SLIDERS
    def lb_slider(key, label, value):
        w = widgets.IntSlider(
            description=f"[LB] {label}",
            value=value,
            min=0,
            max=50,
            step=1,
            layout=widgets.Layout(width="95%"),
            style={"description_width": "350px"},
        )
        WIDGET_REGISTRY[key] = w
        SLIDERS[key] = w
        return w

    def val_slider(key, label, value, max_val=3.0):
        w = widgets.FloatSlider(
            description=label,
            value=value,
            min=0.0,
            max=max_val,
            step=0.05,
            layout=widgets.Layout(width="95%"),
            style={"description_width": "350px"},
        )
        WIDGET_REGISTRY[key] = w
        SLIDERS[key] = w
        return w

    # --- pattern setting functions ---
    def settings_hammer():
        apply_setting(CST.BodyShort, RT.RealBody, "HAMMER_BODY_SHORT_LB", "HAMMER_BODY_SHORT")
        apply_setting(CST.ShadowLong, RT.RealBody, "HAMMER_SHADOW_LONG_LB", "HAMMER_SHADOW_LONG")
        apply_setting(CST.ShadowVeryShort, RT.HighLow, "HAMMER_SHADOW_VERY_SHORT_LB", "HAMMER_SHADOW_VERY_SHORT")
        apply_setting(CST.Near, RT.HighLow, "HAMMER_NEAR_LB", "HAMMER_NEAR")

    def settings_inverted_hammer():
        apply_setting(CST.BodyShort, RT.RealBody, "INVERTED_HAMMER_BODY_SHORT_LB", "INVERTED_HAMMER_BODY_SHORT")
        apply_setting(CST.ShadowLong, RT.RealBody, "INVERTED_HAMMER_SHADOW_LONG_LB", "INVERTED_HAMMER_SHADOW_LONG")
        apply_setting(CST.ShadowVeryShort, RT.HighLow, "INVERTED_HAMMER_SHADOW_VERY_SHORT_LB", "INVERTED_HAMMER_SHADOW_VERY_SHORT")

    def settings_shooting_star():
        apply_setting(CST.BodyShort, RT.RealBody, "SHOOTING_STAR_BODY_SHORT_LB", "SHOOTING_STAR_BODY_SHORT")
        apply_setting(CST.ShadowLong, RT.RealBody, "SHOOTING_STAR_SHADOW_LONG_LB", "SHOOTING_STAR_SHADOW_LONG")
        apply_setting(CST.ShadowVeryShort, RT.HighLow, "SHOOTING_STAR_SHADOW_VERY_SHORT_LB", "SHOOTING_STAR_SHADOW_VERY_SHORT")

    def settings_hanging_man():
        apply_setting(CST.BodyShort, RT.RealBody, "HANGING_MAN_BODY_SHORT_LB", "HANGING_MAN_BODY_SHORT")
        apply_setting(CST.ShadowLong, RT.RealBody, "HANGING_MAN_SHADOW_LONG_LB", "HANGING_MAN_SHADOW_LONG")
        apply_setting(CST.ShadowVeryShort, RT.HighLow, "HANGING_MAN_SHADOW_VERY_SHORT_LB", "HANGING_MAN_SHADOW_VERY_SHORT")
        apply_setting(CST.Near, RT.HighLow, "HANGING_MAN_NEAR_LB", "HANGING_MAN_NEAR")

    def settings_doji():
        apply_setting(CST.BodyDoji, RT.HighLow, "DOJI_BODY_LB", "DOJI_BODY")

    def settings_long_legged_doji():
        settings_doji()
        apply_setting(CST.ShadowLong, RT.RealBody, "LONG_LEGGED_DOJI_SHADOW_LONG_LB", "LONG_LEGGED_DOJI_SHADOW_LONG")

    def settings_dragonfly_doji():
        settings_doji()
        apply_setting(CST.ShadowLong, RT.RealBody, "DRAGONFLY_DOJI_SHADOW_LONG_LB", "DRAGONFLY_DOJI_SHADOW_LONG")
        apply_setting(CST.ShadowVeryShort, RT.HighLow, "DRAGONFLY_DOJI_SHADOW_VERY_SHORT_LB", "DRAGONFLY_DOJI_SHADOW_VERY_SHORT")

    def settings_gravestone_doji():
        settings_doji()
        apply_setting(CST.ShadowLong, RT.RealBody, "GRAVESTONE_DOJI_SHADOW_LONG_LB", "GRAVESTONE_DOJI_SHADOW_LONG")
        apply_setting(CST.ShadowVeryShort, RT.HighLow, "GRAVESTONE_DOJI_SHADOW_VERY_SHORT_LB", "GRAVESTONE_DOJI_SHADOW_VERY_SHORT")

    def settings_spinning_top():
        apply_setting(CST.BodyShort, RT.RealBody, "SPINNING_TOP_BODY_SHORT_LB", "SPINNING_TOP_BODY_SHORT")

    def settings_marubozu():
        apply_setting(CST.BodyLong, RT.RealBody, "MARUBOZU_BODY_LONG_LB", "MARUBOZU_BODY_LONG")
        apply_setting(CST.ShadowVeryShort, RT.HighLow, "MARUBOZU_SHADOW_VERY_SHORT_LB", "MARUBOZU_SHADOW_VERY_SHORT")

    def settings_piercing():
        apply_setting(CST.BodyLong, RT.RealBody, "PIERCING_BODY_LONG_LB", "PIERCING_BODY_LONG")

    def settings_abandoned_baby():
        apply_setting(CST.BodyLong, RT.RealBody, "ABANDONED_BABY_BODY_LONG_LB", "ABANDONED_BABY_BODY_LONG")
        apply_setting(CST.BodyDoji, RT.HighLow, "ABANDONED_BABY_DOJI_LB", "ABANDONED_BABY_DOJI")

    def settings_dark_cloud():
        apply_setting(CST.BodyLong, RT.RealBody, "DARK_CLOUD_BODY_LONG_LB", "DARK_CLOUD_BODY_LONG")

    def settings_morning_star():
        apply_setting(CST.BodyLong, RT.RealBody, "MORNING_STAR_BODY_LONG_LB", "MORNING_STAR_BODY_LONG")
        apply_setting(CST.BodyShort, RT.RealBody, "MORNING_STAR_BODY_SHORT_LB", "MORNING_STAR_BODY_SHORT")

    def settings_morning_doji_star():
        apply_setting(CST.BodyLong, RT.RealBody, "MORNING_DOJI_STAR_BODY_LONG_LB", "MORNING_DOJI_STAR_BODY_LONG")
        apply_setting(CST.BodyDoji, RT.HighLow, "MORNING_DOJI_STAR_DOJI_LB", "MORNING_DOJI_STAR_DOJI")

    def settings_evening_star():
        apply_setting(CST.BodyLong, RT.RealBody, "EVENING_STAR_BODY_LONG_LB", "EVENING_STAR_BODY_LONG")
        apply_setting(CST.BodyShort, RT.RealBody, "EVENING_STAR_BODY_SHORT_LB", "EVENING_STAR_BODY_SHORT")

    def settings_evening_doji_star():
        apply_setting(CST.BodyLong, RT.RealBody, "EVENING_DOJI_STAR_BODY_LONG_LB", "EVENING_DOJI_STAR_BODY_LONG")
        apply_setting(CST.BodyDoji, RT.HighLow, "EVENING_DOJI_STAR_DOJI_LB", "EVENING_DOJI_STAR_DOJI")

    def settings_three_white_soldiers():
        apply_setting(CST.BodyLong, RT.RealBody, "THREE_WHITE_SOLDIERS_BODY_LONG_LB", "THREE_WHITE_SOLDIERS_BODY_LONG")

    def settings_three_black_crows():
        apply_setting(CST.BodyLong, RT.RealBody, "THREE_BLACK_CROWS_BODY_LONG_LB", "THREE_BLACK_CROWS_BODY_LONG")

    # --- aggregator that applies all settings ---
    def apply_all_settings(_=None):
        reset_all()

        settings_hammer()
        settings_inverted_hammer()
        settings_shooting_star()
        settings_hanging_man()

        settings_doji()
        settings_long_legged_doji()
        settings_dragonfly_doji()
        settings_gravestone_doji()
        settings_spinning_top()
        settings_marubozu()

        settings_piercing()
        settings_dark_cloud()
        settings_abandoned_baby()

        settings_morning_star()
        settings_morning_doji_star()
        settings_evening_star()
        settings_evening_doji_star()
        settings_three_white_soldiers()
        settings_three_black_crows()

    # --- construct UI ---
    load_saved_settings()

    single_candle_box = widgets.VBox(
        [
            widgets.HTML("<h2>Single Candle Patterns</h2>"),

            widgets.HTML("<h3>Hammer</h3>"),
            lb_slider("HAMMER_BODY_SHORT_LB", "Wielkość body względem poprzednich świec", get_value('HAMMER_BODY_SHORT_LB')),
            val_slider("HAMMER_BODY_SHORT", "Wielkość body względem poprzednich świec", get_value('HAMMER_BODY_SHORT')),
            lb_slider("HAMMER_SHADOW_LONG_LB", "Dolny knot względem body", get_value('HAMMER_SHADOW_LONG_LB')),
            val_slider("HAMMER_SHADOW_LONG", "Dolny knot względem body", get_value('HAMMER_SHADOW_LONG')),
            lb_slider("HAMMER_SHADOW_VERY_SHORT_LB", "Górny knot względem body", get_value('HAMMER_SHADOW_VERY_SHORT_LB')),
            val_slider("HAMMER_SHADOW_VERY_SHORT", "Górny knot względem body", get_value('HAMMER_SHADOW_VERY_SHORT')),
            lb_slider("HAMMER_NEAR_LB", "Porównanie ze średnim gapem", get_value('HAMMER_NEAR_LB')),
            val_slider("HAMMER_NEAR", "Porównanie ze średnim gapem", get_value('HAMMER_NEAR')),

            widgets.HTML("<h3>Inverted Hammer</h3>"),
            lb_slider("INVERTED_HAMMER_BODY_SHORT_LB", "Wielkość body względem poprzednich świec", get_value('INVERTED_HAMMER_BODY_SHORT_LB')),
            val_slider("INVERTED_HAMMER_BODY_SHORT", "Wielkość body względem poprzednich świec", get_value('INVERTED_HAMMER_BODY_SHORT')),
            lb_slider("INVERTED_HAMMER_SHADOW_LONG_LB", "Górny knot względem body", get_value('INVERTED_HAMMER_SHADOW_LONG_LB')),
            val_slider("INVERTED_HAMMER_SHADOW_LONG", "Górny knot względem body", get_value('INVERTED_HAMMER_SHADOW_LONG')),
            lb_slider("INVERTED_HAMMER_SHADOW_VERY_SHORT_LB", "Dolny knot względem body", get_value('INVERTED_HAMMER_SHADOW_VERY_SHORT_LB')),
            val_slider("INVERTED_HAMMER_SHADOW_VERY_SHORT", "Dolny knot względem body", get_value('INVERTED_HAMMER_SHADOW_VERY_SHORT')),
        ],
        layout=widgets.Layout(width="70%"),
    )

    double_triple_box = widgets.VBox(
        [
            widgets.HTML("<h2>Double & Triple Candle Patterns</h2>"),
            widgets.HTML("<h3>Piercing</h3>"),
            lb_slider("PIERCING_BODY_LONG_LB", "Wielkość body względem poprzednich świec", get_value('PIERCING_BODY_LONG_LB')),
            val_slider("PIERCING_BODY_LONG", "Wielkość body względem poprzednich świec", get_value('PIERCING_BODY_LONG')),
        ],
        layout=widgets.Layout(width="70%"),
    )

    save_btn = widgets.Button(description="Save Settings", button_style="success", layout=widgets.Layout(width="70%"))
    save_btn.on_click(save_settings_to_file)

    apply_btn = widgets.Button(description="Apply TA-Lib Candle Settings", button_style="success", layout=widgets.Layout(width="70%"))
    apply_btn.on_click(apply_all_settings)

    reset_btn = widgets.Button(description="Reset settings to deafult", button_style="info", layout=widgets.Layout(width="70%"))
    reset_btn.on_click(reset_all_files)

    settings_panel = widgets.VBox(
        [
            single_candle_box,
            double_triple_box,
            save_btn,
            # apply_btn, kept for manual use
            reset_btn,
        ],
        layout=widgets.Layout(width="80%"),
    )

    # Return the panel, the apply function, and the registry so caller can inspect values
    return settings_panel, apply_all_settings, WIDGET_REGISTRY
