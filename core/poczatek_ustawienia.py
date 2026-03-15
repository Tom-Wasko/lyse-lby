import json
import os
from typing import Tuple, Dict, Any

import ipywidgets as widgets
from IPython.display import display, clear_output


def create_settings_ui(settings_path: str = "global_settings.json") -> Tuple[widgets.VBox, Dict[str, Any]]:
    """
    Create and return the settings UI panel and the settings dictionary.
    
    Args:
        settings_path: Path to the JSON settings file (default: "global_settings.json")
    
    Returns:
        Tuple containing:
            - settings_panel: The main widget panel to display
            - settings_dict: Dictionary with all current settings
    """
    
    # Create widgets
    file_path_widget = widgets.Text(
        value="",
        placeholder="/path/to/data.csv",
        description="File path:",
        layout=widgets.Layout(width="600px")
    )

    market_widget = widgets.ToggleButtons(
        options=[
            ("Krypto", "crypto"),
            ("Spółki", "stocks"),
            ("Surowce", "commodities"),
        ],
        description="Market:",
        button_style=""
    )

    interval_widget = widgets.ToggleButtons(
        options=[
            ("dzienny", "1d"),
            ("tygodniowy", "1week"),
        ],
        description="Interwal:"
    )

    week_start_widget = widgets.Dropdown(
        options=[
            #("standardowo", "BASE"),
            #("Sunday", "SUN"),
            ("poniedzialek", "MON"),
            ("wtorek", "TUE"),
            ("sroda", "WED"),
            ("czwartek", "THU"),
            ("piatek", "FRI"),
            #("Saturday", "SAT"),
        ],
        value="MON",
        description="Kiedy sie tydzien zaczyna gosciu:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"),
    )

    upload_widget = widgets.FileUpload(
        accept=".csv",
        multiple=False
    )

    save_button = widgets.Button(
        description="Save settings",
        button_style="success",
        icon="save"
    )

    status_out = widgets.Output()

    # Volume filter widgets
    vol_ratio_window_widget = widgets.IntSlider(
        value=20,
        min=5,
        max=50,
        step=1,
        description="Vol window:",
        continuous_update=False,
        style={"description_width": "120px"},
    )

    vol_ratio_threshold_widget = widgets.FloatSlider(
        value=1.5,
        min=0.1,
        max=2.0,
        step=0.1,
        description="Vol threshold:",
        continuous_update=False,
        style={"description_width": "120px"},
    )

    vol_enabled_widget = widgets.Checkbox(
        value=True,
        description="Use volume filter",
        indent=False
    )

    # CMO filter widgets
    cmo_len_widget = widgets.IntSlider(
        value=5,
        min=1,
        max=50,
        step=1,
        description="CMO length:",
        continuous_update=False,
        style={"description_width": "120px"},
    )

    cmo_thres_widget = widgets.IntSlider(
        value=20,
        min=-100,
        max=100,
        step=5,
        description="Cmo thres (current candle):",
        continuous_update=False,
        style={"description_width": "120px"},
    )

    cmo_thres_prev_widget = widgets.IntSlider(
        value=20,
        min=-100,
        max=100,
        step=5,
        description="Cmo thres (previous candle):",
        continuous_update=False,
        style={"description_width": "120px"},
    )

    cmo_enabled_widget = widgets.Checkbox(
        value=True,
        description="Use cmo filter",
        indent=False
    )

    # Save settings function
    def save_settings(_):
        settings = {
            "market": market_widget.value,
            "interval": interval_widget.value,
            'week_start': week_start_widget.value,
            "vol_enabled": vol_enabled_widget.value,
            "vol_ratio_window": vol_ratio_window_widget.value,
            "vol_ratio_threshold": vol_ratio_threshold_widget.value,
            "cmo_enabled": cmo_enabled_widget.value,
            "cmo_len": cmo_len_widget.value,
            "cmo_thres": cmo_thres_widget.value,
            "cmo_thres_prev": cmo_thres_prev_widget.value,
        }

        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)

        with status_out:
            status_out.clear_output()

    save_button.on_click(save_settings)

    # Load settings function
    def load_settings():
        if not os.path.exists(settings_path):
            return

        with open(settings_path, "r") as f:
            settings = json.load(f)

        # Use `.get()` with the existing widget default to avoid KeyError
        market_widget.value = settings.get("market", market_widget.value)
        interval_widget.value = settings.get("interval", interval_widget.value)
        week_start_widget.value = settings.get("week_start", week_start_widget.value)
        vol_enabled_widget.value = settings.get("vol_enabled", vol_enabled_widget.value)
        vol_ratio_window_widget.value = settings.get("vol_ratio_window", vol_ratio_window_widget.value)
        vol_ratio_threshold_widget.value = settings.get("vol_ratio_threshold", vol_ratio_threshold_widget.value)
        cmo_enabled_widget.value = settings.get("cmo_enabled", cmo_enabled_widget.value)
        cmo_len_widget.value = settings.get("cmo_len", cmo_len_widget.value)
        cmo_thres_widget.value = settings.get("cmo_thres", cmo_thres_widget.value)
        cmo_thres_prev_widget.value = settings.get("cmo_thres_prev", cmo_thres_prev_widget.value)

    # Show/hide week_start depending on interval selection
    def update_week_start_visibility(change=None):
        try:
            is_weekly = interval_widget.value == "1week"
        except Exception:
            is_weekly = False

        if is_weekly:
            week_start_widget.layout.display = ""
            week_start_widget.disabled = False
        else:
            week_start_widget.layout.display = "none"
            week_start_widget.disabled = True

    interval_widget.observe(update_week_start_visibility, names="value")
    # set initial visibility according to loaded/initial value
    update_week_start_visibility()

    load_settings()

    # Layout configuration
    PANEL_WIDTH = "1000px"

    MAIN_BOX = widgets.Layout(
        width=PANEL_WIDTH,
        padding="16px",
        margin="8px 0",
        border="1px solid #ddd",
        align_items="flex-start"
    )

    SECTION_BOX = widgets.Layout(
        width="95%",
        padding="12px",
        margin="12px 0",
        border="1px solid #eee",
    )

    ROW_GRID = widgets.Layout(
        width="100%",
        display="grid",
        grid_template_columns="220px 1fr 1fr",
        align_items="center",
        gap="12px",
    )

    CHECKBOX_COL = widgets.Layout(width="220px")
    SLIDER_COL = widgets.Layout(width="95%")

    # Apply layouts
    vol_enabled_widget.layout = CHECKBOX_COL
    vol_ratio_window_widget.layout = SLIDER_COL
    vol_ratio_threshold_widget.layout = SLIDER_COL

    cmo_enabled_widget.layout = CHECKBOX_COL
    cmo_len_widget.layout = SLIDER_COL
    cmo_thres_widget.layout = SLIDER_COL
    cmo_thres_prev_widget.layout = SLIDER_COL

    # Build panels
    area_panel = widgets.VBox(
        [
            widgets.HTML("<h3>Area picking</h3>"),
            widgets.VBox(
                [
                    market_widget,
                    interval_widget,
                    week_start_widget,
                ],
                layout=widgets.Layout(
                    width="100%",
                    gap="10px",
                ),
            ),
        ],
        layout=SECTION_BOX,
    )


    volume_panel = widgets.VBox(
        [
            widgets.HTML("<h3>Volume Filter Settings</h3>"),

            widgets.Box(
                [
                    # LEFT column: Enabled only
                    vol_enabled_widget,

                    # RIGHT column: other settings stacked
                    widgets.VBox(
                        [
                            vol_ratio_window_widget,
                            vol_ratio_threshold_widget,
                        ],
                        layout=widgets.Layout(gap="8px"),
                    ),
                ],
                layout=widgets.Layout(
                    width="100%",
                    display="grid",
                    grid_template_columns="220px 1fr",
                    #align_items="start",
                    gap="16px",
                ),
            ),
        ],
        layout=SECTION_BOX,
    )


    cmo_panel = widgets.VBox(
        [
            widgets.HTML("<h3>CMO Filter Settings</h3>"),

            widgets.Box(
                [
                    # LEFT column: Enabled only
                    cmo_enabled_widget,

                    # RIGHT column: other settings stacked
                    widgets.VBox(
                        [
                            cmo_len_widget,
                            cmo_thres_widget,
                            cmo_thres_prev_widget,
                        ],
                        layout=widgets.Layout(gap="8px"),
                    ),
                ],
                layout=widgets.Layout(
                    width="100%",
                    display="grid",
                    grid_template_columns="220px 1fr",
                    #align_items="start",
                    gap="16px",
                ),
            ),
        ],
        layout=SECTION_BOX,
    )



    save_button.layout = widgets.Layout(width="260px", height="48px")

    save_row = widgets.HBox(
        [save_button],
        layout=widgets.Layout(
            width="100%",
            justify_content="center",
            margin="12px 0"
        )
    )

    settings_panel = widgets.VBox(
        [
            area_panel,
            widgets.HTML("<hr>"),
            status_out,
            #widgets.HTML("<hr>"),
            #widgets.HTML("<hr>"),
            volume_panel,
            cmo_panel,
            widgets.HTML("<hr>"),
            save_row,
        ],
        layout=MAIN_BOX
    )

    # Get current settings
    settings_dict = {
        "market": market_widget.value,
        "interval": interval_widget.value,
        'week_start': week_start_widget.value,
        "vol_enabled": vol_enabled_widget.value,
        "vol_ratio_window": vol_ratio_window_widget.value,
        "vol_ratio_threshold": vol_ratio_threshold_widget.value,
        "cmo_enabled": cmo_enabled_widget.value,
        "cmo_len": cmo_len_widget.value,
        "cmo_thres": cmo_thres_widget.value,
        "cmo_thres_prev": cmo_thres_prev_widget.value,
    }

    return settings_panel, settings_dict


# Example usage
if __name__ == "__main__":
    settings_panel, settings = create_settings_ui()
    display(settings_panel)
    print(settings)
