from typing import Dict, Any

from acestep.gradio_ui.ui_settings import save_ui_settings


def save_generation_settings(autosave_outputs: bool, autosave_dir: str) -> Dict[str, Any]:
    settings = save_ui_settings(
        {
            "autosave_outputs": bool(autosave_outputs),
            "autosave_dir": autosave_dir or "",
        }
    )
    return settings
