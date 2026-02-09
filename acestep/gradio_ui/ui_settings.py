import json
import os
from typing import Dict, Any

from loguru import logger


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "ui_settings.json")


def load_ui_settings() -> Dict[str, Any]:
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load UI settings: {e}")
        return {}


def save_ui_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    settings = load_ui_settings()
    settings.update(updates or {})
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save UI settings: {e}")
    return settings
