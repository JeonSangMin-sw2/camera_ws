"""
Backward compatibility bridge for core.language.
Redirects I18nManager and tr to core.language.LanguageManager.
"""
from core.language import (
    LanguageManager as I18nManager,
    LanguageManager,
    tr,
    UI_DROPDOWNS,
    DARK_STYLESHEET,
    load_ui_dropdowns,
    load_stylesheet,
)

__all__ = [
    "I18nManager",
    "LanguageManager",
    "tr",
    "UI_DROPDOWNS",
    "DARK_STYLESHEET",
    "load_ui_dropdowns",
    "load_stylesheet",
]
