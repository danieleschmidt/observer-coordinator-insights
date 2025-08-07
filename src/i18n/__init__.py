"""
Internationalization (i18n) Framework for Global-First Neuromorphic Clustering System
Provides multi-language support, localized error messages, and cultural adaptations

Supported Languages:
- English (en)
- Spanish (es) 
- French (fr)
- German (de)
- Japanese (ja)
- Chinese (zh)
"""

# Avoid circular imports by importing translator function conditionally
try:
    from .translator import Translator
    _translator_available = True
except ImportError:
    _translator_available = False

def get_translator(language=None):
    """Get translator instance avoiding circular imports"""
    if not _translator_available:
        from .translator import Translator
    from .translator import get_translator as _get_translator
    return _get_translator(language)
from .locale_manager import LocaleManager
from .cultural_adapter import CulturalAdapter
from .errors import I18nError, UnsupportedLocaleError
from .formatters import LocalizedFormatter

__all__ = [
    'Translator',
    'get_translator', 
    'LocaleManager',
    'CulturalAdapter',
    'I18nError',
    'UnsupportedLocaleError',
    'LocalizedFormatter'
]

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Español', 
    'fr': 'Français',
    'de': 'Deutsch',
    'ja': '日本語',
    'zh': '中文'
}

# Default language
DEFAULT_LANGUAGE = 'en'