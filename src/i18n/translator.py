"""Translation engine for multi-language support
"""

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Union

from . import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
from .errors import UnsupportedLocaleError


logger = logging.getLogger(__name__)


class Translator:
    """Main translator class for handling multi-language translations"""

    def __init__(self, translations_dir: Union[str, Path] = None):
        """Initialize translator
        
        Args:
            translations_dir: Directory containing translation files
        """
        self.translations_dir = Path(translations_dir) if translations_dir else Path(__file__).parent.parent.parent / 'locales'
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.current_locale = DEFAULT_LANGUAGE
        self._lock = Lock()

        # Load all available translations
        self._load_translations()

    def _load_translations(self):
        """Load all translation files"""
        with self._lock:
            for locale in SUPPORTED_LANGUAGES.keys():
                try:
                    translation_file = self.translations_dir / f"{locale}.json"
                    if translation_file.exists():
                        with open(translation_file, encoding='utf-8') as f:
                            self.translations[locale] = json.load(f)
                        logger.info(f"Loaded translations for locale: {locale}")
                    else:
                        logger.warning(f"Translation file not found: {translation_file}")
                        # Initialize empty translation dict
                        self.translations[locale] = {}
                except Exception as e:
                    logger.error(f"Failed to load translations for {locale}: {e}")
                    self.translations[locale] = {}

    def set_locale(self, locale: str):
        """Set current locale"""
        if locale not in SUPPORTED_LANGUAGES:
            raise UnsupportedLocaleError(locale, list(SUPPORTED_LANGUAGES.keys()))

        with self._lock:
            self.current_locale = locale
            logger.info(f"Locale changed to: {locale}")

    def get_locale(self) -> str:
        """Get current locale"""
        return self.current_locale

    def translate(self, key: str, locale: str = None, **kwargs) -> str:
        """Translate a key to the target locale
        
        Args:
            key: Translation key (dot-separated path)
            locale: Target locale (uses current if not specified)
            **kwargs: Variables for string interpolation
        
        Returns:
            Translated string
        """
        target_locale = locale or self.current_locale

        if target_locale not in SUPPORTED_LANGUAGES:
            raise UnsupportedLocaleError(target_locale, list(SUPPORTED_LANGUAGES.keys()))

        # Get translation from cache
        translation = self._get_nested_translation(target_locale, key)

        # Fallback to English if translation not found
        if translation is None and target_locale != DEFAULT_LANGUAGE:
            logger.debug(f"Translation not found for '{key}' in {target_locale}, falling back to {DEFAULT_LANGUAGE}")
            translation = self._get_nested_translation(DEFAULT_LANGUAGE, key)

        # Fallback to key if still not found
        if translation is None:
            logger.warning(f"Translation not found for key: {key}")
            translation = key

        # Perform string interpolation if needed
        if kwargs and isinstance(translation, str):
            try:
                translation = translation.format(**kwargs)
            except KeyError as e:
                logger.error(f"Missing variable for translation interpolation: {e}")
                # Return unformatted string rather than failing

        return translation

    def _get_nested_translation(self, locale: str, key: str) -> Optional[str]:
        """Get nested translation value using dot notation"""
        try:
            translations = self.translations.get(locale, {})
            keys = key.split('.')
            value = translations

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return None

            return value if isinstance(value, str) else None
        except Exception:
            return None

    def has_translation(self, key: str, locale: str = None) -> bool:
        """Check if translation exists for key"""
        target_locale = locale or self.current_locale
        return self._get_nested_translation(target_locale, key) is not None

    def get_available_locales(self) -> Dict[str, str]:
        """Get all available locales with their display names"""
        return SUPPORTED_LANGUAGES.copy()

    def reload_translations(self):
        """Reload all translation files"""
        logger.info("Reloading translations...")
        self._load_translations()

    def add_translation(self, locale: str, key: str, value: str):
        """Add or update a translation
        
        Args:
            locale: Target locale
            key: Translation key (dot-separated path)
            value: Translation value
        """
        if locale not in SUPPORTED_LANGUAGES:
            raise UnsupportedLocaleError(locale, list(SUPPORTED_LANGUAGES.keys()))

        with self._lock:
            if locale not in self.translations:
                self.translations[locale] = {}

            # Handle nested keys
            keys = key.split('.')
            current = self.translations[locale]

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

    def __call__(self, key: str, locale: str = None, **kwargs) -> str:
        """Allow translator to be called as a function"""
        return self.translate(key, locale, **kwargs)


# Global translator instance
_global_translator: Optional[Translator] = None
_translator_lock = Lock()


def get_translator() -> Translator:
    """Get global translator instance (singleton)"""
    global _global_translator

    if _global_translator is None:
        with _translator_lock:
            if _global_translator is None:
                _global_translator = Translator()

    return _global_translator


def set_global_locale(locale: str):
    """Set locale for global translator"""
    translator = get_translator()
    translator.set_locale(locale)


def t(key: str, locale: str = None, **kwargs) -> str:
    """Shorthand function for translation"""
    translator = get_translator()
    return translator.translate(key, locale, **kwargs)
