"""Locale management for detecting and setting user preferences
"""

import logging
import os
from threading import Lock
from typing import List, Optional

from .errors import UnsupportedLocaleError


# Define constants directly to avoid circular imports
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Español',
    'fr': 'Français',
    'de': 'Deutsch',
    'ja': '日本語',
    'zh': '中文'
}
DEFAULT_LANGUAGE = 'en'

logger = logging.getLogger(__name__)


class LocaleManager:
    """Manages locale detection and user preferences"""

    def __init__(self):
        self.user_preferences: dict = {}
        self._lock = Lock()

    def detect_locale(self,
                     accept_language: str = None,
                     user_preference: str = None,
                     fallback: str = DEFAULT_LANGUAGE) -> str:
        """Detect best locale from various sources
        
        Args:
            accept_language: HTTP Accept-Language header
            user_preference: User's saved preference
            fallback: Fallback locale
            
        Returns:
            Best matching locale
        """
        # Priority order:
        # 1. User preference (if valid)
        # 2. Accept-Language header
        # 3. Environment variable
        # 4. Fallback

        candidates = []

        # User preference
        if user_preference and self.is_supported_locale(user_preference):
            candidates.append(user_preference)

        # HTTP Accept-Language header
        if accept_language:
            header_locales = self._parse_accept_language(accept_language)
            candidates.extend([loc for loc in header_locales if self.is_supported_locale(loc)])

        # Environment variables
        env_locale = self._get_env_locale()
        if env_locale and self.is_supported_locale(env_locale):
            candidates.append(env_locale)

        # Fallback
        if self.is_supported_locale(fallback):
            candidates.append(fallback)

        # Return first valid candidate
        for candidate in candidates:
            if candidate in SUPPORTED_LANGUAGES:
                return candidate

        # Ultimate fallback
        return DEFAULT_LANGUAGE

    def _parse_accept_language(self, accept_language: str) -> List[str]:
        """Parse HTTP Accept-Language header
        
        Args:
            accept_language: Accept-Language header value
            
        Returns:
            List of locales ordered by preference
        """
        locales = []

        try:
            # Split by comma and parse quality values
            entries = []
            for entry in accept_language.split(','):
                parts = entry.strip().split(';q=')
                locale = parts[0].strip().lower()
                quality = float(parts[1]) if len(parts) > 1 else 1.0
                entries.append((locale, quality))

            # Sort by quality (descending)
            entries.sort(key=lambda x: x[1], reverse=True)

            # Extract locales
            for locale, _ in entries:
                # Handle both 'en-US' and 'en' formats
                if '-' in locale:
                    primary = locale.split('-')[0]
                    locales.extend([locale, primary])
                else:
                    locales.append(locale)

        except Exception as e:
            logger.debug(f"Error parsing Accept-Language header: {e}")

        return locales

    def _get_env_locale(self) -> Optional[str]:
        """Get locale from environment variables"""
        env_vars = ['LC_ALL', 'LC_MESSAGES', 'LANG', 'LANGUAGE']

        for var in env_vars:
            value = os.environ.get(var)
            if value:
                # Extract locale code (handle formats like 'en_US.UTF-8')
                locale = value.split('.')[0].split('_')[0].lower()
                return locale

        return None

    def is_supported_locale(self, locale: str) -> bool:
        """Check if locale is supported"""
        return locale in SUPPORTED_LANGUAGES

    def get_supported_locales(self) -> dict:
        """Get all supported locales"""
        return SUPPORTED_LANGUAGES.copy()

    def set_user_preference(self, user_id: str, locale: str):
        """Set user's locale preference
        
        Args:
            user_id: User identifier
            locale: Preferred locale
        """
        if not self.is_supported_locale(locale):
            raise UnsupportedLocaleError(locale, list(SUPPORTED_LANGUAGES.keys()))

        with self._lock:
            self.user_preferences[user_id] = locale

        logger.info(f"Set locale preference for user {user_id}: {locale}")

    def get_user_preference(self, user_id: str) -> Optional[str]:
        """Get user's locale preference
        
        Args:
            user_id: User identifier
            
        Returns:
            User's preferred locale or None
        """
        return self.user_preferences.get(user_id)

    def remove_user_preference(self, user_id: str):
        """Remove user's locale preference
        
        Args:
            user_id: User identifier
        """
        with self._lock:
            self.user_preferences.pop(user_id, None)

    def get_locale_info(self, locale: str) -> dict:
        """Get detailed information about a locale
        
        Args:
            locale: Locale code
            
        Returns:
            Dictionary with locale information
        """
        if not self.is_supported_locale(locale):
            raise UnsupportedLocaleError(locale, list(SUPPORTED_LANGUAGES.keys()))

        # RTL (Right-to-Left) languages
        rtl_languages = {'ar', 'he', 'fa', 'ur'}

        # Language direction and formatting info
        info = {
            'code': locale,
            'name': SUPPORTED_LANGUAGES[locale],
            'native_name': SUPPORTED_LANGUAGES[locale],
            'direction': 'rtl' if locale in rtl_languages else 'ltr',
            'decimal_separator': '.' if locale in ['en', 'zh', 'ja'] else ',',
            'thousand_separator': ',' if locale in ['en'] else '.' if locale in ['de', 'es'] else ' ',
            'date_format': self._get_date_format(locale),
            'time_format': '24h' if locale in ['de', 'fr', 'es'] else '12h'
        }

        return info

    def _get_date_format(self, locale: str) -> str:
        """Get date format for locale"""
        formats = {
            'en': 'MM/DD/YYYY',
            'es': 'DD/MM/YYYY',
            'fr': 'DD/MM/YYYY',
            'de': 'DD.MM.YYYY',
            'ja': 'YYYY/MM/DD',
            'zh': 'YYYY-MM-DD'
        }
        return formats.get(locale, 'DD/MM/YYYY')
