"""
I18n error classes for the internationalization framework
"""


class I18nError(Exception):
    """Base exception for i18n-related errors"""
    pass


class UnsupportedLocaleError(I18nError):
    """Raised when an unsupported locale is requested"""
    
    def __init__(self, locale: str, supported_locales: list):
        self.locale = locale
        self.supported_locales = supported_locales
        super().__init__(
            f"Locale '{locale}' is not supported. "
            f"Supported locales: {', '.join(supported_locales)}"
        )


class TranslationNotFoundError(I18nError):
    """Raised when a translation key is not found"""
    
    def __init__(self, key: str, locale: str):
        self.key = key
        self.locale = locale
        super().__init__(
            f"Translation not found for key '{key}' in locale '{locale}'"
        )


class InvalidTranslationFormatError(I18nError):
    """Raised when translation format is invalid"""
    
    def __init__(self, key: str, locale: str, reason: str):
        self.key = key
        self.locale = locale
        self.reason = reason
        super().__init__(
            f"Invalid translation format for key '{key}' in locale '{locale}': {reason}"
        )