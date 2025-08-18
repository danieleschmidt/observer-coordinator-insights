"""Localized formatters for numbers, dates, currencies, and data processing
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Optional, Union

from . import SUPPORTED_LANGUAGES


logger = logging.getLogger(__name__)


class LocalizedFormatter:
    """Handles locale-aware formatting of numbers, dates, and other data"""

    def __init__(self, locale: str = 'en'):
        """Initialize formatter for specific locale
        
        Args:
            locale: Target locale code
        """
        self.locale = locale
        self.locale_config = self._get_locale_config(locale)

    def _get_locale_config(self, locale: str) -> dict:
        """Get formatting configuration for locale"""
        configs = {
            'en': {
                'decimal_separator': '.',
                'thousand_separator': ',',
                'currency_symbol': '$',
                'currency_position': 'before',
                'date_format': '%m/%d/%Y',
                'time_format': '%I:%M %p',
                'datetime_format': '%m/%d/%Y %I:%M %p',
                'first_day_of_week': 0  # Sunday
            },
            'es': {
                'decimal_separator': ',',
                'thousand_separator': '.',
                'currency_symbol': '€',
                'currency_position': 'after',
                'date_format': '%d/%m/%Y',
                'time_format': '%H:%M',
                'datetime_format': '%d/%m/%Y %H:%M',
                'first_day_of_week': 1  # Monday
            },
            'fr': {
                'decimal_separator': ',',
                'thousand_separator': ' ',
                'currency_symbol': '€',
                'currency_position': 'after',
                'date_format': '%d/%m/%Y',
                'time_format': '%H:%M',
                'datetime_format': '%d/%m/%Y %H:%M',
                'first_day_of_week': 1  # Monday
            },
            'de': {
                'decimal_separator': ',',
                'thousand_separator': '.',
                'currency_symbol': '€',
                'currency_position': 'after',
                'date_format': '%d.%m.%Y',
                'time_format': '%H:%M',
                'datetime_format': '%d.%m.%Y %H:%M',
                'first_day_of_week': 1  # Monday
            },
            'ja': {
                'decimal_separator': '.',
                'thousand_separator': ',',
                'currency_symbol': '¥',
                'currency_position': 'before',
                'date_format': '%Y/%m/%d',
                'time_format': '%H:%M',
                'datetime_format': '%Y/%m/%d %H:%M',
                'first_day_of_week': 0  # Sunday
            },
            'zh': {
                'decimal_separator': '.',
                'thousand_separator': ',',
                'currency_symbol': '¥',
                'currency_position': 'before',
                'date_format': '%Y-%m-%d',
                'time_format': '%H:%M',
                'datetime_format': '%Y-%m-%d %H:%M',
                'first_day_of_week': 1  # Monday
            }
        }

        return configs.get(locale, configs['en'])

    def format_number(self, value: Union[int, float, Decimal], decimal_places: int = 2) -> str:
        """Format number according to locale conventions
        
        Args:
            value: Number to format
            decimal_places: Number of decimal places
            
        Returns:
            Formatted number string
        """
        try:
            # Handle None or non-numeric values
            if value is None:
                return "N/A"

            if not isinstance(value, (int, float, Decimal)):
                return str(value)

            # Round to specified decimal places
            if isinstance(value, float):
                value = round(value, decimal_places)

            # Split integer and decimal parts
            if decimal_places > 0:
                formatted = f"{value:.{decimal_places}f}"
            else:
                formatted = str(int(value))

            # Handle locale-specific formatting
            if '.' in formatted:
                integer_part, decimal_part = formatted.split('.')
            else:
                integer_part = formatted
                decimal_part = None

            # Add thousand separators
            if len(integer_part) > 3:
                # Add separators from right to left
                chars = list(integer_part)
                for i in range(len(chars) - 3, 0, -3):
                    chars.insert(i, self.locale_config['thousand_separator'])
                integer_part = ''.join(chars)

            # Combine with decimal separator
            if decimal_part and int(decimal_part) > 0:
                result = integer_part + self.locale_config['decimal_separator'] + decimal_part
            else:
                result = integer_part

            return result

        except Exception as e:
            logger.error(f"Error formatting number {value}: {e}")
            return str(value)

    def format_percentage(self, value: Union[int, float], decimal_places: int = 1) -> str:
        """Format percentage according to locale conventions
        
        Args:
            value: Percentage value (0.0 to 1.0 or 0 to 100)
            decimal_places: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        try:
            if value is None:
                return "N/A"

            # Convert to percentage if needed (assume values > 1 are already percentages)
            if value <= 1:
                percentage = value * 100
            else:
                percentage = value

            formatted_number = self.format_number(percentage, decimal_places)
            return f"{formatted_number}%"

        except Exception as e:
            logger.error(f"Error formatting percentage {value}: {e}")
            return f"{value}%"

    def format_currency(self, value: Union[int, float, Decimal],
                       currency_code: str = None, decimal_places: int = 2) -> str:
        """Format currency according to locale conventions
        
        Args:
            value: Currency value
            currency_code: Currency code (e.g., 'USD', 'EUR')
            decimal_places: Number of decimal places
            
        Returns:
            Formatted currency string
        """
        try:
            if value is None:
                return "N/A"

            formatted_number = self.format_number(value, decimal_places)

            # Use provided currency or default
            symbol = currency_code or self.locale_config['currency_symbol']

            # Position currency symbol
            if self.locale_config['currency_position'] == 'before':
                return f"{symbol}{formatted_number}"
            else:
                return f"{formatted_number} {symbol}"

        except Exception as e:
            logger.error(f"Error formatting currency {value}: {e}")
            return f"{self.locale_config['currency_symbol']}{value}"

    def format_date(self, date_obj: Union[datetime, date], format_string: str = None) -> str:
        """Format date according to locale conventions
        
        Args:
            date_obj: Date to format
            format_string: Custom format string (optional)
            
        Returns:
            Formatted date string
        """
        try:
            if date_obj is None:
                return "N/A"

            if not isinstance(date_obj, (datetime, date)):
                return str(date_obj)

            format_str = format_string or self.locale_config['date_format']
            return date_obj.strftime(format_str)

        except Exception as e:
            logger.error(f"Error formatting date {date_obj}: {e}")
            return str(date_obj)

    def format_time(self, time_obj: Union[datetime, date], format_string: str = None) -> str:
        """Format time according to locale conventions
        
        Args:
            time_obj: Time to format
            format_string: Custom format string (optional)
            
        Returns:
            Formatted time string
        """
        try:
            if time_obj is None:
                return "N/A"

            if not isinstance(time_obj, datetime):
                return str(time_obj)

            format_str = format_string or self.locale_config['time_format']
            return time_obj.strftime(format_str)

        except Exception as e:
            logger.error(f"Error formatting time {time_obj}: {e}")
            return str(time_obj)

    def format_datetime(self, datetime_obj: datetime, format_string: str = None) -> str:
        """Format datetime according to locale conventions
        
        Args:
            datetime_obj: Datetime to format
            format_string: Custom format string (optional)
            
        Returns:
            Formatted datetime string
        """
        try:
            if datetime_obj is None:
                return "N/A"

            if not isinstance(datetime_obj, datetime):
                return str(datetime_obj)

            format_str = format_string or self.locale_config['datetime_format']
            return datetime_obj.strftime(format_str)

        except Exception as e:
            logger.error(f"Error formatting datetime {datetime_obj}: {e}")
            return str(datetime_obj)

    def format_list(self, items: list, separator: str = None,
                   conjunction: str = None) -> str:
        """Format list according to locale conventions
        
        Args:
            items: List of items to format
            separator: Custom separator (optional)
            conjunction: Conjunction word for last item (optional)
            
        Returns:
            Formatted list string
        """
        try:
            if not items:
                return ""

            if len(items) == 1:
                return str(items[0])

            # Default separators and conjunctions by locale
            default_separators = {
                'en': ', ',
                'es': ', ',
                'fr': ', ',
                'de': ', ',
                'ja': '、',
                'zh': '、'
            }

            default_conjunctions = {
                'en': ' and ',
                'es': ' y ',
                'fr': ' et ',
                'de': ' und ',
                'ja': 'と',
                'zh': '和'
            }

            sep = separator or default_separators.get(self.locale, ', ')
            conj = conjunction or default_conjunctions.get(self.locale, ' and ')

            if len(items) == 2:
                return f"{items[0]}{conj}{items[1]}"
            else:
                return sep.join(items[:-1]) + conj + str(items[-1])

        except Exception as e:
            logger.error(f"Error formatting list {items}: {e}")
            return str(items)

    def parse_number(self, number_string: str) -> Optional[float]:
        """Parse number from localized string
        
        Args:
            number_string: Localized number string
            
        Returns:
            Parsed number or None if invalid
        """
        try:
            # Remove thousand separators and replace decimal separator
            cleaned = number_string.replace(self.locale_config['thousand_separator'], '')
            cleaned = cleaned.replace(self.locale_config['decimal_separator'], '.')

            return float(cleaned)

        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing number '{number_string}': {e}")
            return None

    def get_locale_info(self) -> dict:
        """Get locale configuration information"""
        return {
            'locale': self.locale,
            'config': self.locale_config.copy(),
            'supported_locales': list(SUPPORTED_LANGUAGES.keys())
        }
