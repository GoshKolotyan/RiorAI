import sys
import logging
from pathlib import Path
from typing import Optional, Union


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        'DEBUG': '\033[90m',    # Dark gray
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        # Color only the level name
        colored_level = f"{log_color}[{record.levelname}]{self.RESET}"
        # Replace the original level in the format
        formatted_message = super().format(record)
        formatted_message = formatted_message.replace(f"[{record.levelname}]", colored_level)
        return formatted_message


def setup_loger(
    name: str,
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    enable_colors: bool = True
) -> logging.Logger:

    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Set logging level
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level
    logger.setLevel(numeric_level)

    # Prevent propagation to avoid duplicate messages
    logger.propagate = False

    # Default format
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    if enable_colors and sys.stdout.isatty():
        console_formatter = ColorFormatter(format_string)
        console_formatter.datefmt = "%Y-%m-%d"
    else:
        console_formatter = logging.Formatter(format_string)
        console_formatter.datefmt = "%Y-%m-%d"
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(format_string)
        file_formatter.datefmt = "%Y-%m-%d"
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger