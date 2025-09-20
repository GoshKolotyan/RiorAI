import sys
import logging
from pathlib import Path


class ColorFormatter(logging.Formatter):
    """Add colors to log levels."""
    
    COLORS = {
        'DEBUG': '\033[90m',    # Gray
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Get the formatted message first
        message = super().format(record)
        
        # Add color to the level name
        color = self.COLORS.get(record.levelname, '')
        colored_level = f"{color}[{record.levelname}]{self.RESET}"
        
        # Replace the level in the message
        return message.replace(f"[{record.levelname}]", colored_level)


def setup_logger(name, level="INFO", log_file=None, enable_colors=True):
    """Create a logger with console and optional file output."""
    
    logger = logging.getLogger(name)
    
    # Skip if already configured
    if logger.handlers:
        return logger
    
    # Set level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    
    # Format for messages
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Console output
    console = logging.StreamHandler(sys.stdout)
    if enable_colors and sys.stdout.isatty():
        console.setFormatter(ColorFormatter(fmt))
    else:
        console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)
    
    # File output (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)
    
    return logger


# Example usage
# if __name__ == "__main__":
#     # Basic usage
#     log = setup_logger("myapp")
#     log.info("This is an info message")
#     log.warning("This is a warning")
#     log.error("This is an error")
    
#     # With file logging
#     log2 = setup_logger("fileapp", level="DEBUG", log_file="app.log")
#     log2.debug("Debug message")
#     log2.info("Info message")