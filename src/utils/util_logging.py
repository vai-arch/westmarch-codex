"""
Westmarch's Codex - Logging Module
Provides consistent logging across the project.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import get_config
from src.paths import get_paths

config = get_config()
paths = get_paths()


# Color codes for console output
class LogColors:
    """ANSI color codes for console logging"""

    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output.
    """

    COLORS = {
        "DEBUG": LogColors.CYAN,
        "INFO": LogColors.GREEN,
        "WARNING": LogColors.YELLOW,
        "ERROR": LogColors.RED,
        "CRITICAL": LogColors.RED + LogColors.BOLD,
    }

    def format(self, record):
        """Format log record with colors"""
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{LogColors.RESET}"

        # Format the message
        result = super().format(record)

        # Reset levelname for other handlers
        record.levelname = levelname

        return result


def set_global_log_level(level: str):
    """
    Safely change the logging level for all existing loggers at runtime.

    Args:
        level: Log level name as string (DEBUG, INFO, WARNING, ERROR, CRITICAL, or custom)
    """
    # Map custom levels
    level_name = level.upper()

    # Convert to numeric level
    numeric_level = getattr(logging, level_name, None)
    if numeric_level is None:
        raise ValueError(f"Invalid log level: {level}")

    # Change root logger
    logging.getLogger().setLevel(numeric_level)

    # Change all existing named loggers
    for logger_name, logger_obj in logging.Logger.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger):
            logger_obj.setLevel(numeric_level)


def setup_logging(
    name="westmarch_codex",
    log_file=None,
    log_level="INFO",
    console_output=True,
    file_output=True,
    max_bytes=10485760,  # 10MB
    backup_count=5,
):
    """
    Set up logging configuration.

    Args:
        name: Logger name
        log_file: Path to log file (if None, uses logs/westmarch_codex.log)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console
        file_output: Whether to output to file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    console_format = "%(levelname)-8s | %(name)-20s | %(message)s"
    file_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

    console_formatter = ColoredFormatter(console_format)
    file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output:
        if log_file is None:
            log_file = Path("logs") / "westmarch_codex.log"
        else:
            log_file = Path(log_file)

        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name, log_level=None):
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name (typically __name__ from calling module)
        log_level: Optional log level override

    Returns:
        Logger instance
    """
    # Try to load configuration
    try:
        default_file = paths.FILE_MAIN_LOG
        default_level = config.LOG["LEVEL"]
        max_bytes = config.LOG["MAX_BYTES"]
        backup_count = config.LOG["BACKUP_COUNT"]
    except (ImportError, Exception):
        # Fallback if config not available
        default_level = "INFO"
        default_file = Path("logs") / "westmarch_codex.log"
        max_bytes = 10485760
        backup_count = 5

    level = log_level or default_level

    # Check if logger already exists
    logger = logging.getLogger(name)

    # If no handlers, set up logging
    if not logger.handlers:
        logger = setup_logging(name=name, log_file=default_file, log_level=level, max_bytes=max_bytes, backup_count=backup_count)

    return logger


# Main project logger
def get_main_logger():
    """Get the main Westmarch's Codex logger"""
    return get_logger("westmarch_codex")


# Convenience logging functions
def log_info(message, logger_name="westmarch_codex"):
    """Log an info message"""
    get_logger(logger_name).info(message)


def log_warning(message, logger_name="westmarch_codex"):
    """Log a warning message"""
    get_logger(logger_name).warning(message)


def log_error(message, logger_name="westmarch_codex"):
    """Log an error message"""
    get_logger(logger_name).error(message)


def log_debug(message, logger_name="westmarch_codex"):
    """Log a debug message"""
    get_logger(logger_name).debug(message)


def log_critical(message, logger_name="westmarch_codex"):
    """Log a critical message"""
    get_logger(logger_name).critical(message)


# Context manager for timing operations
class LogTimer:
    """Context manager for timing and logging operations"""

    def __init__(self, operation_name, logger=None):
        """
        Initialize timer.

        Args:
            operation_name: Name of operation being timed
            logger: Logger instance (if None, uses main logger)
        """
        self.operation_name = operation_name
        self.logger = logger or get_main_logger()
        self.start_time = None

    def __enter__(self):
        """Start timing"""
        import time

        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration"""
        import time

        duration = time.time() - self.start_time

        if exc_type is not None:
            self.logger.error(f"Failed: {self.operation_name} (after {duration:.2f}s)")
        else:
            self.logger.info(f"Completed: {self.operation_name} ({duration:.2f}s)")


if __name__ == "__main__":
    # Test logging
    print("Testing Westmarch's Codex Logging System\n")

    logger = get_logger("test")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("\nTesting LogTimer context manager:")

    with LogTimer("Test operation", logger):
        import time

        time.sleep(1)

    print("\n✓ Logging test complete!")
    print("  Check logs/westmarch_codex.log for file output")
