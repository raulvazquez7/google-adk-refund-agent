"""
Structured logging utility for consistent, parseable logs.

This module provides a structured logger that outputs JSON-formatted logs
for easy parsing and analysis in production environments.
"""
import logging
import json
import sys
from typing import Any, Dict, Optional
from datetime import datetime


class StructuredLogger:
    """
    Structured logger that outputs JSON-formatted logs.

    Usage:
        logger = get_logger(__name__)
        logger.info("user_action", user_id="123", action="refund_requested")

    Output:
        {"timestamp": "2025-09-30T10:00:00", "level": "INFO", "message": "user_action", "user_id": "123", "action": "refund_requested"}
    """

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))

        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """
        Internal method to structure and log messages.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: Main log message
            **kwargs: Additional structured fields
        """
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        }

        # Add all additional fields
        log_entry.update(kwargs)

        self.logger.log(
            getattr(logging, level),
            json.dumps(log_entry, default=str)
        )

    def info(self, message: str, **kwargs: Any) -> None:
        """Log INFO level message with structured fields."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log WARNING level message with structured fields."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        """
        Log ERROR level message with structured fields.

        Args:
            message: Error message
            error: Optional exception object
            **kwargs: Additional fields
        """
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)

        self._log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log DEBUG level message with structured fields."""
        self._log("DEBUG", message, **kwargs)


# Cache for logger instances
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """
    Get or create a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, level)

    return _loggers[name]
