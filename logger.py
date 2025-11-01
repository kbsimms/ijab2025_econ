"""
Logging Module for IJAB Economic Scenario Analysis.

Provides structured logging to replace print statements throughout the codebase.
Supports different log levels, optional file output, and formatted progress messages.

Usage:
    from logger import get_logger, LogLevel
    
    logger = get_logger(__name__, level=LogLevel.INFO)
    logger.info("Starting optimization")
    logger.debug("Debug details")
    logger.warning("Warning message")
    logger.error("Error occurred")
"""

import sys
from enum import IntEnum
from typing import Optional, TextIO
from pathlib import Path
from datetime import datetime


class LogLevel(IntEnum):
    """Log levels from least to most verbose."""
    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3


class Logger:
    """
    Simple logger with level-based filtering and optional file output.
    
    Provides structured logging with timestamps and context information.
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        file_path: Optional[Path] = None,
        stream: TextIO = sys.stdout
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name (typically module name)
            level: Minimum log level to display
            file_path: Optional file path for log output
            stream: Output stream (default: stdout)
        """
        self.name = name
        self.level = level
        self.stream = stream
        self.file_path = file_path
        self._file_handle: Optional[TextIO] = None
        
        if file_path:
            self._open_log_file()
    
    def _open_log_file(self) -> None:
        """Open log file for writing."""
        if self.file_path:
            try:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                self._file_handle = open(self.file_path, 'a', encoding='utf-8')
            except Exception as e:
                self._write(LogLevel.ERROR, f"Failed to open log file: {e}")
    
    def _format_message(self, level: LogLevel, message: str) -> str:
        """
        Format log message with timestamp and level.
        
        Args:
            level: Log level
            message: Message to format
            
        Returns:
            Formatted message string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = level.name.ljust(7)  # Pad to 7 chars for alignment
        return f"[{timestamp}] {level_name} [{self.name}] {message}"
    
    def _write(self, level: LogLevel, message: str) -> None:
        """
        Write message to output if level is sufficient.
        
        Args:
            level: Log level of message
            message: Message to write
        """
        if level <= self.level:
            formatted = self._format_message(level, message)
            
            # Write to stream
            print(formatted, file=self.stream)
            
            # Write to file if configured
            if self._file_handle:
                try:
                    self._file_handle.write(formatted + '\n')
                    self._file_handle.flush()
                except Exception as e:
                    print(f"Error writing to log file: {e}", file=sys.stderr)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self._write(LogLevel.DEBUG, message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self._write(LogLevel.INFO, message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self._write(LogLevel.WARNING, message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self._write(LogLevel.ERROR, message)
    
    def section(self, title: str, width: int = 80) -> None:
        """
        Log a section header (always shown, regardless of level).
        
        Args:
            title: Section title
            width: Width of separator line
        """
        separator = "=" * width
        # Force output by using ERROR level (always shown)
        original_level = self.level
        self.level = LogLevel.ERROR
        self._write(LogLevel.ERROR, separator)
        self._write(LogLevel.ERROR, title.center(width))
        self._write(LogLevel.ERROR, separator)
        self.level = original_level
    
    def subsection(self, title: str, width: int = 80) -> None:
        """
        Log a subsection header.
        
        Args:
            title: Subsection title
            width: Width of separator line
        """
        separator = "-" * width
        self.info(separator)
        self.info(title.center(width))
        self.info(separator)
    
    def progress(self, message: str, current: int, total: int) -> None:
        """
        Log progress message with counter.
        
        Args:
            message: Progress message
            current: Current item number
            total: Total number of items
        """
        percentage = (current / total * 100) if total > 0 else 0
        self.info(f"[{current}/{total} - {percentage:.1f}%] {message}")
    
    def close(self) -> None:
        """Close log file if open."""
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
            finally:
                self._file_handle = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global logger registry
_loggers: dict[str, Logger] = {}
_global_level: LogLevel = LogLevel.INFO
_global_log_file: Optional[Path] = None


def get_logger(
    name: str,
    level: Optional[LogLevel] = None,
    file_path: Optional[Path] = None
) -> Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level (uses global if not specified)
        file_path: Optional log file path (uses global if not specified)
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        actual_level = level if level is not None else _global_level
        actual_file = file_path if file_path is not None else _global_log_file
        _loggers[name] = Logger(name, actual_level, actual_file)
    
    return _loggers[name]


def set_global_level(level: LogLevel) -> None:
    """
    Set global log level for all loggers.
    
    Args:
        level: New log level
    """
    global _global_level
    _global_level = level
    
    # Update existing loggers
    for logger in _loggers.values():
        logger.level = level


def set_global_log_file(file_path: Optional[Path]) -> None:
    """
    Set global log file for all loggers.
    
    Args:
        file_path: Path to log file (None to disable file logging)
    """
    global _global_log_file
    _global_log_file = file_path
    
    # Update existing loggers
    for logger in _loggers.values():
        if logger._file_handle:
            logger.close()
        logger.file_path = file_path
        if file_path:
            logger._open_log_file()


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    file_path: Optional[Path] = None,
    verbose: bool = False
) -> None:
    """
    Configure global logging settings.
    
    Args:
        level: Default log level
        file_path: Optional log file path
        verbose: If True, set level to DEBUG
    """
    actual_level = LogLevel.DEBUG if verbose else level
    set_global_level(actual_level)
    set_global_log_file(file_path)


def close_all_loggers() -> None:
    """Close all logger file handles."""
    for logger in _loggers.values():
        logger.close()