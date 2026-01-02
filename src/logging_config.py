"""
Centralized logging configuration for the entire project
Đảm bảo logging đồng bộ và consistent across all modules
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

_logging_configured = False
_log_file_path: Optional[Path] = None


def setup_logging(
    log_file: Optional[Path] = None,
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    force_reconfigure: bool = False
) -> logging.Logger:
    """
    Setup centralized logging configuration
    
    Args:
        log_file: Path to log file (None = auto-generate in project root)
        log_level: Logging level (default: INFO)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        force_reconfigure: Force reconfiguration even if already configured
        
    Returns:
        Root logger instance
    """
    global _logging_configured, _log_file_path
    
    if _logging_configured and not force_reconfigure:
        return logging.getLogger()
    
    project_root = Path(__file__).parent.parent
    
    if log_file is None and log_to_file:
        log_file = project_root / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    _log_file_path = log_file
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    if force_reconfigure:
        root_logger.handlers.clear()
    elif _logging_configured:
        has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) 
                         for h in root_logger.handlers)
        has_file = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        
        if (log_to_console and not has_console) or (log_to_file and not has_file):
            pass
        else:
            return root_logger
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    if log_to_file and log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            root_logger.warning(f"Failed to setup file logging: {str(e)}, using console only")
    
    _logging_configured = True
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for a module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    if not _logging_configured:
        setup_logging(log_to_file=False)
    
    return logging.getLogger(name)


def get_log_file_path() -> Optional[Path]:
    """Get current log file path"""
    return _log_file_path

