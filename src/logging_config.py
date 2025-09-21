"""Logging configuration for the lyrics search application."""

import logging
import sys
from typing import Literal, Optional

VerboseLevel = Literal["normal", "verbose", "very_verbose"]


def setup_logging(verbose_level: VerboseLevel = "normal") -> None:
    """Configure logging for the application.

    Args:
        verbose_level: Logging verbosity level
            - "normal": INFO level for our app, WARNING+ for libraries
            - "verbose": DEBUG level for our app, INFO+ for libraries
            - "very_verbose": DEBUG level for everything
    """
    # Configure the root logger first
    logging.basicConfig(
        level=logging.DEBUG,  # Allow all messages through initially
        format="%(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    # Get the root logger
    root_logger = logging.getLogger()

    if verbose_level == "normal":
        # INFO for our app, WARNING+ for libraries
        root_logger.setLevel(logging.WARNING)
        _set_our_loggers_level(logging.INFO)

    elif verbose_level == "verbose":
        # DEBUG for our app, INFO+ for libraries
        root_logger.setLevel(logging.INFO)
        _set_our_loggers_level(logging.DEBUG)

        # Reduce noise from some chatty libraries
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.INFO)

    elif verbose_level == "very_verbose":
        # DEBUG for everything
        root_logger.setLevel(logging.DEBUG)
        _set_our_loggers_level(logging.DEBUG)


def _set_our_loggers_level(level: int) -> None:
    """Set logging level for our application modules."""
    our_modules = [
        "src.main",
        "src.nodes.analysis",
        "src.nodes.search",
        "src.nodes.formatting",
        "src.nodes.translation",
        "src.nodes.facts",
        "src.state",
        "src.graph",
        "src.config",
    ]

    for module in our_modules:
        logging.getLogger(module).setLevel(level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name, defaults to calling module name

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name or __name__)
