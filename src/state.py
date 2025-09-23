"""State definition and utilities for the lyrics search agent."""

from typing import List, TypedDict

from .logging_config import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict):
    """State object that flows through the agent graph."""

    user_query: str
    target_language: str | None
    song_title: str
    song_artist: str
    search_results: List[str]
    formatted_lyrics: str
    translated_lyrics: str
    interspersed_lyrics: str
    curious_facts: str
    error_message: str


def log_debug_state(node_name: str, state: AgentState):
    """Logs the current state for debugging purposes."""
    logger.debug(f"--- After {node_name} ---")
    for key, value in state.items():
        if key == "search_results" and value:
            logger.debug(f"  - {key}: {len(value)} items found.")
            for i, result in enumerate(value):
                # Log a snippet of each search result to inspect its quality
                snippet = result[:150].replace("\n", " ")
                logger.debug(f"    - Result {i + 1}: {snippet}...")
        elif isinstance(value, str) and len(value) > 250:
            logger.debug(f"  - {key}: {value[:250]}... (truncated)")
        else:
            logger.debug(f"  - {key}: {value}")
    logger.debug("--- End Debug ---")
