"""Query analysis node for identifying song and artist."""

import json

from ..config import deepseek_client
from ..logging_config import get_logger
from ..state import AgentState, log_debug_state

logger = get_logger(__name__)


def analyze_query_node(state: AgentState) -> dict:
    """Analyzes the user's query to determine the actual song title and artist."""
    user_query = state["user_query"]
    logger.info(f"🤔 Analyzing your request: '{user_query}'...")
    system_prompt = (
        "You are an expert musicologist. Your task is to analyze a user's query about a song and "
        "determine the precise song title and artist. Respond ONLY with a single, valid JSON object "
        "with two keys: 'title' and 'artist'. The artist can be null if unknown. "
        "If you cannot deduce a specific song, return the original query in the 'title' field. "
        'Always return valid JSON format like: {"title": "Song Name", "artist": "Artist Name"}'
    )
    user_prompt = f"Analyze this query: '{user_query}'"
    try:
        # Use LangChain's ChatOpenAI for DeepSeek
        messages = [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
        # Note: LangChain's ChatOpenAI doesn't support response_format directly
        # We'll need to parse JSON from the response
        response = deepseek_client.invoke(messages)

        # Handle empty or invalid response
        if not response.content or not response.content.strip():
            logger.warning("Empty response from LLM, using fallback")
            result = {"title": user_query, "artist": None}
        else:
            try:
                result = json.loads(response.content.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}, using fallback")
                # Fallback to original query
                result = {"title": user_query, "artist": None}
        if result.get("title") != user_query:
            artist_info = f" by {result.get('artist')}" if result.get("artist") else ""
            logger.info(
                f"🧠 I believe you're looking for '{result.get('title')}'{artist_info}."
            )
        update = {
            "song_title": result.get("title"),
            "song_artist": result.get("artist"),
        }
        log_debug_state("analyze_query_node", {**state, **update})
        return update
    except Exception as e:
        logger.exception(f"ERROR in analyze_query_node: {e}")
        return {"error_message": "An error occurred during query analysis."}
