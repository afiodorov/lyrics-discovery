"""Combined node for filtering and extracting lyrics in one step."""

from typing import Any, Dict

from ..config import deepseek_client
from ..logging_config import get_logger

logger = get_logger(__name__)


def extract_lyrics_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combined node that filters search results and extracts lyrics in a single LLM call.
    This replaces both filter_results_node and format_lyrics_node.
    """
    search_results = state.get("search_results", [])
    song_title = state.get("song_title", "")
    song_artist = state.get("song_artist", "")

    if not search_results:
        logger.warning("No search results to process")
        return {"formatted_lyrics": "", "error_message": "No search results found"}

    logger.info("🔍 Extracting lyrics from search results...")

    # Prepare the combined prompt
    search_context = "\n\n---SOURCE---\n\n".join(search_results[:5])  # Limit to top 5

    prompt = f"""You are a lyrics extraction expert. Given search results for the song "{song_title}" by {song_artist},
extract and format the complete lyrics.

Instructions:
1. Find the source that contains the most complete lyrics
2. Extract the FULL lyrics (all verses, choruses, bridges)
3. Format them properly with clear verse/chorus structure
4. If lyrics are in multiple sources, combine them to get the complete version
5. Remove any website navigation, ads, or non-lyric content

Search results:
{search_context}

Return ONLY the formatted lyrics, nothing else. If no lyrics are found, return "LYRICS_NOT_FOUND"."""

    try:
        response = deepseek_client.invoke(prompt)
        formatted_lyrics = response.content.strip()

        if formatted_lyrics == "LYRICS_NOT_FOUND":
            logger.warning("Could not extract lyrics from search results")
            return {
                "formatted_lyrics": "",
                "error_message": "Could not find lyrics in search results",
            }

        logger.info(f"✅ Successfully extracted lyrics ({len(formatted_lyrics)} chars)")
        return {"formatted_lyrics": formatted_lyrics}

    except Exception as e:
        logger.error(f"Error extracting lyrics: {e}")
        return {
            "formatted_lyrics": "",
            "error_message": f"Failed to extract lyrics: {str(e)}",
        }
