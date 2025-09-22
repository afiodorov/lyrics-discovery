"""Search and filtering nodes for finding lyrics."""

from ..config import llm_client, tavily_client
from ..logging_config import get_logger
from ..state import AgentState, log_debug_state

logger = get_logger(__name__)


def search_lyrics_node(state: AgentState) -> dict:
    """Searches for lyrics using the Tavily Search API."""
    title, artist = state["song_title"], state["song_artist"]
    artist_info = f" by {artist}" if artist else ""
    logger.info(f"🔎 Searching for lyrics for '{title}'{artist_info}...")
    # More specific query to get cleaner, lyrics-focused results
    query = f"full complete song lyrics for '{title}'" + (
        f" by {artist}" if artist else ""
    )
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_raw_content="text",  # Get full page content instead of snippets
        )
        # Prefer raw_content over content snippets when available
        content = []
        for result in response["results"]:
            if result.get("raw_content"):
                # Use full page content
                content.append(result["raw_content"])
                logger.debug(
                    f"    - Using raw content ({len(result['raw_content'])} chars) from {result.get('url', 'unknown')}"
                )
            else:
                # Fallback to snippet
                content.append(result["content"])
                logger.debug(
                    f"    - Using content snippet ({len(result['content'])} chars) from {result.get('url', 'unknown')}"
                )
        update = {"search_results": content}
        log_debug_state("search_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        logger.exception(f"    - ❌ ERROR in search_lyrics_node: {e}")
        return {"error_message": "An error occurred during the web search."}


def filter_results_node(state: AgentState) -> dict:
    """
    Uses an LLM to analyze search results and pick the best one.
    Combines fragmented lyrics when necessary.
    """
    search_results = state["search_results"]
    logger.info("🔍 Filtering search results to find the best source...")

    # First, check if any result contains what looks like lyrics
    # This is a heuristic to avoid rejecting partial but valid lyrics
    lyrics_indicators = [
        "стою",
        "пред",
        "душу",
        "птица",
        "купола",  # Russian lyrics keywords
        "verse",
        "chorus",
        "lyrics",  # Common lyrics structure
    ]

    has_potential_lyrics = any(
        any(indicator.lower() in result.lower() for indicator in lyrics_indicators)
        for result in search_results
    )

    if has_potential_lyrics:
        # Combine all results that might contain lyrics fragments
        # Sometimes lyrics are split across multiple search results
        combined_results = []
        for i, result in enumerate(search_results):
            # Check if this result likely contains lyrics content
            result_lower = result.lower()
            if any(
                word in result_lower
                for word in [
                    "купола",
                    "высоцк",
                    "lyrics",
                    "стою",
                    "пред",
                    "душу",
                    "птица",
                ]
            ):
                combined_results.append(result)

        if combined_results:
            # Merge the results for better context
            merged_content = "\n\n---\n\n".join(combined_results)
            update = {"search_results": [merged_content]}
            log_debug_state("filter_results_node", {**state, **update})
            return update

    system_prompt = (
        "You are a web scraping expert. Your task is to analyze a list of text snippets from web pages "
        "and identify content that contains lyrics for a song. Even partial lyrics are acceptable. "
        "Look for poetic text, verses, repeated structures, or any content that resembles song lyrics. "
        "Combine multiple snippets if they appear to be parts of the same song. "
        "Respond with the combined text of ALL snippets that contain lyrics. "
        "Only respond with 'No suitable source found' if absolutely NO lyrics content is present."
    )

    # We'll number the results to make it easier for the LLM to differentiate
    numbered_results = "\n\n".join(
        [f"--- RESULT {i + 1} ---\n{res}" for i, res in enumerate(search_results)]
    )
    user_prompt = (
        f"Analyze the following search results for the song '{state['song_title']}' "
        f"{f'by {state["song_artist"]}' if state['song_artist'] else ''} "
        f"and return ALL text that contains lyrics (even partial):\n\n"
        f"{numbered_results}"
    )

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        best_result = response.choices[0].message.content.strip()

        if "No suitable source found" in best_result:
            # Fallback: just combine all results and let the format node handle it
            combined_all = "\n\n---\n\n".join(search_results)
            update = {"search_results": [combined_all]}
            log_debug_state("filter_results_node (fallback)", {**state, **update})
            return update

        # Update the search_results to contain only the single best one
        update = {"search_results": [best_result]}
        log_debug_state("filter_results_node", {**state, **update})
        return update
    except Exception as e:
        logger.exception(f"    - ❌ ERROR in filter_results_node: {e}")
        # On error, pass through all results combined
        combined_all = "\n\n---\n\n".join(search_results)
        return {"search_results": [combined_all]}
