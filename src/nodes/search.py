"""Search and filtering nodes for finding lyrics."""

from ..config import (
    cache_search_results,
    deepseek_client,
    get_cached_search,
    tavily_search,
)
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
        # Check cache first
        cached_results = get_cached_search(query)
        if cached_results:
            logger.debug(
                f"    - Using cached search results ({len(cached_results.get('results', []))} results)"
            )
            content = cached_results.get("content", [])
            update = {"search_results": content}
            log_debug_state("search_lyrics_node (cached)", {**state, **update})
            return update

        # Use LangChain's TavilySearch tool
        search_response = tavily_search.invoke(query)

        # Extract content from results - TavilySearch returns a dict with 'results' key
        # Prefer raw_content over content snippets when available
        content = []
        results = search_response.get("results", [])
        for result in results:
            if result.get("raw_content"):
                # Use full page content
                text = result["raw_content"]
                content.append(text)
                logger.debug(
                    f"    - Using raw content ({len(text)} chars) from {result.get('url', 'unknown')}"
                )
            else:
                # Fallback to snippet
                text = result.get("content", "")
                if text:
                    content.append(text)
                    logger.debug(
                        f"    - Using content snippet ({len(text)} chars) from {result.get('url', 'unknown')}"
                    )

        if not content:
            logger.warning("    - No results found from Tavily search")
            return {"search_results": []}

        # Cache the results
        cache_data = {"results": search_response.get("results", []), "content": content}
        cache_search_results(query, cache_data)

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
        # Use LangChain's ChatOpenAI for DeepSeek
        messages = [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
        response = deepseek_client.invoke(messages)
        best_result = (response.content or "").strip()

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
