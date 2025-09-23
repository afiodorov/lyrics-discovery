"""Formatting nodes for lyrics processing."""

from ..config import deepseek_client
from ..logging_config import get_logger
from ..state import AgentState, log_debug_state

logger = get_logger(__name__)


def format_lyrics_node(state: AgentState) -> dict:
    """Uses an LLM to extract and format lyrics from the single best search result."""
    # This node now receives a much cleaner, pre-filtered search result
    search_context = "\n\n".join(state["search_results"] or [])
    title, artist = state["song_title"], state["song_artist"]
    logger.info("🤖 Asking the LLM to format the lyrics...")
    system_prompt = (
        "You are an expert assistant specializing in formatting song lyrics. Your task is to analyze the "
        "provided text and reconstruct the complete song lyrics from fragments if necessary. "
        "The text may contain partial or fragmented lyrics - piece them together to form the complete song. "
        "Format the lyrics cleanly with proper verse/stanza breaks. "
        "Do not include commentary or metadata. Return ONLY the lyrics text. "
        "If the lyrics appear incomplete, still return what you can extract and format properly."
    )
    user_prompt = (
        f"Extract and format the lyrics for '{title}'{f' by {artist}' if artist else ''} from this source:\n\n"
        f"{search_context}\n\n"
        "Format the lyrics with proper line breaks and stanza separation."
    )
    try:
        # Use LangChain's ChatOpenAI for DeepSeek
        messages = [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
        response = deepseek_client.invoke(messages, max_tokens=8192)
        formatted = (response.content or "").strip()

        # Check for potential truncation
        # Note: LangChain responses have response_metadata with finish_reason
        finish_reason = response.response_metadata.get("finish_reason", "stop")
        if finish_reason == "length":
            logger.warning(
                "    - ⚠️ WARNING: Response was truncated due to token limit!"
            )
        elif finish_reason == "content_filter":
            logger.warning("    - ⚠️ WARNING: Response was filtered by content policy!")
            # Should rarely happen with DeepSeek
        logger.debug(f"    - Formatted lyrics length: {len(formatted)} characters")
        logger.debug(f"    - Finish reason: {finish_reason}")

        # Be more lenient - accept any reasonable lyrics output
        if len(formatted) < 10:
            return {
                "error_message": "Could not extract meaningful lyrics from the search results."
            }
        update = {"formatted_lyrics": formatted}
        log_debug_state("format_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        logger.exception(f"    - ❌ ERROR in format_lyrics_node: {e}")
        return {"error_message": "An error occurred while formatting lyrics."}


def intersperse_lyrics_node(state: AgentState) -> dict:
    """Combines original and translated lyrics into an interspersed format."""
    original, translated, language = (
        state["formatted_lyrics"],
        state["translated_lyrics"],
        state["target_language"],
    )

    if not original or not translated:
        return {
            "error_message": "Cannot intersperse: missing original or translated lyrics"
        }

    logger.info("🎨 Combining original and translated lyrics...")
    system_prompt = (
        "You are a text formatting expert. Your task is to combine original song lyrics with their translation. "
        "For each line from the original, add the corresponding translated line immediately below it. "
        "Preserve stanza breaks. Do not add any commentary. Simply provide the final text."
    )
    user_prompt = (
        f"Please combine the following original lyrics with their translation into {language}. "
        f"The final output should have the original line, and then the translated line on the next line, separated by a blank line for stanza breaks.\n"
        f"Example:\nOriginal Line 1\nTranslated Line 1\n\n"
        f"--- ORIGINAL LYRICS ---\n{original}\n\n"
        f"--- TRANSLATED LYRICS ---\n{translated}\n\n"
        "Combine them now."
    )
    try:
        # Use LangChain's ChatOpenAI for DeepSeek
        messages = [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
        response = deepseek_client.invoke(messages, max_tokens=8192)
        interspersed = (response.content or "").strip()

        # Check for potential truncation
        # Note: LangChain responses have response_metadata with finish_reason
        finish_reason = response.response_metadata.get("finish_reason", "stop")
        if finish_reason == "length":
            logger.warning(
                "    - ⚠️ WARNING: Interspersed lyrics were truncated due to token limit!"
            )
        logger.debug(f"    - Original lyrics length: {len(original)} characters")
        logger.debug(f"    - Translated lyrics length: {len(translated)} characters")
        logger.debug(
            f"    - Interspersed lyrics length: {len(interspersed)} characters"
        )
        logger.debug(f"    - Finish reason: {finish_reason}")

        update = {"interspersed_lyrics": interspersed}
        log_debug_state("intersperse_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        logger.exception(f"    - ❌ ERROR in intersperse_lyrics_node: {e}")
        return {"error_message": "An error occurred during final formatting."}
