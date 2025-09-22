"""Facts finding node for discovering interesting information about songs."""

import wikipedia

from ..config import llm_client, tavily_client
from ..logging_config import get_logger
from ..state import AgentState, log_debug_state

logger = get_logger(__name__)


def find_curious_facts_node(state: AgentState) -> dict:
    """Searches for curious facts, first on Wikipedia, then via web search as a fallback."""
    title, artist = state["song_title"], state["song_artist"]
    logger.info(f"🧐 Searching for curious facts about '{title}'...")
    facts_content = None

    try:
        search_query = f"{title} (song)" + (f" ({artist} song)" if artist else "")
        page = wikipedia.page(search_query, auto_suggest=True, redirect=True)
        facts_content = page.content
        logger.info("    - Found Wikipedia page, summarizing...")
    except Exception as e:
        logger.exception(f"    - ⚠️ Wikipedia search failed with error: {e}")
        logger.info(
            "    - Could not find a specific Wikipedia page. Trying web search as a fallback."
        )
        facts_content = None

    if not facts_content:
        try:
            search_query = f"interesting facts about the song '{title}'" + (
                f" by '{artist}'" if artist else ""
            )
            response = tavily_client.search(
                query=search_query, search_depth="basic", max_results=3
            )
            results = response.get("results", [])
            if results:
                valid_content = [
                    result.get("content", "")
                    for result in results
                    if result.get("content")
                ]
                facts_content = "\n\n".join(valid_content) if valid_content else None
            else:
                facts_content = None
            logger.info("    - Found web search results, summarizing...")
        except Exception as e:
            logger.exception(f"    - ⚠️ Tavily facts search failed with error: {e}")
            logger.warning("    - Web search for facts also failed.")
            return {}

    if not facts_content:
        return {}

    try:
        system_prompt = (
            "You are a research assistant. Your task is to read the provided text "
            "about a song and extract 1 to 3 curious or interesting facts. Format them as a short, "
            "bulleted list. If no interesting facts can be found, respond with 'No specific facts found.'"
        )
        user_prompt = f"Extract 1-3 curious facts from this article about '{title}':\n\n{facts_content[:4000]}"
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        facts = (response.choices[0].message.content or "").strip()
        if "No specific facts found" in facts:
            return {}

        # Translate facts if target language is specified
        if state.get("target_language"):
            facts = _translate_facts(facts, state["target_language"], title)
        else:
            # If no translation requested, detect song language and translate facts to match
            detected_lang = _detect_song_language(state, title, artist)
            if detected_lang and detected_lang.lower() not in ["en", "english"]:
                logger.info(
                    f"    - Detected song is in {detected_lang}, translating facts to match..."
                )
                facts = _translate_facts(facts, detected_lang, title)

        update = {"curious_facts": facts}
        log_debug_state("find_curious_facts_node",  state | update)
        return update
    except Exception as e:
        logger.exception(f"    - ⚠️ LLM fact extraction failed with error: {e}")
        logger.warning("    - An error occurred during LLM fact extraction.")
        return {}


def _detect_song_language(state: AgentState, title: str, artist: str) -> str | None:
    """Detect the language of the song based on its lyrics or metadata."""
    # Try to detect from formatted lyrics if available
    lyrics_snippet = (
        state.get("formatted_lyrics", "")[:500] if state.get("formatted_lyrics") else ""
    )

    if not lyrics_snippet:
        # No lyrics to analyze
        return None

    try:
        # Use LLM to detect language
        prompt = (
            "Analyze this song text and identify the language. "
            "Respond with ONLY the language name (e.g., 'Spanish', 'Portuguese', 'Italian', 'French'). "
            "If you cannot determine or it's English, respond with 'English'."
        )

        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Song: '{title}' by {artist}\n\nLyrics excerpt:\n{lyrics_snippet}",
                },
            ],
            temperature=0.0,
            max_tokens=50,
        )

        detected = (response.choices[0].message.content or "").strip()
        logger.debug(f"    - Detected language: {detected}")
        return detected

    except Exception as e:
        logger.error(f"    - Could not detect language: {e}")
        return None


def _translate_facts(facts: str, target_language: str, title: str) -> str:
    """Translate curious facts to the target language."""
    logger.info(f"    - Translating facts to {target_language}...")

    try:
        system_prompt = (
            f"You are a professional translator. Translate the following facts about '{title}' "
            f"to {target_language}. Maintain the bullet list format and factual accuracy. "
            "Only translate the text, keep any formatting like bullet points or dashes."
        )

        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": facts},
            ],
            temperature=0.0,
        )

        translated_facts = (response.choices[0].message.content or "").strip()
        logger.debug("    - Facts translated successfully")
        return translated_facts

    except Exception as e:
        logger.exception(f"    - Could not translate facts: {e}")
        # Return original facts if translation fails
        return facts
