"""Translation node for lyrics."""

from ..config import llm_client
from ..logging_config import get_logger
from ..state import AgentState, log_debug_state

logger = get_logger(__name__)


def translate_lyrics_node(state: AgentState) -> dict:
    """Translates the lyrics to the target language."""
    lyrics, language = state["formatted_lyrics"], state["target_language"]

    if not lyrics or not language:
        return {"error_message": "Cannot translate: missing lyrics or target language"}

    logger.info(f"🈯 Translating lyrics to {language}...")
    system_prompt = "You are a world-class polyglot and translator. Your task is to translate the provided song lyrics into the specified target language. Retain the poetic structure and meaning as best as possible. Do not add any commentary or introductory text, only the translated lyrics."
    user_prompt = f"Please translate the following lyrics into {language}:\n\n--- LYRICS ---\n{lyrics}\n--- END OF LYRICS ---"
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=8000,  # Increased from 4000
        )
        translated = response.choices[0].message.content.strip()

        # Check for potential truncation
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            logger.warning(
                "    - ⚠️ WARNING: Translation was truncated due to token limit!"
            )
        logger.debug(f"    - Original lyrics length: {len(lyrics)} characters")
        logger.debug(f"    - Translated lyrics length: {len(translated)} characters")
        logger.debug(f"    - Finish reason: {finish_reason}")

        update = {"translated_lyrics": translated}
        log_debug_state("translate_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        logger.error(f"    - ❌ ERROR in translate_lyrics_node: {e}")
        return {"error_message": "An error occurred during translation."}
