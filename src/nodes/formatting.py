"""Formatting nodes for lyrics processing."""

from ..config import llm_client
from ..state import AgentState, print_debug_log


def format_lyrics_node(state: AgentState) -> dict:
    """Uses an LLM to extract and format lyrics from the single best search result."""
    # This node now receives a much cleaner, pre-filtered search result
    search_context = "\n\n".join(state["search_results"])
    title, artist = state["song_title"], state["song_artist"]
    print("🤖 Asking the LLM to format the lyrics...")
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
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        formatted = response.choices[0].message.content.strip()
        # Be more lenient - accept any reasonable lyrics output
        if len(formatted) < 10:
            return {
                "error_message": "Could not extract meaningful lyrics from the search results."
            }
        update = {"formatted_lyrics": formatted}
        print_debug_log("format_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ❌ ERROR in format_lyrics_node: {e}")
        return {"error_message": "An error occurred while formatting lyrics."}


def intersperse_lyrics_node(state: AgentState) -> dict:
    """Combines original and translated lyrics into an interspersed format."""
    original, translated, language = (
        state["formatted_lyrics"],
        state["translated_lyrics"],
        state["target_language"],
    )
    print("🎨 Combining original and translated lyrics...")
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
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4095,
        )
        interspersed = response.choices[0].message.content.strip()
        update = {"interspersed_lyrics": interspersed}
        print_debug_log("intersperse_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ❌ ERROR in intersperse_lyrics_node: {e}")
        return {"error_message": "An error occurred during final formatting."}
