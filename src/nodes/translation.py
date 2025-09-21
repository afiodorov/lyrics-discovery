"""Translation node for lyrics."""

from ..config import llm_client
from ..state import AgentState, print_debug_log


def translate_lyrics_node(state: AgentState) -> dict:
    """Translates the lyrics to the target language."""
    lyrics, language = state["formatted_lyrics"], state["target_language"]
    print(f"🈯 Translating lyrics to {language}...")
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
            max_tokens=4000,
        )
        translated = response.choices[0].message.content.strip()
        update = {"translated_lyrics": translated}
        print_debug_log("translate_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ❌ ERROR in translate_lyrics_node: {e}")
        return {"error_message": "An error occurred during translation."}
