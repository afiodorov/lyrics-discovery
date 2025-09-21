"""Gradio web UI for the lyrics search application."""

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from .graph import create_workflow
from .logging_config import get_logger, setup_logging
from .state import AgentState

# Set up logging
setup_logging("normal")
logger = get_logger(__name__)

# Cache directory
CACHE_DIR = Path(".cache/lyrics_search")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(query: str, language: Optional[str]) -> str:
    """Generate a cache key for the query and language combination."""
    cache_str = f"{query}:{language or 'none'}"
    return hashlib.md5(cache_str.encode()).hexdigest()


def load_from_cache(query: str, language: Optional[str]) -> Optional[dict]:
    """Load results from cache if they exist."""
    cache_key = get_cache_key(query, language)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    return None


def save_to_cache(query: str, language: Optional[str], results: dict):
    """Save results to cache."""
    cache_key = get_cache_key(query, language)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


def search_lyrics(query: str, translate_to: str) -> Tuple[str, str, str]:
    """
    Search for lyrics with progress tracking.
    Returns: (progress_log, lyrics_output, facts_output)
    """
    if not query.strip():
        return "Please enter a song name or description.", "", ""

    # Check cache first
    target_lang = translate_to.strip() if translate_to.strip() else None
    cached = load_from_cache(query, target_lang)
    if cached:
        return cached["progress"], cached["lyrics"], cached["facts"]

    # Initialize progress log
    progress_log = []

    # Create initial state
    initial_state = AgentState(
        user_query=query,
        target_language=target_lang,
        song_title="",
        song_artist="",
        search_results=[],
        formatted_lyrics="",
        translated_lyrics="",
        interspersed_lyrics="",
        curious_facts="",
        error_message="",
    )

    # Create and run the graph
    graph = create_workflow()
    app = graph.compile()

    try:
        # Log progress messages based on what will happen
        progress_log.append("🤔 Analyzing your request...")

        # Run the entire graph
        result_state = app.invoke(initial_state)

        # Add progress messages based on what was done
        if result_state.get("song_title"):
            title = result_state.get("song_title", "")
            artist = result_state.get("song_artist", "")
            artist_info = f" by {artist}" if artist else ""
            progress_log.append(f"🧠 Identified: '{title}'{artist_info}")

        if result_state.get("search_results"):
            progress_log.append(
                f"🔎 Searched and found {len(result_state['search_results'])} sources"
            )
            progress_log.append("🔍 Filtered to best source")

        if result_state.get("formatted_lyrics"):
            progress_log.append(
                f"🤖 Formatted lyrics ({len(result_state['formatted_lyrics'])} chars)"
            )

        if target_lang and result_state.get("translated_lyrics"):
            progress_log.append(f"🈯 Translated to {target_lang}")
            progress_log.append("🎨 Combined original and translated lyrics")

        if result_state.get("curious_facts"):
            progress_log.append("🧐 Found curious facts")

        # Check for errors
        if result_state.get("error_message"):
            error_msg = f"❌ Error: {result_state['error_message']}"
            progress_log.append(error_msg)
            final_progress = "\n".join(progress_log)
            save_to_cache(
                query,
                target_lang,
                {"progress": final_progress, "lyrics": "", "facts": ""},
            )
            return final_progress, "", ""

        # Prepare output
        if target_lang and result_state.get("interspersed_lyrics"):
            lyrics_output = result_state["interspersed_lyrics"]
        else:
            lyrics_output = result_state.get("formatted_lyrics", "No lyrics found.")

        facts_output = result_state.get("curious_facts", "No facts found.")

        # Add completion message
        progress_log.append("✅ Complete!")

        final_progress = "\n".join(progress_log)

        # Cache the results
        save_to_cache(
            query,
            target_lang,
            {
                "progress": final_progress,
                "lyrics": lyrics_output,
                "facts": facts_output,
            },
        )

        return final_progress, lyrics_output, facts_output

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        progress_log.append(error_msg)
        return "\n".join(progress_log), "", ""


# Create the Gradio interface
def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="🎵 Lyrics Search & Translate") as demo:
        gr.Markdown(
            """
            # 🎵 Lyrics Search & Translate

            Search for song lyrics, get translations, and discover curious facts about songs!
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Song Query",
                    placeholder="Enter song name or description (e.g., 'Bella Ciao', 'that Beatles song about yesterday')",
                    value="Bella Ciao",  # Pre-filled example
                    lines=1,
                )

            with gr.Column(scale=1):
                translate_input = gr.Textbox(
                    label="Translate to (optional)",
                    placeholder="e.g., 'ru', 'es', 'fr'",
                    value="en",  # Pre-filled with English
                    lines=1,
                )

        search_button = gr.Button("🔍 Search", variant="primary")

        # Progress display
        progress_output = gr.Textbox(
            label="Progress", lines=8, interactive=False, value="Ready to search..."
        )

        with gr.Row():
            # Lyrics output
            with gr.Column():
                lyrics_output = gr.Textbox(
                    label="📜 Lyrics", lines=20, interactive=False
                )

            # Facts output
            with gr.Column():
                facts_output = gr.Textbox(
                    label="🧐 Curious Facts", lines=20, interactive=False
                )

        # Examples
        gr.Examples(
            examples=[
                ["Bella Ciao", "en"],
                ["Gracias a la vida", "ru"],
                ["Yesterday by Beatles", "es"],
                ["Imagine John Lennon", "fr"],
                ["that song that goes 'we will rock you'", ""],
            ],
            inputs=[query_input, translate_input],
        )

        # Set up the search action
        search_button.click(
            fn=search_lyrics,
            inputs=[query_input, translate_input],
            outputs=[progress_output, lyrics_output, facts_output],
        )

    return demo


# Main entry point
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Create a public URL
    )
