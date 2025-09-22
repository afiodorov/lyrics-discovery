"""Simple working Gradio web UI with proper streaming."""

import hashlib
import json
import time
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


def search_lyrics_simple(query: str, translate_to: str):
    """
    Simple generator function that works with Gradio streaming.
    Yields: (progress_log, lyrics_output, facts_output)
    """
    logger.info(
        f"WEB UI FUNCTION CALLED: query='{query}', translate_to='{translate_to}'"
    )

    if not query.strip():
        yield "Please enter a song name or description.", "", ""
        return

    # Check cache first
    target_lang = translate_to.strip() if translate_to.strip() else None
    cached = load_from_cache(query, target_lang)
    if cached:
        logger.info(f"WEB UI: Found cached result for '{query}' - returning from cache")
        yield cached["progress"], cached["lyrics"], cached["facts"]
        return
    logger.info(f"WEB UI: No cache found for '{query}' - proceeding with live search")

    # Initialize
    progress_log = []
    current_lyrics = ""
    current_facts = ""

    try:
        logger.info(
            f"WEB UI: Starting search for '{query}' with translation to '{target_lang}'"
        )

        # Step 1: Start
        progress_log.append("🤔 Analyzing your request...")
        yield "\n".join(progress_log), current_lyrics, current_facts

        # Create state
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

        # Track accumulated state
        result_state = {}

        # Stream through the graph execution
        logger.info(f"WEB UI: About to start streaming...")
        chunk_count = 0
        for chunk in app.stream(initial_state):
            chunk_count += 1
            logger.info(f"WEB UI: Got chunk {chunk_count}: {list(chunk.keys())}")
            for node_name, node_output in chunk.items():
                # Update accumulated result state
                result_state.update(node_output)
                logger.info(
                    f"WEB UI: Processing {node_name} with keys: {list(node_output.keys())}"
                )

                # Handle each node type (using correct node names from graph.py)
                if node_name == "analyze_query":
                    if node_output.get("song_title"):
                        title = node_output.get("song_title", "")
                        artist = node_output.get("song_artist", "")
                        artist_info = f" by {artist}" if artist else ""
                        progress_log.append(f"🧠 Identified: '{title}'{artist_info}")
                    yield "\n".join(progress_log), current_lyrics, current_facts

                elif node_name == "search_lyrics":
                    if node_output.get("search_results"):
                        progress_log.append(
                            f"🔎 Found {len(node_output['search_results'])} sources"
                        )
                    yield "\n".join(progress_log), current_lyrics, current_facts

                elif node_name == "filter_results":
                    progress_log.append("🔍 Filtered to best source")
                    yield "\n".join(progress_log), current_lyrics, current_facts

                elif node_name == "format_lyrics":
                    if node_output.get("formatted_lyrics"):
                        progress_log.append(
                            f"🤖 Formatted lyrics ({len(node_output['formatted_lyrics'])} chars)"
                        )
                        current_lyrics = node_output["formatted_lyrics"]
                    yield "\n".join(progress_log), current_lyrics, current_facts

                elif node_name == "translate_lyrics":
                    if target_lang:
                        progress_log.append(f"🈯 Translated to {target_lang}")
                    yield "\n".join(progress_log), current_lyrics, current_facts

                elif node_name == "intersperse_lyrics":
                    if target_lang and node_output.get("interspersed_lyrics"):
                        progress_log.append(
                            "🎨 Combined original and translated lyrics"
                        )
                        current_lyrics = node_output["interspersed_lyrics"]
                    yield "\n".join(progress_log), current_lyrics, current_facts

                elif node_name == "find_curious_facts":
                    if node_output.get("curious_facts"):
                        progress_log.append("🧐 Found curious facts")
                        current_facts = node_output["curious_facts"]
                    yield "\n".join(progress_log), current_lyrics, current_facts

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
            yield final_progress, "", ""
            return

        # Add completion message and make final yield
        progress_log.append("✅ Complete!")
        final_progress = "\n".join(progress_log)

        # Cache the results
        save_to_cache(
            query,
            target_lang,
            {
                "progress": final_progress,
                "lyrics": current_lyrics,
                "facts": current_facts,
            },
        )

        yield final_progress, current_lyrics, current_facts

    except Exception as e:
        logger.error(f"WEB UI: Error in search_lyrics_simple: {e}")
        error_msg = f"❌ Error: {str(e)}"
        progress_log.append(error_msg)
        yield "\n".join(progress_log), current_lyrics, current_facts


# Create the Gradio interface
def create_simple_interface():
    """Create the simple Gradio interface."""

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

        # Set up the search action with streaming enabled
        search_button.click(
            fn=search_lyrics_simple,
            inputs=[query_input, translate_input],
            outputs=[progress_output, lyrics_output, facts_output],
            show_progress=True,  # Enable Gradio's built-in progress
        )

    return demo


# Main entry point
if __name__ == "__main__":
    demo = create_simple_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Create a public URL
    )
