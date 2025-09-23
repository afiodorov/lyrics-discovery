"""Fixed Gradio web UI with proper URL parameter handling and sharing."""

from typing import Optional

import gradio as gr

from .graph import create_workflow
from .logging_config import get_logger, setup_logging
from .state import AgentState

# Set up logging
setup_logging("normal")
logger = get_logger(__name__)


def search_lyrics_simple(query: str, translate_to: str):
    """
    Simple generator function that works with Gradio streaming.
    Yields: (progress_log, lyrics_output, facts_output)
    """
    if not query.strip():
        yield "Please enter a song name or description.", "", ""
        return

    # Initialize
    target_lang = translate_to.strip() if translate_to.strip() else None

    # Create workflow
    app = create_workflow()

    # Set initial state
    initial_state = AgentState(
        user_query=query,
        target_language=target_lang,
    )

    progress_log = []
    current_lyrics = ""
    current_facts = ""

    try:
        progress_log.append(f"🎵 Searching for: {query}")
        if target_lang:
            progress_log.append(f"🌍 Will translate to: {target_lang}")
        yield "\n".join(progress_log), "", ""

        # Stream the results
        for event in app.stream(initial_state):
            node_name = list(event.keys())[0]
            result_state = event[node_name]

            # Map node names to user-friendly messages
            node_messages = {
                "analyze_query": "🤔 Analyzing your request...",
                "search_lyrics": "🔍 Searching for lyrics...",
                "filter_results": "📋 Filtering search results...",
                "format_lyrics": "✨ Formatting lyrics...",
                "translate_lyrics": f"🌍 Translating to {target_lang}...",
                "intersperse_lyrics": "🎨 Combining original and translated lyrics...",
                "find_curious_facts": "🧐 Finding curious facts...",
            }

            if node_name in node_messages:
                progress_log.append(node_messages[node_name])
                yield "\n".join(progress_log), current_lyrics, current_facts

            # Update outputs as data becomes available
            if "formatted_lyrics" in result_state:
                current_lyrics = result_state["formatted_lyrics"]
                yield "\n".join(progress_log), current_lyrics, current_facts

            if "interspersed_lyrics" in result_state:
                current_lyrics = result_state["interspersed_lyrics"]
                yield "\n".join(progress_log), current_lyrics, current_facts

            if "curious_facts" in result_state:
                current_facts = result_state["curious_facts"]
                yield "\n".join(progress_log), current_lyrics, current_facts

            # Handle errors
            if "error_message" in result_state:
                error_msg = f"❌ Error: {result_state['error_message']}"
                progress_log.append(error_msg)
                final_progress = "\n".join(progress_log)
                yield final_progress, "", ""
                return

    except Exception as e:
        error_msg = f"❌ An error occurred: {str(e)}"
        progress_log.append(error_msg)
        yield "\n".join(progress_log), "", ""
        return

    # Final update
    progress_log.append("✅ Complete!")
    yield "\n".join(progress_log), current_lyrics, current_facts


def create_simple_interface():
    """Create the simple Gradio interface with working URL parameters."""

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
                    placeholder="Enter song name or description",
                    value="",  # Start empty
                    lines=1,
                    elem_id="query_input",
                )

            with gr.Column(scale=1):
                translate_input = gr.Textbox(
                    label="Translate to (optional)",
                    placeholder="e.g., 'ru', 'es', 'fr'",
                    value="",  # Start empty
                    lines=1,
                    elem_id="translate_input",
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

        # Set up search action
        search_button.click(
            fn=search_lyrics_simple,
            inputs=[query_input, translate_input],
            outputs=[progress_output, lyrics_output, facts_output],
            show_progress="full",
        )

        # Handle URL parameters and auto-search on load
        def load_and_search_from_url(request: gr.Request):
            """Load query parameters from URL and auto-search if present."""
            if request:
                query = request.query_params.get("q", "")
                translate = request.query_params.get("t", "")

                print(f"📎 Loading from URL: q='{query}', t='{translate}'")

                if query:
                    print(f"🔍 Auto-searching for: {query}")
                    # Start the search immediately and return results
                    results = list(search_lyrics_simple(query, translate))
                    if results:
                        # Get the final result
                        progress, lyrics, facts = results[-1]
                        return query, translate, progress, lyrics, facts
                    else:
                        return query, translate, "Search completed", "", ""
                else:
                    # Just populate fields without searching
                    return (
                        query,
                        translate,
                        "Ready to search... (loaded from URL)",
                        "",
                        "",
                    )

            return "", "", "Ready to search...", "", ""

        # Set up load handler to populate fields and auto-search from URL
        demo.load(
            fn=load_and_search_from_url,
            inputs=[],
            outputs=[
                query_input,
                translate_input,
                progress_output,
                lyrics_output,
                facts_output,
            ],
        )

    return demo


# For running with gradio command
demo = create_simple_interface()

if __name__ == "__main__":
    print("🎵 Starting Lyrics Search Web Interface...")
    print("✨ Features:")
    print("  • URL parameter support: ?q=song&t=language")
    print("  • Auto-search from URL parameters")
    print("  • Redis caching for fast repeated searches")
    print("  • Advanced Tavily search with raw content")
    print()
    demo.launch(server_name="0.0.0.0", server_port=7860)
