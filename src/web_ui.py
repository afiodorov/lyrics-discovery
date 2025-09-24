"""Fixed Gradio web UI with proper URL parameter handling and sharing."""

import os

import gradio as gr

from .graph import create_workflow
from .logging_config import get_logger, setup_logging
from .resources.defaults import (
    get_default_facts,
    get_default_language,
    get_default_lyrics,
    get_default_progress,
    get_default_query,
)
from .state import AgentState

# Set up logging
setup_logging("normal")
logger = get_logger(__name__)


def search_lyrics_simple(query: str, translate_to: str):
    """
    Simple generator function that works with Gradio streaming.
    Yields: (progress_log, lyrics_output, facts_output)
    """
    # Handle None values
    query = query or ""
    translate_to = translate_to or ""

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
                "extract_lyrics": "✨ Extracting and formatting lyrics...",
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

    with gr.Blocks(
        title="🎵 Lyrics Search & Translate",
        head="""
        <script>
        window.updateUrlWithSearch = function(query, translate) {
            // Handle null/undefined values
            query = query || '';
            translate = translate || '';

            console.log('🔗 Updating URL with:', query, translate);

            // Build URL parameters
            const params = new URLSearchParams();
            if (query && query.trim()) {
                params.set('q', query.trim());
            }
            if (translate && translate.trim()) {
                params.set('t', translate.trim());
            }

            // Update URL without page reload
            const newUrl = params.toString() ?
                window.location.pathname + '?' + params.toString() :
                window.location.pathname;

            window.history.pushState({}, '', newUrl);
            console.log('🔗 Updated URL to:', newUrl);
        };
        </script>
        """,
    ) as demo:
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
                    value=get_default_query(),
                    lines=1,
                    elem_id="query_input",
                )

            with gr.Column(scale=1):
                translate_input = gr.Textbox(
                    label="Translate to (optional)",
                    placeholder="e.g., 'ru', 'es', 'fr'",
                    value=get_default_language(),
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

        # Hidden HTML component for JavaScript execution
        html_output = gr.HTML(visible=False)

        def search_and_update_url(query: str, translate_to: str):
            """Search for lyrics and update URL in browser."""
            # Handle None values
            query = query or ""
            translate_to = translate_to or ""

            # Use the existing search generator
            for result in search_lyrics_simple(query, translate_to):
                yield result + ("",)  # Add empty string for the HTML output

        # Set up search action
        search_button.click(
            fn=search_and_update_url,
            inputs=[query_input, translate_input],
            outputs=[progress_output, lyrics_output, facts_output, html_output],
            show_progress="full",
        )

        # Add JavaScript click handler to update URL
        search_button.click(
            fn=None,
            inputs=[query_input, translate_input],
            outputs=[],
            js="(query, translate) => { console.log('Updating URL:', query, translate); window.updateUrlWithSearch(query, translate); }",
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
                        return query, translate, progress, lyrics, facts, ""
                    else:
                        return query, translate, "Search completed", "", "", ""
                else:
                    # No URL params - use pre-computed defaults from resource files
                    default_query = get_default_query()
                    default_translate = get_default_language()
                    default_progress = get_default_progress()
                    default_lyrics = get_default_lyrics()
                    default_facts = get_default_facts()

                    print(f"📎 Loading pre-computed defaults for: {default_query}")

                    return (
                        default_query,
                        default_translate,
                        default_progress,
                        default_lyrics,
                        default_facts,
                        "",
                    )

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
                html_output,
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
