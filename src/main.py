"""Main entry point for the lyrics search application."""

import argparse

from .config import OPENAI_API_KEY, TAVILY_API_KEY
from .graph import create_workflow
from .logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def display_results(final_state: dict) -> None:
    """Display the results of the lyrics search."""
    print("\n" + "=" * 50)

    if final_state.get("error_message"):
        print(f"🔥 An error occurred: {final_state['error_message']}")
    elif final_state.get("interspersed_lyrics"):
        title, artist, lang = (
            final_state["song_title"],
            final_state["song_artist"],
            final_state["target_language"],
        )
        artist_info = f" by {artist}" if artist else ""
        print(f"🎶 Lyrics for '{title}'{artist_info} (with {lang} translation)")
        print("=" * 50 + "\n")
        print(final_state["interspersed_lyrics"])
    elif final_state.get("formatted_lyrics"):
        title, artist = final_state["song_title"], final_state["song_artist"]
        artist_info = f" by {artist}" if artist else ""
        print(f"🎶 Lyrics for '{title}'{artist_info}")
        print("=" * 50 + "\n")
        print(final_state["formatted_lyrics"])
    else:
        print("🤷 The agent finished without finding lyrics or an error occurred.")

    if final_state.get("curious_facts"):
        print("\n" + "-" * 50)
        print("🧐 Curious Facts")
        print("-" * 50 + "\n")
        print(final_state["curious_facts"])

    print("\n" + "=" * 50)


def main():
    """Main function that sets up and runs the lyrics search agent."""
    parser = argparse.ArgumentParser(
        description="Find song lyrics using a LangGraph agent."
    )
    parser.add_argument(
        "query", type=str, help="The song title or a description of the song."
    )
    parser.add_argument(
        "-t", "--translate", type=str, help="Optional: Target language for translation."
    )
    # Verbose options following Unix conventions
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (debug output from our application only)",
    )
    parser.add_argument(
        "-vvv",
        "--very-verbose",
        action="store_true",
        help="Enable very verbose logging (debug output from all libraries including OpenAI, httpx, etc.)",
    )
    # Legacy compatibility
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Legacy alias for --verbose (deprecated, use -v instead)",
    )
    args = parser.parse_args()

    # Setup logging based on verbosity level
    if args.very_verbose:
        setup_logging(verbose_level="very_verbose")
    elif args.verbose or args.debug:  # Include legacy --debug flag
        setup_logging(verbose_level="verbose")
        if args.debug:
            logger.warning("--debug is deprecated, use -v/--verbose instead")
    else:
        setup_logging(verbose_level="normal")

    if not TAVILY_API_KEY or not OPENAI_API_KEY:
        logger.error(
            "❌ Error: TAVILY_API_KEY and OPENAI_API_KEY environment variables must be set."
        )
        return

    # Create and compile the workflow
    workflow = create_workflow()
    app = workflow.compile()

    # Set initial state (no more debug_mode needed)
    initial_state = {
        "user_query": args.query,
        "target_language": args.translate,
    }

    # Run the agent
    final_state = app.invoke(initial_state)

    # Display results
    display_results(final_state)


if __name__ == "__main__":
    main()
