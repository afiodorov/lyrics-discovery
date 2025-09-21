"""Main entry point for the lyrics search application."""

import argparse

from .config import OPENAI_API_KEY, TAVILY_API_KEY
from .graph import create_workflow


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
        print(
            f"🎶 Lyrics for '{title}'{f' by {artist}' if artist else ''} (with {lang} translation)"
        )
        print("=" * 50 + "\n")
        print(final_state["interspersed_lyrics"])
    elif final_state.get("formatted_lyrics"):
        title, artist = final_state["song_title"], final_state["song_artist"]
        print(f"🎶 Lyrics for '{title}'{f' by {artist}' if artist else ''}")
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to inspect agent state.",
    )
    args = parser.parse_args()

    if not TAVILY_API_KEY or not OPENAI_API_KEY:
        print(
            "❌ Error: TAVILY_API_KEY and OPENAI_API_KEY environment variables must be set."
        )
        return

    # Create and compile the workflow
    workflow = create_workflow()
    app = workflow.compile()

    # Set initial state
    initial_state = {
        "user_query": args.query,
        "target_language": args.translate,
        "debug_mode": args.debug,
    }

    # Run the agent
    final_state = app.invoke(initial_state)

    # Display results
    display_results(final_state)


if __name__ == "__main__":
    main()
