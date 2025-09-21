"""State definition and utilities for the lyrics search agent."""

from typing import List, TypedDict


class AgentState(TypedDict):
    """State object that flows through the agent graph."""

    user_query: str
    target_language: str
    song_title: str
    song_artist: str
    search_results: List[str]
    formatted_lyrics: str
    translated_lyrics: str
    interspersed_lyrics: str
    curious_facts: str
    error_message: str
    debug_mode: bool


def print_debug_log(node_name: str, state: AgentState):
    """Prints a formatted log of the current state if debug mode is on."""
    if not state.get("debug_mode"):
        return
    print("\n" + f"--- DEBUG: After {node_name} ---")
    for key, value in state.items():
        if key == "search_results" and value:
            print(f"  - {key}: {len(value)} items found.")
            for i, result in enumerate(value):
                # Print a snippet of each search result to inspect its quality
                print(f"    - Result {i + 1}: {result[:150].replace('\n', ' ')}...")
        elif isinstance(value, str) and len(value) > 250:
            print(f"  - {key}: {value[:250]}... (truncated)")
        elif key != "debug_mode":
            print(f"  - {key}: {value}")
    print("--- END DEBUG ---" + "\n")
