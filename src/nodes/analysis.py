"""Query analysis node for identifying song and artist."""

import json

from ..config import llm_client
from ..state import AgentState, print_debug_log


def analyze_query_node(state: AgentState) -> dict:
    """Analyzes the user's query to determine the actual song title and artist."""
    user_query = state["user_query"]
    print(f"🤔 Analyzing your request: '{user_query}'...")
    system_prompt = (
        "You are an expert musicologist. Your task is to analyze a user's query about a song and "
        "determine the precise song title and artist. Respond ONLY with a single, valid JSON object "
        "with two keys: 'title' and 'artist'. The artist can be null if unknown. "
        "If you cannot deduce a specific song, return the original query in the 'title' field."
    )
    user_prompt = f"Analyze this query: '{user_query}'"
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(response.choices[0].message.content)
        if result.get("title") != user_query:
            print(
                f"🧠 I believe you're looking for '{result.get('title')}'{f' by {result.get("artist")}' if result.get('artist') else ''}."
            )
        update = {
            "song_title": result.get("title"),
            "song_artist": result.get("artist"),
        }
        print_debug_log("analyze_query_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ❌ ERROR in analyze_query_node: {e}")
        return {"error_message": "An error occurred during query analysis."}
