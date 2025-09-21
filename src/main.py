import os
import argparse
import json
from typing import TypedDict, List
from tavily import TavilyClient
import openai
import wikipedia
from langgraph.graph import StateGraph, END

# --- Configuration ---
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# It's good practice to have a centralized client
llm_client = openai.OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# --- 1. Define the State for our Graph ---
class AgentState(TypedDict):
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
    debug_mode: bool # New field for debugging

# --- Utility Function for Debugging ---
def print_debug_log(node_name: str, state: AgentState):
    """Prints a formatted log of the current state if debug mode is on."""
    if not state.get("debug_mode"):
        return
    print("\n" + f"--- DEBUG: After {node_name} ---")
    for key, value in state.items():
        if key == 'search_results' and value:
            print(f"  - {key}: {len(value)} items found.")
            for i, result in enumerate(value):
                # Print a snippet of each search result to inspect its quality
                print(f"    - Result {i+1}: {result[:150].replace('\n', ' ')}...")
        elif isinstance(value, str) and len(value) > 250:
            print(f"  - {key}: {value[:250]}... (truncated)")
        elif key != 'debug_mode':
            print(f"  - {key}: {value}")
    print("--- END DEBUG ---" + "\n")

# --- 2. Define the Nodes for our Graph ---

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
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(response.choices[0].message.content)
        if result.get("title") != user_query:
            print(f"🧠 I believe you're looking for '{result.get('title')}'{f' by {result.get('artist')}' if result.get('artist') else ''}.")
        update = {"song_title": result.get("title"), "song_artist": result.get("artist")}
        print_debug_log("analyze_query_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ❌ ERROR in analyze_query_node: {e}")
        return {"error_message": "An error occurred during query analysis."}

def search_lyrics_node(state: AgentState) -> dict:
    """Searches for lyrics using the Tavily Search API."""
    title, artist = state["song_title"], state["song_artist"]
    print(f"🔎 Searching for lyrics for '{title}'{f' by {artist}' if artist else ''}...")
    # More specific query to get cleaner, lyrics-focused results
    query = f"full complete song lyrics for '{title}'" + (f" by {artist}" if artist else "")
    try:
        response = tavily_client.search(query=query, search_depth="advanced", max_results=5)
        content = [result["content"] for result in response["results"]]
        update = {"search_results": content}
        print_debug_log("search_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ❌ ERROR in search_lyrics_node: {e}")
        return {"error_message": "An error occurred during the web search."}

def filter_results_node(state: AgentState) -> dict:
    """
    NEW NODE: Uses an LLM to analyze search results and pick the best one.
    """
    search_results = state["search_results"]
    print("🔍 Filtering search results to find the best source...")

    system_prompt = (
        "You are a web scraping expert. Your task is to analyze a list of text snippets from web pages "
        "and identify the one that most likely contains the complete and accurate lyrics for a song. "
        "Ignore snippets that are mostly user comments, tracklists, or incomplete. "
        "Respond with ONLY the full text of the best snippet. "
        "If no snippet appears to be a good source for lyrics, respond with the single phrase: 'No suitable source found.'"
    )

    # We'll number the results to make it easier for the LLM to differentiate
    numbered_results = "\n\n".join([f"--- RESULT {i+1} ---\n{res}" for i, res in enumerate(search_results)])
    user_prompt = (
        f"Analyze the following search results and return the full text of the best one for extracting lyrics:\n\n"
        f"{numbered_results}"
    )

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
        )
        best_result = response.choices[0].message.content.strip()

        if "No suitable source found" in best_result:
            return {"error_message": "Could not find a suitable source for lyrics in search results."}
        
        # Update the search_results to contain only the single best one
        update = {"search_results": [best_result]}
        print_debug_log("filter_results_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ❌ ERROR in filter_results_node: {e}")
        return {"error_message": "An error occurred while filtering search results."}


def format_lyrics_node(state: AgentState) -> dict:
    """Uses an LLM to extract and format lyrics from the single best search result."""
    # This node now receives a much cleaner, pre-filtered search result
    search_context = "\n\n".join(state["search_results"])
    title, artist = state["song_title"], state["song_artist"]
    print("🤖 Asking the LLM to format the lyrics...")
    system_prompt = (
        "You are an expert assistant specializing in formatting song lyrics. Your task is to analyze the "
        "provided text, which has been pre-selected as the best source, and format the lyrics cleanly. "
        "Ensure you provide the FULL, COMPLETE lyrics. Do not truncate them. "
        "Do not include commentary or introductory phrases. If you cannot find lyrics, respond with the "
        "single phrase: 'Could not find lyrics.'"
    )
    user_prompt = (
        f"Here is the best available source for '{title}'{f' by {artist}' if artist else ''}:\n\n"
        f"--- SOURCE TEXT ---\n{search_context}\n--- END OF SOURCE ---\n\n"
        "Please extract and format the full, complete lyrics."
    )
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=4000
        )
        formatted = response.choices[0].message.content.strip()
        if "Could not find lyrics" in formatted or len(formatted) < 20:
            return {"error_message": "Could not find lyrics after formatting."}
        update = {"formatted_lyrics": formatted}
        print_debug_log("format_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ❌ ERROR in format_lyrics_node: {e}")
        return {"error_message": "An error occurred while formatting lyrics."}

def translate_lyrics_node(state: AgentState) -> dict:
    """Translates the lyrics to the target language."""
    lyrics, language = state["formatted_lyrics"], state["target_language"]
    print(f"🈯 Translating lyrics to {language}...")
    system_prompt = "You are a world-class polyglot and translator. Your task is to translate the provided song lyrics into the specified target language. Retain the poetic structure and meaning as best as possible. Do not add any commentary or introductory text, only the translated lyrics."
    user_prompt = f"Please translate the following lyrics into {language}:\n\n--- LYRICS ---\n{lyrics}\n--- END OF LYRICS ---"
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=4000
        )
        translated = response.choices[0].message.content.strip()
        update = {"translated_lyrics": translated}
        print_debug_log("translate_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ❌ ERROR in translate_lyrics_node: {e}")
        return {"error_message": "An error occurred during translation."}

def intersperse_lyrics_node(state: AgentState) -> dict:
    """Combines original and translated lyrics into an interspersed format."""
    original, translated, language = state["formatted_lyrics"], state["translated_lyrics"], state["target_language"]
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
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=4095
        )
        interspersed = response.choices[0].message.content.strip()
        update = {"interspersed_lyrics": interspersed}
        print_debug_log("intersperse_lyrics_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ❌ ERROR in intersperse_lyrics_node: {e}")
        return {"error_message": "An error occurred during final formatting."}

def find_curious_facts_node(state: AgentState) -> dict:
    """Searches for curious facts, first on Wikipedia, then via web search as a fallback."""
    title, artist = state["song_title"], state["song_artist"]
    print(f"🧐 Searching for curious facts about '{title}'...")
    facts_content = None

    try:
        search_query = f"{title} (song)" + (f" ({artist} song)" if artist else "")
        page = wikipedia.page(search_query, auto_suggest=True, redirect=True)
        facts_content = page.content
        print("    - Found Wikipedia page, summarizing...")
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ⚠️ Wikipedia search failed with error: {e}")
        print("    - Could not find a specific Wikipedia page. Trying web search as a fallback.")
        facts_content = None

    if not facts_content:
        try:
            search_query = f"interesting facts about the song '{title}'" + (f" by '{artist}'" if artist else "")
            response = tavily_client.search(query=search_query, search_depth="basic", max_results=3)
            facts_content = "\n\n".join([result["content"] for result in response["results"]])
            print("    - Found web search results, summarizing...")
        except Exception as e:
            if state.get("debug_mode"):
                print(f"    - ⚠️ Tavily facts search failed with error: {e}")
            print(f"    - Web search for facts also failed.")
            return {}

    if not facts_content: return {}

    try:
        system_prompt = (
            "You are a research assistant. Your task is to read the provided text "
            "about a song and extract 1 to 3 curious or interesting facts. Format them as a short, "
            "bulleted list. If no interesting facts can be found, respond with 'No specific facts found.'"
        )
        user_prompt = f"Extract 1-3 curious facts from this article about '{title}':\n\n{facts_content[:4000]}"
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
        )
        facts = response.choices[0].message.content.strip()
        if "No specific facts found" in facts: return {}
        update = {"curious_facts": facts}
        print_debug_log("find_curious_facts_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ⚠️ LLM fact extraction failed with error: {e}")
        print(f"    - An error occurred during LLM fact extraction.")
        return {}


# --- 3. Define Conditional Logic ---
def should_continue_after_search(state: AgentState) -> str:
    return "end_with_error" if state.get("error_message") or not state.get("search_results") else "filter_results"

def should_format_lyrics(state: AgentState) -> str:
    return "end_with_error" if state.get("error_message") or not state.get("search_results") else "format_lyrics"

def should_translate(state: AgentState) -> str:
    if state.get("error_message"): return "end_with_error"
    return "translate_lyrics" if state.get("target_language") and state.get("formatted_lyrics") else "find_facts"

# --- 4. Build and Run the Graph ---
def main():
    parser = argparse.ArgumentParser(description="Find song lyrics using a LangGraph agent.")
    parser.add_argument("query", type=str, help="The song title or a description of the song.")
    parser.add_argument("-t", "--translate", type=str, help="Optional: Target language for translation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to inspect agent state.")
    args = parser.parse_args()

    if not TAVILY_API_KEY or not OPENAI_API_KEY:
        print("❌ Error: TAVILY_API_KEY and OPENAI_API_KEY environment variables must be set.")
        return

    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("search_lyrics", search_lyrics_node)
    workflow.add_node("filter_results", filter_results_node) # New node
    workflow.add_node("format_lyrics", format_lyrics_node)
    workflow.add_node("translate_lyrics", translate_lyrics_node)
    workflow.add_node("intersperse_lyrics", intersperse_lyrics_node)
    workflow.add_node("find_curious_facts", find_curious_facts_node)

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "search_lyrics")

    # Updated graph flow
    workflow.add_conditional_edges("search_lyrics", should_continue_after_search, {"filter_results": "filter_results", "end_with_error": END})
    workflow.add_conditional_edges("filter_results", should_format_lyrics, {"format_lyrics": "format_lyrics", "end_with_error": END})
    
    workflow.add_conditional_edges("format_lyrics", should_translate, {"translate_lyrics": "translate_lyrics", "find_facts": "find_curious_facts", "end_with_error": END})
    workflow.add_edge("translate_lyrics", "intersperse_lyrics")
    workflow.add_edge("intersperse_lyrics", "find_curious_facts")
    workflow.add_edge("find_curious_facts", END)

    app = workflow.compile()
    initial_state = {"user_query": args.query, "target_language": args.translate, "debug_mode": args.debug}
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 50)
    if final_state.get("error_message"):
        print(f"🔥 An error occurred: {final_state['error_message']}")
    elif final_state.get("interspersed_lyrics"):
        title, artist, lang = final_state['song_title'], final_state['song_artist'], final_state['target_language']
        print(f"🎶 Lyrics for '{title}'{f' by {artist}' if artist else ''} (with {lang} translation)")
        print("=" * 50 + "\n")
        print(final_state['interspersed_lyrics'])
    elif final_state.get("formatted_lyrics"):
        title, artist = final_state['song_title'], final_state['song_artist']
        print(f"🎶 Lyrics for '{title}'{f' by {artist}' if artist else ''}")
        print("=" * 50 + "\n")
        print(final_state['formatted_lyrics'])
    else:
        print("🤷 The agent finished without finding lyrics or an error occurred.")

    if final_state.get("curious_facts"):
        print("\n" + "-" * 50)
        print("🧐 Curious Facts")
        print("-" * 50 + "\n")
        print(final_state['curious_facts'])
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()

