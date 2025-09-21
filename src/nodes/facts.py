"""Facts finding node for discovering interesting information about songs."""

import wikipedia

from ..config import llm_client, tavily_client
from ..state import AgentState, print_debug_log


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
        print(
            "    - Could not find a specific Wikipedia page. Trying web search as a fallback."
        )
        facts_content = None

    if not facts_content:
        try:
            search_query = f"interesting facts about the song '{title}'" + (
                f" by '{artist}'" if artist else ""
            )
            response = tavily_client.search(
                query=search_query, search_depth="basic", max_results=3
            )
            facts_content = "\n\n".join(
                [result["content"] for result in response["results"]]
            )
            print("    - Found web search results, summarizing...")
        except Exception as e:
            if state.get("debug_mode"):
                print(f"    - ⚠️ Tavily facts search failed with error: {e}")
            print(f"    - Web search for facts also failed.")
            return {}

    if not facts_content:
        return {}

    try:
        system_prompt = (
            "You are a research assistant. Your task is to read the provided text "
            "about a song and extract 1 to 3 curious or interesting facts. Format them as a short, "
            "bulleted list. If no interesting facts can be found, respond with 'No specific facts found.'"
        )
        user_prompt = f"Extract 1-3 curious facts from this article about '{title}':\n\n{facts_content[:4000]}"
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        facts = response.choices[0].message.content.strip()
        if "No specific facts found" in facts:
            return {}
        update = {"curious_facts": facts}
        print_debug_log("find_curious_facts_node", {**state, **update})
        return update
    except Exception as e:
        if state.get("debug_mode"):
            print(f"    - ⚠️ LLM fact extraction failed with error: {e}")
        print(f"    - An error occurred during LLM fact extraction.")
        return {}
