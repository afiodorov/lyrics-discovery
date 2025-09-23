"""Graph construction and conditional logic for the lyrics search agent."""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    analyze_query_node,
    extract_lyrics_node,
    find_curious_facts_node,
    intersperse_lyrics_node,
    search_lyrics_node,
    translate_lyrics_node,
)
from .state import AgentState


def should_continue_after_search(state: AgentState) -> str:
    """Determines whether to continue after search or end with error."""
    return (
        "end_with_error"
        if state.get("error_message") or not state.get("search_results")
        else "extract_lyrics"
    )


def should_translate(state: AgentState) -> str:
    """Determines whether to translate lyrics, find facts, or end with error."""
    if state.get("error_message"):
        return "end_with_error"
    return (
        "translate_lyrics"
        if state.get("target_language") and state.get("formatted_lyrics")
        else "find_facts"
    )


def create_workflow() -> CompiledStateGraph:
    """Creates and configures the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("search_lyrics", search_lyrics_node)
    workflow.add_node("extract_lyrics", extract_lyrics_node)  # Combined filter+format
    workflow.add_node("translate_lyrics", translate_lyrics_node)
    workflow.add_node("intersperse_lyrics", intersperse_lyrics_node)
    workflow.add_node("find_curious_facts", find_curious_facts_node)

    # Set entry point
    workflow.set_entry_point("analyze_query")

    # Add edges
    workflow.add_edge("analyze_query", "search_lyrics")

    # Conditional edges
    workflow.add_conditional_edges(
        "search_lyrics",
        should_continue_after_search,
        {"extract_lyrics": "extract_lyrics", "end_with_error": END},
    )
    workflow.add_conditional_edges(
        "extract_lyrics",
        should_translate,
        {
            "translate_lyrics": "translate_lyrics",
            "find_facts": "find_curious_facts",
            "end_with_error": END,
        },
    )

    # Final edges
    workflow.add_edge("translate_lyrics", "intersperse_lyrics")
    workflow.add_edge("intersperse_lyrics", "find_curious_facts")
    workflow.add_edge("find_curious_facts", END)

    # Compile the workflow to get a runnable app
    return workflow.compile()
