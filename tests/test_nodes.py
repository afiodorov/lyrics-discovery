"""Tests for individual nodes with mocked responses to catch NoneType errors."""

from unittest.mock import Mock, patch

import pytest

from src.nodes.facts import (
    _detect_song_language,
    _translate_facts,
    find_curious_facts_node,
)
from src.nodes.formatting import format_lyrics_node, intersperse_lyrics_node
from src.nodes.translation import translate_lyrics_node
from src.state import AgentState


class MockOpenAIResponse:
    """Mock OpenAI API response."""

    def __init__(self, content="Mock response", finish_reason="stop"):
        self.choices = [Mock()]
        self.choices[0].message = Mock()
        self.choices[0].message.content = content
        self.choices[0].finish_reason = finish_reason


@pytest.fixture
def basic_state():
    """Basic state for testing."""
    return AgentState(
        user_query="test song",
        target_language="en",
        song_title="Test Song",
        song_artist="Test Artist",
        search_results=["Mock search result content"],
        formatted_lyrics="Test lyrics content",
        translated_lyrics="Translated test lyrics",
        interspersed_lyrics="",
        curious_facts="",
        error_message="",
    )


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch("src.config.llm_client") as mock_client:
        mock_client.chat.completions.create.return_value = MockOpenAIResponse()
        yield mock_client


@pytest.fixture
def mock_tavily():
    """Mock Tavily client."""
    with patch("src.config.tavily_client") as mock_client:
        mock_client.search.return_value = {
            "results": [{"content": "Mock fact content"}]
        }
        yield mock_client


@pytest.fixture
def mock_wikipedia():
    """Mock Wikipedia."""
    with patch("wikipedia.page") as mock_page:
        mock_page.return_value = Mock(content="Mock Wikipedia content")
        yield mock_page


class TestFactsNode:
    """Test facts node with various edge cases."""

    def test_facts_with_none_tavily_results(
        self, basic_state, mock_openai, mock_tavily, mock_wikipedia
    ):
        """Test facts node when Tavily returns None results."""
        # Wikipedia fails, Tavily returns None
        mock_wikipedia.side_effect = Exception("Wikipedia failed")
        mock_tavily.search.return_value = None

        result = find_curious_facts_node(basic_state)

        # Should return empty dict, not crash with NoneType error
        assert result == {}

    def test_facts_with_empty_tavily_results(
        self, basic_state, mock_openai, mock_tavily, mock_wikipedia
    ):
        """Test facts node when Tavily returns empty results."""
        mock_wikipedia.side_effect = Exception("Wikipedia failed")
        mock_tavily.search.return_value = {"results": []}

        result = find_curious_facts_node(basic_state)

        # Should return empty dict
        assert result == {}

    def test_facts_with_tavily_missing_results_key(
        self, basic_state, mock_openai, mock_tavily, mock_wikipedia
    ):
        """Test facts node when Tavily response has no 'results' key."""
        mock_wikipedia.side_effect = Exception("Wikipedia failed")
        mock_tavily.search.return_value = {"other_key": "value"}

        result = find_curious_facts_node(basic_state)

        # Should handle missing 'results' key gracefully
        assert result == {}

    def test_facts_with_none_content_in_results(
        self, basic_state, mock_openai, mock_tavily, mock_wikipedia
    ):
        """Test facts node when Tavily results have None content."""
        mock_wikipedia.side_effect = Exception("Wikipedia failed")
        mock_tavily.search.return_value = {
            "results": [
                {"content": None, "url": "test.com"},
                {"content": "Valid content", "url": "test2.com"},
            ]
        }

        # Mock successful fact extraction - should not contain "No specific facts found"
        mock_openai.chat.completions.create.return_value = MockOpenAIResponse(
            "• Test fact about the song"
        )

        result = find_curious_facts_node(basic_state)

        # Should handle None content and extract facts from valid content
        assert "curious_facts" in result
        assert "Test fact about the song" in result["curious_facts"]

    def test_facts_with_missing_content_key(
        self, basic_state, mock_openai, mock_tavily, mock_wikipedia
    ):
        """Test facts node when Tavily results missing 'content' key."""
        mock_wikipedia.side_effect = Exception("Wikipedia failed")
        mock_tavily.search.return_value = {
            "results": [
                {"url": "test.com"},  # Missing 'content' key
                {"content": "Valid content", "url": "test2.com"},
            ]
        }

        # Mock successful fact extraction
        mock_openai.chat.completions.create.return_value = MockOpenAIResponse(
            "• Test fact"
        )

        result = find_curious_facts_node(basic_state)

        # Should handle missing 'content' key gracefully
        assert "curious_facts" in result

    def test_language_detection_with_none_lyrics(self, basic_state):
        """Test language detection when lyrics are None."""
        state_with_none_lyrics = basic_state.copy()
        state_with_none_lyrics["formatted_lyrics"] = None

        result = _detect_song_language(
            state_with_none_lyrics, "Test Song", "Test Artist"
        )

        # Should return None, not crash
        assert result is None

    def test_language_detection_with_empty_lyrics(self, basic_state):
        """Test language detection when lyrics are empty string."""
        state_with_empty_lyrics = basic_state.copy()
        state_with_empty_lyrics["formatted_lyrics"] = ""

        result = _detect_song_language(
            state_with_empty_lyrics, "Test Song", "Test Artist"
        )

        # Should return None for empty lyrics
        assert result is None

    def test_translate_facts_with_none_facts(self):
        """Test translating None facts."""
        result = _translate_facts(None, "Spanish", "Test Song")

        # Should handle None facts gracefully
        assert result is None


class TestFormattingNode:
    """Test formatting nodes with edge cases."""

    def test_format_lyrics_with_none_search_results(self, mock_openai):
        """Test format_lyrics_node with None search results."""
        state = AgentState(
            user_query="test",
            target_language="en",
            song_title="Test Song",
            song_artist="Test Artist",
            search_results=None,  # None instead of list
            formatted_lyrics="",
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
        )

        # Should handle None search_results without crashing
        result = format_lyrics_node(state)

        # May return error or handle gracefully
        assert isinstance(result, dict)

    def test_format_lyrics_with_empty_search_results(self, mock_openai):
        """Test format_lyrics_node with empty search results."""
        state = AgentState(
            user_query="test",
            target_language="en",
            song_title="Test Song",
            song_artist="Test Artist",
            search_results=[],  # Empty list
            formatted_lyrics="",
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
        )

        mock_openai.chat.completions.create.return_value = MockOpenAIResponse(
            "Mock lyrics"
        )

        result = format_lyrics_node(state)

        # Should handle empty list gracefully
        assert isinstance(result, dict)

    def test_intersperse_with_none_lyrics(self, mock_openai):
        """Test intersperse_lyrics_node with None lyrics."""
        state = AgentState(
            user_query="test",
            target_language="en",
            song_title="Test Song",
            song_artist="Test Artist",
            search_results=["test"],
            formatted_lyrics=None,  # None lyrics
            translated_lyrics=None,  # None translation
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
        )

        # Should handle None lyrics without crashing
        result = intersperse_lyrics_node(state)

        assert isinstance(result, dict)


class TestTranslationNode:
    """Test translation node with edge cases."""

    def test_translate_with_none_lyrics(self, mock_openai):
        """Test translate_lyrics_node with None lyrics."""
        state = AgentState(
            user_query="test",
            target_language="en",
            song_title="Test Song",
            song_artist="Test Artist",
            search_results=["test"],
            formatted_lyrics=None,  # None lyrics
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
        )

        # Should handle None lyrics gracefully
        result = translate_lyrics_node(state)

        assert isinstance(result, dict)

    def test_translate_with_none_target_language(self, mock_openai):
        """Test translate_lyrics_node with None target language."""
        state = AgentState(
            user_query="test",
            target_language=None,  # None target language
            song_title="Test Song",
            song_artist="Test Artist",
            search_results=["test"],
            formatted_lyrics="Test lyrics",
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
        )

        # Should handle None target language
        result = translate_lyrics_node(state)

        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
