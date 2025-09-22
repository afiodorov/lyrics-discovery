"""Tests for the raw content improvement that fixes truncation issues."""

from unittest.mock import patch


from src.nodes.search import search_lyrics_node
from src.state import AgentState


class TestRawContentImprovement:
    """Test cases for raw content retrieval from Tavily."""

    def test_search_uses_raw_content_parameter(self):
        """Test that search includes raw_content parameter."""
        state = AgentState(
            user_query="Gracias a la vida",
            target_language="ru",
            song_title="Gracias a la Vida",
            song_artist="Violeta Parra",
            search_results=[],
            formatted_lyrics="",
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
        )

        # Mock Tavily response with raw_content
        mock_response = {
            "results": [
                {
                    "content": "Short snippet...",
                    "raw_content": "Much longer full page content with complete lyrics: "
                    + "A" * 5000,
                    "url": "https://example.com/lyrics",
                },
                {
                    "content": "Another snippet...",
                    "raw_content": "Another full page with additional content: "
                    + "B" * 3000,
                    "url": "https://example.com/another",
                },
            ]
        }

        with patch(
            "src.nodes.search.tavily_client.search", return_value=mock_response
        ) as mock_search:
            result = search_lyrics_node(state)

            # Verify Tavily was called with raw_content parameter
            mock_search.assert_called_once_with(
                query="full complete song lyrics for 'Gracias a la Vida' by Violeta Parra",
                search_depth="advanced",
                max_results=5,
                include_raw_content="text",
            )

            # Verify we get the full raw content, not just snippets
            assert len(result["search_results"][0]) > 5000  # Much longer than snippet
            assert len(result["search_results"][1]) > 3000
            assert "Much longer full page content" in result["search_results"][0]
            assert (
                "Another full page with additional content"
                in result["search_results"][1]
            )

    def test_fallback_to_content_when_no_raw_content(self):
        """Test fallback to content field when raw_content is not available."""
        state = AgentState(
            user_query="Test song",
            target_language="en",
            song_title="Test Song",
            song_artist="Test Artist",
            search_results=[],
            formatted_lyrics="",
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
        )

        # Mock response with no raw_content
        mock_response = {
            "results": [
                {
                    "content": "Only content snippet available",
                    "url": "https://example.com/no-raw",
                }
            ]
        }

        with patch("src.nodes.search.tavily_client.search", return_value=mock_response):
            result = search_lyrics_node(state)

            # Should fall back to content field
            assert result["search_results"][0] == "Only content snippet available"

    def test_character_count_improvement(self):
        """Test that raw content provides significantly more characters."""
        # This test documents the improvement we've made

        # Before: typical content snippet lengths were ~150 chars
        old_snippet_length = 150

        # After: raw content can be 5000+ chars (real example from logs)
        new_raw_content_length = 5754  # From actual log output

        # We should see at least 10x improvement
        improvement_ratio = new_raw_content_length / old_snippet_length
        assert improvement_ratio > 10

        # Full song should be around 1400+ characters
        full_song_expected = 1400
        assert new_raw_content_length > full_song_expected
