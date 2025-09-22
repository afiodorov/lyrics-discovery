"""Tests for web UI streaming functionality with mocked API responses."""

from unittest.mock import Mock, patch

import pytest

from src.web_ui import search_lyrics_simple


class MockOpenAIResponse:
    """Mock OpenAI API response."""

    def __init__(self, content="Mock response", finish_reason="stop"):
        self.choices = [Mock()]
        self.choices[0].message = Mock()
        self.choices[0].message.content = content
        self.choices[0].finish_reason = finish_reason


class MockTavilyResponse:
    """Mock Tavily API response."""

    def __init__(self, results=None):
        if results is None:
            results = [{"content": "Mock content", "url": "http://example.com"}]
        self.data = {"results": results}

    def get(self, key, default=None):
        return self.data.get(key, default)


class MockWikipediaPage:
    """Mock Wikipedia page."""

    def __init__(self, content="Mock Wikipedia content"):
        self.content = content


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
        mock_client.search.return_value = MockTavilyResponse()
        yield mock_client


@pytest.fixture
def mock_wikipedia():
    """Mock Wikipedia."""
    with patch("wikipedia.page") as mock_page:
        mock_page.return_value = MockWikipediaPage()
        yield mock_page


@pytest.fixture
def mock_all_apis(mock_openai, mock_tavily, mock_wikipedia):
    """Fixture that mocks all external APIs."""
    return {"openai": mock_openai, "tavily": mock_tavily, "wikipedia": mock_wikipedia}


def test_streaming_basic_flow(mock_all_apis):
    """Test that streaming yields progress updates at each step."""
    query = "Bella Ciao"
    translate_to = "en"

    # Configure mock responses for different nodes
    mock_all_apis["openai"].chat.completions.create.side_effect = [
        # analyze_query_node response
        MockOpenAIResponse(
            '{"song_title": "Bella Ciao", "song_artist": "Italian Folk"}'
        ),
        # format_lyrics_node response
        MockOpenAIResponse(
            "Oh bella ciao, bella ciao, bella ciao ciao ciao\nUna mattina mi son svegliato"
        ),
        # translate_lyrics_node response
        MockOpenAIResponse(
            "Oh beautiful goodbye, beautiful goodbye\nOne morning I woke up"
        ),
        # intersperse_lyrics_node response
        MockOpenAIResponse(
            "Oh bella ciao, bella ciao, bella ciao ciao ciao\nOh beautiful goodbye, beautiful goodbye"
        ),
        # find_curious_facts_node response
        MockOpenAIResponse(
            "• This is an Italian partisan song from WWII\n• It became a symbol of resistance"
        ),
        # Language detection response
        MockOpenAIResponse("Italian"),
    ]

    # Collect all streaming outputs
    outputs = list(search_lyrics_simple(query, translate_to))

    # Should have multiple streaming updates
    assert len(outputs) > 5, f"Expected multiple streaming updates, got {len(outputs)}"

    # Check that progress messages appear in order
    progress_messages = [output[0] for output in outputs]
    final_progress = progress_messages[-1]

    assert "🤔 Analyzing your request..." in final_progress
    assert "🧠 Identified:" in final_progress
    assert "🔎 Found" in final_progress
    assert "✅ Complete!" in final_progress


def test_streaming_with_empty_query(mock_all_apis):
    """Test streaming with empty query."""
    outputs = list(search_lyrics_simple("", ""))

    assert len(outputs) == 1
    assert "Please enter a song name" in outputs[0][0]
    assert outputs[0][1] == ""  # No lyrics
    assert outputs[0][2] == ""  # No facts


def test_streaming_with_tavily_none_results(mock_all_apis):
    """Test streaming when Tavily returns None results."""
    # Mock Tavily to return None results
    mock_all_apis["tavily"].search.return_value = MockTavilyResponse([])

    # Mock other API calls
    mock_all_apis["openai"].chat.completions.create.side_effect = [
        MockOpenAIResponse('{"song_title": "Unknown Song", "song_artist": "Unknown"}'),
        MockOpenAIResponse("No lyrics found"),
        MockOpenAIResponse("English"),
    ]

    outputs = list(search_lyrics_simple("unknown song", "en"))

    # Should complete without crashing
    assert len(outputs) > 0
    final_output = outputs[-1]

    # Should show progress but may have empty results
    assert "🤔 Analyzing your request..." in final_output[0]


def test_streaming_with_wikipedia_exception(mock_all_apis):
    """Test streaming when Wikipedia throws exception."""
    # Mock Wikipedia to throw exception
    mock_all_apis["wikipedia"].side_effect = Exception("Wikipedia error")

    # Mock other responses
    mock_all_apis["openai"].chat.completions.create.side_effect = [
        MockOpenAIResponse('{"song_title": "Test Song", "song_artist": "Test Artist"}'),
        MockOpenAIResponse("Test lyrics content"),
        MockOpenAIResponse("English"),
        MockOpenAIResponse("• Test fact about the song"),
    ]

    outputs = list(search_lyrics_simple("test song", ""))

    # Should complete without crashing despite Wikipedia error
    assert len(outputs) > 0
    final_output = outputs[-1]
    assert "✅ Complete!" in final_output[0] or "❌ Error:" in final_output[0]


def test_streaming_with_openai_none_response(mock_all_apis):
    """Test streaming when OpenAI returns None content."""
    # Mock OpenAI to return None content
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = None
    mock_response.choices[0].finish_reason = "stop"

    mock_all_apis["openai"].chat.completions.create.return_value = mock_response

    outputs = list(search_lyrics_simple("test song", "en"))

    # Should handle None content gracefully
    assert len(outputs) > 0
    # May result in error but shouldn't crash with 'NoneType' object is not iterable


def test_facts_node_with_empty_tavily_results(mock_all_apis):
    """Test facts processing with empty Tavily results."""
    # Mock empty Tavily response
    mock_all_apis["tavily"].search.return_value = {"results": []}

    # Mock Wikipedia to fail so it falls back to Tavily
    mock_all_apis["wikipedia"].side_effect = Exception("No Wikipedia page")

    # Mock basic OpenAI responses
    mock_all_apis["openai"].chat.completions.create.side_effect = [
        MockOpenAIResponse('{"song_title": "Test Song", "song_artist": ""}'),
        MockOpenAIResponse("Test lyrics"),
        MockOpenAIResponse("English"),
    ]

    outputs = list(search_lyrics_simple("test song", ""))

    # Should complete without NoneType iteration error
    assert len(outputs) > 0
    final_output = outputs[-1]

    # Facts might be empty but shouldn't crash
    assert final_output[2] == "" or "No specific facts found" in final_output[2]


def test_facts_node_with_none_tavily_response(mock_all_apis):
    """Test facts processing when Tavily returns None."""
    # Mock Tavily to return None response
    mock_all_apis["tavily"].search.return_value = None

    # Mock Wikipedia to fail
    mock_all_apis["wikipedia"].side_effect = Exception("Wikipedia failed")

    # Mock OpenAI responses
    mock_all_apis["openai"].chat.completions.create.side_effect = [
        MockOpenAIResponse('{"song_title": "Test Song", "song_artist": ""}'),
        MockOpenAIResponse("Test lyrics"),
        MockOpenAIResponse("English"),
    ]

    outputs = list(search_lyrics_simple("test song", ""))

    # Should handle None Tavily response without crashing
    assert len(outputs) > 0


def test_language_detection_with_none_lyrics(mock_all_apis):
    """Test language detection when lyrics are None."""
    # Mock to return None/empty lyrics at various stages
    mock_all_apis["openai"].chat.completions.create.side_effect = [
        MockOpenAIResponse('{"song_title": "Test Song", "song_artist": ""}'),
        MockOpenAIResponse(""),  # Empty formatted lyrics
        MockOpenAIResponse("English"),  # Language detection
    ]

    outputs = list(search_lyrics_simple("test song", "en"))

    # Should handle empty lyrics gracefully
    assert len(outputs) > 0


def test_cache_functionality():
    """Test that caching works properly."""
    from src.web_ui import get_cache_key, load_from_cache, save_to_cache

    query = "test song"
    language = "en"
    test_results = {
        "progress": "Test progress",
        "lyrics": "Test lyrics",
        "facts": "Test facts",
    }

    # Test cache key generation
    cache_key = get_cache_key(query, language)
    assert isinstance(cache_key, str)
    assert len(cache_key) == 32  # MD5 hash length

    # Test saving and loading
    save_to_cache(query, language, test_results)
    loaded = load_from_cache(query, language)

    assert loaded is not None
    assert loaded["progress"] == test_results["progress"]
    assert loaded["lyrics"] == test_results["lyrics"]
    assert loaded["facts"] == test_results["facts"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
