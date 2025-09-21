from unittest.mock import Mock, patch

import pytest

from src.nodes.search import filter_results_node
from src.state import AgentState


class TestFilterResultsNode:
    """Test cases for the filter_results_node function."""

    def test_filter_results_with_vysotsky_kupola_data(self):
        """Test the actual failure case with Высоцкий's Купола search results."""
        # These are the actual search results from the debug output
        state = AgentState(
            user_query="Купола высоцкого",
            target_language=None,
            song_title="Купола",
            song_artist="Владимир Высоцкий",
            search_results=[
                "Чтобы чаще Господь замечал.  Я стою, как перед вечною загадкою,  Пред великою да сказочной страною —  Перед солоно- да горько-кисло-сладкою,  Голубою, родниковою, ржаною.  Грязью чавкая жирной да ржавою,  Вязнут лошади по стремена,  Но влекут меня сонной державою,  Что раскисла, опухла от сна.  Словно семь заветных струн  Зазвенели в свой черед —  Это птица Гамаюн  Надежду подает!  Душу, сбитую утратами да тратами,  Душу, стертую перекатами, —  Если до крови лоскут истончал, —  Залатаю золотыми я заплатами",
                "что услышится! Птицы вещие поют, и все из сказок. Птица сирин мне радостно скалится, Веселит, зазывает из гнезд, А напротив тоскует, печалится, Травит душу чудной алконост. Словно семь богатых лун На пути моем встает, Это птица гамаюн Надежду подает! Птица сирин мне радостно скалится, Веселит, зазывает из гнезд, А напротив тоскует, печалится, Травит душу чудной алконост. Словно семь богатых лун На пути моем встает, Это птица гамаюн Надежду подает!",
                "Lyrics for Купола российские by Vladimir Vysotsky ... Надежду подает! В синем небе, колокольнями проколотом. Медный колокол, медный колокол. То ль возвестит о моей доле, как о заколотой, То ли мне в этом звоне",
                "Купола в России кроют чистым золотом, Чтобы чаще Господь замечал. Я стою, как перед вечною загадкою, Пред великою да сказочной страною, Перед солоно да горько-кисло-сладкою, Голубую, родниковую, ржаною. Грязью, чавкая",
                "Купола song by Vladimir Vysotsky  # Купола  Песни для театра и кино  Vladimir Vysotsky January 1, 2002  ## More By Vladimir Vysotsky  Владимир Высоцкий. Гамлет  Vladimir Vysotsky June 16, 2023  Архивные записи - 20. 1976  Vladimir Vysotsky 1993",
            ],
            formatted_lyrics="",
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
            debug_mode=True,
        )

        # The function should now use heuristics to combine lyrics fragments
        result = filter_results_node(state)

        # The function should return combined results, not an error
        assert "error_message" not in result or result["error_message"] == ""
        assert "search_results" in result
        assert len(result["search_results"]) == 1
        # Check that it combined multiple fragments that contain lyrics keywords
        assert "стою" in result["search_results"][0]
        assert "птица" in result["search_results"][0].lower()

    def test_filter_results_with_no_suitable_source(self):
        """Test when LLM genuinely finds no suitable source - fallback behavior."""
        state = AgentState(
            user_query="test query",
            target_language=None,
            song_title="Test Song",
            song_artist="Test Artist",
            search_results=[
                "Random text about something else",
                "Another unrelated snippet",
                "Nothing here related to music",
            ],
            formatted_lyrics="",
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
            debug_mode=True,
        )

        # This test should trigger the heuristic path since we have newline in lyrics_indicators
        result = filter_results_node(state)

        # Since none of the specific lyrics keywords match, it should return the last result
        assert "search_results" in result
        assert len(result["search_results"]) == 1

    def test_filter_results_with_good_source(self):
        """Test when a good lyrics source is found."""
        full_lyrics = """
        Verse 1:
        This is the first line
        This is the second line

        Chorus:
        This is the chorus
        Repeating the chorus

        Verse 2:
        Another verse here
        With more lyrics
        """

        state = AgentState(
            user_query="test query",
            target_language=None,
            song_title="Test Song",
            song_artist="Test Artist",
            search_results=[
                "Some metadata about the song",
                full_lyrics,
                "User comments and ratings",
            ],
            formatted_lyrics="",
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
            debug_mode=True,
        )

        with patch("src.nodes.search.llm_client") as mock_llm:
            # Mock the OpenAI response selecting the full lyrics
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = full_lyrics
            mock_llm.chat.completions.create.return_value = mock_response

            result = filter_results_node(state)

            # Should return the selected lyrics
            assert "search_results" in result
            assert len(result["search_results"]) == 1
            assert result["search_results"][0] == full_lyrics
            assert "error_message" not in result or result["error_message"] == ""
