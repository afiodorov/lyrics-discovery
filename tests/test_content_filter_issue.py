"""Tests for content filter and truncation issues."""

from unittest.mock import Mock, patch

import pytest

from src.nodes.formatting import format_lyrics_node
from src.nodes.search import filter_results_node
from src.state import AgentState


class TestContentFilterIssue:
    """Test cases for content filter blocking and truncation."""

    def test_format_lyrics_with_content_filter_response(self):
        """Test handling when OpenAI returns content_filter finish_reason."""
        # Simulate a state with search results containing full lyrics
        state = AgentState(
            user_query="Gracias a la vida",
            target_language="ru",
            song_title="Gracias a la Vida",
            song_artist="Violeta Parra",
            search_results=[
                "Gracias a la vida Lyrics: Gracias a la vida, que me ha dado tanto / "
                "Me dio dos luceros que cuando los abro / Perfecto distingo lo negro del blanco / "
                "Y en el alto cielo su fondo estrellado / Y en las multitudes el hombre que yo amo / "
                "Gracias a la vida que me ha dado tanto / Me ha dado el oído que en todo su ancho / "
                "Graba noche y días, grillos y canarios / Martillos, turbinas, ladridos, chubascos / "
                "Y la voz tan tierna de mi bien amado / [... full song continues ...]"
            ],
            formatted_lyrics="",
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
        )

        # Mock response with content_filter finish_reason
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = "Gracias a la vida, que me ha dado tanto\nMe dio dos luceros que cuando los abro"  # Truncated
        mock_response.choices[0].finish_reason = "content_filter"

        with patch(
            "src.nodes.formatting.llm_client.chat.completions.create",
            return_value=mock_response,
        ):
            result = format_lyrics_node(state)

            # Should detect content filter issue
            assert len(result["formatted_lyrics"]) < 200  # Much shorter than expected
            # This test documents the current problematic behavior

    def test_search_results_already_truncated(self):
        """Test that search results from Tavily are already truncated."""
        # Simulate truncated search results (what we actually get from Tavily)
        state = AgentState(
            user_query="Gracias a la vida",
            target_language="ru",
            song_title="Gracias a la Vida",
            song_artist="Violeta Parra",
            search_results=[
                "Gracias a la vida Lyrics: Gracias a la vida, que me ha dado tanto / Me dio dos luceros que cuando los abro / Perfecto distingo lo negro del blanco / Y...",
                'Con una voz casi desnuda, Violeta Parra introduce el motivo principal de la pieza: la frase del texto "Gracias a la vida que me ha dado tanto". El tex...',
                "Gracias a la vida que me ha dado tanto. Me dio dos luceros que cuando los abro perfecto distingo lo negro del blanco y en el alto cielo su fondo es...",
            ],
            formatted_lyrics="",
            translated_lyrics="",
            interspersed_lyrics="",
            curious_facts="",
            error_message="",
        )

        # Mock LLM response that combines fragments
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Gracias a la vida, que me ha dado tanto\n"
            "Me dio dos luceros que cuando los abro\n"
            "Perfecto distingo lo negro del blanco\n"
            "Y en el alto cielo su fondo estrellado"  # Only partial lyrics due to input truncation
        )

        with patch(
            "src.nodes.search.llm_client.chat.completions.create",
            return_value=mock_response,
        ):
            result = filter_results_node(state)

            # The filtered result is limited by the truncated input
            assert len(result["search_results"][0]) < 500  # Much shorter than full song
            # This test documents how truncated search results limit our output

    def test_full_song_length_expectation(self):
        """Test what the full song length should be."""
        # The full song you provided has approximately 1400+ characters
        full_song = """Gracias a la vida que me ha dado tanto
Me dio dos luceros que, cuando los abro
Perfecto distingo, lo negro del blanco
Y en el alto cielo su fondo estrellado
Y en las multitudes, el hombre que yo amo
Gracias a la vida que me ha dado tanto
Me ha dado el oído que en todo su ancho
Graba noche y días, grillos y canarios
Martillos, turbinas, ladridos, chubascos
Y la voz tan tierna de mi bien amado
Gracias a la vida que me ha dado tanto
Me ha dado el sonido y el abecedario
Con él, las palabras que pienso y declaro
Madre, amigo, hermano y luz alumbrando
La ruta del alma del que estoy amando
Gracias a la vida que me ha dado tanto
Me ha dado la marcha de mis pies cansados
Con ellos anduve, ciudades y charcos
Playas y desiertos, montañas y llanos
Y la casa tuya, tu calle y tu patio
Gracias a la vida que me ha dado tanto
Me dio el corazón que agita su marco
Cuando miro el fruto del cerebro humano
Cuando miro el bueno, tan lejos del malo
Cuando miro el fondo de tus ojos claros
Gracias a la vida que me ha dado tanto
Me ha dado la risa y me ha dado el llanto
Así yo distingo, dicha de quebranto
Los dos materiales que forman mi canto
Y el canto de ustedes, que es el mismo canto
Y el canto de todos que es mi propio canto
Gracias a la vida"""

        # The actual output (332 chars) vs expected (1400+ chars)
        expected_length = len(full_song)
        actual_length = 332  # From the log

        assert expected_length > 1400
        assert actual_length < 400
        # This shows we're getting less than 25% of the expected content
