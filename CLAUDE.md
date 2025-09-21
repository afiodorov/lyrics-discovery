# Lyrics Search Research - Developer Documentation

## Project Overview
A LangGraph-based agent system that searches for song lyrics, formats them, optionally translates them, and finds interesting facts about songs.

## Quick Start
```bash
# Install dependencies
uv sync

# Run the application
uv run python -m src.main "song name or description"

# With translation
uv run python -m src.main "song name" -t "target_language"

# With verbose logging (our app debug only)
uv run python -m src.main "song name" -v

# With very verbose logging (all libraries debug)
uv run python -m src.main "song name" -vvv
```

## Development Setup

### Code Quality Tools
The project uses `ruff` for code formatting and linting:

```bash
# Format code
uv run ruff format src/ tests/

# Fix import sorting
uv run ruff check --select I --fix src/ tests/

# Run linting
uv run ruff check src/ tests/
```

### Testing
```bash
# Run all tests
uv run python -m pytest tests/ -xvs

# Run specific test
uv run python -m pytest tests/test_filter_node.py::TestFilterResultsNode::test_filter_results_with_vysotsky_kupola_data -xvs
```

## Architecture

### Project Structure
```
src/
├── __init__.py              # Package initialization
├── config.py                # Configuration and API clients
├── state.py                 # AgentState definition and utilities
├── graph.py                 # Graph building and conditional logic
├── main.py                  # CLI entry point and result display
└── nodes/                   # Individual node implementations
    ├── __init__.py          # Node exports
    ├── analysis.py          # Query analysis
    ├── search.py            # Search and filtering
    ├── formatting.py        # Lyrics formatting
    ├── translation.py       # Translation
    └── facts.py             # Facts discovery
```

### LangGraph Agent Pipeline
The system uses a state graph with the following nodes:

1. **analyze_query_node**: Analyzes user input to extract song title and artist
2. **search_lyrics_node**: Uses Tavily API to search for lyrics online
3. **filter_results_node**: Filters and combines search results containing lyrics
4. **format_lyrics_node**: Uses LLM to extract and format clean lyrics
5. **translate_lyrics_node**: Translates lyrics to target language (optional)
6. **intersperse_lyrics_node**: Combines original and translated lyrics
7. **find_curious_facts_node**: Searches for interesting facts about the song

### Key Improvements Made

#### 1. Robust Lyrics Filtering (Bug Fix)
The `filter_results_node` was enhanced to handle fragmented search results:
- Added heuristic detection for lyrics keywords
- Automatically combines fragments containing lyrics
- Fallback mechanism if LLM fails to identify lyrics
- More lenient filtering to avoid false negatives

#### 2. Enhanced Formatting
The `format_lyrics_node` now:
- Reconstructs complete songs from fragments
- Handles partial lyrics gracefully
- Maintains proper verse/stanza structure

## Testing Strategy

### Unit Tests
Located in `tests/test_filter_node.py`:

1. **test_filter_results_with_vysotsky_kupola_data**: Tests handling of Russian lyrics fragments
2. **test_filter_results_with_no_suitable_source**: Tests fallback behavior with non-lyrics content
3. **test_filter_results_with_good_source**: Tests extraction of well-formatted lyrics

## Environment Variables
Required environment variables:
- `TAVILY_API_KEY`: API key for Tavily search
- `OPENAI_API_KEY`: API key for OpenAI LLM

## Common Issues and Solutions

### Issue: "Could not find a suitable source for lyrics"
**Solution**: The filter_results_node has been updated with:
- Heuristic detection for lyrics indicators
- Automatic combination of fragments
- Fallback mechanism to pass all results to formatting node

### Issue: Incomplete lyrics extraction
**Solution**: The system now:
- Combines multiple search results containing lyrics fragments
- Uses more lenient filtering criteria
- Reconstructs complete songs from partial results

## Code Style
The project follows Python best practices:
- Modular architecture with clear separation of concerns
- Import sorting with `ruff check --select I`
- Code formatting with `ruff format`
- Type hints where appropriate
- Comprehensive error handling

### Module Organization
- **config.py**: Centralized configuration and API client setup
- **state.py**: State management and debugging utilities
- **nodes/**: Each node in its own file for maintainability
- **graph.py**: Graph construction separate from business logic
- **main.py**: Clean CLI interface with result display

## Logging and Debugging

The application uses Python's logging module with Unix-style verbose levels:

### Normal Mode (INFO level)
```bash
uv run python -m src.main "song name"
```
Shows:
- Main processing steps with emojis
- Progress through the pipeline
- Final results
- WARNING+ logs from all libraries

### Verbose Mode (`-v` or `--verbose`)
```bash
uv run python -m src.main "song name" -v
```
Additionally shows:
- DEBUG logs from our application only
- Detailed state after each node execution
- Character counts and token usage
- LLM finish reasons (complete vs truncated)
- INFO+ logs from libraries (OpenAI, httpx, etc.)

### Very Verbose Mode (`-vvv` or `--very-verbose`)
```bash
uv run python -m src.main "song name" -vvv
```
Shows everything including:
- DEBUG logs from all libraries
- HTTP request/response details
- OpenAI API call traces
- Low-level networking information

### Legacy Debug Flag
```bash
uv run python -m src.main "song name" --debug
```
Equivalent to `-v` for backward compatibility.

### Logging Features
- **Unix-style verbosity**: Follows standard `-v`, `-vv`, `-vvv` conventions
- **Hierarchical filtering**: Libraries only show debug when explicitly requested
- **Structured logging**: Proper log levels (INFO, DEBUG, WARNING, ERROR)
- **Truncation detection**: Warnings when LLM responses are cut off
- **Character tracking**: Monitor content length through the pipeline
- **Module-specific loggers**: Each component has its own logger
- **Clean output**: User-facing results separate from debug info

## Future Improvements
- Add caching for repeated searches
- Support multiple language translations simultaneously
- Add more comprehensive test coverage
- Implement retry logic for API failures
- Add support for more search providers