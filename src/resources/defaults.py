"""Load default values from resource files."""

from importlib import resources


def get_default_query() -> str:
    """Get the default search query from resource file."""
    with resources.open_text("src.resources", "init_query.txt") as f:
        return f.read().strip()


def get_default_language() -> str:
    """Get the default translation language from resource file."""
    with resources.open_text("src.resources", "init_language.txt") as f:
        return f.read().strip()


def get_default_progress() -> str:
    """Get the pre-computed progress from resource file."""
    with resources.open_text("src.resources", "init_progress.txt") as f:
        return f.read().strip()


def get_default_lyrics() -> str:
    """Get the pre-computed lyrics from resource file."""
    with resources.open_text("src.resources", "init_lyrics.txt") as f:
        return f.read()


def get_default_facts() -> str:
    """Get the pre-computed facts from resource file."""
    with resources.open_text("src.resources", "init_curious.txt") as f:
        return f.read()
