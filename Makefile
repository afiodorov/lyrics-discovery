.PHONY: format lint clean

# Default target - run both formatting and linting
format:
	uv run ruff check --select I --fix .
	uv run ruff format .

# Just run import sorting
imports:
	uv run ruff check --select I --fix .

# Just run code formatting
fmt:
	uv run ruff format .

# Run linting (without fixes)
lint:
	uv run ruff check .

# Clean cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +