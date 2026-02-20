.PHONY: install run test format lint

install:
	uv sync

run:
	uv run python src/{{ cookiecutter.package_name }}/infrastructure/api/main.py

test:
	uv run pytest tests/

format:
	uv run ruff format .

lint:
	uv run ruff check .