.PHONY: install test format
all: install format test
install:
	poetry install
format:
	poetry run autoflake -ri --remove-all-unused-imports dreml/ tests/
	poetry run isort dreml tests
	poetry run black dreml tests
test:
	poetry run pytest