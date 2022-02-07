.PHONY: install test format
install:
	poetry install
format:
	poetry run isort dreml tests
	poetry run black dreml tests
	poetry run autoflake -ri --remove-all-unused-imports dreml/
test:
	poetry run pytest
all: format install test