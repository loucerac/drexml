.PHONY: install test format
all: format install test
install:
	poetry install
format:
	poetry run isort dreml tests
	poetry run black dreml tests
	poetry run autoflake -ri --remove-all-unused-imports dreml/
test:
	poetry run pytest