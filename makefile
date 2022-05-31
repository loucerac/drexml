.PHONY: install test format
all: install format test
install:
	poetry env use 3.8.13
	poetry install
format:
	poetry run autoflake -ri --remove-all-unused-imports dreml/ tests/
	poetry run isort dreml tests
	poetry run black dreml tests
	(cd docs && poetry run make html)
test:
	poetry run pytest