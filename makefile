.PHONY: install test format
all: install format test
install:
	poetry env use 3.10
	poetry install
	poetry run pytest
format:
	poetry run isort drexml tests
	poetry run black drexml tests
	(cd docs && poetry run make html)
test:
	tox
