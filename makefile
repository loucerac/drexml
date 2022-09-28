.PHONY: install test format
all: install format test
install:
	poetry env use 3.8.13
	poetry install
format:
	poetry run isort drexml tests
	poetry run black drexml tests
	(cd docs && poetry run make html)
test:
	poetry run tox
