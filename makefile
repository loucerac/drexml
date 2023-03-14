.PHONY: install test format
all: install format test
install:
	conda create -p ./.venv --override-channels -c "nvidia/label/cuda-11.8.0" -c conda-forge cuda cuda-nvcc cuda-toolkit gxx=11.2 python=3.10
	poetry install
	poetry run pytest
format:
	poetry run isort drexml tests
	poetry run black drexml tests
	(cd docs && poetry run make html)
test:
	tox
