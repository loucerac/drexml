.PHONY: install test format
.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate

all: install format test

install:
ifeq ($(use_gpu),1)
	conda create -y -p ./.venv --override-channels -c "nvidia/label/cuda-11.8.0" -c conda-forge cuda cuda-nvcc cuda-toolkit gxx=11.2 python=3.10
else
	conda create -y -p ./.venv --override-channels -c conda-forge python=3.10
endif
	$(CONDA_ACTIVATE) ./.venv
	poetry install
	poetry run pytest
format:
	$(CONDA_ACTIVATE) ./.venv
	poetry run isort drexml tests
	poetry run black drexml tests
	(cd docs && poetry run make html)
test:
ifeq ($(use_gpu),1)
	nox -- "gpu"
else
	nox
endif
