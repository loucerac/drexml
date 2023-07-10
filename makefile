.PHONY: install test format
.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate

all: install format test

install:
ifeq ($(use_gpu),1)
	conda create -y -p ./.venv --override-channels -c "nvidia/label/cuda-11.8.0" \
	-c conda-forge cuda cuda-nvcc cuda-toolkit gxx=11.2 python=3.10
else
	conda create -y -p ./.venv --override-channels -c conda-forge python=3.10
endif
	$(CONDA_ACTIVATE) ./.venv
	
	PDM_NO_BINARY=shap pdm install
	pdm run pytest
	python -c 'import shap; shap.utils.assert_import("cext_gpu")'
format:
	$(CONDA_ACTIVATE) ./.venv
	autoflake  --remove-all-unused-imports --ignore-init-module-imports \
	--remove-unused-variables -i drexml/*.py
	autoflake  --remove-all-unused-imports --ignore-init-module-imports \
	--remove-unused-variables -i tests/*.py
	pdm run isort drexml tests noxfile.py
	pdm run black drexml tests noxfile.py
	(cd docs && pdm run make html)
test:
ifeq ($(use_gpu),1)
	nox -- "gpu"
else
	nox
endif
cover:
	$(CONDA_ACTIVATE) ./.venv
	pdm run coverage run -m pytest tests/ -v && poetry run coverage report -m
build:
	rm -rf dist
	rm -rf ./.venv
ifeq ($(use_gpu),1)
	conda create -y -p ./.venv --override-channels -c "nvidia/label/cuda-11.8.0" \
	-c conda-forge cuda cuda-nvcc cuda-toolkit gxx=11.2 python=3.10
else
	conda create -y -p ./.venv --override-channels -c conda-forge python=3.10
endif
	$(CONDA_ACTIVATE) ./.venv

	PDM_NO_BINARY=shap pdm install	
	pdm publish --repository testpypi
	wait
	sleep 60
	pip install --no-cache-dir --no-binary=shap  -i https://test.pypi.org/simple/ \
	drexml==0.11.2 --extra-index-url=https://pypi.org/simple
ifeq ($(use_gpu),1)
	python -c 'import shap; shap.utils.assert_import("cext_gpu")'
endif
	python -c 'import drexml'
