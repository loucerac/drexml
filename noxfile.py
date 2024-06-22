# -*- coding: utf-8 -*-
"""
Nox test suite.
"""

import nox


@nox.session(venv_backend="conda")
@nox.parametrize("python", ["3.10", "3.9", "3.8"])
def tests(session):
    """Test with conda."""
    pdm_env = {"PDM_IGNORE_SAVED_PYTHON": "1"}
    if session.posargs:
        if any(("gpu" in arg for arg in session.posargs)):
            session.conda_install(
                "cuda",
                "cuda-nvcc",
                "cuda-toolkit",
                "gxx=11.2",
                "-c",
                "nvidia/label/cuda-11.8.0",
                "-c",
                "conda-forge",
                "--override-channels",
            )

            pdm_env["PDM_NO_BINARY"] = "shap"

    #session.conda_install("pytest=7", "-c", "conda-forge")

    session.run("pdm", "install", "-vd", external=True, env=pdm_env)
    session.run("pdm", "run", "pytest", external=True, env=pdm_env)
