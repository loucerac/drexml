import nox
from nox_poetry import session


@session(venv_backend="conda")
@nox.parametrize("python", ["3.10", "3.9", "3.8"])
def tests(session):
    if session.posargs:
        if any(["gpu" in arg for arg in session.posargs]):
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
 
    session.install("pytest", ".")
    session.run("pytest")
