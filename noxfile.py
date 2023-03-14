from nox_poetry import session


@session(python=["3.10", "3.9", "3.8"], venv_backend="conda")
def tests(session):
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
