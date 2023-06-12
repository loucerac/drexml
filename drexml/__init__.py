from importlib import metadata

def get_package_version(package_name):
    """
    Get the version number of a package.

    Parameters
    ----------
    package_name : str
        The name of the package.

    Returns
    -------
    version : str
        The version number of the package.

    Examples
    --------
    >>> get_package_version('numpy')
    '1.20.3'
    >>> get_package_version('requests')
    '2.26.0'

    Notes
    -----
    This function uses the `importlib.metadata` module to retrieve the version
    number of the package. The `metadata` module is available in Python 3.8 and
    above.

    The `metadata` module provides access to the distribution metadata for a
    package, including its version number. It is a recommended way to retrieve
    package metadata in Python.

    The `metadata.version` function is used to retrieve the version number of
    the package specified by `package_name`. If the package is not installed or
    if its version number cannot be determined, this function will raise an
    exception.

    After retrieving the version number, the `metadata` module is deleted from
    the namespace to avoid polluting the results of `dir(package_name)`.

    See Also
    --------
    metadata.version : Get the version number of a package using the
                       `importlib.metadata` module.
    """
    version = metadata.version(package_name)
    del metadata
    return version
