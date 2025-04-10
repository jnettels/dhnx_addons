"""Collection of generalized functions for ALKIS and OpenStreetMap data."""
from importlib.metadata import version, PackageNotFoundError

try:
    dist_name = 'dhnx_addons'
    # Try to get the version name from the installed package
    __version__ = version(dist_name)
except PackageNotFoundError:
    try:
        # If package is not installed, try to get version from git
        from setuptools_scm import get_version
        __version__ = get_version(version_scheme='post-release',
                                  root='..', relative_to=__file__)
    except (LookupError, Exception) as e:
        print(e)
        __version__ = '0.0.0'
try:
    import dhnx_addons.fix_gdal_import  # needs further testing
except Exception as e:
    print(e)
    pass

from .dhnx_addons import *
from .cbc_installer import *
