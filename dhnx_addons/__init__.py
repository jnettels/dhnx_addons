"""Collection of generalized functions for ALKIS and OpenStreetMap data."""
from pkg_resources import get_distribution, DistributionNotFound

try:
    dist_name = 'dhnx_addons'
    # Try to get the version name from the installed package
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    try:
        # If package is not installed, try to get version from git
        from setuptools_scm import get_version
        __version__ = get_version(version_scheme='post-release',
                                  root='..', relative_to=__file__)
    except (LookupError, Exception) as e:
        print(e)
        __version__ = '0.0.0'

from .dhnx_addons import *
from .cbc_installer import *
