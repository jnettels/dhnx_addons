# MIT License

# Copyright (c) 2024 Joris Zimmermann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Collection of generalized functions for ALKIS and OpenStreetMap data.

DHNx Addons
=====

This package contains a collection of functions useful for workflows with
DHNx (https://github.com/oemof/DHNx), 
LPagg (https://github.com/jnettels/lpagg),
and GIS-data in general in the context of municipal heat planning.

Setup script
------------

Run one of the following commands to install into your Python environment:

.. code:: sh

    python setup.py install

    pip install -e <path to this folder>

"""
from setuptools import setup
from setuptools_scm import get_version


try:
    version = get_version(version_scheme='post-release')
except LookupError:
    version = '0.0.0'
    print('Warning: setuptools-scm requires an intact git repository to detect'
          ' the version number for this build.')

print('Building dhnx_addons with version tag: ' + version)

# The setup function
setup(
    name='dhnx_addons',
    version=version,
    description='Collection of GIS functions',
    long_description=open('README.md').read(),
    license='MIT',
    author='Joris Zimmermann',
    author_email='joris.zimmermann@siz-energieplus.de',
    url='https://github.com/jnettels/dhnx_addons',
    python_requires='>=3.10',
    install_requires=[
        'demandlib @ https://github.com/jnettels/demandlib/archive/features/add-vdi-from-lpagg.tar.gz', # demandlib (custom fork)
        'dhnx @ https://github.com/oemof/DHNx/archive/dev.tar.gz',  # dhnx (branch 'dev')
        'oemof.solph',
        # 'pandapipes @ https://github.com/e2nIEE/pandapipes/archive/v0.11.0.tar.gz',
        'pandapipes',
        'pandapower',
    ],
    extras_require={},
    packages=['dhnx_addons'],
    package_data={},
    entry_points={},
)
