# Copyright (C) 2020 Joris Zimmermann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

"""Collection of generalized functions for ALKIS and OpenStreetMap data.

DHNx Addons
=====

This package contains a collection of functions useful for workflows with
DHNx (https://github.com/oemof/DHNx),
LPagg (https://github.com/jnettels/lpagg),
and GIS-data in general in the context of municipal heat planning.


Module run_test
---------------
Tests to run during build process.
"""

import unittest
import dhnx_addons


def main_test():
    """Run test function."""
    dhnx_addons.setup()
    # dhnx_addons.workflow_example_openstreetmap(show_plot=False)

    return True


class TestMethods(unittest.TestCase):
    """Defines tests."""

    def test(self):
        """Test if the example workflow finishes."""
        self.assertTrue(main_test())


if __name__ == '__main__':
    unittest.main()
