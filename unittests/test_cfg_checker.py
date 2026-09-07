# Copyright (c) 2025 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# This workflow will install Python dependencies, run lint and rat with a variety of Python versions
# For more information see: https://help.github.com/actions/language-and-framework-guides/using-python-with-github-actions

import os
import unittest

from spinn_utilities.configs.config_checker import ConfigChecker

from spynnaker.pyNN.config_setup import unittest_setup


class TestCfgChecker(unittest.TestCase):

    def setUp(self):
        unittest_setup()

    def test_config_checks(self):
        unittests = os.path.dirname(__file__)
        parent = os.path.dirname(unittests)
        cfg = os.path.join(parent, "spynnaker.cfg")
        integration_tests = os.path.join(parent, "integration_tests")
        microcircuit = os.path.join(parent, "microcircuit")

        cc = ConfigChecker([cfg, integration_tests, microcircuit, unittests])
        cc.check(local_defaults=False)
