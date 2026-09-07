#!/bin/bash

# Copyright (c) 2026 The University of Manchester
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

# This bash assumes that other repositories are installed in parallel
# ruffs SpiNNUtils, spinn_machine and unittests

if [ "$#" -eq  "0" ]
  then
    echo "Using previous setup. Provide an argument to run setup"
    source ../SupportScripts/venv/ruff_runner/bin/activate
else
  python3 -m venv ../SupportScripts/venv/ruff_runner
  source ../SupportScripts/venv/ruff_runner/bin/activate
  python3 -m pip install --upgrade ruff flake8
fi

echo ruffusing ruff_ignore.toml
ruff check microcircuit unittests integration_tests  \
     --target-version py310 --config ../SupportScripts/actions/ruff/ruff_ignore.toml --fix
echo flake8
flake8 microcircuit unittests integration_tests