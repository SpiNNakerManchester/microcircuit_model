# Copyright (c) 2017 The University of Manchester
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

import os
import stat
from spinnaker_testbase import ScriptChecker


class TestMicrocircuit(ScriptChecker):
    """
    Runs the Microcircuit script and check for the existence of result files
    """

    def test_microcircuit(self):
        """
        Runs the Microcircuit with safety code

        :return:
        """
        self.runsafe(self.microcircuit)

    def microcircuit(self):
        """
        Runs the Microcircuit script and check for the existence of results
        """
        self.check_script("run_microcircuit.py")
        for result_file in [
                "spikes_L23E.pkl", "spikes_L23I.pkl",
                "spikes_L4E.pkl", "spikes_L4I.pkl",
                "spikes_L5E.pkl", "spikes_L5I.pkl",
                "spikes_L6E.pkl", "spikes_L6I.pkl",
                "spiking_activity.png"]:
            result_path = os.path.join("results", result_file)
            assert os.path.exists(result_path)
            assert os.stat(result_path)[stat.ST_SIZE]
