# Copyright (c) 2017 Ebrains project and The University of Manchester
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

###################################################
# Simulation parameters
###################################################
from inspect import getfullargspec

NEST_SIM = "NEST"
SPINNAKER_SIM = "SPINNAKER"


class SimParams(object):
    """
    Shared parameters for simulations.
    """

    __slots__ = [
        # pylint: disable=wrong-spelling-in-comment
        # sim time step
        'timestep',
        # sim duration
        'sim_duration',
        # min delay
        'min_delay',
        # max delay
        'max_delay',
        # file name for standard output
        'outfile',
        # file name for error output
        'errfile',
        # absolute path to which the output files should be written
        'output_path',
        # output format for spike data (h5 or dat)
        'output_format',
        # Directory for connectivity I/O
        'conn_dir',
        # setup params as dict
        'setup_params'
    ]

    def __init__(
            self, timestep, sim_duration, min_delay, max_delay, outfile,
            errfile, output_path, output_format, conn_dir):
        self.timestep = timestep
        self.sim_duration = sim_duration
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.outfile = outfile
        self.errfile = errfile
        self.output_path = output_path
        self.output_format = output_format
        self.conn_dir = conn_dir
        self.setup_params = {
            'timestep': self.timestep, 'min_delay': self.min_delay,
            'max_delay': self.max_delay
        }


class NestParams(SimParams):
    """
    Nest parameters for simulations.

    Not used by sPyNNaker
    """

    __slots__ = [
        # number of nodes
        'n_nodes',
        # number of MPI processes per node
        'n_procs_per_node',
        # wall time for simulation
        'wall_time',
        # total memory for simulation
        # For 12 or 24 MPI processes, 4gb is OK. For 48 MPI processes,
        # 8gb doesn't work, 24gb does.
        'memory',
        # path to the MPI shell script
        'mpi_path',
        # path to back-end (not needed for standard NEST versions,
        # which are loaded as modules)
        'backend_path',
        # path to PyNN installation
        'pynn_path']

    def __init__(
            self, timestep=0.1, sim_duration=10000.0, min_delay=0.1,
            max_delay=100.0, n_nodes=1, outfile='output.txt',
            errfile='errors.txt', output_path='results', output_format='pkl',
            conn_dir='connectivity', n_procs_per_node=24, wall_time='8:0:0',
            memory='4gb',
            mpi_path=(
                '/usr/local/mpi/openmpi/1.4.3/gcc64/bin/'
                'mpivars_openmpi-1.4.3_gcc64.sh'),
            backend_path='/path/to/backend', pynn_path='/path/to/pyNN'):
        super(NestParams, self).__init__(
            timestep, sim_duration, min_delay, max_delay, outfile,
            errfile, output_path, output_format, conn_dir)
        self.n_nodes = n_nodes
        self.n_procs_per_node = n_procs_per_node
        self.wall_time = wall_time
        self.memory = memory
        self.mpi_path = mpi_path
        self.backend_path = backend_path
        self.pynn_path = pynn_path


class SpinnakerParams(SimParams):
    """
    Spynnaker parameters
    """


def add_subparser(subparsers, command, method):
    # pylint:disable = wrong-spelling-in-docstring
    """
    Adds simulation subparsers.

    :param subparsers:
    :param command:
    :param method:
    :return:
    """
    argspec = getfullargspec(method)
    args_with_defaults = argspec.args
    args_without_defaults = []
    if argspec.defaults:
        args_with_defaults = argspec.args[-len(argspec.defaults):]
        args_without_defaults = argspec.args[:-len(argspec.defaults)]

    args = subparsers.add_parser(command)
    for arg in args_without_defaults:
        if arg != "self":
            args.add_argument(arg, action="store")
    if argspec.defaults:
        for arg, default in zip(args_with_defaults, argspec.defaults):
            args.add_argument("--" + arg, action="store", default=default)
    return args
