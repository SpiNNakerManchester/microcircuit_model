

class SimParams(object):

    __slots__ = [
        #
        "_timestep",
        #
        "_threads",
        #
        "_sim_duration",
        #
        "_min_delay",
        #
        "_max_delay",
        # number of nodes
        '_n_nodes',
        # number of MPI processes per node
        '_n_procs_per_code',
        # walltime for simulation
        "_walltime",
        # total memory for simulation
        "_memory",
        # file name for standard output
        "_outfile",
        # file name for error output
        "_errfile",
        # absolute path to which the output files should be written
        "_output_path",
        # output format for spike data (h5 or dat)
        "_output_format",
        # Directory for connectivity I/O
        "_conn_dir",
        # path to the MPI shell script
        "_mpi_path",
        # path to back-end (not needed for standard NEST versions on Blaustein,
        # which are loaded as modules)
        "_backend_path",
        # path to pyNN installation
        "_pyNN_path"
    ]

    def __init__(self, sim_name):
        if sim_name == 'nest':
            self._timestep = 0.1
            self._threads = 1
            self._sim_duration = 10000.0
            self._min_delay = 0.1
            self._max_delay = 100.0
        else:
            self._timestep = 0.1
            self._sim_duration = 1000.0
            self._min_delay = 0.1
            self._max_delay = 14.4

        self._n_nodes = 1
        self._n_procs_per_code = 24
        self._walltime = '8:0:0'
        self._memory = '4gb'
        self._outfile = 'output.txt'
        self._errfile = 'errors.txt'
        self._output_path = 'results'
        self._output_format = 'pkl'
        self._conn_dir = 'connectivity'
        self._mpi_path = (
            '/usr/local/mpi/openmpi/1.4.3/gcc64/bin/' 
            'mpivars_openmpi-1.4.3_gcc64.sh')
        self._backend_path = '/path/to/backend'
        self._pyNN_path = '/path/to/pyNN'

    @property
    def timestep(self):
        return self._timestep

    @property
    def threads(self):
        return self._threads

    @property
    def sim_duration(self):
        return self._sim_duration

    @property
    def min_delay(self):
        return self._min_delay

    @property
    def max_delay(self):
        return self._max_delay

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def n_procs_per_code(self):
        return self._n_procs_per_code

    @property
    def walltime(self):
        return self._walltime

    @property
    def memory(self):
        return self._memory

    @property
    def outfile(self):
        return self._outfile

    @property
    def errfile(self):
        return self._errfile

    @property
    def output_path(self):
        return self._output_path

    @property
    def output_format(self):
        return self._output_format

    @property
    def conn_dir(self):
        return self._conn_dir

    @property
    def mpi_path(self):
        return self._mpi_path

    @property
    def backend_path(self):
        return self._backend_path

    @property
    def pyNN_path(self):
        return self._pyNN_path
