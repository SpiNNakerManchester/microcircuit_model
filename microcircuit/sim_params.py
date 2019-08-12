###################################################
###     	Simulation parameters		###
###################################################

import os

simulator_params = {
    'nest':
    {
      'timestep'        : 0.1,
      'threads'         : 1,
      'sim_duration'    : 10000.,
      'min_delay'       : 0.1,
      'max_delay'       : 100.
      # Set max_delay to avoid error due to requested delays being larger
      # than the default maximum delay of 10 ms.
      # Do not set to np.inf: the simulation then fails as buffers are
      # probably too large.
    },
    'spiNNaker':
    {
      'timestep'        : 0.1,
      'min_delay'       : 0.1,
      'max_delay'       : 14.4,
      'sim_duration'    : 10000.0, #43200000.0,
      'n_sub_runs'      : 1
    }	
}

system_params = {
    # number of nodes
    'n_nodes'           : 1,
    # number of MPI processes per node
    'n_procs_per_node'  : 24,
    # walltime for simulation
    'walltime'          : '8:0:0',
    # total memory for simulation
    'memory'            : '4gb', # For 12 or 24 MPI processes, 4gb is OK. For 48 MPI processes, 8gb doesn't work, 24gb does.
    # file name for standard output
    'outfile'           : 'output.txt',
    # file name for error output
    'errfile'           : 'errors.txt',
    # absolute path to which the output files should be written
    'output_path'       : 'results',
    # output format for spike data (h5 or dat)
    'output_format'     : 'pkl',
    # Directory for connectivity I/O
    'conn_dir'		: 'connectivity',
    # path to the MPI shell script
    'mpi_path'          : '/usr/local/mpi/openmpi/1.4.3/gcc64/bin/mpivars_openmpi-1.4.3_gcc64.sh',
    # path to back-end (not needed for standard NEST versions on Blaustein,
    # which are loaded as modules)
    'backend_path'      : '/path/to/backend',
    # path to pyNN installation
    'pyNN_path'         : '/path/to/pyNN'
}

# make any changes to the parameters
if 'custom_sim_params.py' in os.listdir('.'):
    execfile('custom_sim_params.py')
