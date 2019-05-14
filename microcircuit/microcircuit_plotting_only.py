###################################################
###     	Main script			###
###################################################

import neo
import sys
from sim_params import *
sys.path.append(system_params['backend_path'])
sys.path.append(system_params['pyNN_path'])
from network_params import *
import pyNN
import time
import plotting_to_display as plotting
import numpy as np

# system_params['output_path'] = '/Users/oliver/Documents/microcircuit_data/poisson_master'
# system_params['output_path'] = '/Users/oliver/Documents/microcircuit_data/DC_master'
# system_params['output_path'] = '/Users/oliver/Documents/microcircuit_data/subthreshold_init_poisson_master'
# system_params['output_path'] = '/Users/oliver/Documents/microcircuit_data/subthreshold_init_dc_master'
system_params['output_path'] = '/Users/oliver/Desktop/poisson_orig_init'
# system_params['output_path'] = '/Users/oliver/Desktop/dc_orig_init'

# system_params['output_path'] = '/Users/oliver/Desktop/poisson_julich_init'

# prepare simulation
#exec('import pyNN.%s as sim' %simulator)
import spynnaker8 as sim

sim.setup(**simulator_params[simulator])

if simulator == 'nest':
    n_vp = sim.nest.GetKernelStatus('total_num_virtual_procs')
    if sim.rank() == 0:
        print 'n_vp: ', n_vp
        print 'master_seed: ', master_seed
    sim.nest.SetKernelStatus({'print_time' : False,
                              'dict_miss_is_error': False,
                              'grng_seed': master_seed,
                              'rng_seeds': range(master_seed + 1, master_seed + n_vp + 1)})

if simulator == 'spiNNaker':
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 128)
    sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 128)

import network

# create network
start_netw = time.time()
n = network.Network(sim)
n.setup(sim)
end_netw = time.time()

if sim.rank() == 0:
    print 'Creating the network took ', end_netw - start_netw, ' s'

if simulator == 'nest':
    # determine memory consumption
    sim.nest.sli_run("memory_thisjob")
    print 'memory usage after network creation:', sim.nest.sli_pop(), 'kB'



if plot_spiking_activity and sim.rank()==0:
    plotting.plot_raster_bars( \
        raster_t_min, raster_t_max, n_rec, frac_to_plot, \
        system_params['output_path'])

sim.end()
print "job done"
