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

system_params['output_path'] = '/Users/oliver/Documents/microcircuit_data/poisson_master'
# system_params['output_path'] = '/Users/oliver/Documents/microcircuit_data/DC_master'
# system_params['output_path'] = '/Users/oliver/Documents/microcircuit_data/subthreshold_init_poisson_master'
# system_params['output_path'] = '/Users/oliver/Documents/microcircuit_data/subthreshold_init_dc_master'

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
#
# # simulate
# if sim.rank() == 0:
#     print "Simulating..."
# start_sim = time.time()
# t = sim.run(simulator_params[simulator]['sim_duration'])
# end_sim = time.time()
# if sim.rank() == 0:
#     print 'Simulation took ', end_sim - start_sim, ' s'
#
# if simulator == 'nest':
#     # determine memory consumption
#     sim.nest.sli_run("memory_thisjob")
#     print 'memory usage after simulation:', sim.nest.sli_pop(), 'kB'
#
# start_writing = time.time()
# for layer in layers:
#     for pop in pops:
#         filename = system_params['output_path'] + '/spikes_' + layer + pop + '.' + system_params['output_format']
#         # n.pops[layer][pop].printSpikes(filename, gather=True)
#         n.pops[layer][pop].write_data(io=filename, variables='spikes')
#
# if record_v:
#     for layer in layers:
#         for pop in pops:
#             filename = system_params['output_path'] + '/voltages_' + layer + pop + '.dat'
#             n.pops[layer][pop].print_v(filename, gather=True)
#
# if simulator == 'nest':
#     if record_corr:
#         if sim.nest.GetStatus(n.corr_detector, 'local')[0]:
#             print 'getting count_covariance on rank ', sim.rank()
#             cov_all = sim.nest.GetStatus(n.corr_detector, 'count_covariance')[0]
#             delta_tau = sim.nest.GetStatus(n.corr_detector, 'delta_tau')[0]
#
#             cov = {}
#             for target_layer in np.sort(layers.keys()):
#                 for target_pop in pops:
#                     target_index = structure[target_layer][target_pop]
#                     cov[target_index] = {}
#                     for source_layer in np.sort(layers.keys()):
#                         for source_pop in pops:
#                             source_index = structure[source_layer][source_pop]
#                             cov[target_index][source_index] = np.array(list(cov_all[target_index][source_index][::-1]) \
#                             + list(cov_all[source_index][target_index][1:]))
#
#             f = open(system_params['output_path'] + '/covariances.dat', 'w')
#             print >>f, 'tau_max: ', tau_max
#             print >>f, 'delta_tau: ', delta_tau
#             print >>f, 'simtime: ', simulator_params[simulator]['sim_duration'], '\n'
#
#             for target_layer in np.sort(layers.keys()):
#                 for target_pop in pops:
#                     target_index = structure[target_layer][target_pop]
#                     for source_layer in np.sort(layers.keys()):
#                         for source_pop in pops:
#                             source_index = structure[source_layer][source_pop]
#                             print >>f, target_layer, target_pop, '-', source_layer, source_pop
#                             print >>f, 'n_events_target: ', sim.nest.GetStatus(n.corr_detector, 'n_events')[0][target_index]
#                             print >>f, 'n_events_source: ', sim.nest.GetStatus(n.corr_detector, 'n_events')[0][source_index]
#                             for i in xrange(len(cov[target_index][source_index])):
#                                 print >>f, cov[target_index][source_index][i]
#                             print >>f, ''
#             f.close()
#
#
# end_writing = time.time()
# print "Writing data took ", end_writing - start_writing, " s"

if plot_spiking_activity and sim.rank()==0:
    plotting.plot_raster_bars( \
        raster_t_min, raster_t_max, n_rec, frac_to_plot, \
        system_params['output_path'])

sim.end()
print "job done"