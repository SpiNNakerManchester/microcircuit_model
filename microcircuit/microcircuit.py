###################################################
#  Main script
###################################################
import sys
import time
from past.builtins import xrange
import plotting
import numpy as np
from network import Network
from sim_params import NEST_SIM, SPINNAKER_SIM, SIMULATOR
from network_params import NestNetworkParams, SpinnakerNetworkParams
from common_params import CommonParams

# build sim params
if SIMULATOR == NEST_SIM:
    simulator_params = NestNetworkParams()
else:
    simulator_params = SpinnakerNetworkParams()

# build common params
common_params = CommonParams(simulator_params)

# do nest'y things
if SIMULATOR == NEST_SIM:
    sys.path.append(simulator_params.backend_path)
    sys.path.append(simulator_params.pynn_path)

# prepare simulation
if SIMULATOR == SPINNAKER_SIM:
    import spynnaker8 as sim
else:
    #?????????
    pass

sim.setup(**simulator_params.setup_params)

if SIMULATOR == NEST_SIM:
    n_vp = sim.nest.GetKernelStatus('total_num_virtual_procs')
    if sim.rank() == 0:
        print('n_vp: ', n_vp)
        print('master_seed: ', simulator_params.master_seed)
    sim.nest.SetKernelStatus(
        {'print_time': False, 'dict_miss_is_error': False,
         'grng_seed': simulator_params.master_seed,
         'rng_seeds': range(
             simulator_params.master_seed + 1,
             simulator_params.master_seed + n_vp + 1)})

if SIMULATOR == SPINNAKER_SIM:
    neurons_per_core = 255
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, neurons_per_core)
    sim.set_number_of_neurons_per_core(
        sim.SpikeSourcePoisson, neurons_per_core)

# create network
start_netw = time.time()
network = Network()
network.setup(sim, simulator_params, common_params)
end_netw = time.time()
if sim.rank() == 0:
    print( 'Creating the network took ', end_netw - start_netw, ' s')

if SIMULATOR == NEST_SIM:
    # determine memory consumption
    sim.nest.sli_run("memory_thisjob")
    print('memory usage after network creation:', sim.nest.sli_pop(), 'kB')

# simulate
if sim.rank() == 0:
    print("Simulating...")
start_sim = time.time()
t = sim.run(simulator_params.sim_duration)
end_sim = time.time()
if sim.rank() == 0:
    print('Simulation took ', end_sim - start_sim, ' s')

if SIMULATOR == NEST_SIM:
    # determine memory consumption
    sim.nest.sli_run("memory_thisjob")
    print('memory usage after simulation:', sim.nest.sli_pop(), 'kB')

start_writing = time.time()
for layer in common_params.layers:
    for pop in common_params.pops:
        filename = (
            simulator_params.output_path + '/spikes_' + layer + pop + '.' +
            simulator_params.output_format)
        network.pops[layer][pop].write_data(io=filename, variables='spikes')

if simulator_params.record_v:
    for layer in common_params.layers:
        for pop in common_params.pops:
            filename = (
                simulator_params.output_path + '/voltages_' + layer +
                pop + '.dat')
            network.pops[layer][pop].print_v(filename, gather=True)

if SIMULATOR == NEST_SIM:
    if simulator_params.record_corr:
        if sim.nest.GetStatus(network.corr_detector, 'local')[0]:
            print('getting count_covariance on rank ', sim.rank())
            cov_all = (sim.nest.GetStatus(
                network.corr_detector, 'count_covariance')[0])
            delta_tau = sim.nest.GetStatus(
                network.corr_detector, 'delta_tau')[0]

            cov = {}
            for target_layer in np.sort(common_params.layers.keys()):
                for target_pop in common_params.pops:
                    target_index = (
                        common_params.structure[target_layer][target_pop])
                    cov[target_index] = {}
                    for source_layer in np.sort(common_params.layers.keys()):
                        for source_pop in common_params.pops:
                            source_index = (
                                common_params.structure[
                                    source_layer][source_pop])
                            cov[target_index][source_index] = (
                                np.array(list(
                                    cov_all[target_index][source_index][::-1])
                                         + list(
                                    cov_all[source_index][target_index][1:])))

            f = open(simulator_params.output_path + '/covariances.dat', 'w')
            f.write('tau_max: {}'.format(common_params.tau_max))
            f.write('delta_tau: {}'.format(delta_tau))
            f.write('simtime: {}\n'.format(simulator_params.sim_duration))

            for target_layer in np.sort(common_params.layers.keys()):
                for target_pop in common_params.pops:
                    target_index = (
                        common_params.structure[target_layer][target_pop])
                    for source_layer in np.sort(common_params.layers.keys()):
                        for source_pop in common_params.pops:
                            source_index = (
                                common_params.structure[source_layer][
                                    source_pop])
                            f.write("{}{} - {}{}".format(
                                target_layer, target_pop, source_layer,
                                source_pop))
                            f.write('n_events_target: {}'.format(
                                sim.nest.GetStatus(
                                    network.corr_detector, 'n_events')[0][
                                    target_index]))
                            f.write('n_events_source: {}'.format(
                                sim.nest.GetStatus(
                                    network.corr_detector, 'n_events')[0][
                                    source_index]))
                            for i in xrange(
                                    len(cov[target_index][source_index])):
                                f.write(cov[target_index][source_index][i])
                            f.write('')
            f.close()


end_writing = time.time()
print("Writing data took ", end_writing - start_writing, " s")

if common_params.plot_spiking_activity and sim.rank() == 0:
    plotting.plot_raster_bars(
        common_params.raster_t_min, common_params.raster_t_max,
        common_params.n_rec, common_params.frac_to_plot,
        simulator_params.output_path, common_params)

sim.end()
