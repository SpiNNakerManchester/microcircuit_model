###################################################
#  Main script
###################################################
import sys
import time
import plotting
from network import Network
from sim_params import NEST_SIM, SPINNAKER_SIM, SIMULATOR
from spinnaker_specific_info import SpinnakerSimulatorInfo
from nest_specific_info import NestSimulatorInfo
from common_params import CommonParams

# build sim params
if SIMULATOR == NEST_SIM:
    simulator_specific_stuff = NestSimulatorInfo()
else:
    simulator_specific_stuff = SpinnakerSimulatorInfo()

# build common params
common_params = CommonParams(simulator_specific_stuff)

# do nest'y things
if SIMULATOR == NEST_SIM:
    sys.path.append(simulator_specific_stuff.backend_path)
    sys.path.append(simulator_specific_stuff.pynn_path)

# prepare simulation
if SIMULATOR == SPINNAKER_SIM:
    import spynnaker8 as sim

sim.setup(**simulator_specific_stuff.setup_params)
simulator_specific_stuff.after_setup_stuff(sim)

# create network
start_netw = time.time()
network = Network()
network.setup(sim, simulator_specific_stuff, common_params)
end_netw = time.time()
if sim.rank() == 0:
    print( 'Creating the network took ', end_netw - start_netw, ' s')

if SIMULATOR == NEST_SIM:
    simulator_specific_stuff.memory_print(sim)

# simulate
if sim.rank() == 0:
    print("Simulating...")
start_sim = time.time()
t = sim.run(simulator_specific_stuff.sim_duration)
end_sim = time.time()
if sim.rank() == 0:
    print('Simulation took ', end_sim - start_sim, ' s')

if SIMULATOR == NEST_SIM:
    simulator_specific_stuff.memory_print(sim)

start_writing = time.time()
for layer in common_params.layers:
    for pop in common_params.pops:
        filename = (
            simulator_specific_stuff.output_path + '/spikes_' + layer + pop + '.' +
            simulator_specific_stuff.output_format)
        network.pops[layer][pop].write_data(io=filename, variables='spikes')

if simulator_specific_stuff.record_v:
    for layer in common_params.layers:
        for pop in common_params.pops:
            filename = (
                simulator_specific_stuff.output_path + '/voltages_' + layer +
                pop + '.dat')
            network.pops[layer][pop].print_v(filename, gather=True)

if SIMULATOR == NEST_SIM:
    simulator_specific_stuff.after_run_stuff(sim, common_params)


end_writing = time.time()
print("Writing data took ", end_writing - start_writing, " s")

if common_params.plot_spiking_activity and sim.rank() == 0:
    plotting.plot_raster_bars(
        common_params.raster_t_min, common_params.raster_t_max,
        common_params.n_rec, common_params.frac_to_plot,
        simulator_specific_stuff.output_path, common_params)

sim.end()
