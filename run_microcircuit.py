###################################################
#  Main script
###################################################
import sys
import time
from argparse import ArgumentParser

from microcircuit import plotting
from microcircuit.network import Network
from microcircuit.sim_params import NEST_SIM, SPINNAKER_SIM, add_subparser
from microcircuit.spinnaker_specific_info import SpinnakerSimulatorInfo
from microcircuit.nest_specific_info import NestSimulatorInfo
from microcircuit.common_params import CommonParams

sim_parser = ArgumentParser(add_help=False)
sim_parser.add_argument("simulator", nargs="?", action="store")
sim_parser.add_argument("-h", "--help", action="store_true")
args, extras = sim_parser.parse_known_args()
argv = list()
if args.simulator is not None:
    argv.append(args.simulator)
if args.help:
    argv.append("-h")
elif args.simulator is None:
    argv.append("spinnaker")
argv.extend(extras)

parser = ArgumentParser()
subparsers = parser.add_subparsers(dest="simulator")
add_subparser(subparsers, "spinnaker", SpinnakerSimulatorInfo.__init__)
add_subparser(subparsers, "nest", NestSimulatorInfo.__init__)
args = parser.parse_args(argv)

simulator = args.simulator.upper()
arg_dict = vars(args)
del arg_dict["simulator"]

# build sim params
if simulator == NEST_SIM:
    simulator_specific_info = NestSimulatorInfo(**arg_dict)
else:
    simulator_specific_info = SpinnakerSimulatorInfo(**arg_dict)

# build common params
common_params = CommonParams(simulator_specific_info)

# do nest'y things
if simulator == NEST_SIM:
    import pyNN.nest as sim
    sys.path.append(simulator_specific_info.backend_path)
    sys.path.append(simulator_specific_info.pynn_path)

# prepare simulation
if simulator == SPINNAKER_SIM:
    import spynnaker8 as sim

sim.setup(**simulator_specific_info.setup_params)
simulator_specific_info.after_setup_info(sim)

# create network
start_netw = time.time()
network = Network(simulator)
network.setup(sim, simulator_specific_info, common_params)
end_netw = time.time()
if sim.rank() == 0:
    print('Creating the network took ', end_netw - start_netw, ' s')

if simulator == NEST_SIM:
    simulator_specific_info.memory_print(sim)

# simulate
if sim.rank() == 0:
    print("Simulating...")
start_sim = time.time()
t = sim.run(simulator_specific_info.sim_duration)
end_sim = time.time()
if sim.rank() == 0:
    print('Simulation took ', end_sim - start_sim, ' s')

if simulator == NEST_SIM:
    simulator_specific_info.memory_print(sim)

start_writing = time.time()
for layer in common_params.layers:
    for pop in common_params.pops:
        filename = (
            simulator_specific_info.output_path + '/spikes_' + layer + pop +
            '.' + simulator_specific_info.output_format)
        network.pops[layer][pop].write_data(io=filename, variables='spikes')

if simulator_specific_info.record_v:
    for layer in common_params.layers:
        for pop in common_params.pops:
            filename = (
                simulator_specific_info.output_path + '/voltages_' + layer +
                pop + '.dat')
            network.pops[layer][pop].print_v(filename, gather=True)

if simulator == NEST_SIM:
    simulator_specific_info.after_run_info(sim, common_params)


end_writing = time.time()
print("Writing data took ", end_writing - start_writing, " s")

if common_params.plot_spiking_activity and sim.rank() == 0:
    plotting.plot_raster_bars(
        common_params.raster_t_min, common_params.raster_t_max,
        common_params.n_rec, common_params.frac_to_plot,
        simulator_specific_info.output_path, common_params)

sim.end()
