###################################################
###     	Main script			###
###################################################

import neo
import sys
from sim_params import *
from spinn_front_end_common.utilities import globals_variables

sys.path.append(system_params['backend_path'])
sys.path.append(system_params['pyNN_path'])
from network_params import *
import pyNN
import time
import plotting
import numpy as np

# prepare simulation
#exec('import pyNN.%s as sim' %simulator)
import spynnaker8 as sim


class MicroCircuit(object):

    @staticmethod
    def name():
        return "MicroCircuit"

    def __call__(
            self, mapping_algorithms, loading_algorithms, time_step,
            time_scale_factor):

        simulator_params["extra_mapping_algorithms"] = mapping_algorithms
        simulator_params["extra_load_algorithms"] = loading_algorithms
        simulator_params["timestep"] = time_step
        simulator_params["time_scale_factor"] = time_scale_factor
        sim.setup(**simulator_params[simulator])

        if simulator == 'spiNNaker':
            sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 255)
            sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 255)

        import network

        # create network
        start_netw = time.time()
        n = network.Network(sim)
        n.setup(sim)
        end_netw = time.time()
        if sim.rank() == 0:
            print( 'Creating the network took ', end_netw - start_netw, ' s')

        # simulate
        if sim.rank() == 0:
            print("Simulating...")

        try:
            sim.run(simulator_params[simulator]['sim_duration'])
        except Exception:
            return globals_variables.get_simulator(), sim, False
        return globals_variables.get_simulator(), sim, True

if __name__ == "__main__":
    run = MicroCircuit()
    run(mapping_algorithms=None,
        loading_algorithms=None,
        time_step=1000,
        time_scale_factor=100)
