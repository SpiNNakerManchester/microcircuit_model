import math
import os
import numpy as np


def create_weight_matrix(common_params):
    """
    Creates the wight array based on the parameters

    :param  CommonParams common_params:
    :rtype: nparray
    """
    w = np.zeros(
        [common_params.n_layers * common_params.n_pops_per_layer,
         common_params.n_layers * common_params.n_pops_per_layer])
    for target_layer in common_params.layers:
        for target_pop in common_params.pops:
            target_index = common_params.structure[target_layer][target_pop]
            for source_layer in common_params.layers:
                for source_pop in common_params.pops:
                    source_index = (
                        common_params.structure[source_layer][source_pop])
                    if source_pop == 'E':
                        if (source_layer == 'L4' and
                                target_layer == 'L23' and target_pop == 'E'):
                            w[target_index][source_index] = (
                                common_params.w_234)
                        else:
                            w[target_index][source_index] = (
                                common_params.w_mean)
                    else:
                        w[target_index][source_index] = (
                            common_params.g * common_params.w_mean)
    return w


def get_init_voltages_from_file(pop, simulator_params):
    """
    Reads the voltages from the file based on population label

    :param Population pop:
    :param SpinnakerParams simulator_params:
    :rtype: nparray
    """
    voltages = np.zeros(len(pop))
    for filename in os.listdir(simulator_params.input_dir):
        if filename == f'voltages_{pop.label}.dat':
            print(f'Reading voltages from {filename}')
            f = open(os.path.join(simulator_params.input_dir, filename),
                     encoding="utf8")
            for line in f:
                if not line.startswith("#"):
                    line = line.strip()
                    (voltage, neuron_id) = line.split()
                    neuron_id = int(math.floor(float(neuron_id)))
                    voltage = float(voltage)
                    if voltages[neuron_id] == 0:
                        voltages[neuron_id] = voltage
    return voltages
