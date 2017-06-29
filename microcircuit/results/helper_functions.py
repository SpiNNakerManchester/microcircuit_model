from network_params import *
import numpy as np
import os
import math

def create_weight_matrix(neuron_model, **kwargs):
    w = np.zeros([n_layers * n_pops_per_layer, n_layers * n_pops_per_layer])
    if neuron_model == 'IF_curr_exp':
        for target_layer in layers:
            for target_pop in pops:
                target_index = structure[target_layer][target_pop]
                for source_layer in layers:
                    for source_pop in pops:
                        source_index = structure[source_layer][source_pop]
                        if source_pop == 'E':
                            if source_layer == 'L4' and target_layer == 'L23' and target_pop == 'E':
                                w[target_index][source_index] = w_234
                            else:
                                w[target_index][source_index] = w_mean
                        else:
                            w[target_index][source_index] = g * w_mean
    elif neuron_model == 'IF_cond_exp':
        g_e = kwargs.get('g_e', None)
        g_i = kwargs.get('g_i', None)
        if conn_routine == 'fixed_total_number' and simulator == 'nest':
        # inhibitory conductances get a minus sign
            for target_layer in layers:
                for target_pop in pops:
                    target_index = structure[target_layer][target_pop]
                    w[target_index] = n_layers * [g_e[target_layer][target_pop], -g_i[target_layer][target_pop]]
        else :
            for target_layer in layers:
                for target_pop in pops:
                    target_index = structure[target_layer][target_pop]
                    w[target_index] = n_layers * [g_e[target_layer][target_pop], g_i[target_layer][target_pop]]

    return w



def get_init_voltages_from_file(pop):
    voltages = np.zeros(len(pop))
    for filename in os.listdir(input_dir):
        if filename == ('voltages_{}.dat'.format(pop.label)):
            print 'Reading voltages from {}'.format(filename)
            f = open(os.path.join(input_dir, filename))
            for line in f:
                if not line.startswith("#"):
                    line = line.strip()
                    (voltage, neuron_id) = line.split()
                    neuron_id = int(math.floor(float(neuron_id)))
                    voltage = float(voltage)
                    if voltages[neuron_id] == 0:
                        voltages[neuron_id] = voltage
    return voltages
