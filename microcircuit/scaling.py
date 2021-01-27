#############################################################################
# Functions for computing and adjusting connection and input parameters ###
#############################################################################

import numpy as np
from constants import POISSON


def get_indegrees(common_params):
    '''Get in-degrees for each connection for the full-scale (1 mm^2) model'''
    K = np.zeros(
        [common_params.n_layers * common_params.n_pops_per_layer,
         common_params.n_layers * common_params.n_pops_per_layer])
    for target_layer in common_params.layers:
        for target_pop in common_params.pops:
            for source_layer in common_params.layers:
                for source_pop in common_params.pops:
                    target_index = (
                        common_params.structure[target_layer][target_pop])
                    source_index = (
                        common_params.structure[source_layer][source_pop])
                    n_target = common_params.n_full[target_layer][target_pop]
                    n_source = common_params.n_full[source_layer][source_pop]
                    K[target_index][source_index] = (
                        round(np.log(
                            1. - common_params.conn_probs[
                                target_index][source_index]) / np.log(
                            (n_target * n_source - 1.) /
                            (n_target * n_source))) / n_target)
    return K


def adjust_w_and_ext_to_K(
        K_full, K_scaling, w, DC, common_params, simulation_params):
    '''Adjust synaptic weights and external drive to the in-degrees
     to preserve mean and variance of inputs in the diffusion approximation'''

    internal_scaling = K_scaling
   
    w_new = w / np.sqrt(internal_scaling)
    w_ext_new = None
    I_ext = {}
    for target_layer in common_params.layers:
        I_ext[target_layer] = {}
        for target_pop in common_params.pops:
            target_index = common_params.structure[target_layer][target_pop]
            x1 = 0
            for source_layer in common_params.layers:
                for source_pop in common_params.pops:
                    source_index = (
                        common_params.structure[source_layer][source_pop])
                    x1 += (
                        w[target_index][source_index] *
                        K_full[target_index][source_index] *
                        common_params.full_mean_rates[
                            source_layer][source_pop])

            if simulation_params.input_type == POISSON:
                x1_ext = (
                    common_params.w_mean *
                    common_params.k_ext[target_layer][target_pop] *
                    common_params.bg_rate)
                external_scaling = K_scaling
                w_ext_new = common_params.w_mean / np.sqrt(external_scaling)
                I_ext[target_layer][target_pop] = (
                    0.001 * simulation_params.neuron_params['tau_syn_E'] *
                    ((1. - np.sqrt(internal_scaling)) * x1 +
                     (1. - np.sqrt(external_scaling)) * x1_ext) +
                    DC[target_layer][target_pop])
            else:
                w_ext_new = np.nan
                I_ext[target_layer][target_pop] = (
                    0.001 * simulation_params.neuron_params['tau_syn_E'] *
                    ((1. - np.sqrt(internal_scaling)) * x1) +
                    DC[target_layer][target_pop])

    return w_new, w_ext_new, I_ext


def adjust_w_and_g_to_K(K_full, K_scaling, w, common_params):
    '''
    Calculate target-population-specific synaptic weights that
    approximately preserve the mean and variance of the population activities
    for the given full-scale in-degrees, making all excitatory weights equal
    for the given target population
    '''
    g_new = {}
    w_new = {}
    w_m_matrix = np.zeros(
        [common_params.n_layers * common_params.n_pops_per_layer,
         common_params.n_layers * common_params.n_pops_per_layer])
    for target_layer in common_params.layers:
        g_new[target_layer] = {}
        w_new[target_layer] = {}
        for target_pop in common_params.pops:
            target_index = common_params.structure[target_layer][target_pop]
            x0 = {'E': 0, 'I': 0}
            x1 = {'E': 0, 'I': 0}
            x2 = {'E': 0, 'I': 0}
            for source_layer in common_params.layers:
                for source_pop in common_params.pops:
                    source_index = (
                        common_params.structure[source_layer][source_pop])
                    x0[source_pop] += (
                        K_full[target_index][source_index] *
                        common_params.full_mean_rates[
                            source_layer][source_pop])
                    x1[source_pop] += (
                        w[target_index][source_index] *
                        K_full[target_index][source_index] *
                        common_params.full_mean_rates[
                            source_layer][source_pop])
                    x2[source_pop] += ((
                        w[target_index][source_index]) ** 2 *
                        K_full[target_index][source_index] *
                        common_params.full_mean_rates[
                            source_layer][source_pop])

            N_min = (
                (x1['E'] + x1['I'])**2 /
                ((x0['E'] + x0['I']) * (x2['E'] + x2['I'])))
            w_new[target_layer][target_pop] = (
                (np.sqrt(x0['E']) * (x1['E'] + x1['I'])
                + np.sqrt(x0['I'] * K_scaling * (x0['E'] + x0['I']) *
                (x2['E'] + x2['I']) - x0['I'] * (x1['E'] + x1['I'])**2)) /
                (K_scaling * np.sqrt(x0['E']) * (x0['E'] + x0['I'])))
            g_new[target_layer][target_pop] = (
                (x1['E'] + x1['I'] - w_new[target_layer][target_pop] *
                 K_scaling * x0['E']) /
                (w_new[target_layer][target_pop] * K_scaling * x0['I']))

    for target_layer in common_params.layers:
        for target_pop in common_params.pops:
            target_index = common_params.structure[target_layer][target_pop]
            x2_new = {'E': 0, 'I': 0}
            for source_layer in common_params.layers:
                for source_pop in common_params.pops:
                    source_index = (
                        common_params.structure[source_layer][source_pop])
                    if source_pop == 'E':
                        w_m_matrix[target_index][source_index] = (
                            w_new[target_layer][target_pop])
                    else:
                        w_m_matrix[target_index][source_index] = (
                            g_new[target_layer][target_pop] *
                            w_new[target_layer][target_pop])
                    # test if new variance equals old variance
                    x2_new[source_pop] += (
                        (w_m_matrix[target_index][source_index])**2 *
                        K_scaling * K_full[target_index][source_index] *
                        common_params.full_mean_rates[
                            source_layer][source_pop])
    return w_m_matrix
