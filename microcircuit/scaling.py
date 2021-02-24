#############################################################################
# Functions for computing and adjusting connection and input parameters
#############################################################################

import numpy as np
from .constants import POISSON


def get_in_degrees(common_params):
    """
    Get in-degrees for each connection for the full-scale (1 mm^2) model
    :param common_params: network common params
    :return: k
    :rtype: numpy array
    """
    k = np.zeros(
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
                    k[target_index][source_index] = (
                        round(np.log(
                            1. - common_params.conn_probs[
                                target_index][source_index]) / np.log(
                            (n_target * n_source - 1.) /
                            (n_target * n_source))) / n_target)
    return k


def adjust_w_and_ext_to_k(
        k_full, k_scaling, w, dc, common_params, simulation_params):
    """
    Adjust synaptic weights and external drive to the in-degrees
    to preserve mean and variance of inputs in the diffusion approximation

    :param k_full:
    :param k_scaling:
    :param w:
    :param dc:
    :param common_params:
    :param simulation_params:
    :return: w_new, w_ext_new, i_ext
    """
    internal_scaling = k_scaling

    w_new = w / np.sqrt(internal_scaling)
    w_ext_new = None
    i_ext = {}
    for target_layer in common_params.layers:
        i_ext[target_layer] = {}
        for target_pop in common_params.pops:
            target_index = common_params.structure[target_layer][target_pop]
            x1 = 0
            for source_layer in common_params.layers:
                for source_pop in common_params.pops:
                    source_index = (
                        common_params.structure[source_layer][source_pop])
                    x1 += (
                        w[target_index][source_index] *
                        k_full[target_index][source_index] *
                        common_params.full_mean_rates[
                            source_layer][source_pop])

            if simulation_params.input_type == POISSON:
                x1_ext = (
                    common_params.w_mean *
                    common_params.k_ext[target_layer][target_pop] *
                    common_params.bg_rate)
                external_scaling = k_scaling
                w_ext_new = common_params.w_mean / np.sqrt(external_scaling)
                i_ext[target_layer][target_pop] = (
                    0.001 * simulation_params.neuron_params['tau_syn_E'] *
                    ((1. - np.sqrt(internal_scaling)) * x1 +
                     (1. - np.sqrt(external_scaling)) * x1_ext) +
                    dc[target_layer][target_pop])
            else:
                w_ext_new = np.nan
                i_ext[target_layer][target_pop] = (
                    0.001 * simulation_params.neuron_params['tau_syn_E'] *
                    ((1. - np.sqrt(internal_scaling)) * x1) +
                    dc[target_layer][target_pop])

    return w_new, w_ext_new, i_ext
