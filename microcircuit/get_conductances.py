from network_params import *
from scaling import get_indegrees, adjust_w_and_ext_to_K, adjust_w_and_g_to_K
from network import *
from helper_functions import *

def get_cond(DC_amp):
    # parameters for conductance-based model
    cm_cond = neuron_params['cm']
    e_rev_e = v_offset + v_scaling * neuron_params['e_rev_E']
    e_rev_i = v_offset + v_scaling * neuron_params['e_rev_I']

    # parameters for original model
    tau_m_curr = 10.0 * time_scaling
    tau_syn = 0.5 * time_scaling
    V_rest_curr = -65.0
    cm_curr = 0.25 # nF

    K_full = get_indegrees()

    w = create_weight_matrix('IF_curr_exp')

    # In the original model, L4e -> L23e has twice the weight of the other excitatory connections.
    # Find unique Je, Ji for each target population, such that mean and variance of inputs 
    # to each population are preserved.
    w_eq = adjust_w_and_g_to_K(K_full, 1., w)

    # scaling
    w, w_ext, DC_amp_new = adjust_w_and_ext_to_K(K_full, K_scaling, w_eq, DC_amp)
 
    K = K_scaling * K_full

    # mean and variance of membrane potentials
    # total internal input rates to each population
    x1 = {}
    x2 = {}
    R_e = {}
    R_i = {}
    R_ext = {}
    # mean membrane potential
    Vmeancurr = {}
    for target_layer in layers:
        x1[target_layer] = {}
        x2[target_layer] = {}
        R_e[target_layer] = {}
        R_i[target_layer] = {}
        R_ext[target_layer] = {}
        Vmeancurr[target_layer] = {}
        for target_pop in pops:
            target_index = structure[target_layer][target_pop]
            x1[target_layer][target_pop] = 0
            x2[target_layer][target_pop] = 0
            R_e[target_layer][target_pop] = 0
            R_i[target_layer][target_pop] = 0
            for source_layer in layers:
                for source_pop in pops:
                    source_index = structure[source_layer][source_pop]
                    # rates in ms**-1 since time constants are in ms
                    x1[target_layer][target_pop] += w[target_index][source_index] * \
                        K[target_index][source_index] * full_mean_rates[source_layer][source_pop] / 1000.
                    x2[target_layer][target_pop] += (w[target_index][source_index])**2 * \
                        K[target_index][source_index] * full_mean_rates[source_layer][source_pop] / 1000.
                    if source_pop == 'E':
                        R_e[target_layer][target_pop] += K[target_index][source_index] * \
                            full_mean_rates[source_layer]['E'] / 1000.
                    if source_pop == 'I':
                        R_i[target_layer][target_pop] += K[target_index][source_index] * \
                            full_mean_rates[source_layer]['I'] / 1000.
            # account for external drive
            x1[target_layer][target_pop] += w_ext * K_scaling * K_ext[target_layer][target_pop] * bg_rate / 1000.
            x2[target_layer][target_pop] += w_ext**2 * K_scaling * K_ext[target_layer][target_pop] * bg_rate / 1000.
            R_ext[target_layer][target_pop] = K_scaling * K_ext[target_layer][target_pop] * bg_rate / 1000.
    
            Vmeancurr[target_layer][target_pop] = V_rest_curr + tau_m_curr / cm_curr * \
                (tau_syn * x1[target_layer][target_pop] + DC_amp_new[target_layer][target_pop])

    g_e = {}
    g_i = {}
    g_ext = {}
    v_rest_intended = {}
    tau_m = {}

    for target_layer in layers:
        g_e[target_layer] = {}
        g_i[target_layer] = {}
        g_ext[target_layer] = {}
        v_rest_intended[target_layer] = {}
        tau_m[target_layer] = {}

        for target_pop in pops:
            target_index = structure[target_layer][target_pop]

            Je = w[target_index][0]
            Ji = w[target_index][1]
            Je_tilde = Je * (cm_cond / cm_curr)
            Ji_tilde = Ji * (cm_cond / cm_curr)
            Jext_tilde = w_ext * (cm_cond / cm_curr)
            # conductances in microSiemens
            g_e[target_layer][target_pop] = -Je_tilde / (Vmeancurr[target_layer][target_pop] - e_rev_e)    
            g_i[target_layer][target_pop] = -Ji_tilde / (Vmeancurr[target_layer][target_pop] - e_rev_i)
            g_ext[target_layer][target_pop] = -Jext_tilde / (Vmeancurr[target_layer][target_pop] - e_rev_e)

            ge0 = g_e[target_layer][target_pop] * tau_syn * R_e[target_layer][target_pop] + \
                g_ext[target_layer][target_pop]*tau_syn*R_ext[target_layer][target_pop]
            gi0 = g_i[target_layer][target_pop] * tau_syn * R_i[target_layer][target_pop]
            tau_m[target_layer][target_pop] = cm_cond / (cm_cond / tau_m_curr - ge0 - gi0)

            # resting membrane potential that would yield membrane potential distribution
            # close to that in the original model (model value, not PyNN value)
            # The equation below takes into account that no DC input can be provided on the ESS
            # 'DC_amp_new' is that of the current-based model.
            # That of the conductance-based model is taken to be zero.
            v_rest_intended[target_layer][target_pop] = tau_m[target_layer][target_pop]*( V_rest_curr/tau_m_curr + \
                                                        DC_amp_new[target_layer][target_pop]/cm_curr + \
                                                        Vmeancurr[target_layer][target_pop] \
                                                        / (Vmeancurr[target_layer][target_pop] - e_rev_e) * tau_syn \
                                                        * Je * R_e[target_layer][target_pop] / cm_curr + \
                                                        Vmeancurr[target_layer][target_pop] \
                                                        / (Vmeancurr[target_layer][target_pop] - e_rev_e) * tau_syn \
                                                        * w_ext * R_ext[target_layer][target_pop] / cm_curr + \
                                                        Vmeancurr[target_layer][target_pop] \
                                                        / (Vmeancurr[target_layer][target_pop] - e_rev_i) * tau_syn \
                                                        * Ji * R_i[target_layer][target_pop] / cm_curr )

    return g_e, g_i, g_ext, tau_m, v_rest_intended



