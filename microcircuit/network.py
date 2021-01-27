from connectivity import build_from_list_connect
from constants import (
    DC, NEST_NERUON_MODEL, SPINNAKER_NEURON_MODEL, POISSON, CONN_ROUTINE)
from sim_params import SIMULATOR, NEST_SIM, SPINNAKER_SIM
from scaling import get_in_degrees, adjust_w_and_ext_to_k
from helper_functions import (
    create_weight_matrix, get_init_voltages_from_file)
from pyNN.random import NumpyRNG, RandomDistribution
import numpy as np


class Network:
    """ builds the pynn network

    """

    __slots__ = [
        'pops',
    ]

    def __init__(self):
        self.pops = {}

    def setup(self, sim, simulator_specific_stuff, common_params):
        """ creates the pynn network

        :param sim: the simulator
        :param simulator_specific_stuff: \
            the holder for simulator specific params.
        :param common_params: the holder for common params.
        :rtype: None
        """

        # if parallel_safe=False, PyNN offsets the seeds by 1 for each rank
        script_rng = NumpyRNG(
            seed=simulator_specific_stuff.pyseed,
            parallel_safe=simulator_specific_stuff.parallel_safe)

        # Compute DC input before scaling
        if simulator_specific_stuff.input_type == DC:
            dc_amp = {}
            for target_layer in common_params.layers:
                dc_amp[target_layer] = {}
                for target_pop in common_params.pops:
                    dc_amp[target_layer][target_pop] = (
                        common_params.bg_rate *
                        common_params.k_ext[
                            target_layer][target_pop] *
                        common_params.w_mean *
                        common_params.tau_syn_name / 1000.0)
        else:
            dc_amp = {
                'L23': {'E': 0.0, 'I': 0.0},
                'L4': {'E': 0.0, 'I': 0.0},
                'L5': {'E': 0.0, 'I': 0.0},
                'L6': {'E': 0.0, 'I': 0.0}}

        # In-degrees of the full-scale and scaled models
        k_full = get_in_degrees(common_params)
        k = simulator_specific_stuff.k_scaling * k_full

        k_ext = {}
        for layer in common_params.layers:
            k_ext[layer] = {}
            for pop in common_params.pops:
                k_ext[layer][pop] = (
                    simulator_specific_stuff.k_scaling *
                    common_params.k_ext[layer][pop])

        w = create_weight_matrix(common_params)

        # Network scaling
        if simulator_specific_stuff.k_scaling != 1:
            w, w_ext, dc_amp = adjust_w_and_ext_to_k(
                k_full, simulator_specific_stuff.k_scaling, w, dc_amp,
                common_params, simulator_specific_stuff)
        else:
            w_ext = common_params.w_mean

        # Initial membrane potential distribution
        v_dist_all = RandomDistribution(
            'normal', [common_params.v0_mean, common_params.v0_sd],
            rng=script_rng)

        # Improved initialisation from Julich
        v_dist = {
            'L23E': RandomDistribution(
                'normal',
                [common_params.v0_l23e_mean, common_params.v0_l23e_sd],
                rng=script_rng),
            'L23I': RandomDistribution(
                'normal',
                [common_params.v0_l23i_mean, common_params.v0_l23i_sd],
                rng=script_rng),
            'L4E': RandomDistribution(
                'normal',
                [common_params.v0_l4e_mean, common_params.v0_l4e_sd],
                rng=script_rng),
            'L4I': RandomDistribution(
                'normal',
                [common_params.v0_l4i_mean, common_params.v0_l4i_sd],
                rng=script_rng),
            'L5E': RandomDistribution(
                'normal',
                [common_params.v0_l5e_mean, common_params.v0_l5e_sd],
                rng=script_rng),
            'L5I': RandomDistribution(
                'normal',
                [common_params.v0_l5i_mean, common_params.v0_l5i_sd],
                rng=script_rng),
            'L6E': RandomDistribution(
                'normal',
                [common_params.v0_l6e_mean, common_params.v0_l6e_sd],
                rng=script_rng),
            'L6I': RandomDistribution(
                'normal',
                [common_params.v0_l6i_mean, common_params.v0_l6i_sd],
                rng=script_rng)}

        if simulator_specific_stuff.neuron_model == NEST_NERUON_MODEL:
            from pyNN.nest import native_cell_type
            model = native_cell_type('iaf_psc_exp_ps')
        else:
            model = getattr(sim, simulator_specific_stuff.neuron_model)

        if SIMULATOR == NEST_SIM:
            simulator_specific_stuff.record_corr_stuff(sim, common_params)

        if sim.rank() == 0:
            print('neuron_params:', simulator_specific_stuff.neuron_params)
            print('k: ', k)
            print('k_ext: ', k_ext)
            print('w: ', w)
            print('w_ext: ', w_ext)
            print('dc_amp: ', dc_amp)
            print('n_rec:')
            for layer in sorted(common_params.layers):
                for pop in sorted(common_params.pops):
                    print(layer, pop, common_params.n_rec[layer][pop])
                    if SIMULATOR == NEST_SIM:
                        simulator_specific_stuff.rank_stuff(
                            common_params, layer, pop)

        # Create cortical populations
        global_neuron_id = 1
        base_neuron_ids = {}
        for layer in sorted(common_params.layers):
            self.pops[layer] = {}
            for pop in sorted(common_params.pops):
                self.pops[layer][pop] = sim.Population(
                    int(round(common_params.n_full[layer][pop] *
                              simulator_specific_stuff.n_scaling)),
                    model, cellparams=simulator_specific_stuff.neuron_params,
                    label=layer+pop)
                this_pop = self.pops[layer][pop]

                # Provide DC input
                if (simulator_specific_stuff.neuron_model ==
                        SPINNAKER_NEURON_MODEL):
                    this_pop.set(i_offset=dc_amp[layer][pop])
                if simulator_specific_stuff.neuron_model == NEST_NERUON_MODEL:
                    this_pop.set(I_e=1000 * dc_amp[layer][pop])

                base_neuron_ids[this_pop] = global_neuron_id
                global_neuron_id += len(this_pop) + 2

                if simulator_specific_stuff.voltage_input_type == 'random':
                    this_pop.initialize(v=v_dist_all)
                elif (simulator_specific_stuff.voltage_input_type ==
                        'pop_random'):
                    this_pop.initialize(v=v_dist[this_pop.label])
                elif (simulator_specific_stuff.voltage_input_type ==
                        'from_list'):
                    this_pop.initialize(v=get_init_voltages_from_file(
                        this_pop, simulator_specific_stuff))

                # Spike recording
                this_pop[0: common_params.n_rec[layer][pop]].record("spikes")

                # Membrane potential recording
                if simulator_specific_stuff.record_v:
                    simulator_specific_stuff.set_record_v(this_pop)

                # Correlation recording
                if SIMULATOR == NEST_SIM:
                    simulator_specific_stuff.set_corr_recording(
                        layer, pop, common_params, sim, this_pop)

        if SIMULATOR == NEST_SIM:
            simulator_specific_stuff.set_defaults(sim)

        thalamic_population = None
        if common_params.thalamic_input:
            # Create thalamic population
            thalamic_population = sim.Population(
                common_params.thal_params['n_thal'],
                sim.SpikeSourcePoisson, {
                    'rate': common_params.thal_params['rate'],
                    'start': common_params.thal_params['start'],
                    'duration': common_params.thal_params['duration']},
                label='thalamic_population',
                additional_parameters={
                    'seed': simulator_specific_stuff.pyseed})
            base_neuron_ids[thalamic_population] = global_neuron_id
            global_neuron_id += len(thalamic_population) + 2

        possible_targets = ['inhibitory', 'excitatory']

        # Connect
        for target_layer in sorted(common_params.layers):
            for target_pop in sorted(common_params.pops):
                target_index = (
                    common_params.structure[target_layer][target_pop])
                this_target_pop = self.pops[target_layer][target_pop]
                w_ext = w_ext
                # External inputs
                if simulator_specific_stuff.input_type == POISSON:
                    rate = (
                        common_params.bg_rate *
                        k_ext[target_layer][target_pop])

                    if SIMULATOR == NEST_SIM:
                        simulator_specific_stuff.create_poissons(
                            sim, target_layer, target_pop, rate,
                            this_target_pop, w_ext, common_params)
                    else:
                        simulator_specific_stuff.create_poissons(
                            sim, target_layer, target_pop, rate,
                            this_target_pop, w_ext)

                if common_params.thalamic_input:
                    # Thalamic inputs
                    if sim.rank() == 0:
                        print('creating thalamic connections to {} {}'.format(
                            target_layer, target_pop))
                    c_thal = (
                        common_params.thal_params[
                            'C'][target_layer][target_pop])
                    n_target = (
                        common_params.n_full[target_layer][target_pop])
                    k_thal = (
                        round(np.log(1 - c_thal) / np.log(
                            (n_target *
                             common_params.thal_params['n_thal'] - 1.) /
                            (n_target *
                             common_params.thal_params['n_thal']))) /
                        n_target * simulator_specific_stuff.k_scaling)

                    if simulator_specific_stuff.conn_routine == CONN_ROUTINE:
                        if SIMULATOR == SPINNAKER_SIM:
                            simulator_specific_stuff.fixed_tot_number_connect(
                                sim, thalamic_population, this_target_pop,
                                k_thal, w_ext, common_params.w_rel * w_ext,
                                common_params.d_mean['E'],
                                common_params.d_sd['E'], 'excitatory',
                                script_rng)
                        else:
                            simulator_specific_stuff.fixed_tot_number_connect(
                                sim, thalamic_population, this_target_pop,
                                k_thal, w_ext, common_params.w_rel * w_ext,
                                common_params.d_mean['E'],
                                common_params.d_sd['E'])
                    elif simulator_specific_stuff.conn_routine == 'from_list':
                        build_from_list_connect(
                            sim, thalamic_population, this_target_pop,
                            'excitatory', base_neuron_ids,
                            simulator_specific_stuff)

                # Recurrent inputs
                for source_layer in sorted(common_params.layers):
                    for source_pop in sorted(common_params.pops):
                        source_index = (
                            common_params.structure[source_layer][source_pop])
                        this_source_pop = self.pops[source_layer][source_pop]
                        weight = w[target_index][source_index]

                        conn_type = possible_targets[
                            int((np.sign(weight) + 1) / 2)]

                        if sim.rank() == 0:
                            print(
                                'creating connections from {} {} to {} '
                                '{}'.format(
                                    source_layer, source_pop, target_layer,
                                    target_pop))

                        if (source_pop == 'E' and source_layer == 'L4'
                                and target_layer == 'L23'
                                and target_pop == 'E'):
                            w_sd = weight * common_params.w_rel_234
                        else:
                            w_sd = abs(weight * common_params.w_rel)

                        if (simulator_specific_stuff.conn_routine ==
                                CONN_ROUTINE):
                            if SIMULATOR == SPINNAKER_SIM:
                                simulator_specific_stuff.\
                                    fixed_tot_number_connect(
                                        sim, this_source_pop, this_target_pop,
                                        k[target_index][source_index], weight,
                                        w_sd, common_params.d_mean[source_pop],
                                        common_params.d_sd[source_pop],
                                        conn_type, script_rng)
                            else:
                                simulator_specific_stuff.\
                                    fixed_tot_number_connect(
                                        sim, this_source_pop, this_target_pop,
                                        k[target_index][source_index], weight,
                                        w_sd, common_params.d_mean[source_pop],
                                        common_params.d_sd[source_pop])
                        elif (simulator_specific_stuff.conn_routine ==
                                'from_list'):
                            build_from_list_connect(
                                sim, this_source_pop, this_target_pop,
                                conn_type, base_neuron_ids,
                                simulator_specific_stuff)
