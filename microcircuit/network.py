from connectivity import (
    fixed_total_number_connect_spinnaker,
    fixed_total_number_connect_nest, build_from_list_connect)
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
        'corr_detector'
    ]

    def __init__(self):
        self.pops = {}
        self.corr_detector = None

    def setup(self, sim, simulator_params, common_params):
        """ creates the pynn network

        :param sim: the simulator
        :param simulator_params: the holder for simulator specific params.
        :param common_params: the holder for common params.
        :rtype: None
        """

        # if parallel_safe=False, PyNN offsets the seeds by 1 for each rank
        script_rng = NumpyRNG(
            seed=simulator_params.pyseed,
            parallel_safe=simulator_params.parallel_safe)

        # Compute DC input before scaling
        if simulator_params.input_type == DC:
            dc_amp = {}
            for target_layer in common_params.layers:
                dc_amp[target_layer] = {}
                for target_pop in common_params.pops:
                    dc_amp[target_layer][target_pop] = (
                        simulator_params.bg_rate *
                        simulator_params.k_ext[target_layer][target_pop] *
                        simulator_params.w_mean *
                        simulator_params.tau_syn_name / 1000.0)
        else:
            dc_amp = {
                'L23': {'E': 0.0, 'I': 0.0},
                'L4': {'E': 0.0, 'I': 0.0},
                'L5': {'E': 0.0, 'I': 0.0},
                'L6': {'E': 0.0, 'I': 0.0}}

        # In-degrees of the full-scale and scaled models
        k_full = get_in_degrees(common_params)
        k = simulator_params.k_scaling * k_full

        k_ext = {}
        for layer in common_params.layers:
            k_ext[layer] = {}
            for pop in common_params.pops:
                k_ext[layer][pop] = (
                    simulator_params.k_scaling *
                    common_params.k_ext[layer][pop])

        w = create_weight_matrix(common_params)
        # Network scaling
        if simulator_params.k_scaling != 1:
            w, w_ext, dc_amp = adjust_w_and_ext_to_k(
                k_full, simulator_params.k_scaling, w, dc_amp,
                common_params, simulator_params)
        else:
            w_ext = common_params.w_mean

        # Initial membrane potential distribution
        v_dist_all = RandomDistribution(
            'normal', [common_params.v0_mean, common_params.v0_sd],
            rng=script_rng)

        # Improved initialisation from Julich
        v_dist = {'L23E': RandomDistribution(
            'normal',
            [common_params.v0_l23e_mean, common_params.v0_l23e_sd],
            rng=script_rng), 'L23I': RandomDistribution(
            'normal',
            [common_params.v0_l23i_mean, common_params.v0_l23i_sd],
            rng=script_rng), 'L4E': RandomDistribution(
            'normal',
            [common_params.v0_l4e_mean, common_params.v0_l4e_sd],
            rng=script_rng), 'L4I': RandomDistribution(
            'normal',
            [common_params.v0_l4i_mean, common_params.v0_l4i_sd],
            rng=script_rng), 'L5E': RandomDistribution(
            'normal',
            [common_params.v0_l5e_mean, common_params.v0_l5e_sd],
            rng=script_rng), 'L5I': RandomDistribution(
            'normal',
            [common_params.v0_l5i_mean, common_params.v0_l5i_sd],
            rng=script_rng), 'L6E': RandomDistribution(
            'normal',
            [common_params.v0_l6e_mean, common_params.v0_l6e_sd],
            rng=script_rng), 'L6I': RandomDistribution(
            'normal',
            [common_params.v0_l6i_mean, common_params.v0_l6i_sd],
            rng=script_rng)}

        if simulator_params.neuron_model == NEST_NERUON_MODEL:
            from pyNN.nest import native_cell_type
            model = native_cell_type('iaf_psc_exp_ps')
        else:
            model = getattr(sim, simulator_params.neuron_model)

        if SIMULATOR == NEST_SIM:
            if simulator_params.record_corr:
                # Create correlation recording device
                sim.nest.SetDefaults(
                    'correlomatrix_detector', {'delta_tau': 0.5})
                self.corr_detector = (
                    sim.nest.Create('correlomatrix_detector'))
                sim.nest.SetStatus(
                    self.corr_detector,
                    {'N_channels': (
                         common_params.n_layers *
                         common_params.n_pops_per_layer),
                     'tau_max': common_params.tau_max,
                     'Tstart': common_params.tau_max})

        if sim.rank() == 0:
            print('neuron_params:', simulator_params.neuron_params)
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
                        if (not simulator_params.record_fraction and
                            simulator_params.n_record > int(
                                round(common_params.n_full[layer][pop] *
                                      simulator_params.n_scaling))):
                            print(
                                'Note that requested number of neurons '
                                'to record exceeds {} {} population '
                                'size'.format(layer, pop))

        # Create cortical populations
        global_neuron_id = 1
        base_neuron_ids = {}
        for layer in sorted(common_params.layers):
            self.pops[layer] = {}
            for pop in sorted(common_params.pops):
                self.pops[layer][pop] = sim.Population(
                    int(round(common_params.n_full[layer][pop] *
                              simulator_params.n_scaling)),
                    model, cellparams=simulator_params.neuron_params,
                    label=layer+pop)
                this_pop = self.pops[layer][pop]

                # Provide DC input
                if simulator_params.neuron_model == SPINNAKER_NEURON_MODEL:
                    this_pop.set(i_offset=dc_amp[layer][pop])
                if simulator_params.neuron_model == NEST_NERUON_MODEL:
                    this_pop.set(I_e=1000 * dc_amp[layer][pop])

                base_neuron_ids[this_pop] = global_neuron_id
                global_neuron_id += len(this_pop) + 2

                if simulator_params.voltage_input_type == 'random':
                    this_pop.initialize(v=v_dist_all)
                elif simulator_params.voltage_input_type == 'pop_random':
                    this_pop.initialize(v=v_dist[this_pop.label])
                elif simulator_params.voltage_input_type == 'from_list':
                    this_pop.initialize(v=get_init_voltages_from_file(
                        this_pop, simulator_params))

                # Spike recording
                this_pop[0: common_params.n_rec[layer][pop]].record("spikes")

                # Membrane potential recording
                if simulator_params.record_v:
                    if SIMULATOR == SPINNAKER_SIM:
                        this_pop.record_v()
                    else:
                        if simulator_params.record_fraction:
                            n_rec_v = round(
                                this_pop.size * simulator_params.frac_record_v)
                        else :
                            n_rec_v = simulator_params.n_record_v
                        if simulator_params.neuron_model == NEST_NERUON_MODEL:
                            this_pop.celltype.recordable = ['V_m', 'spikes']
                            this_pop[0: n_rec_v]._record('V_m')
                        else:
                            this_pop[0: n_rec_v].record_v()

                # Correlation recording
                if SIMULATOR == NEST_SIM:
                    if simulator_params.record_corr:
                        index = simulator_params.structure[layer][pop]
                        sim.nest.SetDefaults(
                            'static_synapse', {'receptor_type': index})
                        sim.nest.ConvergentConnect(
                            list(this_pop.all_cells), self.corr_detector)

        if SIMULATOR == NEST_SIM:
            if simulator_params.record_corr:
                # reset receptor_type
                sim.nest.SetDefaults('static_synapse', {'receptor_type': 0})

        thalamic_population = None
        if common_params.thalamic_input:
            # Create thalamic population
            thalamic_population = sim.Population(
                simulator_params.thal_params['n_thal'],
                sim.SpikeSourcePoisson, {
                    'rate': simulator_params.thal_params['rate'],
                    'start': simulator_params.thal_params['start'],
                    'duration': simulator_params.thal_params['duration']},
                label='thalamic_population')
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
                if simulator_params.input_type == POISSON:
                    rate = (
                        common_params.bg_rate *
                        k_ext[target_layer][target_pop])

                    if SIMULATOR == NEST_SIM:
                        # create only a single Poisson generator for
                        # each population, since the native NEST implementation
                        # sends independent spike trains to all targets
                        if sim.rank() == 0:
                            print('connecting Poisson generator to {} {} '
                                  'via SLI'.format(target_layer, target_pop))
                        sim.nest.sli_run(
                            '/poisson_generator Create /poisson_generator_e '
                            'Set poisson_generator_e << /rate '
                            + str(rate) + ' >> SetStatus')
                        sim.nest.sli_run(
                            "poisson_generator_e " + str(list(
                                this_target_pop.all_cells)).replace(',', '')
                            + " [" + str(1000 * w_ext) + "] [" +
                            str(common_params.d_mean['E']) +
                            "] DivergentConnect")
                    else:
                        if sim.rank() == 0:
                            print(
                                'connecting Poisson generators to'
                                ' {} {}'.format(target_layer, target_pop))
                        poisson_generator = sim.Population(
                            this_target_pop.size, sim.SpikeSourcePoisson,
                            {'rate': rate})
                        conn = sim.OneToOneConnector()
                        syn = sim.StaticSynapse(weight=w_ext)
                        sim.Projection(
                            poisson_generator, this_target_pop, conn, syn,
                            receptor_type='excitatory')

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
                             simulator_params.thal_params['n_thal'] - 1.) /
                            (n_target *
                             simulator_params.thal_params['n_thal']))) /
                        n_target * simulator_params.k_scaling)

                    if simulator_params.conn_routine == CONN_ROUTINE:
                        if SIMULATOR == SPINNAKER_SIM:
                            fixed_total_number_connect_spinnaker(
                                sim, thalamic_population, this_target_pop,
                                k_thal, w_ext, common_params.w_rel * w_ext,
                                common_params.d_mean['E'],
                                common_params.d_sd['E'], 'excitatory',
                                script_rng, simulator_params)
                        else:
                            fixed_total_number_connect_nest(
                                sim, thalamic_population, this_target_pop,
                                k_thal, w_ext, common_params.w_rel * w_ext,
                                common_params.d_mean['E'],
                                common_params.d_sd['E'], simulator_params)
                    elif simulator_params.conn_routine == 'from_list':
                        build_from_list_connect(
                            sim, thalamic_population, this_target_pop,
                            'excitatory', base_neuron_ids, simulator_params)

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

                        if simulator_params.conn_routine == CONN_ROUTINE:
                            if SIMULATOR == SPINNAKER_SIM:
                                fixed_total_number_connect_spinnaker(
                                    sim, this_source_pop, this_target_pop,
                                    k[target_index][source_index], weight,
                                    w_sd, common_params.d_mean[source_pop],
                                    common_params.d_sd[source_pop], conn_type,
                                    script_rng, simulator_params)
                            else:
                                fixed_total_number_connect_nest(
                                    sim, this_source_pop, this_target_pop,
                                    k[target_index][source_index], weight,
                                    w_sd, common_params.d_mean[source_pop],
                                    common_params.d_sd[source_pop],
                                    simulator_params)
                        elif simulator_params.conn_routine == 'from_list':
                            build_from_list_connect(
                                sim, this_source_pop, this_target_pop,
                                conn_type, base_neuron_ids,
                                simulator_params)
