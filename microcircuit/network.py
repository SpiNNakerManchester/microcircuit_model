from network_params import *
from sim_params import *
from scaling import get_indegrees, adjust_w_and_ext_to_K
from connectivity import *
from helper_functions import create_weight_matrix, get_init_voltages_from_file
import pyNN
from pyNN.random import NumpyRNG, RandomDistribution
import numpy as np
import sys
if simulator == 'nest' and neuron_model == 'iaf_psc_exp_ps':
    from pyNN.nest import native_cell_type

class Network:

    def __init__(self, sim):
        return None

    def setup(self,sim):

        # if parallel_safe=False, PyNN offsets the seeds by 1 for each rank
        script_rng = NumpyRNG(seed=pyseed, parallel_safe=parallel_safe)

        # Compute DC input before scaling
        if input_type == 'DC':
            self.DC_amp = {}
            for target_layer in layers:
                self.DC_amp[target_layer] = {}
                for target_pop in pops:
                    if neuron_model == 'iaf_psc_exp_ps':
                        tau_syn = neuron_params['tau_syn_ex']
                    else:
                        tau_syn = neuron_params['tau_syn_E']
                    self.DC_amp[target_layer][target_pop] = bg_rate * \
                    K_ext[target_layer][target_pop] * w_mean * tau_syn / 1000.
        else:
            self.DC_amp = {'L23': {'E':0., 'I':0.},
                           'L4' : {'E':0., 'I':0.},
                           'L5' : {'E':0., 'I':0.},
                           'L6' : {'E':0., 'I':0.}}

        # In-degrees of the full-scale and scaled models
        K_full = get_indegrees()
        self.K = K_scaling * K_full

        self.K_ext = {}
        for layer in layers:
            self.K_ext[layer] = {}
            for pop in pops:
                self.K_ext[layer][pop] = K_scaling * K_ext[layer][pop]

        self.w = create_weight_matrix('IF_curr_exp')
        # Network scaling
        if K_scaling != 1:
            self.w, self.w_ext, self.DC_amp = adjust_w_and_ext_to_K(K_full, K_scaling, self.w, self.DC_amp)
        else:
            self.w_ext = w_mean

        # Initial membrane potential distribution
#         V_dist = RandomDistribution('normal', [V0_mean, V0_sd], rng=script_rng)


        # Improved initialisation from Julich
        V_dist={}

        V_dist['L23E'] = RandomDistribution('normal', [-64.28, 4.36], rng=script_rng)
        V_dist['L23I'] = RandomDistribution('normal', [-59.16, 3.57], rng=script_rng)

        V_dist['L4E'] = RandomDistribution('normal', [-59.33, 3.74], rng=script_rng)
        V_dist['L4I'] = RandomDistribution('normal', [-59.45, 3.94], rng=script_rng)

        V_dist['L5E'] = RandomDistribution('normal', [-59.11, 3.94], rng=script_rng)
        V_dist['L5I'] = RandomDistribution('normal', [-57.66, 3.55], rng=script_rng)

        V_dist['L6E'] = RandomDistribution('normal', [-62.72, 4.46], rng=script_rng)
        V_dist['L6I'] = RandomDistribution('normal', [-57.43, 3.48], rng=script_rng)

        if neuron_model == 'iaf_psc_exp_ps':
            model = native_cell_type('iaf_psc_exp_ps')
        else:
            model = getattr(sim, neuron_model)

        if simulator == 'nest':
            if record_corr:
                # Create correlation recording device
                sim.nest.SetDefaults('correlomatrix_detector', {'delta_tau': 0.5})
                self.corr_detector = sim.nest.Create('correlomatrix_detector')
                sim.nest.SetStatus(self.corr_detector, {'N_channels': n_layers*n_pops_per_layer, \
                                                        'tau_max': tau_max, 'Tstart': tau_max})


        if sim.rank() == 0:
            print 'neuron_params:', neuron_params
            print 'K: ', self.K
            print 'K_ext: ', self.K_ext
            print 'w: ', self.w
            print 'w_ext: ', self.w_ext
            print 'DC_amp: ', self.DC_amp
            print 'n_rec:'
            for layer in sorted(layers):
                for pop in sorted(pops):
                    print layer, pop, n_rec[layer][pop]
                    if simulator == 'nest':
                        if not record_fraction and n_record > int(round(N_full[layer][pop] * N_scaling)):
                            print 'Note that requested number of neurons to record exceeds ', \
                                   layer, pop, ' population size'


        # Create cortical populations
        self.pops = {}
        global_neuron_id = 1
        self.base_neuron_ids = {}
        for layer in sorted(layers):
            self.pops[layer] = {}
            for pop in sorted(pops):
                self.pops[layer][pop] = sim.Population( \
                    int(round(N_full[layer][pop] * N_scaling)), \
                    model, cellparams=neuron_params, label=layer+pop)
                this_pop = self.pops[layer][pop]

                # Provide DC input
                if neuron_model == 'IF_curr_exp':
                    this_pop.set(i_offset=self.DC_amp[layer][pop])
                if neuron_model == 'iaf_psc_exp_ps':
                    this_pop.set(I_e=1000*self.DC_amp[layer][pop])

                self.base_neuron_ids[this_pop] = global_neuron_id
                global_neuron_id += len(this_pop) + 2

                if voltage_input_type == 'random':
                    this_pop.initialize(v=V_dist)
                elif voltage_input_type == 'pop_random':
                    this_pop.initialize(v=V_dist[this_pop.label])
                elif voltage_input_type == 'from_list':
                    this_pop.initialize(v=get_init_voltages_from_file(this_pop))

                # Spike recording
                this_pop[0:n_rec[layer][pop]].record("spikes")

                # Membrane potential recording
                if record_v:
                    if simulator == 'spiNNaker':
                        this_pop.record(["v", "gsyn_exc", "gsyn_inh", 'synapse'])#,
#                                         indexes=range(0, 1000, 64))
                    else:
                        if record_fraction:
                            n_rec_v = round(this_pop.size * frac_record_v)
                        else :
                            n_rec_v = n_record_v
                        if neuron_model == 'iaf_psc_exp_ps':
                            this_pop.celltype.recordable = ['V_m', 'spikes']
                            this_pop[0 : n_rec_v]._record('V_m')
                        else:
                            this_pop[0 : n_rec_v].record("v")

                # Correlation recording
                if simulator == 'nest':
                    if record_corr:
                        index = structure[layer][pop]
                        sim.nest.SetDefaults('static_synapse', {'receptor_type': index})
                        sim.nest.ConvergentConnect(list(this_pop.all_cells), self.corr_detector)


        if simulator == 'nest':
            if record_corr:
                # reset receptor_type
                sim.nest.SetDefaults('static_synapse', {'receptor_type': 0})

        if thalamic_input:
        # Create thalamic population
            self.thalamic_population = sim.Population( \
                thal_params['n_thal'], sim.SpikeSourcePoisson, {'rate': thal_params['rate'], \
                'start': thal_params['start'], 'duration': thal_params['duration']}, \
                label='thalamic_population')
            self.base_neuron_ids[self.thalamic_population] = global_neuron_id
            global_neuron_id += len(self.thalamic_population) + 2

        possible_targets = ['inhibitory', 'excitatory']

        # Connect

        for target_layer in sorted(layers):
            for target_pop in sorted(pops):
                target_index = structure[target_layer][target_pop]
                this_target_pop = self.pops[target_layer][target_pop]
                w_ext = self.w_ext
                # External inputs
                if input_type == 'poisson':
                    rate = bg_rate * self.K_ext[target_layer][target_pop]

                    if simulator == 'nest':
                    # create only a single Poisson generator for each population,
                    # since the native NEST implementation sends independent spike trains to all targets
                        if sim.rank() == 0:
                            print 'connecting Poisson generator to', target_layer, target_pop, ' via SLI'
                        sim.nest.sli_run('/poisson_generator Create /poisson_generator_e Set poisson_generator_e << /rate ' \
                            + str(rate) + ' >> SetStatus')
                        sim.nest.sli_run("poisson_generator_e " + str(list(this_target_pop.all_cells)).replace(',', '') \
                            + " [" + str(1000 * w_ext) + "] [" + str(d_mean['E']) + "] DivergentConnect")
                    else:
                        if sim.rank() == 0:
                            print 'connecting Poisson generators to', target_layer, target_pop
                        poisson_generator = sim.Population(this_target_pop.size, \
                            sim.SpikeSourcePoisson, {'rate': rate})
                        conn = sim.OneToOneConnector()
                        syn = sim.StaticSynapse(weight=w_ext)
                        sim.Projection(poisson_generator, this_target_pop, conn, syn, receptor_type='excitatory')

                if thalamic_input:
                    # Thalamic inputs
                    if sim.rank() == 0:
                        print 'creating thalamic connections to ' + target_layer + target_pop
                    C_thal = thal_params['C'][target_layer][target_pop]
                    n_target = N_full[target_layer][target_pop]
                    K_thal = round(np.log(1 - C_thal) / np.log((n_target * thal_params['n_thal'] - 1.)/ \
                             (n_target * thal_params['n_thal']))) / n_target * K_scaling
                    if conn_routine == 'fixed_total_number':
                        if simulator == 'spiNNaker':
                            FixedTotalNumberConnect_SpiNNaker(sim, self.thalamic_population, \
                                this_target_pop, K_thal, w_ext, w_rel * w_ext, \
                                d_mean['E'], d_sd['E'], 'excitatory', script_rng)
                        else:
                            FixedTotalNumberConnect_NEST(sim, self.thalamic_population, \
                                this_target_pop, K_thal, w_ext, w_rel * w_ext, \
                                d_mean['E'], d_sd['E'])
                    elif conn_routine == 'from_list':
                        FromListConnect(sim, self.thalamic_population, this_target_pop, 'excitatory', self.base_neuron_ids)

                # Recurrent inputs
                for source_layer in sorted(layers):
                    for source_pop in sorted(pops):
                        source_index = structure[source_layer][source_pop]
                        this_source_pop = self.pops[source_layer][source_pop]
                        weight = self.w[target_index][source_index]

                        conn_type = possible_targets[int((np.sign(weight)+1)/2)]

                        if sim.rank() == 0:
                            print 'creating connections from ' + source_layer + \
                            source_pop + ' to ' + target_layer + target_pop

                        if source_pop == 'E' and source_layer == 'L4' and target_layer == 'L23' and target_pop == 'E':
                            w_sd = weight * w_rel_234
                        else:
                            w_sd = abs(weight * w_rel)

                        if conn_routine == 'fixed_total_number':
                            if simulator == 'spiNNaker':
                                FixedTotalNumberConnect_SpiNNaker( \
                                    sim, this_source_pop, this_target_pop, \
                                    self.K[target_index][source_index], weight, w_sd, \
                                    d_mean[source_pop], d_sd[source_pop], conn_type, script_rng)
                            else :
                                FixedTotalNumberConnect_NEST( \
                                    sim, this_source_pop, this_target_pop, \
                                    self.K[target_index][source_index], weight, w_sd, \
                                    d_mean[source_pop], d_sd[source_pop])
                        elif conn_routine == 'from_list' :
                            FromListConnect(sim, this_source_pop, this_target_pop, conn_type, self.base_neuron_ids)
