# Copyright (c) 2017 Ebrains project and The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os

from .sim_params import NestParams
from .constants import DC, NEST_NEURON_MODEL, CONN_ROUTINE
import numpy
from pyNN.random import NumpyRNG  # type: ignore[import]

# pylint: skip-file


class NestSimulatorInfo(NestParams):
    """
    NEST specific params.
    """

    __slots__ = [
        # Whether to make random numbers independent of the number of processes
        'parallel_safe',
        # Fraction of neurons to simulate
        'n_scaling',
        # Scaling factor for in-degrees. Upon downscaling, synaptic weights are
        # taken proportional to 1/sqrt(in-degree) and external drive is
        # adjusted to preserve mean and variances of activity in the diffusion
        # approximation. In-degrees and weights of both intrinsic and
        # extrinsic inputs are adjusted. This scaling was not part of the
        # original study, but this option is included here to enable
        # simulations on small systems that give results similar to
        # full-scale simulations.
        'k_scaling',
        # Neuron model. Possible values: 'IF_curr_exp', 'iaf_psc_exp_ps'
        'neuron_model',
        # Connection routine
        # 'fixed_total_number' reproduces the connectivity from
        # Potjans & Diesmann (2014), establishing a fixed number of synapses
        # between each pair of populations. This function is available for
        # the NEST and SpiNNaker back-ends. 'from_list' reads in the
        # connections from file
        'conn_routine',
        # Whether to save connections to file. See README.txt for known
        # issues with using  save_connections in parallel simulations.
        'save_connections',
        # Initialization of membrane potentials
        # 'from_list' uses a set of initial neuron voltages read from a file,
        # 'random' uses randomized voltages
        'voltage_input_type',
        # Delay distribution. Possible values: 'normal' and 'uniform'.
        # The original model has normally distributed delays.
        'delay_dist_type',
        # Type of background input. Possible values: 'poisson' and 'DC'
        # If 'DC' is chosen, a constant external current is provided,
        # equal to the mean current due to the Poisson input used in the
        # default version of the model.
        'input_type',
        # Whether to record from a fixed fraction of neurons in each
        # population. If False, a fixed number of neurons is recorded.
        'record_fraction',
        # Number of neurons from which to record spikes
        # when record_fraction = False
        'n_record',
        # Fraction of neurons from which to record spikes
        # when record_fraction = True
        'frac_record_spikes',
        # Whether to record membrane potentials
        # (not yet working for iaf_psc_exp_ps)
        'record_v',
        # Fixed number of neurons from which to record membrane potentials when
        # record_v=True and record_fraction = False
        'n_record_v',
        # Fraction of neurons from which to record membrane potentials when
        # record_v=True and record_fraction = True
        'frac_record_v',
        # Whether to record correlations
        'record_corr',
        # random number generator seeds for V and connectivity.
        # When parallel_safe is True, only the first is used.
        # When parallel_safe is False, the first num_processes are used.
        'pyseed',
        # random number generator seed for NEST Poisson generators
        'master_seed',
        # neuron params
        'neuron_params',
        # tau_syn param name
        'tau_syn_name',
        # correlation detector
        'corr_detector',
        # The RNG to use
        'script_rng'
    ]

    def __init__(
            self, timestep=0.1, sim_duration=10000.0, min_delay=0.1,
            max_delay=100.0, n_nodes=1, outfile='output.txt',
            errfile='errors.txt', output_path='results', output_format='pkl',
            conn_dir='connectivity', n_procs_per_node=24, wall_time='8:0:0',
            memory='4gb',
            mpi_path=(
                '/usr/local/mpi/openmpi/1.4.3/gcc64/bin/'
                'mpivars_openmpi-1.4.3_gcc64.sh'),
            backend_path='/path/to/backend', pynn_path='/path/to/pyNN',
            parallel_safe=True, n_scaling=1.0,
            k_scaling=1.0, neuron_model=NEST_NEURON_MODEL,
            conn_routine=CONN_ROUTINE, save_connections=False,
            voltage_input_type='random', delay_dist_type='normal',
            input_type=DC, record_fraction=True, n_record=100,
            frac_record_spikes=1.0, record_v=False, n_record_v=20,
            frac_record_v=0.1, record_corr=False, pyseed=2563297,
            master_seed=124678, tau_syn_name='tau_syn_ex'):
        super(NestSimulatorInfo, self).__init__(
            timestep, sim_duration, min_delay, max_delay, n_nodes, outfile,
            errfile, output_path, output_format, conn_dir, n_procs_per_node,
            wall_time, memory, mpi_path, backend_path, pynn_path)
        self.parallel_safe = bool(parallel_safe)
        self.n_scaling = float(n_scaling)
        self.k_scaling = float(k_scaling)
        self.neuron_model = neuron_model
        self.conn_routine = conn_routine
        self.save_connections = bool(save_connections)
        self.voltage_input_type = voltage_input_type
        self.delay_dist_type = delay_dist_type
        self.input_type = input_type
        self.record_fraction = bool(record_fraction)
        self.n_record = int(n_record)
        self.frac_record_spikes = float(frac_record_spikes)
        self.record_v = bool(record_v)
        self.n_record_v = int(n_record_v)
        self.frac_record_v = float(frac_record_v)
        self.record_corr = bool(record_corr)
        self.pyseed = int(pyseed)
        self.master_seed = int(master_seed)
        self.tau_syn_name = tau_syn_name
        self.neuron_params = {
            'C_m': 250.0,  # pF
            'I_e': 0.0,  # pA
            'tau_m': 10.0,  # ms
            't_ref': 2.0,  # ms
            'tau_syn_ex': 0.5,  # ms
            'tau_syn_in': 0.5,  # ms
            'V_reset': -65.0,  # mV
            'E_L': -65.0,  # mV
            'V_th': -50.0  # mV
        }
        self.corr_detector = None

        # if parallel_safe=False, PyNN offsets the seeds by 1 for each rank
        self.script_rng = NumpyRNG(
            seed=self.pyseed, parallel_safe=self.parallel_safe)

    def after_setup_info(self, sim):
        n_vp = sim.nest.GetKernelStatus('total_num_virtual_procs')
        if sim.rank() == 0:
            print('n_vp: ', n_vp)
            print('master_seed: ', self.master_seed)
        sim.nest.SetKernelStatus(
            {'print_time': False, 'dict_miss_is_error': False,
             'grng_seed': self.master_seed,
             'rng_seeds': range(
                 self.master_seed + 1,
                 self.master_seed + n_vp + 1)})

    def after_run_info(self, sim, common_params):
        if self.record_corr:
            if sim.nest.GetStatus(self.corr_detector, 'local')[0]:
                print('getting count_covariance on rank ', sim.rank())
                cov_all = (sim.nest.GetStatus(
                    self.corr_detector, 'count_covariance')[0])
                delta_tau = sim.nest.GetStatus(
                    self.corr_detector, 'delta_tau')[0]

                cov = {}
                for target_layer in numpy.sort(common_params.layers.keys()):
                    for target_pop in common_params.pops:
                        target_index = (
                            common_params.structure[target_layer][target_pop])
                        cov[target_index] = {}
                        for source_layer in numpy.sort(
                                common_params.layers.keys()):
                            for source_pop in common_params.pops:
                                source_index = (
                                    common_params.structure[
                                        source_layer][source_pop])
                                cov[target_index][source_index] = (
                                    numpy.array(list(
                                        cov_all[target_index][source_index][
                                            ::-1])
                                             + list(
                                        cov_all[source_index][target_index][
                                            1:])))

                f = open(self.output_path + '/covariances.dat', 'w')
                f.write('tau_max: {}'.format(common_params.tau_max))
                f.write('delta_tau: {}'.format(delta_tau))
                f.write('simtime: {}\n'.format(self.sim_duration))

                for target_layer in numpy.sort(common_params.layers.keys()):
                    for target_pop in common_params.pops:
                        target_index = (
                            common_params.structure[target_layer][target_pop])
                        for source_layer in numpy.sort(
                                common_params.layers.keys()):
                            for source_pop in common_params.pops:
                                source_index = (
                                    common_params.structure[source_layer][
                                        source_pop])
                                f.write("{}{} - {}{}".format(
                                    target_layer, target_pop, source_layer,
                                    source_pop))
                                f.write('n_events_target: {}'.format(
                                    sim.nest.GetStatus(
                                        self.corr_detector,
                                        'n_events')[0][target_index]))
                                f.write('n_events_source: {}'.format(
                                    sim.nest.GetStatus(
                                        self.corr_detector,
                                        'n_events')[0][source_index]))
                                for i in range(
                                        len(cov[target_index][source_index])):
                                    f.write(cov[target_index][source_index][i])
                                f.write('')
                f.close()

    def record_corr_info(self, sim, common_params):
        if self.record_corr:
            # Create correlation recording device
            sim.nest.SetDefaults(
                'correlomatrix_detector', {'delta_tau': 0.5})
            self.corr_detector = (sim.nest.Create('correlomatrix_detector'))
            sim.nest.SetStatus(
                self.corr_detector,
                {'N_channels': (
                    common_params.n_layers * common_params.n_pops_per_layer),
                 'tau_max': common_params.tau_max,
                 'Tstart': common_params.tau_max})

    def rank_info(self, common_params, layer, pop):
        if (not self.record_fraction and self.n_record > int(
                round(common_params.n_full[layer][pop] * self.n_scaling))):
            print(
                'Note that requested number of neurons '
                'to record exceeds {} {} population '
                'size'.format(layer, pop))

    def set_record_v(self, this_pop):
        if self.record_fraction:
            n_rec_v = round(this_pop.size * self.frac_record_v)
        else:
            n_rec_v = self.n_record_v
        if self.neuron_model == NEST_NEURON_MODEL:
            this_pop.celltype.recordable = ['V_m', 'spikes']
            this_pop[0: n_rec_v]._record('V_m')
        else:
            this_pop[0: n_rec_v].record_v()

    def set_corr_recording(
            self, layer, pop, common_params, sim, this_pop):
        if self.record_corr:
            index = common_params.structure[layer][pop]
            sim.nest.SetDefaults(
                'static_synapse', {'receptor_type': index})
            sim.nest.ConvergentConnect(
                list(this_pop.all_cells), self.corr_detector)

    def set_defaults(self, sim):
        if self.record_corr:
            # reset receptor_type
            sim.nest.SetDefaults('static_synapse', {'receptor_type': 0})

    def create_neural_population(self, sim, n_neurons, layer, pop):
        from pyNN.nest import native_cell_type  # type: ignore[import]
        model = native_cell_type('iaf_psc_exp_ps')
        return sim.Population(
            int(round(n_neurons * self.n_scaling)),
            model, cellparams=self.neuron_params,
            label=layer+pop)

    @staticmethod
    def create_poissons(
            sim, target_layer, target_pop, rate, this_target_pop,
            w_ext, common_params):
        # create only a single Poisson generator for
        # each population, since the native NEST implementation
        # sends independent spike trains to all targets
        if sim.rank() == 0:
            print('connecting Poisson generator to {} {} '
                  'via SLI'.format(target_layer, target_pop))
        sim.nest.sli_run(
            '/poisson_generator Create /poisson_generator_e '
            'Set poisson_generator_e << /rate ' + str(rate) + ' >> SetStatus')
        sim.nest.sli_run(
            "poisson_generator_e " + str(list(
                this_target_pop.all_cells)).replace(',', '')
            + " [" + str(1000 * w_ext) + "] [" +
            str(common_params.d_mean['E']) + "] DivergentConnect")

    def fixed_tot_number_connect(
            self, sim, pop1, pop2, k, w_mean, w_sd, d_mean, d_sd):
        """
        Function connecting two populations with multapses and a fixed
        total number of synapses Using new NEST implementation of Connect

        :param sim:
        :param pop1:
        :param pop2:
        :param k:
        :param w_mean:
        :param w_sd:
        :param d_mean:
        :param d_sd:
        :return:
        """

        if not k:
            return

        source_neurons = list(pop1.all_cells)
        target_neurons = list(pop2.all_cells)
        n_syn = int(round(k * len(target_neurons)))
        # weights are multiplied by 1000 because NEST uses pA whereas PyNN
        # uses nA RandomPopulationConnectD is called on each process with the
        # full sets of source and target neurons, and internally only
        # connects the target neurons on the current process.

        conn_dict = {'rule': 'fixed_total_number',
                     'N': n_syn}

        syn_dict = {'model': 'static_synapse',
                    'weight': {
                        'distribution': 'normal_clipped',
                        'mu': 1000. * w_mean,
                        'sigma': 1000. * w_sd},
                    'delay': {
                        'distribution': 'normal_clipped',
                        'low': self.min_delay,
                        'mu': d_mean,
                        'sigma': d_sd}}
        if w_mean > 0:
            syn_dict['weight']['low'] = 0.0
        if w_mean < 0:
            syn_dict['weight']['high'] = 0.0

        sim.nest.sli_push(source_neurons)
        sim.nest.sli_push(target_neurons)
        sim.nest.sli_push(conn_dict)
        sim.nest.sli_push(syn_dict)
        sim.nest.sli_run("Connect")

        if self.save_connections:
            # - weights are in pA
            # - no header lines
            # - one file for each MPI process
            # - GIDs

            # get connections to target on this MPI process
            conn = sim.nest.GetConnections(
                source=source_neurons, target=target_neurons)
            conns = sim.nest.GetStatus(
                conn, ['source', 'target', 'weight', 'delay'])
            if not os.path.exists(self.conn_dir):
                try:
                    os.makedirs(self.conn_dir)
                except OSError as e:
                    if e.errno != 17:
                        raise
                    pass
            f = open(
                "{}/{}_{}'.conn{}".format(
                    self.conn_dir, pop1.label, pop2.label, str(sim.rank())),
                'w')
            for c in conns:
                f.write(
                    str(c).replace('(', '').replace(')', '').replace(
                        ', ', '\t'))
            f.close()

    @staticmethod
    def memory_print(sim):
        # determine memory consumption
        sim.nest.sli_run("memory_thisjob")
        print('memory usage after simulation:', sim.nest.sli_pop(), 'kB')
