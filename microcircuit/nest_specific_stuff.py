from past.builtins import xrange
from sim_params import NestParams
from constants import DC, NEST_NERUON_MODEL, CONN_ROUTINE
import numpy


class NestSimulatorStuff(NestParams):
    """
    nest specific params.
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
        # ????????
        'corr_detector'
    ]

    def __init__(
            self, parallel_safe=True, n_scaling=1.0, k_scaling=1.0,
            neuron_model=NEST_NERUON_MODEL, conn_routine=CONN_ROUTINE,
            save_connections=False, voltage_input_type='random',
            delay_dist_type='normal', input_type=DC, record_fraction=True,
            n_record=100, frac_record_spikes=1.0, record_v=False,
            n_record_v=20, frac_record_v=0.1, record_corr=False,
            pyseed=2563297, master_seed=124678,
            tau_syn_name='tau_syn_ex'):
        super(NestSimulatorStuff, self).__init__()
        self.parallel_safe = parallel_safe
        self.n_scaling = n_scaling
        self.k_scaling = k_scaling
        self.neuron_model = neuron_model
        self.conn_routine = conn_routine
        self.save_connections = save_connections
        self.voltage_input_type = voltage_input_type
        self.delay_dist_type = delay_dist_type
        self.input_type = input_type
        self.record_fraction = record_fraction
        self.n_record = n_record
        self.frac_record_spikes = frac_record_spikes
        self.record_v = record_v
        self.n_record_v = n_record_v
        self.frac_record_v = frac_record_v
        self.record_corr = record_corr
        self.pyseed = pyseed
        self.master_seed = master_seed
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

    def after_setup_stuff(self, sim):
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

    def after_run_stuff(self, sim, common_params):
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
                                for i in xrange(
                                        len(cov[target_index][source_index])):
                                    f.write(cov[target_index][source_index][i])
                                f.write('')
                f.close()
