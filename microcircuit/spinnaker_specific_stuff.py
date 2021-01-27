from sim_params import SpinnakerParams
from constants import POISSON, SPINNAKER_NEURON_MODEL, CONN_ROUTINE


class SpinnakerSimulatorStuff(SpinnakerParams):
    """
    spinnaker specific params.
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
        # Fraction of neurons from which to record membrane potentials when
        # record_v=True and record_fraction = True
        'frac_record_v',
        # random number generator seeds for V and connectivity.
        # When parallel_safe is True, only the first is used.
        # When parallel_safe is False, the first num_processes are used.
        'pyseed',
        # Whether to send output live
        'live_output',
        # ???????????
        'input_dir',
        # neuron params
        'neuron_params',
        # tau syn name
        'tau_syn_name'
    ]

    def __init__(
            self, parallel_safe=True, n_scaling=1.0, k_scaling=1.0,
            neuron_model=SPINNAKER_NEURON_MODEL, conn_routine=CONN_ROUTINE,
            save_connections=False, voltage_input_type='pop_random',
            delay_dist_type='normal', input_dir='voltages_0.1_0.1_delays',
            input_type=POISSON, record_fraction=True,
            n_record=100, frac_record_spikes=1.0, record_v=False,
            frac_record_v=0.1, pyseed=2563297, live_output=False,
            tau_syn_name='tau_syn_E'):
        super(SpinnakerSimulatorStuff, self).__init__()
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
        self.frac_record_v = frac_record_v
        self.pyseed = pyseed
        self.live_output = live_output
        self.input_dir = input_dir
        self.tau_syn_name = tau_syn_name
        self.neuron_params = {
            'cm': 0.25,  # nF
            'i_offset': 0.0,   # nA
            'tau_m': 10.0,  # ms
            'tau_refrac': 2.0,   # ms
            'tau_syn_E': 0.5,   # ms
            'tau_syn_I': 0.5,   # ms
            'v_reset': -65.0,  # mV
            'v_rest': -65.0,  # mV
            'v_thresh': -50.0  # mV
        }

    @staticmethod
    def after_setup_stuff(sim):
        neurons_per_core = 255
        sim.set_number_of_neurons_per_core(
            sim.IF_curr_exp, neurons_per_core)
        sim.set_number_of_neurons_per_core(
            sim.SpikeSourcePoisson, neurons_per_core)