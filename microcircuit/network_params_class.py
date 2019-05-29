from collections import defaultdict
import math


class NetworkParams(object):

    __slots__ = [
        # Whether to make random numbers independent of the number of processes
        '_parallel_safe',
        # Fraction of neurons to simulate
        '_n_scaling',
        # Scaling factor for in-degrees. Upon downscaling, synaptic weights are
        # taken proportional to 1/sqrt(in-degree) and external drive is
        # adjusted to preserve mean and variances of activity in the diffusion
        # approximation. In-degrees and weights of both intrinsic and
        # extrinsic inputs are adjusted. This scaling was not part of the
        # original study, but this option is included here to enable
        # simulations on small systems that give results similar to
        # full-scale simulations.
        '_k_scaling',
        # Neuron model. Possible values: 'IF_curr_exp', 'iaf_psc_exp_ps'
        '_neuron_model',
        # Connection routine
        # 'fixed_total_number' reproduces the connectivity from
        # Potjans & Diesmann (2014), establishing a fixed number of synapses
        # between each pair of populations. This function is available for
        # the NEST and SpiNNaker back-ends. 'from_list' reads in the
        # connections from file
        '_conn_routine',
        # Whether to save connections to file. See README.txt for known issues
        # with using save_connections in parallel simulations.
        '_save_connections',
        # Initialization of membrane potentials
        # 'from_list' uses a set of initial neuron voltages read from a file,
        # 'random' uses randomized voltages
        '_voltage_input_type',
        # Delay distribution. Possible values: 'normal' and 'uniform'.
        # The original model has normally distributed delays.
        '_delay_dist_type',
        # Type of background input. Possible values: 'poisson' and 'DC' If
        # 'DC' is chosen, a constant external current is provided, equal to the
        #  mean current due to the Poisson input used in the default version
        # of the model.
        '_input_type',
        # Whether to record from a fixed fraction of neurons in each
        # population. If False, a fixed number of neurons is recorded.
        '_record_fraction',
        # Number of neurons from which to record spikes when record_fraction
        # = False
        '_n_record',
        # Fraction of neurons from which to record spikes when record_fraction
        # = True
        '_frac_record_spikes',
        # Whether to record membrane potentials (not yet working for
        #  iaf_psc_exp_ps)
        '_record_v',
        # Fixed number of neurons from which to record membrane potentials when
        # record_v=True and record_fraction = False
        '_n_record_v',
        # Fraction of neurons from which to record membrane potentials when
        # record_v=True and record_fraction = True
        '_frac_record_v',
        # Whether to record correlations
        '_record_corr',
        # random number generator seeds for V and connectivity.
        # When parallel_safe is True, only the first is used.
        # When parallel_safe is False, the first num_processes are used.
        '_pyseed',
        # random number generator seed for NEST Poisson generators
        '_master_seed',
        "_input_dir",
        "_live_output",
        # neuron params for the neuron model
        "_neuron_params",
        # enum maybe????
        "_layers",
        # enum maybe?????
        "_pops",
        # enum maybe????
        "_structure",
        # Numbers of neurons in full-scale model
        "_n_full",
        # Probabilities for >=1 connection between neurons in the given
        # populations. The first index is for the target population; the
        # second for the source population
        "_conn_probs",
        # In-degrees for external inputs
        "_k_ext",
        # Mean rates in the full-scale model, necessary for scaling Precise
        # values differ somewhat between network realizations
        "_full_mean_rates",
        # Mean of initial membrane potential distribution
        "_v0_mean",
        # standard deviation of initial membrane potential distribution
        "_v0_sd",
        # Background rate per synapse
        "_bg_rate",
        # Mean synaptic weight for all excitatory projections except L4e->L2/3e
        "_w_mean",
        # Mean synaptic weight for L4e->L2/3e connections
        #  See p. 801 of the paper, second paragraph under
        # 'Model Parameterization', and the caption to Supplementary Fig. 7
        "_w_234",
        # Standard deviation of weight distribution relative to mean for all
        # projections except L4e->L2/3e
        "_w_rel",
        # Standard deviation of weight distribution relative to mean for
        # L4e->L2/3e This value is not mentioned in the paper, but is chosen
        # to match the original code by Tobias Potjans
        "_w_rel_234",
        # Means of delays from given source populations (ms) When
        # delay_dist_type is 'uniform', delays are drawn from
        # [d_mean-d_sd, d_mean+d_sd].
        "_d_mean",
        # standard deviations of delays from given source populations (ms)
        # When delay_dist_type is 'uniform', delays are drawn from
        # [d_mean-d_sd, d_mean+d_sd].
        "_d_sd",
        # Parameters for transient thalamic input
        "_thalamic_input",
        #  ????
        "_thal_params",
        # Maximum delay over which to determine covariances
        "_tau_max",
        # Parameters for plots of spiking activity
        "_plot_spiking_activity",
        # raster_t_min and raster_t_max include the time scaling factor
        "_raster_t_min",
        # ????
        "_raster_t_max",
        # Fraction of recorded neurons to include in raster plot
        "_frac_to_plot",
        # Numbers of neurons from which to record spikes
        "_n_rec"
    ]

    def __init__(self, sim_name, sim_params):
        self._neuron_params = dict()

        if sim_name == "nest":
            self._parallel_safe = True
            self._n_scaling = 1.0
            self._k_scaling = 1.0
            self._neuron_model = 'iaf_psc_exp_ps'
            self._conn_routine = 'fixed_total_number'
            self._save_connections = False
            self._voltage_input_type = 'random'
            self._delay_dist_type = 'normal'
            self._input_type = 'DC'
            self._record_fraction = True
            self._n_record = 100
            self._frac_record_spikes = 1.0
            self._record_v = False
            self._n_record_v = 20
            self._frac_record_v = 0.1
            self._record_corr = False
            self._pyseed = 2563297
            self._master_seed = 124678
            self._input_dir = None
            self._live_output = False
        else:
            self._parallel_safe = True
            self._n_scaling = 1.0
            self._k_scaling = 1.0
            self._neuron_model = 'IF_curr_exp'
            self._conn_routine = 'fixed_total_number'
            self._save_connections = False
            self._voltage_input_type = 'random'
            self._input_dir = 'voltages_0.1_0.1_delays'
            self._delay_dist_type = 'normal'
            self._input_type = 'Poisson'
            self._record_fraction = True
            self._n_record = 100
            self._frac_record_spikes = 1.0
            self._record_v = False
            self._pyseed = 2563297
            self._live_output = False

        if self._neuron_model == 'iaf_psc_exp_ps':
            self._neuron_params["C_m"] = 250.0  # pF
            self._neuron_params["I_e"] = 0.0   # pA
            self._neuron_params["tau_m"] = 10.0  # ms
            self._neuron_params["t_ref"] = 2.0  # ms
            self._neuron_params["tau_syn_ex"] = 0.5  # ms
            self._neuron_params["tau_syn_in"] = 0.5  # ms
            self._neuron_params["V_reset"] = -65.0  # mV
            self._neuron_params["E_L"] = -65.0  # mV
            self._neuron_params["V_th"] = -50.0  # mV
        else:
            self._neuron_params["cm"] = 0.25  # nF
            self._neuron_params["i_offset"] = 0.0  # nA
            self._neuron_params["tau_m"] = 10.0  # ms
            self._neuron_params["tau_refrac"] = 2.0  # ms
            self._neuron_params["tau_syn_E"] = 0.5  # ms
            self._neuron_params["tau_syn_I"] = 0.5  # ms
            self._neuron_params["v_reset"] = -65.0  # mV
            self._neuron_params["v_rest"] = -65.0  # mV
            self._neuron_params["v_thresh"] = -50.0  # mV

        self._layers = dict()
        self._layers['L23'] = 0
        self._layers['L4'] = 1
        self._layers['L5'] = 2
        self._layers["L6"] = 3

        self._pops = dict()
        self._pops['E'] = 0
        self._pops['I'] = 1

        self._structure = defaultdict(dict)
        self._structure['L23']['E'] = 0
        self._structure['L23']['I'] = 1
        self._structure['L4']['E'] = 2
        self._structure['L4']['I'] = 3
        self._structure['L5']['E'] = 4
        self._structure['L5']['I'] = 5
        self._structure['L6']['E'] = 6
        self._structure['L6']['I'] = 7

        self._n_full = defaultdict(dict)
        self._n_full['L23']['E'] = 20683
        self._n_full['L23']['I'] = 5834
        self._n_full['L4']['E'] = 21915
        self._n_full['L4']['I'] = 5479
        self._n_full['L5']['E'] = 4850
        self._n_full['L5']['I'] = 1065
        self._n_full['L6']['E'] = 14395
        self._n_full['L6']['I'] = 2948

        self._conn_probs = [
            # 2/3e    2/3i      4e      4i      5e      5i      6e      6i
            [0.1009,  0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.],
            [0.1346,  0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.],
            [0.0077,  0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.],
            [0.0691,  0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.],
            [0.1004,  0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.],
            [0.0548,  0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.],
            [0.0156,  0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
            [0.0364,  0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]]

        self._k_ext = defaultdict(dict)
        self._k_ext['L23']['E'] = 1600
        self._k_ext['L23']['I'] = 1500
        self._k_ext['L4']['E'] = 2100
        self._k_ext['L4']['I'] = 1900
        self._k_ext['L5']['E'] = 2000
        self._k_ext['L5']['I'] = 1900
        self._k_ext['L6']['E'] = 2900
        self._k_ext['L6']['I'] = 2100

        self._full_mean_rates = defaultdict(dict)
        self._full_mean_rates['L23']['E'] = 0.971
        self._full_mean_rates['L23']['I'] = 2.868
        self._full_mean_rates['L4']['E'] = 4.746
        self._full_mean_rates['L4']['I'] = 5.396
        self._full_mean_rates['L5']['E'] = 8.142
        self._full_mean_rates['L5']['I'] = 9.078
        self._full_mean_rates['L6']['E'] = 0.991
        self._full_mean_rates['L6']['I'] = 7.523

        self._v0_mean = -58.0  # mV
        self._v0_sd = 5.0  # mV
        self._bg_rate = 8.0  # spikes/s
        self._w_mean = 87.8e-3  # nA
        self._w_234 = 2 * self._w_mean  # nA
        self._w_rel = 0.1
        self._w_rel_234 = 0.05

        self._d_mean = dict()
        self._d_mean['E'] = 1.5
        self._d_mean['I'] = 0.75

        self._d_sd = dict()
        self._d_sd['E'] = 0.75
        self._d_sd['I'] = 0.375

        self._thalamic_input = False

        self._thal_params = dict()
        self._thal_params['n_thal'] = 902
        self._thal_params['rate'] = 120.0  # spikes/s
        self._thal_params['start'] = 300.0  # ms
        self._thal_params['duration'] = 10.0  # ms
        self._thal_params['C'] = defaultdict(dict)
        self._thal_params['C']['L23']['E'] = 0
        self._thal_params['C']['L23']['I'] = 0
        self._thal_params['C']['L4']['E'] = 0.0983
        self._thal_params['C']['L4']['I'] = 0.0619
        self._thal_params['C']['L5']['E'] = 0
        self._thal_params['C']['L5']['I'] = 0
        self._thal_params['C']['L6']['E'] = 0.0512
        self._thal_params['C']['L6']['I'] = 0.0196

        self._tau_max = 100.0
        self._plot_spiking_activity = True
        self._raster_t_min = 0  # ms
        self._raster_t_max = sim_params.sim_duration
        self._frac_to_plot = 0.5

        self._n_rec = defaultdict(dict)
        for layer in self._layers:
            for pop in self._pops:
                if self._record_fraction:
                    self._n_rec[layer][pop] = min(
                        int(round(self._n_full[layer][pop] * self._n_scaling *
                                  self._frac_record_spikes)),
                        int(round(self._n_full[layer][pop] * self._n_scaling)))

    @property
    def parallel_safe(self):
        return self._parallel_safe

    @property
    def n_scaling(self):
        return self._n_scaling

    @property
    def k_scaling(self):
        return self._k_scaling

    @property
    def neuron_model(self):
        return self._neuron_model

    @property
    def conn_routine(self):
        return self._conn_routine

    @property
    def save_connections(self):
        return self._save_connections

    @property
    def voltage_input_type(self):
        return self._voltage_input_type

    @property
    def delay_dist_type(self):
        return self._delay_dist_type

    @property
    def input_type(self):
        return self._input_type

    @property
    def record_fraction(self):
        return self._record_fraction

    @property
    def n_record(self):
        return self._n_record

    @property
    def frac_record_spikes(self):
        return self._frac_record_spikes

    @property
    def record_v(self):
        return self._record_v

    @property
    def n_record_v(self):
        return self._n_record_v

    @property
    def frac_record_v(self):
        return self._frac_record_v

    @property
    def record_corr(self):
        return self._record_corr

    @property
    def pyseed(self):
        return self._pyseed

    @property
    def master_seed(self):
        return self._master_seed

    @property
    def input_dir(self):
        return self._input_dir

    @property
    def live_output(self):
        return self._live_output

    @property
    def neuron_params(self):
        return self._neuron_params

    @property
    def layers(self):
        return self._layers

    @property
    def n_layers(self):
        return len(self._layers)

    @property
    def pops(self):
        return self._pops

    @property
    def n_pops_per_layer(self):
        return len(self._pops)

    @property
    def structure(self):
        return self._structure

    @property
    def n_full(self):
        return self._n_full

    @property
    def conn_probs(self):
        return self._conn_probs

    @property
    def k_ext(self):
        return self._k_ext

    @property
    def full_mean_rates(self):
        return self._full_mean_rates

    @property
    def v0_mean(self):
        return self._v0_mean

    @property
    def v0_sd(self):
        return self._v0_sd

    @property
    def bg_rate(self):
        return self._bg_rate

    @property
    def w_mean(self):
        return self._w_mean

    @property
    def w_234(self):
        return self._w_234

    @property
    def w_rel(self):
        return self._w_rel

    @property
    def w_rel_234(self):
        return self._w_rel_234

    @property
    def d_mean(self):
        return self._d_mean

    @property
    def d_sd(self):
        return self._d_sd

    @property
    def thalamic_input(self):
        return self._thalamic_input

    @property
    def thal_params(self):
        return self._thal_params

    @property
    def tau_max(self):
        return self._tau_max

    @property
    def plot_spiking_activity(self):
        return self._plot_spiking_activity

    @property
    def raster_t_min(self):
        return self._raster_t_min

    @property
    def raster_t_max(self):
        return self._raster_t_max

    @property
    def n_rec(self):
        return self._n_rec
