

class CommonParams(object):
    """
    common params for all sims
    """

    __slots__ = [
        # Relative inhibitory synaptic weight
        'g',
        'layers',
        'n_layers',
        'pops',
        'n_pops_per_layer',
        'structure',
        # Numbers of neurons in full-scale model
        'n_full',
        # Probabilities for >=1 connection between neurons in the given
        # populations. The first index is for the target population; the
        # second for the source population
        # 2/3e, 2/3i, 4e, 4i, 5e, 5i, 6e, 6i
        'conn_probs',
        # In-degrees for external inputs
        'k_ext',
        # Mean rates in the full-scale model, necessary for scaling
        # Precise values differ somewhat between network realizations
        'full_mean_rates',
        # Mean and standard deviation of initial membrane potential
        # distribution
        'v0_mean',
        'v0_sd',
        'v0_l23e_mean',
        'v0_l23e_sd',
        'v0_l23i_mean',
        'v0_l23i_sd',
        'v0_l4e_mean',
        'v0_l4e_sd',
        'v0_l4i_mean',
        'v0_l4i_sd',
        'v0_l5e_mean',
        'v0_l5e_sd',
        'v0_l5i_mean',
        'v0_l5i_sd',
        'v0_l6e_mean',
        'v0_l6e_sd',
        'v0_l6i_mean',
        'v0_l6i_sd',
        # Background rate per synapse
        'bg_rate',
        # Mean synaptic weight for all excitatory projections except L4e->L2/3e
        'w_mean',
        # Mean synaptic weight for L4e->L2/3e connections
        # See p. 801 of the paper, second paragraph under
        # 'Model Parameterization',
        # and the caption to Supplementary Fig. 7
        'w_234',
        # Standard deviation of weight distribution relative to mean for
        # all projections except L4e->L2/3e
        'w_rel',
        # Standard deviation of weight distribution relative to mean
        # for L4e->L2/3e
        # This value is not mentioned in the paper, but is chosen to match the
        # original code by Tobias Potjans
        'w_rel_234',
        # Means and standard deviations of delays from given source
        # populations (ms)
        # When delay_dist_type is 'uniform', delays are drawn from
        # [d_mean-d_sd, d_mean+d_sd].
        'd_mean',
        'd_sd',
        # Parameters for transient thalamic input
        'thalamic_input',
        'thal_params',
        # Maximum delay over which to determine covariances
        'tau_max',
        # Parameters for plots of spiking activity
        'plot_spiking_activity',
        # raster_t_min and raster_t_max include the time scaling factor
        'raster_t_min',
        'raster_t_max',
        # Fraction of recorded neurons to include in raster plot
        'frac_to_plot',
        # Numbers of neurons from which to record spikes
        'n_rec'
    ]

    def __init__(self, sim_params):
        self.g = -4.0
        self.layers = {'L23': 0, 'L4': 1, 'L5': 2, 'L6': 3}
        self.n_layers = len(self.layers)
        self.pops = {'E': 0, 'I': 1}
        self.n_pops_per_layer = len(self.pops)
        self.structure = {
            'L23': {'E': 0, 'I': 1},
            'L4': {'E': 2, 'I': 3},
            'L5': {'E': 4, 'I': 5},
            'L6': {'E': 6, 'I': 7}
        }

        self.n_full = {
          'L23': {'E': 20683, 'I': 5834},
          'L4': {'E': 21915, 'I': 5479},
          'L5': {'E': 4850,  'I': 1065},
          'L6': {'E': 14395, 'I': 2948}
        }

        self.conn_probs = [
            [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0],
            [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0],
            [0.0077, 0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.0],
            [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0],
            [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
            [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.0],
            [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
            [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]]

        self.k_ext = {
          'L23': {'E': 1600, 'I': 1500},
          'L4': {'E': 2100, 'I': 1900},
          'L5': {'E': 2000, 'I': 1900},
          'L6': {'E': 2900, 'I': 2100}
        }

        self.full_mean_rates = {
          'L23': {'E': 0.971, 'I': 2.868},
          'L4': {'E': 4.746, 'I': 5.396},
          'L5': {'E': 8.142, 'I': 9.078},
          'L6': {'E': 0.991, 'I': 7.523}
        }

        self.v0_mean = -58.  # mV
        self.v0_sd = 5.     # mV
        self.v0_l23e_mean = -64.28  # mV
        self.v0_l23e_sd = 4.36     # mV
        self.v0_l23i_mean = -59.16  # mV
        self.v0_l23i_sd = 3.57     # mV
        self.v0_l4e_mean = -59.33  # mV
        self.v0_l4e_sd = 3.74     # mV
        self.v0_l4i_mean = -59.45  # mV
        self.v0_l4i_sd = 3.94     # mV
        self.v0_l5e_mean = -59.11  # mV
        self.v0_l5e_sd = 3.94     # mV
        self.v0_l5i_mean = -57.66  # mV
        self.v0_l5i_sd = 3.55     # mV
        self.v0_l6e_mean = -62.72  # mV
        self.v0_l6e_sd = 4.46     # mV
        self.v0_l6i_mean = -57.43  # mV
        self.v0_l6i_sd = 3.48     # mV

        self.bg_rate = 8.  # spikes/s
        self.w_mean = 87.8e-3  # nA
        self.w_234 = 2 * self.w_mean  # nA
        self. w_rel = 0.1
        self.w_rel_234 = 0.05
        self.d_mean = {'E': 1.5, 'I': 0.75}
        self.d_sd = {'E': 0.75, 'I': 0.375}
        self.thalamic_input = False
        self.thal_params = {
          # Number of neurons in thalamic population
          'n_thal': 902,
          # Connection probabilities
          'C': {
              'L23': {'E': 0, 'I': 0},
              'L4': {'E': 0.0983, 'I': 0.0619},
              'L5': {'E': 0, 'I': 0},
              'L6': {'E': 0.0512, 'I': 0.0196}},
          'rate': 120.0,  # spikes/s;
          # Note that the rate is erroneously given as 15 spikes/s in
          # the paper.
          # The rate actually provided was 120 spikes/s.
          'start': 300.0,  # ms
          'duration': 10.  # ms;
        }

        self.tau_max = 100.0
        self.plot_spiking_activity = True
        self.raster_t_min = 0  # ms
        self.raster_t_max = sim_params.sim_duration  # ms
        self.frac_to_plot = 0.5
        self.n_rec = {}
        for layer in self.layers:
            self.n_rec[layer] = {}
            for pop in self.pops:
                if sim_params.record_fraction:
                    self.n_rec[layer][pop] = min(
                        int(round(
                            self.n_full[layer][pop] * sim_params.n_scaling *
                            sim_params.frac_record_spikes)),
                        int(round(
                            self.n_full[layer][pop] * sim_params.n_scaling)))
                else:
                    self.n_rec[layer][pop] = min(
                        sim_params.n_record, int(round(
                            self.n_full[layer][pop] * sim_params.n_scaling)))
