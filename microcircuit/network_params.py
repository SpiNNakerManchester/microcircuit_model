###################################################
###     	Network parameters		###
###################################################

import os
from sim_params import *

params_dict = {
  'nest':
  {
    # Whether to make random numbers independent of the number of processes
    'parallel_safe': True,
    # Fraction of neurons to simulate
    'N_scaling': 1.,
    # Scaling factor for in-degrees. Upon downscaling, synaptic weights are
    # taken proportional to 1/sqrt(in-degree) and external drive is adjusted
    # to preserve mean and variances of activity in the diffusion approximation.
    # In-degrees and weights of both intrinsic and extrinsic inputs are adjusted.
    # This scaling was not part of the original study, but this option is included
    # here to enable simulations on small systems that give results similar to
    # full-scale simulations.
    'K_scaling': 1.,
    # Neuron model. Possible values: 'IF_curr_exp', 'iaf_psc_exp_ps'
    'neuron_model': 'iaf_psc_exp_ps',
    # Connection routine
    # 'fixed_total_number' reproduces the connectivity from Potjans & Diesmann (2014),
    # establishing a fixed number of synapses between each pair of populations.
    # This function is available for the NEST and SpiNNaker back-ends.
    # 'from_list' reads in the connections from file
    'conn_routine': 'fixed_total_number',
    # Whether to save connections to file. See README.txt for known issues with using
    # save_connections in parallel simulations.
    'save_connections': False,
    # Initialization of membrane potentials
    # 'from_list' uses a set of initial neuron voltages read from a file,
    # 'random' uses randomized voltages
    'voltage_input_type': 'random',
    # Delay distribution. Possible values: 'normal' and 'uniform'.
    # The original model has normally distributed delays.
    'delay_dist_type': 'normal',
    # Type of background input. Possible values: 'poisson' and 'DC'
    # If 'DC' is chosen, a constant external current is provided, equal to the mean
    # current due to the Poisson input used in the default version of the model.
    'input_type': 'DC',
    # Whether to record from a fixed fraction of neurons in each population.
    # If False, a fixed number of neurons is recorded.
    'record_fraction': True,
    # Number of neurons from which to record spikes when record_fraction = False
    'n_record': 100,
    # Fraction of neurons from which to record spikes when record_fraction = True
    'frac_record_spikes': 1.,
    # Whether to record membrane potentials (not yet working for iaf_psc_exp_ps)
    'record_v': False,
    # Fixed number of neurons from which to record membrane potentials when
    # record_v=True and record_fraction = False
    'n_record_v': 20,
    # Fraction of neurons from which to record membrane potentials when
    # record_v=True and record_fraction = True
    'frac_record_v': 0.1,
    # Whether to record correlations
    'record_corr': False,
    # random number generator seeds for V and connectivity.
    # When parallel_safe is True, only the first is used.
    # When parallel_safe is False, the first num_processes are used.
    'pyseed': 2563297,
    # random number generator seed for NEST Poisson generators
    'master_seed': 124678
  },

  'spiNNaker':
  {
    # Whether to make random numbers independent of the number of processes
    'parallel_safe': True,
    # Fraction of neurons to simulate
    'N_scaling': 1.0,
    # Scaling factor for in-degrees. Upon downscaling, synaptic weights are
    # taken proportional to 1/sqrt(in-degree) and external drive is adjusted
    # to preserve mean and variances of activity in the diffusion approximation.
    # In-degrees and weights of both intrinsic and extrinsic inputs are adjusted.
    # This scaling was not part of the original study, but this option is included
    # here to enable simulations on small systems that give results similar to
    # full-scale simulations.
    'K_scaling': 1.0,
    # Neuron model. For SpiNNaker, only 'IF_curr_exp' is supported.
    'neuron_model' : 'IF_curr_exp',
    # Connection routine
    # 'fixed_total_number' reproduces the connectivity from Potjans & Diesmann (2014),
    # establishing a fixed number of synapses between each pair of populations.
    # This function is available for the NEST and SpiNNaker back-ends.
    # 'from_list' reads in the connections from file
    'conn_routine': 'fixed_total_number',
    # Whether to save connections to file
    'save_connections': False,
    # Initialization of membrane potentials
    # 'from_list' uses a set of initial neuron voltages read from a file,
    # 'random' uses randomized voltages
    'voltage_input_type': 'pop_random',
    'input_dir': 'voltages_0.1_0.1_delays',
    # Delay distribution. Possible values: 'normal' and 'uniform'.
    # The original model has normally distributed delays.
    'delay_dist_type': 'normal',
    # Type of background input. Possible values: 'poisson' or 'DC'
    # If 'DC' is chosen, a constant external current is provided, equal to the mean
    # current due to the Poisson input used in the default version of the model.
    'input_type': 'poisson',
#     'input_type': 'DC',
    # Whether to write out spikes only for a fixed fraction of neurons in each population.
    # If False, spikes are written out for a fixed number of neurons.
    # Note that spike recording parameters are interpreted slightly differently
    # for SpiNNaker than for NEST, as SpiNNaker always records all spikes.
    # The selection is therefore only made at the output stage.
    # Note that this option only works with the .h5 output format.
    'record_fraction': True,
    # Number of neurons from which to record spikes when record_fraction = False
    'n_record': 1,
    # Fraction of neurons from which to record spikes when record_fraction = True
    'frac_record_spikes': 1.,
    # Whether to record membrane potentials
    'record_v': True,
    # random number generator seed for V and connectivity.
    'pyseed': 2563297,
    # Whether to send output live
    'live_output': False,
  }
}

# Simulator back-end. Choose from 'nest', 'spiNNaker'
simulator = 'spiNNaker'

if simulator == 'spiNNaker':
    record_fraction = True
    frac_record_spikes = 1.

# Load params from params_dict into global namespace
globals().update(params_dict[simulator])

# Relative inhibitory synaptic weight
g = -4.

if neuron_model == 'iaf_psc_exp_ps':
    neuron_params =  {'C_m'       : 250.,  # pF
                      'I_e'  	  : 0.0,   # pA
                      'tau_m'     : 10.0,  # ms
                      't_ref'	  : 2.0,   # ms
                      'tau_syn_ex': 0.5,   # ms
                      'tau_syn_in': 0.5,   # ms
                      'V_reset'   : -65.0, # mV
                      'E_L'	  : -65.0, # mV
                      'V_th'	  : -50.0  # mV
                     }
else:
    neuron_params = {'cm'        : 0.25,  # nF
                     'i_offset'  : 0.0,   # nA
                     'tau_m'     : 10.0,  # ms
                     'tau_refrac': 2.0,   # ms
                     'tau_syn_E' : 0.5,   # ms
                     'tau_syn_I' : 0.5,   # ms
                     'v_reset'   : -65.0, # mV
                     'v_rest'    : -65.0, # mV
                     'v_thresh'  : -50.0  # mV
                    }

layers = {'L23': 0, 'L4': 1, 'L5': 2, 'L6': 3}
n_layers = len(layers)
pops = {'E': 0, 'I': 1}
n_pops_per_layer = len(pops)
structure = {'L23': {'E':0, 'I':1},
             'L4' : {'E':2, 'I':3},
             'L5' : {'E':4, 'I':5},
             'L6' : {'E':6, 'I':7}}

# Numbers of neurons in full-scale model
N_full = {
  'L23': {'E': 20683, 'I': 5834},
  'L4' : {'E': 21915, 'I': 5479},
  'L5' : {'E': 4850,  'I': 1065},
  'L6' : {'E': 14395, 'I': 2948}
}

# Probabilities for >=1 connection between neurons in the given populations.
# The first index is for the target population; the second for the source population
#             2/3e      2/3i    4e      4i      5e      5i      6e      6i
conn_probs = [[0.1009,  0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.    ],
             [0.1346,   0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.    ],
             [0.0077,   0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.    ],
             [0.0691,   0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.    ],
             [0.1004,   0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
             [0.0548,   0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.    ],
             [0.0156,   0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364,   0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]]

# In-degrees for external inputs
K_ext = {
  'L23': {'E': 1600, 'I': 1500},
  'L4' : {'E': 2100, 'I': 1900},
  'L5' : {'E': 2000, 'I': 1900},
  'L6' : {'E': 2900, 'I': 2100}
}

# Mean rates in the full-scale model, necessary for scaling
# Precise values differ somewhat between network realizations
full_mean_rates = {
  'L23': {'E': 0.971, 'I': 2.868},
  'L4' : {'E': 4.746, 'I': 5.396},
  'L5' : {'E': 8.142, 'I': 9.078},
  'L6' : {'E': 0.991, 'I': 7.523}
}

# Mean and standard deviation of initial membrane potential distribution
V0_mean = -58. # mV
V0_sd = 5.     # mV

# Background rate per synapse
bg_rate = 8. # spikes/s

# Mean synaptic weight for all excitatory projections except L4e->L2/3e
w_mean = 87.8e-3 # nA
# Mean synaptic weight for L4e->L2/3e connections
# See p. 801 of the paper, second paragraph under 'Model Parameterization',
# and the caption to Supplementary Fig. 7
w_234 = 2 * w_mean # nA

# Standard deviation of weight distribution relative to mean for
# all projections except L4e->L2/3e
w_rel = 0.1
# Standard deviation of weight distribution relative to mean for L4e->L2/3e
# This value is not mentioned in the paper, but is chosen to match the
# original code by Tobias Potjans
w_rel_234 = 0.05

# Means and standard deviations of delays from given source populations (ms)
# When delay_dist_type is 'uniform', delays are drawn from [d_mean-d_sd, d_mean+d_sd].
d_mean = {'E': 1.5, 'I': 0.75}
d_sd = {'E': 0.75, 'I': 0.375}

# Parameters for transient thalamic input
thalamic_input = False
thal_params = {
  # Number of neurons in thalamic population
  'n_thal'      : 902,
  # Connection probabilities
  'C'           : {'L23': {'E': 0, 'I': 0},
                   'L4' : {'E': 0.0983, 'I': 0.0619},
                   'L5' : {'E': 0, 'I': 0},
                   'L6' : {'E': 0.0512, 'I': 0.0196}},
  'rate'        : 120., # spikes/s;
  # Note that the rate is erroneously given as 15 spikes/s in the paper.
  # The rate actually provided was 120 spikes/s.
  'start'       : 300., # ms
  'duration'    : 10.  # ms;
}

# Maximum delay over which to determine covariances
tau_max = 100.

# Parameters for plots of spiking activity
plot_spiking_activity = True
# raster_t_min and raster_t_max include the time scaling factor
raster_t_min = 0 # ms
raster_t_max = simulator_params[simulator]['sim_duration'] # ms
# Fraction of recorded neurons to include in raster plot
frac_to_plot = 1  #0.05

# Numbers of neurons from which to record spikes
n_rec = {}
for layer in layers:
    n_rec[layer] = {}
    for pop in pops:
        if record_fraction:
            n_rec[layer][pop] = min(int(round(N_full[layer][pop] * N_scaling * frac_record_spikes)), \
                                    int(round(N_full[layer][pop] * N_scaling)))
        else:
            n_rec[layer][pop] = min(n_record, int(round(N_full[layer][pop] * N_scaling)))

# make any changes to the parameters
if 'custom_network_params.py' in os.listdir('.'):
    execfile('custom_network_params.py')
