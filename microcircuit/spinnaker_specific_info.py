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

import numpy
from pyNN.random import RandomDistribution

from .sim_params import SpinnakerParams
from .constants import POISSON, SPINNAKER_NEURON_MODEL, CONN_ROUTINE


class SpinnakerSimulatorInfo(SpinnakerParams):
    """
    spinnaker specific params.
    """

    __slots__ = [
        # pylint: disable=wrong-spelling-in-comment
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
        # input directory where voltages are read from.
        'input_dir',
        # neuron params
        'neuron_params',
        # tau syn name
        'tau_syn_name',
        # The number of neurons per core
        'neurons_per_core',
        # whether to use split synapse neuron model
        'use_split_synapse_neuron_model',
        # If using split synapse neuron model, how many synapse cores?
        'n_synapse_cores',
        # If using split synapse neuron model, how many delay slots?
        'n_delay_slots',
        # Script RNG
        'script_rng'
    ]

    def __init__(
            self, timestep=0.1, sim_duration=1000.0, min_delay=0.1,
            max_delay=12.8, outfile='output.txt', errfile='errors.txt',
            output_path='results', output_format='pkl',
            conn_dir='connectivity', parallel_safe=True, n_scaling=1.0,
            k_scaling=1.0, neuron_model=SPINNAKER_NEURON_MODEL,
            conn_routine=CONN_ROUTINE, save_connections=False,
            voltage_input_type='pop_random', delay_dist_type='normal',
            input_dir='voltages_0.1_0.1_delays', input_type=POISSON,
            record_fraction=True, n_record=100, frac_record_spikes=1.0,
            record_v=False, frac_record_v=0.1, pyseed=2563297,
            live_output=False, tau_syn_name='tau_syn_E',
            neurons_per_core=64, use_split_synapse_neuron_model=True,
            n_synapse_cores=3, n_delay_slots=128):
        super(SpinnakerSimulatorInfo, self).__init__(
            timestep, sim_duration, min_delay, max_delay, outfile, errfile,
            output_path, output_format, conn_dir)
        self.parallel_safe = parallel_safe
        self.n_scaling = float(n_scaling)
        self.k_scaling = float(k_scaling)
        self.neuron_model = neuron_model
        self.conn_routine = conn_routine
        self.save_connections = bool(save_connections)
        self.voltage_input_type = voltage_input_type
        self.delay_dist_type = delay_dist_type
        self.input_type = input_type
        self.record_fraction = record_fraction
        self.n_record = int(n_record)
        self.frac_record_spikes = float(frac_record_spikes)
        self.record_v = bool(record_v)
        self.frac_record_v = float(frac_record_v)
        self.pyseed = int(pyseed)
        self.live_output = bool(live_output)
        self.input_dir = input_dir
        self.tau_syn_name = tau_syn_name
        self.neurons_per_core = int(neurons_per_core)
        self.use_split_synapse_neuron_model = bool(
            use_split_synapse_neuron_model)
        self.n_synapse_cores = int(n_synapse_cores)
        self.n_delay_slots = int(n_delay_slots)
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
        self.script_rng = None

    def after_setup_info(self, sim):
        # pylint: disable=wrong-spelling-in-docstring
        """
        spinnaker related tasks for after setup
        :param sim: sim
        :rtype: None
        """
        sim.set_number_of_neurons_per_core(
            sim.IF_curr_exp, self.neurons_per_core)
        sim.set_number_of_neurons_per_core(
            sim.SpikeSourcePoisson, self.neurons_per_core)

    @staticmethod
    def set_record_v(this_pop):
        """ sets a pop to record v

        :param this_pop:
        :return:
        """
        this_pop.record("v")

    def create_neural_population(self, sim, n_neurons, layer, pop):
        """
        Create a neural population optimized for sPyNNaker
        """
        additional_params = {"seed": self.pyseed}
        if self.use_split_synapse_neuron_model:
            # pylint: disable=import-outside-toplevel
            from spynnaker.pyNN.extra_algorithms.splitter_components import (
                SplitterAbstractPopulationVertexNeuronsSynapses)
            print(f"Using split synapse neuron model with "
                  f"{self.n_synapse_cores} synapse cores and "
                  f"{self.n_delay_slots} delay slots")
            additional_params["splitter"] = \
                SplitterAbstractPopulationVertexNeuronsSynapses(
                    self.n_synapse_cores, self.n_delay_slots, False)
        model = getattr(sim, self.neuron_model)
        return sim.Population(
            int(round(n_neurons * self.n_scaling)),
            model, cellparams=self.neuron_params,
            label=layer+pop, additional_parameters=additional_params)

    def create_poissons(
            self, sim, target_layer, target_pop, rate, this_target_pop, w_ext):
        # pylint: disable=wrong-spelling-in-docstring
        """ Creates the Spike Source Poisson

        :param sim:
        :param target_layer:
        :param target_pop:
        :param rate:
        :param this_target_pop:
        :param w_ext:
        :return:
        """
        if sim.rank() == 0:
            print(
                f'connecting Poisson generators to'
                ' {target_layer} {target_pop}')
        additional_params = {'seed': self.pyseed}
        if self.use_split_synapse_neuron_model:
            # pylint: disable=import-outside-toplevel
            from spynnaker.pyNN.extra_algorithms.splitter_components import (
                SplitterPoissonDelegate)
            additional_params['splitter'] = SplitterPoissonDelegate()
        poisson_generator = sim.Population(
            this_target_pop.size, sim.SpikeSourcePoisson,
            {'rate': rate}, additional_parameters=additional_params)
        conn = sim.OneToOneConnector()
        syn = sim.StaticSynapse(weight=w_ext)
        sim.Projection(
            poisson_generator, this_target_pop, conn, syn,
            receptor_type='excitatory')

    def fixed_tot_number_connect(
            self, sim, pop1, pop2, k, w_mean, w_sd, d_mean, d_sd, conn_type):
        # pylint: disable=wrong-spelling-in-docstring
        """
        SpiNNaker-specific function connecting two populations with multapses
        and a fixed total number of synapses

        :param sim:
        :param pop1:
        :param pop2:
        :param k:
        :param w_mean:
        :param w_sd:
        :param d_mean:
        :param d_sd:
        :param conn_type:
        :return:
        """

        if not k:
            return

        n_syn = int(round(k * len(pop2)))
        d_dist = None

        if self.delay_dist_type == 'normal':
            d_dist = RandomDistribution(
                'normal_clipped', mu=d_mean, sigma=d_sd,
                low=self.min_delay, high=self.max_delay)
        elif self.delay_dist_type == 'uniform':
            d_dist = RandomDistribution(
                'uniform', low=d_mean - d_sd, high=d_mean + d_sd)

        if w_mean > 0:
            w_dist = RandomDistribution(
                'normal_clipped', mu=w_mean, sigma=w_sd,
                low=0., high=numpy.inf)
        else:
            w_dist = RandomDistribution(
                'normal_clipped', mu=w_mean, sigma=w_sd,
                low=-numpy.inf, high=0.)

        syn = sim.StaticSynapse(weight=w_dist, delay=d_dist)
        connector = sim.FixedTotalNumberConnector(n=n_syn)
        proj = sim.Projection(pop1, pop2, connector, syn,
                              receptor_type=conn_type)

        if self.save_connections:
            proj.saveConnections(
                self.conn_dir + '/' + pop1.label + "_" + pop2.label + '.conn',
                gather=True)
