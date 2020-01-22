# Copyright (c) 2017-2019 The University of Manchester
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Synfirechain-like example
"""
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def run_chain():
    runtime = 50
    p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
    nNeurons = 200  # number of neurons in each population
    p.set_number_of_neurons_per_core(p.IF_curr_exp, nNeurons / 2)

    cell_params_lif = {'cm': 0.25,
                       'i_offset': 0.0,
                       'tau_m': 20.0,
                       'tau_refrac': 2.0,
                       'tau_syn_E': 5.0,
                       'tau_syn_I': 5.0,
                       'v_reset': -70.0,
                       'v_rest': -65.0,
                       'v_thresh': -50.0
                       }

    populations = list()
    projections = list()

    weight_to_spike = 2.0
    delay = 17

    loopConnections = list()
    for i in range(0, nNeurons):
        singleConnection = ((i, (i + 1) % nNeurons, weight_to_spike, delay))
        loopConnections.append(singleConnection)

    injectionConnection = [(0, 0)]
    spikeArray = {'spike_times': [[0]]}
    populations.append(
        p.Population(nNeurons, p.IF_curr_exp(**cell_params_lif), label='pop_1'))
    populations.append(
        p.Population(1, p.SpikeSourceArray(**spikeArray), label='inputSpikes_1'))

    projections.append(p.Projection(
        populations[0], populations[0], p.FromListConnector(loopConnections),
        p.StaticSynapse(weight=weight_to_spike, delay=delay)))
    projections.append(p.Projection(
        populations[1], populations[0], p.FromListConnector(injectionConnection),
        p.StaticSynapse(weight=weight_to_spike, delay=1)))

    populations[0].record(['v', 'gsyn_exc', 'gsyn_inh', 'spikes'])

    p.run(runtime)

    # get data (could be done as one, but can be done bit by bit as well)
    v = populations[0].get_data('v')
    gsyn_exc = populations[0].get_data('gsyn_exc')
    gsyn_inh = populations[0].get_data('gsyn_inh')
    spikes = populations[0].get_data('spikes')

    total_sdram = p.globals_variables.get_simulator().get_generated_output(
        "TotalSDRAMTracker")
    matrix = p.globals_variables.get_simulator().get_generated_output(
        "MatrixTracker")
    expander = p.globals_variables.get_simulator().get_generated_output(
        "ExpanderTracker")
    (data_extraction_time, data_loading_time_dsg, data_loading_time_dse,
     data_loading_time_expand) = extract_prov_elements()
    p.end()
    return (
        total_sdram, matrix, expander, data_extraction_time,
        data_loading_time_dsg, data_loading_time_dse, data_loading_time_expand)


def chain_end():
    p.end()


def extract_prov_elements():
    # prov names needed
    GENERATE_DSG = "run_time_of_SpynnakerDataSpecificationWriter"
    LOAD_DSE = "run_time_of_HostExecuteApplicationDataSpecification"
    EXTRACTION = "run_time_of_BufferExtractor"
    EXPANDER_RUNTIME = "run_time_of_SynapseExpander"

    data_extraction_time = None
    data_loading_time_dsg = None
    data_loading_time_dse = None
    data_loading_time_expand = None
    all_prov = p.globals_variables.get_simulator().all_provenance_items
    for items in all_prov:
        for item in items:
            if item.names[-1] == GENERATE_DSG:
                data_loading_time_dsg = item.value
            if item.names[-1] == LOAD_DSE:
                data_loading_time_dse = item.value
            if item.names[-1] == EXTRACTION:
                data_extraction_time = item.value
            if item.names[-1] == EXPANDER_RUNTIME:
                data_loading_time_expand = item.value
    return (
        data_extraction_time, data_loading_time_dsg, data_loading_time_dse,
        data_loading_time_expand)
