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


def run_chain():
    runtime = 2450015

    p.setup(timestep=0.01, min_delay=1.0, max_delay=1.440, n_boards_required=3)
    cores = \
        p.globals_variables.get_simulator().\
        get_number_of_available_cores_on_machine
    p.globals_variables.get_simulator()._machine_outputs[
        "PlanNTimeSteps"] = 2450015
    neurons = cores - 1
    neurons = 5

    nNeurons = 1  # number of neurons in each population

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

    injectionConnection = [(0, 0)]
    spikeArray = {'spike_times': [[0]]}
    populations.append(
        p.Population(
            1, p.SpikeSourceArray(**spikeArray), label='inputSpikes_1'))

    for _ in range(0, neurons):
        populations.append(
            p.Population(nNeurons, p.IF_curr_exp(**cell_params_lif),
                         label='pop_1'))
        populations[-1].record(['v'])

    p.Projection(
        populations[0], populations[1],
        p.FromListConnector(injectionConnection),
        p.StaticSynapse(weight=2, delay=1))
    for pop_id in range(2, neurons):
        p.Projection(
            populations[pop_id-1], populations[pop_id],
            p.AllToAllConnector(),
            p.StaticSynapse(weight=2, delay=1))
    p.Projection(
        populations[-1], populations[1],
        p.AllToAllConnector(),
        p.StaticSynapse(weight=2, delay=1))

    p.run(runtime)

    # get data (could be done as one, but can be done bit by bit as well)
    for id in range(1, neurons):
        v = populations[id].get_data('v')

    total_sdram = p.globals_variables.get_simulator().get_generated_output(
        "TotalSDRAMTracker")
    matrix = p.globals_variables.get_simulator().get_generated_output(
        "MatrixTracker")
    expander = p.globals_variables.get_simulator().get_generated_output(
        "ExpanderTracker")
    io_time = \
        p.globals_variables.get_simulator().get_generated_output(
            "TimeToUseIO")
    print (io_time)
    print (total_sdram)

    (data_extraction_time, data_loading_time_dsg, data_loading_time_dse,
     data_loading_time_expand) = extract_prov_elements()
    try:
        p.end()
    except Exception:
        pass
    return (
        total_sdram, matrix, expander, data_extraction_time,
        data_loading_time_dsg, data_loading_time_dse,
        data_loading_time_expand, io_time)


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


if __name__ == '__main__':
    run_chain()
