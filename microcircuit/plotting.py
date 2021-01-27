import numpy as np
import glob
import neo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_raster_bars(
        t_start, t_stop, n_rec, frac_to_plot, path, common_params):

    # Dictionary of spike arrays, one entry for each population
    spikes = {}

    # Read out spikes for each population

    for layer in common_params.layers:
        spikes[layer] = {}
        for pop in common_params.pops:
            file_start = path + '/spikes_' + layer + pop + '*'
            file_list = glob.glob(file_start)
            pop_spike_array = np.empty((0, 2))
            for file_name in file_list:
                try:
                    if file_name.endswith(".dat"):
                        spike_array = np.loadtxt(file_name)
                        pop_spike_array = (
                            np.vstack((pop_spike_array, spike_array)))
                    else:
                        data = neo.get_io(file_name)
                        spike_trains = (
                            data.read_block().segments[0].spiketrains)
                        spike_array = [
                            [float(spike_trains[i][j]), float(i)]
                            for i in range(len(spike_trains))
                            for j in range(len(spike_trains[i]))]
                        n_rec[layer][pop] = len(spike_trains)
                        pop_spike_array = (
                            np.vstack((pop_spike_array, spike_array)))
                except IOError:
                    print('reading spike data from ', file_name, ' failed')
                    pass
            spikes[layer][pop] = pop_spike_array

    # Plot spike times in raster plot and bar plot with the
    # average firing rates of each population
    layer_list = ['L23', 'L4', 'L5', 'L6']
    pop_list = ['E', 'I']
    pop_labels = ['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    color = {'E': '#595289', 'I': '#af143c'}
    color_list = ['#595289', '#af143c']
    fig = plt.figure()
    axarr = [fig.add_subplot(121), fig.add_subplot(122)]

    # Plot raster plot
    id_count = 0
    print("Mean rates")
    rates = {}
    for layer in layer_list[-1::-1]:
        rates[layer] = {}
        for pop in pop_list[-1::-1]:
            t_spikes = spikes[layer][pop][:, 0]
            ids = spikes[layer][pop][:, 1] + (id_count + 1)
            filtered_times_indices = [
                np.where((t_spikes > t_start) & (t_spikes < t_stop))][0]
            t_spikes = t_spikes[filtered_times_indices]
            ids = ids[filtered_times_indices]

            # Compute rates with all neurons
            rate = 1000 * len(t_spikes) / (t_stop-t_start) / n_rec[layer][pop]
            rates[layer][pop] = rate
            print(layer, pop, np.round(rate, 2))
            # Reduce data for raster plot
            num_neurons = frac_to_plot * n_rec[layer][pop]
            t_spikes = t_spikes[np.where(ids < num_neurons + id_count + 1)[0]]
            ids = ids[np.where(ids < num_neurons + id_count + 1)[0]]
            axarr[0].plot(t_spikes, ids, '.', color=color[pop])
            id_count += num_neurons

    rate_list = np.zeros(
        common_params.n_layers * common_params.n_pops_per_layer)
    for i, layer in enumerate(layer_list):
        for j, pop in enumerate(pop_list):
            rate_list[i * common_params.n_pops_per_layer + j] = (
                rates[layer][pop])

    # Plot bar plot
    axarr[1].barh(
        np.arange(0, 8, 1) + 0.1,
        rate_list[::-1],
        color=color_list[::-1] * 4)

    # Set labels
    axarr[0].set_ylim((0.0, id_count))
    axarr[0].set_yticklabels([])
    axarr[0].set_xlabel('time (ms)')
    axarr[1].set_ylim((0.0, 8.5))
    axarr[1].set_yticks(np.arange(0.5, 8.5, 1.0))
    axarr[1].set_yticklabels(pop_labels[::-1])
    axarr[1].set_xlabel('rate (spikes/s)')

    plt.savefig(path + '/spiking_activity.png')
