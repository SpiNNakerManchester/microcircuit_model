from network_params import *
from sim_params import *
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import neo

def plot_raster_bars(t_start, t_stop, n_rec, frac_to_plot, path):

    # Dictionary of spike arrays, one entry for each population
    spikes = {}

    # Read out spikes for each population

    for layer in layers :
        spikes[layer] = {}
        for pop in pops :
            filestart = path + '/spikes_' + layer + pop + '*'
            filelist = glob.glob(filestart)
            pop_spike_array = np.empty((0,2))
            for file_name in filelist :
                try :
                    if file_name.endswith(".dat"):
                        spike_array = np.loadtxt(file_name)
                        pop_spike_array = np.vstack((pop_spike_array, spike_array))
                    else:
                        data = neo.get_io(file_name)
                        spiketrains = data.read_block().segments[0].spiketrains
                        spike_array = [
                            [float(spiketrains[i][j]), float(i)]
                            for i in range(len(spiketrains))
                            for j in range(len(spiketrains[i]))]

			if spike_array == []:
			    spike_array = [[0.0, 0.0]]

                        n_rec[layer][pop] = len(spiketrains)
                        pop_spike_array = np.vstack((pop_spike_array, spike_array))
                except IOError :
                    print 'reading spike data from ', file_name, ' failed'
                    pass
            spikes[layer][pop] = pop_spike_array

    # Plot spike times in raster plot and bar plot with the average firing rates of each population

    layer_list = ['L23', 'L4', 'L5', 'L6']
    pop_list = ['E', 'I']
    pop_labels = ['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    color = {'E':'#595289', 'I':'#af143c'}
    color_list = ['#595289', '#af143c']
    fig = plt.figure()
    axarr = []
    axarr.append(fig.add_subplot(121))
    axarr.append(fig.add_subplot(122))

#     count_oliver = {}
#
#     for layer in layer_list[-1::-1]:
#         count_oliver[layer] = {}
#         for pop in pop_list[-1::-1]:
#             count_oliver[layer][pop] = []
    print "filtering total incoming E and I spikes"

    # sum excitatory spikes
    exc_counter = []
    inh_counter = []
    total_counter = []
    for i in range(10000):
        time = float(i)/10
        exc_temp_count = 0
        inh_temp_count = 0

#         for l in layer_list:
#             for j in spikes[l]['E']:
#                 if np.round(j[0], 1) == time:
#                     exc_temp_count += 1
#             for j in spikes[l]['I']:
#                 if np.round(j[0], 1) == time:
#                     inh_temp_count += 1
        for l in layer_list:
#             if spikes[l]['E'].__contains__(time):
                exc_temp_count += np.count_nonzero(
                    np.round(spikes[l]['E'][:,0], 1) == time)
#             if spikes[l]['I'].__contains__(time):
                inh_temp_count += np.count_nonzero(
                    np.round(spikes[l]['I'][:,0], 1) == time)

        exc_counter.append(exc_temp_count)
        inh_counter.append(100 + inh_temp_count)
        total_counter.append(200 + exc_temp_count + inh_temp_count)
        print i

# for i in range(10000):
# ...     temp_count = 0
# ...     time = float(i)/10
# ...     for j in spikes['L23']['E']:
# ...         if j[0] == time:
# ...             temp_count += 1
# ...     counter.append(temp_count)

    # Plot raster plot
    id_count = 0
    print "Mean rates"
    rates = {}
    for layer in layer_list[-1::-1] :
        rates[layer] = {}
        for pop in pop_list[-1::-1] :
            rate = 0.0
            t_spikes = spikes[layer][pop][:,0]
            ids = spikes[layer][pop][:,1] + (id_count + 1)
            filtered_times_indices = [np.where((t_spikes >= t_start) & (t_spikes <= t_stop))][0]
            t_spikes = t_spikes[filtered_times_indices]
            ids = ids[filtered_times_indices]

            # Compute rates with all neurons
            rate = 1000*len(t_spikes) / (t_stop-t_start) / n_rec[layer][pop]
            rates[layer][pop] = rate
            print layer, pop, np.round(rate,2)
            # Reduce data for raster plot
            num_neurons = frac_to_plot * n_rec[layer][pop]
            t_spikes = t_spikes[np.where(ids < num_neurons + id_count + 1)[0]]

#             for oi in t_spikes:
#                 if oi == 0.0:
#                     counter_oliver[layer][pop] += 1

            ids = ids[np.where(ids < num_neurons + id_count + 1)[0]]
            axarr[0].plot(t_spikes, ids, '.', color=color[pop])
            id_count += num_neurons

    rate_list = np.zeros(n_layers*n_pops_per_layer)
    for i, layer in enumerate(layer_list) :
        for j, pop in enumerate(pop_list) :
            rate_list[i*n_pops_per_layer + j] = rates[layer][pop]

    # Plot bar plot
    axarr[1].barh(np.arange(0,8,1) + 0.1, rate_list[::-1], color=color_list[::-1] * 4)

    # Set labels
    axarr[0].set_ylim((0.0, id_count))
    axarr[0].set_yticklabels([])
    axarr[0].set_xlabel('time (ms)')

    axarr[1].set_ylim((0.0, 8.5))
    axarr[1].set_yticks(np.arange(0.5, 8.5, 1.0))
    axarr[1].set_yticklabels(pop_labels[::-1])
    axarr[1].set_xlabel('rate (spikes/s)')

    plt.show(block=True)
    plt.savefig(path + '/spiking_activity.png')
#     print "zero spike counter: {}".format(counter_oliver)


    plt.Figure()
    plt.plot(exc_counter, label='Excitatory Spikes')
    plt.plot(inh_counter, label='Inhibitory Spikes')
    plt.plot(total_counter, label='Total Spikes')
    plt.legend()
    plt.show()

