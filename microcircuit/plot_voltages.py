
import spynnaker8 as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import pickle

runtime = 1000

# output_dir = '/Users/oliver/Documents/microcircuit_results/dc_orig_init'
# output_dir = '/Users/oliver/Documents/microcircuit_results/dc_subthresh_init'
# output_dir = '/Users/oliver/Documents/microcircuit_results/poisson_orig_init'
# output_dir = '/Users/oliver/Documents/microcircuit_results/poisson_subthresh_init'

# output_dir = '/Users/oliver/Documents/microcircuit_results/dc_orig_init_255'
# output_dir = '/Users/oliver/Documents/microcircuit_results/dc_subthresh_init_255'
# output_dir = '/Users/oliver/Documents/microcircuit_results/poisson_orig_init_255'
# output_dir = '/Users/oliver/Documents/microcircuit_results/poisson_subthresh_init_255'

# output_dir = '/Users/oliver/Documents/microcircuit_results/poisson_julich_init'


# output_dir = '/Users/oliver/Documents/microcircuit_results/poisson_julich_spike_counts'
# output_dir = '/Users/oliver/Documents/microcircuit_results/dc_julich_spike_counts'
output_dir = '/Users/oliver/Documents/microcircuit_results/dc_julich_spike_counts_255'
# output_dir = '/Users/oliver/Documents/microcircuit_results/poisson_julich_spike_counts_255'

layer_keys = [
    ['L23E', 'L23I'],
    ['L4E', 'L4I'],
    ['L5E', 'L5I'],
    ['L6E', 'L6I']
    ]

plt.figure()
plt.suptitle("Total spikes received per simulation timestep: {}".format(output_dir))
index = 1

time = []
timing = 0.0
for i in range(10000):
    time.append(timing)
    timing += 0.1

# Loop over data keys and append subplots
for k in layer_keys:
    print "Plotting {} total spikes".format(k)
    exc_filename = output_dir + '/voltages_' + k[0] + '.pkl'
    inh_filename = output_dir + '/voltages_' + k[1] + '.pkl'

    pickle_exc = open(exc_filename, "rb")
    exc_data = pickle.load(pickle_exc)

    pickle_inh = open(inh_filename, "rb")
    inh_data = pickle.load(pickle_inh)

    # Create excitatory plot
    plt.subplot(4, 2, index)
    plt.title(k[0])
    plt.plot(time,
        exc_data.segments[0].filter(name='gsyn_exc')[0].magnitude[0:len(time)])
    plt.legend()
    plt.show(block=False)

    index += 1

    # Create inhibitoryp plot
    plt.subplot(4, 2, index)
    plt.title(k[1])
    plt.plot(time,
        inh_data.segments[0].filter(name='gsyn_exc')[0].magnitude[0:len(time)])
    plt.legend()
    plt.show(block=False)

    index += 1

# Plot total processing time
plt.figure()
plt.suptitle("Total processing time per simulation timestep: {}".format(output_dir))
index = 1
for k in layer_keys:

    sh_exc = (exc_data.segments[0].filter(name='v')[0].magnitude[1:len(time)] * 2**15).astype(int)
    sh_inh = (inh_data.segments[0].filter(name='v')[0].magnitude[1:len(time)] * 2**15).astype(int)

    # Create excitatory plot
    plt.subplot(4, 2, index)
    plt.title(k[0])
    plt.plot(time[1:], ((sh_exc[:,0] >> 24) & 0xFF), label='d')
    plt.plot(time[1:], ((sh_exc[:,0] >> 16) & 0xFF), label='c')
    plt.plot(time[1:], ((sh_exc[:,0] >> 8) & 0xFF), label='b')
    plt.plot(time[1:], ((sh_exc[:,0] >> 0) & 0xFF), label='a')
    plt.legend()
    plt.show(block=False)

    index += 1

    # Create inhibitoryp plot
    plt.subplot(4, 2, index)
    plt.title(k[1])
    plt.plot(time[1:], ((sh_inh[:,0] >> 24) & 0xFF), label='d (6+)')
    plt.plot(time[1:], ((sh_inh[:,0] >> 16) & 0xFF), label='c (2-5)')
    plt.plot(time[1:], ((sh_inh[:,0] >> 8) & 0xFF), label='b (1)')
    plt.plot(time[1:], ((sh_inh[:,0] >> 0) & 0xFF), label='a (0)')

    plt.legend()
    plt.show(block=False)



#     print "Plotting {} total processing time".format(k)
#
#     # Create excitatory plot
#     plt.subplot(4, 2, index)
#     plt.title(k[0])
#     plt.plot(time[1:],
#         (20000000 - ((exc_data.segments[0].filter(name='v')[0].magnitude[1:len(time)]) * 2**15)) * 5E-6
#         )
#     plt.legend()
#     plt.show(block=False)
#
#     index += 1
#
#     # Create inhibitoryp plot
#     plt.subplot(4, 2, index)
#     plt.title(k[1])
#     plt.plot(time[1:],
#         (20000000 - ((inh_data.segments[0].filter(name='v')[0].magnitude[1:len(time)])* 2**15)) * 5E-6
#         )
#     plt.legend()
#     plt.show(block=False)

    index += 1


plt.show()
