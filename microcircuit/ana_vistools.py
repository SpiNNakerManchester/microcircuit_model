# Analysis and visualization tools for microcircuit_spinnaker
# Based in part on mlcnm11_vistools.py by Tobias Potjans

# Example usage:
# (cd to directory containing ana_vistools.py)
# ipython
# import ana_vistools as ana
# (cd to directory containing simulation output)
# a = ana.data(simtime=50., n_scaling=0.1)
# a.load_spikes(rec_type='frac', rec=0.1)
# a.load_voltages(rec_type='frac', rec=0.02)
# a.load_conn(source_pop='L23E', target_pop='L4I')
# a.create_pop_rates(200,30000) (parameters are tmin, tmax)
# a.create_pop_cv_isi(200,30000) (parameters are tmin, tmax)
# a.create_pop_phase_sync(10, 50, 1) (parameters are tmin, tmax, dt in ms)
# a.check_vm(dt=0.1)
# a.create_ff(3) # indicates a PSTH bin width of 3 ms
# a.create_conn_stats()
# a.plot_dd_bars()
# a.dot_display(frac_neurons=0.5)

# (cd to a second directory containing simulation output)
# b = ana.data(simtime=50., n_scaling=0.1)
# b.load_conn(source_pop='L23E', target_pop='L4I')
# b.create_conn_stats()

# a.plot_conn(source_pop='L23E', target_pop='L4I', plot_binomial=True, epsilon=0.0691, compare=b)

import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as p
import csv
from scipy import stats
import pylab
import time

class data:
    def __init__(self, simtime, n_scaling):
        # 'n_scaling' indicates the fraction of neurons simulated compared
        # to the full-scale model
        self.spike_name = 'spikes'
        self.vm_name = 'voltages'
        self.conn_name = '.conn'
        self.spike_rec_type = None
        self.vm_rec_type = None
        self.num_layers = 4
        self.num_pop_per_layer = [2, 2, 2, 2]
        self.default_pops = [['L23E', 'L23I'], ['L4E', 'L4I'], ['L5E', 'L5I'], ['L6E', 'L6I']]
        self.default_pops_flat = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
        self.pop_types = [['E', 'I'], ['E', 'I'], ['E', 'I'], ['E', 'I']]
        self.h = 1.  # indicates that times are given in ms
        self.simtime = simtime
        self.set_fig_param_defaults()
        self.default_pop_sizes = [[20683, 5834], [21915, 5479], [4850, 1065], [14395, 2948]]
        self.num_neurons = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.popsize = {}
        for i in xrange(4):
            for j in xrange(2):
                self.num_neurons[i][j] = int(self.default_pop_sizes[i][j] * n_scaling)
                self.popsize[self.default_pops[i][j]] = self.num_neurons[i][j]


    def load_spikes(self, rec_type, rec, dir="."):
        if rec_type == 'frac':
            self.frac_record_spikes = rec
        if rec_type == 'abs':
            self.n_record_spikes = rec
        self.spike_rec_type = rec_type

        files = os.listdir(dir)
        files = np.sort(files)

        spike_files = []
        for f in files:
            if f.find(self.spike_name) > (-1): spike_files.append(f)

        print 'loading spike_data'
        self.spike_data = {}
        self.nspikes = {}
        for f in spike_files:
            f_path = os.path.join(dir, f)
            pop = f.split('_')[1].split('.')[0]
            if pop not in self.nspikes.keys():
                self.nspikes[pop] = 0
            try:
                new_data = np.loadtxt(f_path)
            except:
                new_data = []
                print 'Warning: data not loaded from ' + f
                pass
            if pop in self.spike_data.keys():
                if np.size(new_data) > 0:
                    if len(self.spike_data[pop]) > 0:
                        if np.size(new_data) == 2:
                            if np.size(self.spike_data[pop]) == 2:
                                self.spike_data[pop] = np.append([self.spike_data[pop]], [new_data], axis=0)
                            else:
                                self.spike_data[pop] = np.append(self.spike_data[pop], [new_data], axis=0)
                        else:
                            self.spike_data[pop] = np.append(self.spike_data[pop], new_data, axis=0)
                    else:
                        self.spike_data[pop] = new_data
            else:
                self.spike_data[pop] = new_data
            # load numbers of spikes
            for line in open(f_path):
                if "n =" in line:
                    self.nspikes[pop] += int(line.split('=')[1])
                    break


    def load_voltages(self, rec_type, rec, dir="."):
        if rec_type == 'frac':
            self.frac_record_vm = rec
        if rec_type == 'abs':
            self.n_record_vm = rec
        self.vm_rec_type = rec_type

        files = os.listdir(dir)
        files = np.sort(files)

        vm_files = []
        for f in files:
            if f.find(self.vm_name) > (-1): vm_files.append(f)

        print 'loading vm_data'
        self.vm_data = {}
        for f in vm_files:
            pop = f.split('_')[1].split('.')[0]
            if pop in self.vm_data.keys():
                self.vm_data[pop] = np.append(self.vm_data[pop], np.loadtxt(f), axis=0)
            else:
                self.vm_data[pop] = np.loadtxt(f)



    def load_conn(self, source_pop, target_pop, dir="."):
        files = os.listdir(dir)
        files = np.sort(files)

        conn_files = []
        for f in files:
            source = f.split('_')[0]
            target = f.split('_')[1].split('.')[0]
            if f.find(self.conn_name) > -1 and source == source_pop and target == target_pop: conn_files.append(f)

        print 'loading connectivity data'
        self.conn_data = {}
        for f in conn_files:
            print 'reading from ' + f
            source_pop = f.split('_')[0]
            target_pop = f.split('_')[1].split('.')[0]
            if source_pop not in self.conn_data:
                self.conn_data[source_pop] = {}
            first = True
            with open(f, 'rb') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ')
                for row in csvreader:
                    if row[0] != '#' and row[0] != '':
                        rowdata = np.array(row[0].split('\t'), dtype=float)
                        if first:
                            data = [list(rowdata)]
                            first = False
                        else:
                            data = np.append(data, [list(rowdata)], axis=0)
            if target_pop in self.conn_data[source_pop]:
                self.conn_data[source_pop][target_pop] = np.append(self.conn_data[source_pop][target_pop], data, axis=0)
            else:
                self.conn_data[source_pop][target_pop] = data


################################ Analysis functions

    def check_vm(self, dt):
        '''
        Check if vm data are complete
        '''
        required_len = self.simtime / dt + 1
        not_complete = {}
        neurons_missing = {}
        time_complete = True
        neurons_complete = True
        for pop in self.vm_data:
            if self.vm_rec_type == 'frac':
                required_n_neurons = round(self.frac_record_vm * self.popsize[pop])
            if self.vm_rec_type == 'abs':
                required_n_neurons = self.n_record_vm
            not_complete_pop = {}
            data = self.vm_data[pop]
            neurons = np.unique(data[:, 1])
            if len(neurons) != required_n_neurons:
                neurons_complete = False
                neurons_missing[pop] = required_n_neurons - len(neurons)
            for neuron in neurons:
                neuron_data = data[data[:, 1] == neuron]
                if len(neuron_data) != required_len:
                    time_complete = False
                    not_complete_pop[neuron] = len(neuron_data)
            not_complete[pop] = not_complete_pop
        if time_complete and neurons_complete:
            print "vm data are complete"
        else:
            if not time_complete:
                print 'neurons for which not all vm values are present:', not_complete
            if not neurons_complete:
                print 'numbers of neurons missing:', neurons_missing


    def create_pop_rates(self, tmin, tmax, h):
        self.pop_rates = {}
        # try:
        for pop in self.spike_data:
            spike_times = self.spike_data[pop][:, 0]
            n_spikes = len(self.spike_data[pop][np.all([spike_times >= tmin * h, spike_times < tmax * h], axis=0)])
            if self.spike_rec_type == 'abs':
                self.pop_rates[pop] = 1000.*n_spikes / (self.n_record_spikes * (tmax - tmin))
            if self.spike_rec_type == 'frac':
                self.pop_rates[pop] = 1000.*n_spikes / (round(self.popsize[pop] * self.frac_record_spikes) * (tmax - tmin))
        # create structured_pop_rates according to standard structure
        self.structured_pop_rates = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for i in xrange(4):
            for j in xrange(2):
                self.structured_pop_rates[i][j] = self.pop_rates[self.default_pops[i][j]]
        # except:
        #    print "create_pop_rates failed. You may need to run load_data first."


    def create_pop_cv_isi(self, tmin, tmax, h):
        self.pop_cv_isi = {}
        # try:
        for pop in self.spike_data:
            spike_times = self.spike_data[pop][:, 0]
            data = self.spike_data[pop][np.all([spike_times >= tmin * h, spike_times < tmax * h], axis=0)]
            cv_isi = []
            if len(data) > 2:
                for i in np.unique(data[:, 1]):
                    intervals = np.diff(data[mlab.find(data[:, 1] == i), 0])
                    if (len(intervals) > 1):
                        cv_isi.append(np.std(intervals) / np.mean(intervals))
            self.pop_cv_isi[pop] = np.mean(cv_isi)

        # create structured_cv_isi according to standard structure
        self.structured_cv_isi = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for i in xrange(4):
            for j in xrange(2):
                self.structured_cv_isi[i][j] = self.pop_cv_isi[self.default_pops[i][j]]
        # except:
        #    print "create_pop_cv_isi failed. You may need to run load_data first."


    def create_pop_sync(self, bin_width, tmin, tmax, h):
        self.pop_sync = {}
        for pop in self.spike_data:
            data = self.spike_data[pop]
            self.pop_sync[pop] = sync(data, bin_width, tmin, tmax, h)


    def create_pop_phase_sync(self, tmin, tmax, dt):
        '''Mean phase synchrony for each population between tmin and tmax.'''
        self.pop_phase_sync = {}
        for pop in self.spike_data:
            try:
                data = self.spike_data[pop]
                indices = [n for n in xrange(len(data)) if data[n, 0] >= tmin and data[n, 0] <= tmax]
                if len(indices) == 0:
                    self.pop_phase_sync[pop] = np.nan
                else:
                    phase_sync_vs_t = phase_sync(data[:, 0], data[:, 1], tmin, tmax, dt)
                    self.pop_phase_sync[pop] = np.mean(phase_sync_vs_t)
            except:
                print "create_pop_phase_sync failed for " + pop
        try:
            # create structured_phase_sync according to standard structure
            self.structured_phase_sync = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for i in xrange(4):
                for j in xrange(2):
                    self.structured_phase_sync[i][j] = self.pop_phase_sync[self.default_pops[i][j]]
        except:
            print "creating structured_phase_sync failed"


    def create_pop_v_stats(self, tmin, tmax, dt):
        '''Mean and standard deviation (over time, then averaged across neurons) of membrane potentials between tmin and tmax, when membrane potentials were recorded with time resolution dt.'''
        t = np.arange(0, self.simtime + dt, dt)
        self.pop_v_mean = {}
        self.pop_v_sd = {}
        for pop in self.vm_data:
            pop_data = self.vm_data[pop]
            neurons = np.unique(pop_data[:, 1])
            neuron_means = np.zeros(len(neurons))
            neuron_sds = np.zeros(len(neurons))
            for neuron in neurons:
                neuron_data = pop_data[pop_data[:, 1] == neuron, 0:1]
                data = neuron_data[np.all([t >= tmin, t < tmax], axis=0)]
                neuron_means[neuron] = np.mean(data[:, 0])
                neuron_sds[neuron] = np.std(data[:, 0])
            self.pop_v_mean[pop] = np.mean(neuron_means)
            self.pop_v_sd[pop] = np.mean(neuron_sds)

        # create structured_v_mean and structured_v_sd according to standard structure
        self.structured_v_mean = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.structured_v_sd = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for i in xrange(4):
            for j in xrange(2):
                self.structured_v_mean[i][j] = self.pop_v_mean[self.default_pops[i][j]]
                self.structured_v_sd[i][j] = self.pop_v_sd[self.default_pops[i][j]]


    def create_psth(self, bin_width, *arguments):
        """psth for all layers of spike_data made available
        in psth.

        Parameters:
        - bin_width in [ms]
        - (optional) tmin and tmax in [ms]
        """

        h = self.h
        if (len(arguments) > 0):
            tmin = arguments[0]
        else: tmin = 0
        if (len(arguments) > 1):
            tmax = arguments[1]
        else:
            tmax = self.simtime
        self.set_metadata('psth',
                          {'bin_width': bin_width,
                           'tmin': tmin,
                           'tmax': tmax})
        self.psth = []
        for i in range(self.num_layers):
            layer_psth = []
            for j in range(self.num_pop_per_layer[i]):
                pop = self.default_pops[i][j]
                layer_psth.append(psth(self.spike_data[pop],
                                       bin_width,
                                       tmin,
                                       tmax,
                                       h))
            self.psth.append(layer_psth)



    def create_ff(self, *arguments):
        """create_ff calculates the fano factors for each
        population and makes them available in ff;
        relies on create_psth of class spike_data to create psth

        Parameters:
        - (optional) bin_width in [ms] for calculating underlying psth
        - (optional) tmin and tmax (both in [ms])
          giving optional parameters forces to creation of new psth
        """
        if (len(arguments) > 0):
            bin_width = arguments[0]
        else:
            if hasattr(self, 'psth'):
                print 'create_ff: no psth instance existing and no bin_width given'
                print 'exiting create_ff without creation of ff'
                return
        if (len(arguments) > 1):
            tmin = arguments[1]
        else: tmin = 0.
        if (len(arguments) > 2):
            tmax = arguments[2]
        else:
            tmax = self.simtime
        self.set_metadata('ff',
                          {'bin_width': bin_width,
                           'tmin': tmin,
                           'tmax': tmax})
        if (not hasattr(self, 'psth')) or (len(arguments) > 0):
            print 'create_ff: creating new psth'
            self.create_psth(bin_width, tmin, tmax)
        self.ff = []
        for i in range(self.num_layers):
            this_ff = [0., 0.]
            for j in range(2):
                this_ff[j] = ff(self.psth[i][j])
            self.ff.append(this_ff)


    def create_conn_stats(self):
        """ Compute indegree, outdegree, and multapse distributions
        """
        self.ntps = {}
        self.nspt = {}
        self.multiplicities = {}
        for source_pop in self.conn_data:
            self.ntps[source_pop] = {}
            self.nspt[source_pop] = {}
            self.multiplicities[source_pop] = {}
            for target_pop in self.conn_data[source_pop]:
                # print 'creating connectivity statistics from ' + source_pop + ' to ' + target_pop
                senders = self.conn_data[source_pop][target_pop][:, 0]
                receivers = self.conn_data[source_pop][target_pop][:, 1]
                senders -= min(senders)
                receivers -= min(receivers)
                self.multiplicities[source_pop][target_pop] = np.zeros([self.popsize[source_pop], self.popsize[target_pop]])
                for i in xrange(len(senders)):
                    # multiplicities
                    self.multiplicities[source_pop][target_pop][senders[i], receivers[i]] += 1
                # number of targets per source
                self.ntps[source_pop][target_pop] = sum(np.transpose(self.multiplicities[source_pop][target_pop]))
                # number of sources per target
                self.nspt[source_pop][target_pop] = sum(self.multiplicities[source_pop][target_pop])


################################ Accessory functions for plotting

    def set_fig_param_defaults(self):
        """Sets default values for fig_params.
        """
        self.fig_params = {'figsizebase':(8, 6),
                           'dpi':90,
                           'pop_col':self.create_pop_col(),
                           'pop_col_dict':{'exc':'#595289', 'inh':'#af143c'},
                           'dd_l_col':'#555555',
                           'dd_l_style':['-', '--'],
                           'dd_idlabels':'layer ',
                           'default_axis':True,
                           'ms_':2,
                           'mew_':1,
                           }
        self.set_rc_params('all',
                           'text',
                           {'color': 'k',
                            })

        self.set_rc_params('alltexts',
                           'font',
                           {'size': 20,
                            'style': 'normal',
                            'variant': 'normal',
                            'weight': 'normal'})

        self.set_rc_params('dot_display',
                           'lines',
                           {'marker':'o',
                            'markersize':1,  # 2,
                            'markeredgewidth':0.05,  # 1,
                            }
                           )

        self.set_rc_params('pop_rates',
                           'lines',
                           {'marker':None,
                            'linestyle':'-',
                            'linewidth':0}
                           )

        self.set_rc_params('psth',
                           'lines',
                           {'marker':None,
                            'linestyle':'-',
                            'linewidth':0}
                           )

        self.set_rc_params('cv_isi',
                           'lines',
                           {'marker':None,
                            'linestyle':'-',
                            'linewidth':0}
                           )

        self.set_rc_params('ff',
                           'lines',
                           {'marker':None,
                            'linestyle':'-',
                            'linewidth':0}
                           )


    def set_rc_params(self, plot_type, feature, dict):
        """Sets the rc params for given feature to specified
        dictionary of the data instance that will
        be used for the specified plot_type.
        Example:
        set_rc_params('dot_display', 'font', {'family':'monospace',
                                              'weight':'bold'})
        Arguments:
        - plot_type: string for the plot type, this setting shall
          be applied to. Possible values:
          * 'all': sets for all possible plot types
          * 'dot_display'
          * 'pop_rates'
          * 'cv_isi'
          * 'ff'
          * 'psth'
          * 'alltexts'
          * 'label'
          * 'title'
          * 'ticks'
          * 'text'
          * 'legend'
        - feature: feature of rc_params to be changed, e.g. 'font',
          'lines' or 'axes'. See doc for rc in matplotlib.pyplot
        - dict: dictionary containing parameter settings for feature
          group, e.g. {'lw':2, 'ls':'-'} for feature 'lines'
        """
        if plot_type == 'dot_display' or plot_type == 'all':
            if not hasattr(self, 'dot_display_rc'):
                self.dot_display_rc = {}
            self.dot_display_rc[feature] = dict
        if plot_type == 'pop_rates' or plot_type == 'all':
            if not hasattr(self, 'pop_rates_rc'):
                self.pop_rates_rc = {}
            self.pop_rates_rc[feature] = dict
        if plot_type == 'cv_isi' or plot_type == 'all':
            if not hasattr(self, 'cv_isi_rc'):
                self.cv_isi_rc = {}
            self.cv_isi_rc[feature] = dict
        if plot_type == 'ff' or plot_type == 'all':
            if not hasattr(self, 'ff_rc'):
                self.ff_rc = {}
            self.ff_rc[feature] = dict
        if plot_type == 'psth' or plot_type == 'all':
            if not hasattr(self, 'psth_rc'):
                self.psth_rc = {}
            self.psth_rc[feature] = dict
        if plot_type == 'label' or plot_type == 'alltexts':
            if not hasattr(self, 'label_rc'):
                self.label_rc = {}
            self.label_rc[feature] = dict
        if plot_type == 'title' or plot_type == 'alltexts':
            if not hasattr(self, 'title_rc'):
                self.title_rc = {}
            self.title_rc[feature] = dict
        if plot_type == 'ticks' or plot_type == 'alltexts':
            if not hasattr(self, 'ticks_rc'):
                self.ticks_rc = {}
            self.ticks_rc[feature] = dict
        if plot_type == 'text' or plot_type == 'alltexts':
            if not hasattr(self, 'text_rc'):
                self.text_rc = {}
            self.text_rc[feature] = dict
        if plot_type == 'legend' or plot_type == 'alltexts':
            if not hasattr(self, 'legend_rc'):
                self.legend_rc = {}
            self.legend_rc[feature] = dict


    def create_pop_col(self):
        """creates colors for populations of mlm model
        """
        if not hasattr(self, 'pop_types'):
            print 'data.create_pop_col: pop_types not known, using default mlm'
            return [['#595289', '#af143c'], ['#595289', '#af143c'], ['#595289', '#af143c'], ['#595289', '#af143c']]
        pop_col = []
        for i in range(self.num_layers):
            this_pop_col = []
            for j in range(self.num_pop_per_layer[i]):
                if self.pop_types[i][j] == 'i':
                    this_pop_col.append('#af143c')
                else:
                    this_pop_col.append('#595289')
            pop_col.append(this_pop_col)
        return pop_col


    def set_metadata(self, ana_obj, dict):
        """Sets metadata for analysis object to
        specified dictionary containing parameters necessary for
        the interpretation of the derived data, e.g. bin_width used
        for the creation of a psth. This function is in the normal case
        used by the create_ana_obj routines to update the metadata.
        Example:
        set_obj_params('psth', {'bin_width':50., 'tmin':0., 'tmax':500.})
        Arguments:
        - analysis object: string for the ana_obj, this setting shall
          be applied to. Possible values:
          * 'pop_rates'
          * 'cv_isi'
          * 'ff'
          * 'psth'
        - dict: dictionary containing metadata associated with ana_obj
        """
        if ana_obj == 'pop_rates':
            if not hasattr(self, 'pop_rates_meta'):
                self.pop_rates_meta = {}
            for feature in dict:
                self.pop_rates_meta[feature] = dict[feature]
        if ana_obj == 'cv_isi':
            if not hasattr(self, 'cv_isi_meta'):
                self.cv_isi_meta = {}
            for feature in dict:
                self.cv_isi_meta[feature] = dict[feature]
        if ana_obj == 'ff':
            if not hasattr(self, 'ff_meta'):
                self.ff_meta = {}
            for feature in dict:
                self.ff_meta[feature] = dict[feature]
        if ana_obj == 'psth':
            if not hasattr(self, 'psth_meta'):
                self.psth_meta = {}
            for feature in dict:
                self.psth_meta[feature] = dict[feature]

    def set_tick_font(self):
        """Set tick fonts for given plot according general parameters.
        Arguments:
        - ticks: output from pylab.gca().xaxis.get_major_ticks();
          ticks' font settings will be applied to these ticks
        """
        for tick in p.gca().xaxis.get_major_ticks():
            tick.label1.set_fontsize(self.ticks_rc['font']['size'])
        for tick in p.gca().yaxis.get_major_ticks():
            tick.label1.set_fontsize(self.ticks_rc['font']['size'])

################################ Plotting functions

    def plot_dd_bars(self, *arguments):
        """ Dot display with horizontal bar plots for:
        - rate
        - cv
        - ff

        Parameters:
        - (optional) tmin and tmax (both in [ms]) for dotdisplay
        """
        # h = self.h
        if (len(arguments) > 0):
            tmin = arguments[0]
        else: tmin = 0.
        if (len(arguments) > 1):
            tmax = arguments[1]
        else:
            tmax = self.simtime

        fig = p.figure(figsize=(self.fig_params['figsizebase'][1] + self.fig_params['figsizebase'][1] * 1.5, self.fig_params['figsizebase'][0]),
                 dpi=self.fig_params['dpi'])
        fig.set_facecolor('w')
        ax1 = fig.add_axes([0.2 / (1. + 1.5), 0.1, 0.7 / (1. + 1.5), 0.8])
        set_spines_bottom_left([ax1])
        xoffset = 0.2 / (1. + 1.5) + 0.7 / (1. + 1.5)

        if hasattr(self, 'dot_display_rc'):
            for feature in self.dot_display_rc:
                p.rc(feature, **self.dot_display_rc[feature])

        id_offset = 0
        self.ddbar_dot_display_plots = []
        yticks_ticks = []
        e_bar_locs = []
        i_bar_locs = []
        for i in range(self.num_layers - 1, -1, -1):
            pop_plots = []
            for j in range(self.num_pop_per_layer[i] - 1, -1, -1):
                pop = self.default_pops[i][j]
                if self.spike_rec_type == 'frac':
                    recpopsize = round(self.popsize[pop] * self.frac_record_spikes)
                if self.spike_rec_type == 'abs':
                    recpopsize = self.n_record_spikes

                ax1.scatter(self.spike_data[pop][:, 0],
                         self.spike_data[pop][:, 1] + id_offset,
                         color=self.fig_params['pop_col'][i][j], s=3)
                        # markeredgecolor=self.fig_params['pop_col'][i][j],
                        # linestyle=None)
                # id_offset = id_offset + self.popsize[pop]
                id_offset = id_offset + recpopsize
                pline = True
                if i == 0 and j == 0:
                    pline = False
                if pline:
                    ax1.axhline(y=id_offset,
                              linewidth=2,
                              linestyle=self.fig_params['dd_l_style'][0],
                              marker='None',
                              color=self.fig_params['dd_l_col'])
                if j == 0:
                    e_bar_locs.append(id_offset - 0.6 * recpopsize)
                if j == 1:
                    i_bar_locs.append(id_offset - 0.6 * recpopsize)
            if self.fig_params['dd_idlabels'].find('layer') > -1:
                if self.spike_rec_type == 'frac':
                    yticks_ticks.append(id_offset - 0.5 * sum(np.array(self.num_neurons[i]) * self.frac_record_spikes))
                if self.spike_rec_type == 'abs':
                    yticks_ticks.append(id_offset - self.n_record_spikes)
            self.ddbar_dot_display_plots.append(pop_plots)
        if self.fig_params['dd_idlabels'].find('layer') > -1:
            yticks_items = ['layer 6', 'layer 5', 'layer 4', 'layer 2/3']
            p.yticks(yticks_ticks, yticks_items, rotation='45')
        p.xlabel('time [ms]', **self.label_rc['font'])
        ax1.axis([tmin, tmax, 0, id_offset])
        self.set_tick_font()
        p.xticks(ax1.get_xticks()[0:-1:2])
        p.xlim([tmin, tmax])
        # p.legend((self.ddbar_dot_display_plots[0][1][0], self.ddbar_dot_display_plots[0][0][0]),('exc.', 'inh.'), 'upper right')

        # first bar plot: rate
        if (not hasattr(self, 'pop_rates')):
            print 'plot_dd_bars: trying to plot pop_rates, but not existing!'
            print 'plot_dd_bars: exiting without finishing plot'
            return False

        if hasattr(self, 'pop_rates_rc'):
            for feature in self.pop_rates_rc:
                p.rc(feature, **self.pop_rates_rc[feature])
        ax2 = fig.add_axes([xoffset, 0.1, 0.5 / (1. + 1.5), 0.8])
        set_spines_bottom_left([ax2])
        xoffset = xoffset + 0.5 / (1. + 1.5)
        e_rates = []
        i_rates = []
        for rates in self.structured_pop_rates:
            e_rates.append(rates[0])
            i_rates.append(rates[1])
        e_rates.reverse()
        i_rates.reverse()
        b_e = ax2.barh(yticks_ticks,
                       e_rates,
                       height=max(e_bar_locs) / 15.,
                       left=0.,
                       color=self.fig_params['pop_col_dict']['exc'],
                       edgecolor=self.fig_params['pop_col_dict']['exc'])
        b_i = ax2.barh(np.array(yticks_ticks) - max(e_bar_locs) / 15.*1.2,
                       i_rates,
                       height=max(e_bar_locs) / 15.,
                       color=self.fig_params['pop_col_dict']['inh'],
                       edgecolor=self.fig_params['pop_col_dict']['inh'])
        p.yticks([])
        p.xlabel('rate [Hz]', **self.label_rc['font'])
        self.set_tick_font()
        self.ddbar_pop_rates_plots = [b_e, b_i]
        p.xticks(ax2.get_xticks()[0:-1:2])
        p.xlim([0, 1.2 * max(np.append(np.array(e_rates), np.array(i_rates)))])

        # second bar plot: cv_isi
        if (not hasattr(self, 'pop_cv_isi')):
            print 'plot_dd_bars: trying to plot cv, but not existing!'
            print 'plot_dd_bars: exiting without finishing plot'
            return False

        if hasattr(self, 'cv_isi_rc'):
            for feature in self.cv_isi_rc:
                p.rc(feature, **self.cv_isi_rc[feature])
        ax3 = fig.add_axes([xoffset, 0.1, 0.5 / (1. + 1.5), 0.8])
        set_spines_bottom_left([ax3])
        xoffset = xoffset + 0.5 / (1. + 1.5)
        e_cv = []
        i_cv = []
        for cv in self.structured_cv_isi:
            e_cv.append(cv[0])
            i_cv.append(cv[1])
        e_cv.reverse()
        i_cv.reverse()
        b_e = ax3.barh(yticks_ticks,
                       e_cv,
                       height=max(e_bar_locs) / 15.,
                       left=0.,
                       color=self.fig_params['pop_col_dict']['exc'],
                       edgecolor=self.fig_params['pop_col_dict']['exc'])
        b_i = ax3.barh(np.array(yticks_ticks) - max(e_bar_locs) / 15.*1.2,
                       i_cv,
                       height=max(e_bar_locs) / 15.,
                       color=self.fig_params['pop_col_dict']['inh'],
                       edgecolor=self.fig_params['pop_col_dict']['inh'])
        p.yticks([])
        p.xlabel('cv_isi', **self.label_rc['font'])
        self.set_tick_font()
        self.ddbar_cv_isi_plots = [b_e, b_i]
        p.xticks(ax3.get_xticks()[0:-1:3])
        cv_arr = np.append(np.array(e_cv), np.array(i_cv))
        cv_arr_no_nans = [cv_arr[i] for i in xrange(len(cv_arr)) if not np.isnan(cv_arr[i])]
        p.xlim([0, 1.2 * max(cv_arr_no_nans)])

        # third bar plot: ff
        if (not hasattr(self, 'ff')):
            print 'plot_dd_bars: trying to plot ff, but not existing!'
            print 'plot_dd_bars: exiting without finishing plot'
            return False

        if hasattr(self, 'ff_rc'):
            for feature in self.ff_rc:
                p.rc(feature, **self.ff_rc[feature])
        ax4 = fig.add_axes([xoffset, 0.1, 0.5 / (1. + 1.5), 0.8])
        set_spines_bottom_left([ax4])
        xoffset = xoffset + 0.5 / (1. + 1.5)
        e_ff = []
        i_ff = []
        for ff in self.ff:
            e_ff.append(ff[0])
            i_ff.append(ff[1])
        e_ff.reverse()
        i_ff.reverse()
        b_e = ax4.barh(yticks_ticks,
                       e_ff,
                       height=max(e_bar_locs) / 15.,
                       left=0.,
                       color=self.fig_params['pop_col_dict']['exc'],
                       edgecolor=self.fig_params['pop_col_dict']['exc'])
        b_i = ax4.barh(np.array(yticks_ticks) - max(e_bar_locs) / 15.*1.2,
                       i_ff,
                       height=max(e_bar_locs) / 15.,
                       color=self.fig_params['pop_col_dict']['inh'],
                       edgecolor=self.fig_params['pop_col_dict']['inh'])
        p.yticks([])
        p.xlabel('ff', **self.label_rc['font'])
        self.set_tick_font()
        self.ddbar_ff_plots = [b_e, b_i]
        p.xticks(ax4.get_xticks()[0:-1:2])
        p.xlim([0, max(np.append(np.array(e_ff), np.array(i_ff)))])

        fig.show()


    def dot_display(self, *arguments, **keywords):
        """Layered raster plot.

           Parameters:
	   - (optional) tmin and tmax (both in ms)
           - (optional) frac_neurons: fraction of recorded neurons included in plot
	   - (optional) format of the output file by the optional keyword argument output
	 """
        if (len(arguments) > 0):
            tmin = arguments[0]
        else: 
            tmin = 0
        if (len(arguments) > 1):
            tmax = arguments[1]
        else:
            tmax = self.simtime

        if 'frac_neurons' in keywords:
            frac_neurons = keywords['frac_neurons']
        else:
            frac_neurons = 1.

        fig = p.figure()
        fig.set_facecolor('w')
        ax = fig.add_subplot(111)

    	# Determine number of neurons that will be plotted (for vertical offset)
    	offset = 0
    	n_to_plot = {}
        if self.spike_rec_type == 'frac':
    	    for pop in self.spike_data:
                n_to_plot[pop] = int(round(self.popsize[pop] * self.frac_record_spikes) * frac_neurons)
                offset = offset + n_to_plot[pop]
        if self.spike_rec_type == 'abs':
    	    for pop in self.spike_data:
                n_to_plot[pop] = int(round(self.n_record_spikes * frac_neurons))
                offset = offset + n_to_plot[pop]

        y_max = offset + 1
    	prev_pop = ''
    	yticks = []
        yticklocs = []
    	for pop in self.default_pops_flat:
            pop_data_temp = self.spike_data[pop]
            if pop[0:-1] != prev_pop:
                prev_pop = pop[0:-1]
                yticks.append(pop[0:-1])
                yticklocs.append(offset - 0.5 * n_to_plot[pop])
            if np.size(pop_data_temp) > 2:
                temp = pop_data_temp[pop_data_temp[:, 0] <= tmax]
                pop_data = temp[temp[:, 0] >= tmin]
                pop_spikes = pop_data[:, 0]
                pop_neurons = pop_data[:, 1]
            else:
                if np.size(pop_data_temp) == 2 and pop_data_temp[0] >= tmin and pop_data_temp[0] <= tmax:
                    pop_data = pop_data_temp
                    pop_spikes = np.array([pop_data[0]])
                    pop_neurons = np.array([pop_data[1]])
                if len(pop_data_temp) == 0:
                    pop_neurons = []

            if list(pop_neurons):
                if pop.lower().find('e') > (-1):
                    pcolor = '#595289'
                else:
                    pcolor = '#af143c'

                unique_neurons = np.unique(pop_neurons)
                neurons_to_plot = unique_neurons[unique_neurons < n_to_plot[pop]]

                for k in neurons_to_plot:
                    # we assume that neuron indices start at 0 in each population
                    spike_times = pop_spikes[pop_neurons == k]
                    ax.plot(spike_times, np.zeros(len(spike_times)) + offset - k, '.', color=pcolor, markersize=3)
            offset = offset - n_to_plot[pop]
        y_min = offset
    	ax.set_xlim([tmin, tmax])
    	ax.set_ylim([y_min, y_max])
    	ax.set_xlabel('time [ms]', size=16)
    	ax.set_ylabel('Population', size=16)

    	ax.set_yticks(yticklocs)
    	ax.set_yticklabels(yticks, fontsize='large')
        for tick in ax.xaxis.get_major_ticks() :
            tick.label.set_fontsize('large')
        if 'output' in keywords :
            p.savefig(self.label + '_dotplot.' + keywords['output'])

        fig.show()


    def plot_conn(self, source_pop, target_pop, plot_binomial, epsilon, **keywords):
        """ Plot in- and outdegree and multapse degree distributions
            Arguments:
            - source_pop: 'L23E' etc.
            - target_pop: 'L23E' etc.
            - plot_binomial: whether or not to overplot theoretical binomial distributions.
              If yes, the connection probability epsilon should be given. Uses self.popsize, which requires spikes to have been loaded
            Keywords:
            - compare: an optional second data set with which to compare
              assumed to have the same population sizes and epsilon as the first data set
        """
        if 'compare' in keywords:
            data2 = keywords['compare']

        if plot_binomial:
            n_pre = self.popsize[source_pop]
            n_post = self.popsize[target_pop]
            n_syn = round(np.log(1. - epsilon) / np.log((n_pre * n_post - 1.) / (n_pre * n_post)))

        fig = p.figure()
        ax = fig.add_subplot(131)
        hist1 = p.hist(self.nspt[source_pop][target_pop], normed=True, alpha=0.5)
        if 'compare' in keywords:
            p.hist(data2.nspt[source_pop][target_pop], bins=hist1[1], normed=True, alpha=0.5)
        if plot_binomial:
            x_vals = range(int(0.9 * min(self.nspt[source_pop][target_pop])), int(1.1 * max(self.nspt[source_pop][target_pop])))
            indegrees_theory = stats.binom.pmf(x_vals, n_syn, 1. / n_post)
            p.plot(x_vals, indegrees_theory, c='k', linewidth=2.)
        ax.set_xlabel('In-degree', size=16)
        ax.set_ylabel('Probability', size=16)

        ax = fig.add_subplot(132)
        hist1 = p.hist(self.ntps[source_pop][target_pop], normed=True, alpha=0.5)
        if 'compare' in keywords:
            p.hist(data2.ntps[source_pop][target_pop], bins=hist1[1], normed=True, alpha=0.5)
        if plot_binomial:
            x_vals = range(int(0.9 * min(self.ntps[source_pop][target_pop])), int(1.1 * max(self.ntps[source_pop][target_pop])))
            outdegrees_theory = stats.binom.pmf(x_vals, n_syn, 1. / n_pre)
            p.plot(x_vals, outdegrees_theory, c='k', linewidth=2.)
        ax.set_xlabel('Out-degree', size=16)
        ax.set_ylabel('Probability', size=16)

        ax = fig.add_subplot(133)
        multapse_degrees = np.reshape(self.multiplicities[source_pop][target_pop], np.size(self.multiplicities[source_pop][target_pop]))
        x_vals = range(int(1.2 * max(multapse_degrees)))
        p.hist(multapse_degrees, bins=np.array(x_vals) - 0.5, normed=True, alpha=0.5)
        if 'compare' in keywords:
            multapse_degrees = np.reshape(data2.multiplicities[source_pop][target_pop], np.size(data2.multiplicities[source_pop][target_pop]))
            p.hist(multapse_degrees, bins=np.array(x_vals) - 0.5, normed=True, alpha=0.5)
        if plot_binomial:
            multapses_theory = stats.binom.pmf(x_vals, n_syn, 1. / (n_pre * n_post))
            p.plot(x_vals, multapses_theory, c='k', linewidth=2.)
        ax.set_xlabel('Multapse degree', size=16)
        ax.set_ylabel('Probability', size=16)

        fig.show()

############################## Basic analysis and accessory plotting functions

def ff(psth):
    """calculate FanoFactor for given psth

    Parameters:
    - psth
    """
    if np.mean(psth) != 0.0:
        return np.var(psth) / np.mean(psth)
    else:
        print 'ff: no spikes in psth, returning 0.0'
        return 0.0

def psth(data_array, bin_width, tmin, tmax, h):
    """Create psth of data_array between tmin and tmax
    and for bin_width.

    Arguments:
    - data_array: column 0: neuron_ids, column 1: spike times
    - tmin and tmax in [ms]
    - bin_width [ms]
    - h: transform times in ms to format used in data_array
    """
    p = np.zeros(len(np.arange(tmin, tmax, bin_width)))
    if len(data_array) > 0:
        for event in data_array:
            if ((event[0] > tmin * h) and (event[0] < tmax * h)):
                bin = int(np.floor((event[0] - tmin * h) / (bin_width * h)))
                p[bin] = p[bin] + 1
    return p


def sync(data_array, bin_width, tmin, tmax, h):
    """Compute synchrony as variance of the population signal
       normalized by the mean variance of the individual neurons.
       See e.g. Morrison et al. (2007) Neural Comput, Eq. (5.1)
       Only those neurons that spiked at least once are taken into account.
    """
    senders = data_array[:, 1]
    neurons = np.unique(senders)
    n_neurons = len(neurons)
    pop_signal = psth(data_array, bin_width, tmin, tmax, h)
    neuron_var = 0
    for ii in xrange(n_neurons):
        neuron_data = data_array[pylab.find(senders == neurons[ii])]
        neuron_signal = psth(neuron_data, bin_width, tmin, tmax, h)
        neuron_var += np.var(neuron_signal)
    return np.var(pop_signal) / (neuron_var * n_neurons)


def phase_sync(spikes, senders, tmin, tmax, dt):
    '''Phase synchrony as a function of time. Between each spike pair the phase increases from 0 to 2*pi, and is sampled at step dt. At each point, the phase is 2*pi*(time since last spike)/(interspike interval). This can probably be sped up.'''
    # TODO: reduce size of spikes and senders according to tmin and tmax
    tic = time.time()
    t = np.arange(tmin, tmax + dt, dt)
    neurons = np.unique(senders)
    syncsum = np.zeros(len(t), 'complex')
    for ii in xrange(len(neurons)):
	sp = list(np.sort(spikes[list(pylab.find(senders == neurons[ii]))]))
        # sp = [spikes[n] for n in xrange(len(spikes)) if senders[n]==neurons[ii]]
	mean_ISI = np.mean(np.diff(sp))
	if (not np.isnan(mean_ISI)):
            sp = np.unique([min(sp[0] - mean_ISI, tmin)] + sp + [max(sp[-1] + mean_ISI, tmax)])
	else:
            sp = np.unique([tmin] + sp + [tmax])
	phase = np.zeros(len(t))
	expval = np.zeros(len(phase), 'complex')
	for jj in xrange(len(t)):
	    del_t = t[jj] - sp
	    index = pylab.find(del_t == min(del_t[del_t >= 0]))
	    t_since_last = del_t[index]
	    if t_since_last == 0:
	      phase[jj] = 0
	    else:
	      t_between_spikes = t_since_last - del_t[index + 1]
	      phase[jj] = 2 * np.pi * t_since_last / t_between_spikes
        # syncsum += [np.exp(complex(0,1)*phase[kk]) for kk in xrange(len(phase))]
	for kk in xrange(np.size(phase)):
	    expval[kk] = np.exp(complex(0, 1) * phase[kk])
	syncsum += expval
    sync = abs(syncsum) / len(neurons)
    toc = time.time()
    time_taken = toc - tic
    print 'phase_sync took ' + str(time_taken) + ' s'
    return sync


def set_spines_bottom_left(axs):
    for ax in axs:
        for loc, spine in ax.spines.iteritems():
            if loc in ['right', 'top']:
                spine.set_color('none')  # don't draw spine
            elif not (loc in ['left', 'bottom']):
                raise ValueError('unknown spine location: %s' % loc)
        if loc in ['left', 'bottom']:
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')


