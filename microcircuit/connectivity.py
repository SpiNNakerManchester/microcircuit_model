from network_params import *
import random
from scipy import stats
import os
import math
import pyNN
from pyNN.random import RandomDistribution
from sim_params import *
import numpy as np

def FixedTotalNumberConnect_NEST(sim, pop1, pop2, K, w_mean, w_sd, d_mean, d_sd):
    """Function connecting two populations with multapses and a fixed total number of synapses
       Using new NEST implementation of Connect"""

    if not K:
        return

    source_neurons = list(pop1.all_cells)
    target_neurons = list(pop2.all_cells)
    n_syn = int(round(K*len(target_neurons)))
    # weights are multiplied by 1000 because NEST uses pA whereas PyNN uses nA
    # RandomPopulationConnectD is called on each process with the full sets of
    # source and target neurons, and internally only connects the target
    # neurons on the current process.

    conn_dict = {'rule' : 'fixed_total_number',
                 'N'    : n_syn}

    syn_dict = {'model' : 'static_synapse',
                'weight': {'distribution': 'normal_clipped',
                           'mu': 1000. * w_mean,
                           'sigma': 1000. * w_sd},
                'delay' : {'distribution': 'normal_clipped',
                           'low': simulator_params[simulator]['min_delay'],
                           'mu': d_mean,
                           'sigma': d_sd}}
    if w_mean > 0:
       syn_dict['weight']['low'] = 0.0
    if w_mean < 0:
       syn_dict['weight']['high'] = 0.0

    sim.nest.sli_push(source_neurons)
    sim.nest.sli_push(target_neurons)
    sim.nest.sli_push(conn_dict)
    sim.nest.sli_push(syn_dict)
    sim.nest.sli_run("Connect")

    if save_connections:
        # - weights are in pA
        # - no header lines
        # - one file for each MPI process
        # - GIDs

        # get connections to target on this MPI process
        conn = sim.nest.GetConnections(source=source_neurons, target=target_neurons)
        conns = sim.nest.GetStatus(conn, ['source', 'target', 'weight', 'delay'])
        if not os.path.exists(system_params['conn_dir']):
            try:
                os.makedirs(system_params['conn_dir'])
            except OSError as e:
                if e.errno != 17:
                    raise
                pass
        f = open(system_params['conn_dir'] +  '/' + pop1.label + "_" + \
                 pop2.label + '.conn' + str(sim.rank()), 'w')
        for c in conns:
            f.write(str(c).replace('(','').replace(')','').replace(', ', '\t'))
        f.close()



def FixedTotalNumberConnect_SpiNNaker(sim, pop1, pop2, K, w_mean, w_sd, d_mean, d_sd, conn_type, rng):
    """SpiNNaker-specific function connecting two populations with multapses and a fixed total number of synapses"""

    if not K:
        return

    n_syn = int(round(K*len(pop2)))

    if delay_dist_type == 'normal':
        d_dist = RandomDistribution(
            'normal_clipped', mu=d_mean, sigma=d_sd, rng=rng,
                 low=simulator_params[simulator]['min_delay'],
                 high=simulator_params[simulator]['max_delay'])
    elif delay_dist_type == 'uniform':
        d_dist = RandomDistribution(
            'uniform', low=d_mean - d_sd, high=d_mean + d_sd, rng=rng)

    if w_mean > 0:
        w_dist = RandomDistribution(
            'normal_clipped', mu=w_mean, sigma=w_sd, rng=rng,
            low=0., high=np.inf)
    else:
        w_dist = RandomDistribution(
            'normal_clipped', mu=w_mean, sigma=w_sd, rng=rng,
            low=-np.inf, high=0.)

    if conn_type == "excitatory":
        syn = sim.StaticSynapse(weight=w_dist, delay=d_dist)
    else:
        syn = sim.STDPMechanism(
            timing_dependence=sim.SpikePairRule(),
            weight_dependence=sim.AdditiveWeightDependence(
                w_min=w_mean-w_sd, w_max=w_mean+w_sd),
            weight=w_mean, delay=d_dist)
    connector = sim.FixedTotalNumberConnector(n=n_syn, rng=rng)
    proj = sim.Projection(pop1, pop2, connector, syn, receptor_type=conn_type)

    if save_connections:
        proj.saveConnections(system_params['conn_dir'] + '/' + pop1.label \
                             + "_" + pop2.label + '.conn', gather=True)



def FromListConnect(sim, pop1, pop2, conn_type, base_neuron_ids):
    """Establish connections based on data read from file"""
    connections = list()
    for filename in os.listdir(system_params['conn_dir']):
        if filename.startswith(pop1.label + "_" + pop2.label):
            print("Reading {}".format(filename))
            f = open(os.path.join(system_params['conn_dir'], filename))
            in_comment_bracket = False
            for line in f:
                if line.startswith("#"):
                    if "[" in line:
                        in_comment_bracket = True
                if not line.startswith("#"):
                    if in_comment_bracket:
                        if "]" in line:
                            in_comment_bracket = False
                    else:
                        line = line.strip()
                        (source_id, target_id, weight, delay) = line.split()
                        source_id = int(math.floor(float(source_id))) - base_neuron_ids[pop1]
                        target_id = int(math.floor(float(target_id))) - base_neuron_ids[pop2]
                        if source_id < 0 or target_id < 0:
                            print(line, base_neuron_ids[pop1], base_neuron_ids[pop2])
                        connections.append((source_id, target_id,
                                float(weight) / 1000.0, float(delay)))
            f.close()
    if len(connections) > 0:
        connector = sim.FromListConnector(conn_list=connections)
        sim.Projection(pop1, pop2, connector, receptor_type=conn_type)
