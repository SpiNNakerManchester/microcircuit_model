import os
import math


def build_from_list_connect(
        sim, pop1, pop2, conn_type, base_neuron_ids, simulator_params):
    """
    Establish connections based on data read from file
    :param sim:
    :param pop1:
    :param pop2:
    :param conn_type:
    :param base_neuron_ids:
    :param simulator_params:
    :return:
    """
    connections = list()
    for filename in os.listdir(simulator_params.conn_dir):
        if filename.startswith(pop1.label + "_" + pop2.label):
            print("Reading {}".format(filename))
            f = open(os.path.join(simulator_params.conn_dir, filename),
                     encoding="utf8")
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
                        source_id = (
                            int(math.floor(float(source_id))) -
                            base_neuron_ids[pop1])
                        target_id = (
                            int(math.floor(float(target_id))) -
                            base_neuron_ids[pop2])
                        if source_id < 0 or target_id < 0:
                            print(line, base_neuron_ids[pop1],
                                  base_neuron_ids[pop2])
                        connections.append(
                            (source_id, target_id, float(weight) / 1000.0,
                             float(delay)))
            f.close()
    if len(connections) > 0:
        connector = sim.FromListConnector(conn_list=connections)
        sim.Projection(pop1, pop2, connector, receptor_type=conn_type)
