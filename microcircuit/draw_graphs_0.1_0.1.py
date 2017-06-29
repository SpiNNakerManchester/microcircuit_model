from ana_vistools import data
from matplotlib import pyplot

vis = data(simtime=1000., n_scaling=0.1)
vis.load_spikes(rec_type="frac", rec=1.0, dir="results_0.1_0.1")
vis.dot_display()
pyplot.show()
