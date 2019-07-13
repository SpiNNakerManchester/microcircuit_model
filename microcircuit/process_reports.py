import numpy as np
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib as mpl

# report = "Spin24B_20x_1s_whole"
# report = "Spin24B_30x_0.7s_whole"
# report = "Spin24B_20x_1s_whole_take2"
# # report = "Spin24B_20x_drift_alignment_trial1"
# report = "1x_da_2synexc_opt"

report = '2x_act_check_shift2'


# reports_dir = '/Users/oliver/Documents/Spin24B_testing/reports'
reports_dir = '/Users/oliver/Documents/split_exc_res/reports'
reports_dir = '/Users/oliver/Documents/10x_shift_exploration'

provenence_path = "run_1/provenance_data"

layer_keys = [
    ['L23E', 'L23I'],
    ['L4E', 'L4I'],
    ['L5E', 'L5I'],
    ['L6E', 'L6I']
    ]

out_dir = reports_dir + '/' + report + '/' + provenence_path

file_list = glob.glob(out_dir + "/*L*syn_vertex_*.xml") # all

file_list = glob.glob(out_dir + "/*L*syn_vertex_1*.xml") # inh
# file_list = glob.glob(out_dir + "/*L*low_syn_vertex_0*.xml") # exc low
# file_list = glob.glob(out_dir + "/*L*high_syn_vertex_0*.xml") # exc high
# file_list = glob.glob(out_dir + "/*L*_syn_vertex_0*.xml") # exc high


x_lim = 100
y_lim = 100
max_x = 0
max_y = 0
min_x = x_lim
min_y= y_lim

map = np.zeros([min_x, min_y, 18])

map[:] = -1

field = ["Timestep during which we dropped more spikes",
         "Total dropped spikes"]

for f in file_list:

    # Get chip/core coordinates for plotting
    file_name = f.split('/')[-1]
    chip_x = int(file_name.split('_L')[0].split('_')[0])
    chip_y = int(file_name.split('_L')[0].split('_')[1])
    core = int(file_name.split('_L')[0].split('_')[2])

    if chip_x > max_x:
        max_x = chip_x
    if chip_y > max_y:
        max_y = chip_y

    if chip_x < min_x:
        min_x = chip_x

    if chip_y < min_y:
        min_y = chip_y

    print f.split('/')[-1], chip_x, chip_y, core
    tree = ET.parse(f)
    root = tree.getroot()

    for child in root.getchildren():
        if child.attrib['name'] == field[0]:
            dropped_spikes = int(child.text)
#             print(child.text)
#             if dropped_spikes < 1:
#                 dropped_spikes = 0.01
#                 print "hi"


    map[chip_y, chip_x, core] = dropped_spikes

print min_x, max_x, min_y, max_y

for i in range(x_lim):
    for j in range(y_lim):
        var = np.max(map[i, j, 1:17])

        map[i, j, 0] = var

    print '\n'

plt.figure()
cmap = mpl.colors.ListedColormap(['red', 'orange', 'green', 'blue', 'cyan'])
cmap.set_over('0.25')
cmap.set_under('0.75')

# bounds = [0, 0.000000001, 2500, 5000, 7500, 10000]
# bounds = [0, 0.000000001, 25000/2, 50000/2, 75000/2, 100000/2]
bounds = [0, 1, 2, 3, 4, 5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
#                                 norm=norm,
#                                 boundaries=[0] + bounds + [13],
#                                 extend='both',
#                                 ticks=bounds,
#                                 spacing='proportional',
#                                 orientation='horizontal')
plt.imshow(map[:, :, 0],  origin='lower'
        , cmap=cmap, norm=norm
        )
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
plt.xticks(range(min_x, max_x, 2))
plt.yticks(range(min_y, max_y, 2))
plt.colorbar()

plt.show()


