import numpy as np
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib as mpl

# report = '2019-06-06-19-16-41-770101'
# report = '2019-06-06-18-43-10-793945'


# report = '2019-06-06-16-23-35-727749'

# report = '2019-06-06-18-43-10-793945'
#
# report = '2019-06-06-19-16-41-770101'

# report = '2019-06-07-10-00-38-312734' # test A

report = '2019-06-07-10-24-38-503374' #Works - only 10 ms (not time for drift?)

# report = '2019-06-08-18-56-14-566607'
report = '2019-06-09-10-23-52-830149' # 20x slowdown, 1000ms sim time



reports_dir = '/Users/oliver/Desktop/cspc457_reports/reports'
provenence_path = "run_1/provenance_data"

layer_keys = [
    ['L23E', 'L23I'],
    ['L4E', 'L4I'],
    ['L5E', 'L5I'],
    ['L6E', 'L6I']
    ]

out_dir = reports_dir + '/' + report + '/' + provenence_path

file_list = glob.glob(out_dir + "/*L*syn_vertex*.xml")


map = np.zeros([28,28,18])

max_x = 0
max_y = 0
min_x = 28
min_y= 28



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
        if child.attrib['name'] == field[1]:
            dropped_spikes = int(child.text)
#             print(child.text)
            if dropped_spikes < 1:
                dropped_spikes = 0.01
                print "hi"

    map[chip_y, chip_x, core] = dropped_spikes

print min_x, max_x, min_y, max_y

for i in range(28):
    for j in range(28):
        sum = np.sum(map[i, j, 1:17])

        map[i, j, 0] = sum

    print '\n'

plt.figure()
cmap = mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
cmap.set_over('0.25')
cmap.set_under('0.75')

bounds = [0, 0.001, 1, 100, 1000]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
#                                 norm=norm,
#                                 boundaries=[0] + bounds + [13],
#                                 extend='both',
#                                 ticks=bounds,
#                                 spacing='proportional',
#                                 orientation='horizontal')
plt.imshow(map[:, :, 0],  origin='lower', cmap=cmap, norm=norm)
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
plt.xticks(range(min_x, max_x, 2))
plt.yticks(range(min_y, max_y, 2))
plt.colorbar()

plt.show()


