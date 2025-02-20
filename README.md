[![CI Check](https://github.com/SpiNNakerManchester/microcircuit_model/workflows/Python%20Actions/badge.svg?branch=master)](https://github.com/SpiNNakerManchester/microcircuit_model/actions?query=workflow%3A%22Python+Actions%22+branch%3Amaster)

# Cortical microcircuit simulation: PyNN version
_Stored for easy access for people within the organisation_

**Contributors:**
Sacha van Albada (s.van.albada@fz-juelich.de)
Maximilian Schmidt
Jannis Schücker
Andrew Rowley
Alan Stokes

This is an implementation of the multi-layer microcircuit model of early
sensory cortex published by Potjans and Diesmann (2014) _The cell-type specific
cortical microcircuit: relating structure and activity in a full-scale spiking
network model_. Cerebral Cortex 24 (3): 785-806, [doi:10.1093/cercor/bhs358](https://doi.org/10.1093/cercor/bhs358)

It has been run on three different back-ends: NEST, SpiNNaker, and the ESS (emulator of HMF)

# Instructions

1. Ensure you have the desired back-end.

   For SpiNNaker see https://spinnakermanchester.github.io/latest/spynnaker.html

   For NEST see http://www.nest-initiative.org/index.php/Software:Download
   and to enable full-scale simulation, compile it with MPI support
   (use the --with-mpi option when configuring) according to the instructions on
   http://www.nest-initiative.org/index.php/Software:Installation

2. Install PyNN according to the instructions on
   http://neuralensemble.org/docs/PyNN/installation.html

3. Run the simulation by typing ```python run_microcircuit.py <simulator>``` in
   your terminal in the folder containing this file, where ```<simulator>``` is one
   of ```nest``` or ```spinnaker``` (by default ```spinnaker is selected```).  There
   are several potential arguments which can be seen by typing
   ```python run_microcircuit.py <simulator> -h```.  A few useful ones include:

    - --sim_duration - The simulation duration in milliseconds (default 1000)
    - --output_path  - Where output files should be written (default results)

6. Output files and basic analysis:

   - Spikes are written to .txt files containing IDs of the recorded neurons
     and corresponding spike times in ms.
     Separate files are written out for each population and virtual process.
     File names are formed as 'spikes'+ layer + population + MPI process + .txt
   - Voltages are written to .dat files containing GIDs, times in ms, and the
     corresponding membrane potentials in mV. File names are formed as
     voltmeter label + layer index + population index + spike detector GID +
     virtual process + .dat

   - If 'plot_spiking_activity' is set to True, a raster plot and bar plot
     of the firing rates are created and saved as 'spiking_activity.png'

This simulation is part of the Spynnaker integration tests so is tested daily.
The tests use python 3.12 and the latest possible version of each dependency unless restricted by
https://github.com/SpiNNakerManchester/sPyNNaker/blob/master/requirements.txt

## Known issues:

- At least with PyNN 0.7.5 and NEST revision 10711, ConnectWithoutMultapses
  works correctly on a single process, but not with multiple MPI processes.

- When saving connections to file, ensure that pyNN does not create problems
  with single or nonexistent connections, for instance by adjusting
  lib/python2.6/site-packages/pyNN/nest/__init__.py from line 365 as follows:

      if numpy.size(lines) != 0:
          if numpy.shape(numpy.shape(lines))[0] == 1:
              lines = numpy.array([lines])
              lines[:,2] *= 0.001
              if compatible_output:
                  lines[:,0] = self.pre.id_to_index(lines[:,0])
                  lines[:,1] = self.post.id_to_index(lines[:,1])
          file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})

- To use saveConnections in parallel simulations, additionally ensure that
  pyNN does not cause a race condition where the directory is created by one
  process between the if statement and makedirs on another process: In
  lib/python2.6/site-packages/pyNN/recording/files.py for instance replace

      os.makedirs(dir)

  by

      try:
          os.makedirs(dir)
      except OSError, e:
          if e.errno != 17:
              raise
          pass

Reinstall pyNN after making these adjustments, so that they take effect
in your pyNN installation directory.
