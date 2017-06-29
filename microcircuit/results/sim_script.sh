
#PBS -o results/output.txt
#PBS -e results/errors.txt 
#PBS -l walltime=8:0:0
#PBS -l nodes=1:ppn=24
#PBS -l mem=4gb
. /usr/local/mpi/openmpi/1.4.3/gcc64/bin/mpivars_openmpi-1.4.3_gcc64.sh
mpirun -np 24 python results/microcircuit.py
