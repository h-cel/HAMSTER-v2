#!/bin/bash

#PBS -l nodes=1:ppn=32
#PBS -l walltime=06:00:00
#PBS -l mem=120GB
#PBS -A 2022_204

################################# SET GENERAL #################################

module purge
module load xarray/2023.9.0-gfbf-2023a
module load numba/0.58.1-foss-2023a
module load netcdf4-python/1.6.4-foss-2023a
module load mpi4py/3.1.4-gompi-2023a
module load h5py/3.9.0-foss-2023a

ulimit -s unlimited

#Input data 
DATEI=$1
DATEF=$2
SINK=$3
MASS=$4

#Start and end dates
YEAR=$(printf ${DATEI:0:4})
MONTH=$(printf ${DATEI:5:2})

#Folder paths
base_path=/dodrio/scratch/users/vsc45925/scratch_s4c/JOBS/FLOODS2021/FINAL_TESTS_v11/BACKWARD_parallel_32
run_path=/dodrio/scratch/users/vsc45925/scratch_s4c/JOBS/FLOODS2021/FINAL_TESTS_v11/BACKWARD_parallel_32/RUNS/$YEAR/$MONTH/$SINK
out_path=/dodrio/scratch/users/vsc45925/scratch_s4c/JOBS/FLOODS2021/FINAL_TESTS_v11/BACKWARD_parallel_32/OUTS/$YEAR/$MONTH/$SINK

############################## MODIFY namelist.input##############################

cd $run_path

sed -i -e "s/dateini/$DATEI/g" -i -e "s/datefin/$DATEF/g" -i -e "s/year/$YEAR/g" -i -e "s/month/$MONTH/g" -i -e "s/tma/$MASS/g" -i -e "s/sink/$SINK/g" ./namelist.input

############################## RUN tracking.py ##############################

mpirun -np $PBS_NP python $run_path/tracking.py