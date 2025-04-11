#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=08:00:00
#PBS -l mem=6GB
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
MASS=$3

#Start and end dates
YEAR=$(printf ${DATEI:0:4})
MONTH=$(printf ${DATEI:5:2})

#Folder paths
base_path=/dodrio/scratch/users/vsc45925/scratch_s4c/JOBS/FLOODS2021/FINAL_TESTS_v11/FORWARD_parallel_32
run_path=/dodrio/scratch/users/vsc45925/scratch_s4c/JOBS/FLOODS2021/FINAL_TESTS_v11/FORWARD_parallel_32/RUNS/$YEAR/$MONTH
out_path=/dodrio/scratch/users/vsc45925/scratch_s4c/JOBS/FLOODS2021/FINAL_TESTS_v11/FORWARD_parallel_32/OUTS/$YEAR/$MONTH

############################# MODIFY namelist.input #############################

cd $run_path

sed -i -e "s/dateini/$DATEI/g" -i -e "s/datefin/$DATEF/g" -i -e "s/year/$YEAR/g" -i -e "s/month/$MONTH/g" -i -e "s/tma/$MASS/g" ./namelist.input

############################ Link raw partoutputs ############################
 
mkdir -p $out_path/partoutputs
ln -s /dodrio/scratch/users/vsc45925/scratch_s4c/JOBS/FLOODS2021/VERSION_11_DATA/2021/* $out_path/partoutputs

############################ RUN preprocessing.py ############################

python $run_path/preprocessing.py