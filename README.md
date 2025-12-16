# HAMSTER-v2.0
Parallel version of the Heat And MoiSture Tracking framEwoRk

----

This is an updated and improved version of the *Heat And MoiSture Tracking framEwoRk* (see https://github.com/h-cel/hamster). One of the main updates is that part of the code is now parallelized. The new version follows the same logic as the previous version in that HAMSTER is mainly based on the WaterSip moisture tracking algorithm (https://gfi.wiki.uib.no/WaterSip), but including bias correction by ingesting evaporation and precipitation observations. Furthermore, unlike WaterSip, HAMSTER is also adapted for tracking sensible heat. However, this new version has a completely different code structure to the previous one. HAMSTER v2 now consists of two python scripts (*preprocessing.py* and *tracking.py*), plus an additional one with some useful functions (*utilities.py*). It also includes a control file (*namelist.input*) to set some input variables.

## Context

HAMSTER v2 attributes the moisture sources for a precipitation event in a specific region, which we will normally refer to as the sink  region. This is achieved by tracking the moisture gains and losses of the air parcels contributing to the precipitation event under study in the days prior to the event. The definition of an air parcel must be understood from a Lagrangian point of view, in which the atmosphere is divided into particles that follow the atmospheric flow. HAMSTER therefore needs to be forced with the outputs of a Lagrangian transport model, which provides us with the positions of the parcels as well as their associated specific humidity, among other variables. The code is only set up to read the outputs of the FLEXPART v11 Lagrangian model (see https://www.flexpart.eu). To work with another Lagrangian model, the part of the code responsible for reading the input data would have to be adapted.

Once the variations in the specific humidity of the parcels have been tracked, HAMSTER produces a two-dimensional field of moisture sources, like the one shown below:

<div align="center">
  <img src="https://github.com/user-attachments/assets/c59e8a29-7542-4607-a799-2d4f439220e6" width="500"/>
</div>

This field, which we call E2P, represents, for each cell in the grid, the amount of moisture evaporated in that cell that ends up contributing to the precipitation event of interest. In the case of this figure, the analyzed precipitation event is the one that caused the floods in Germany and Belgium in July 2021 (black box). HAMSTER can produce analogous fields for the contribution of evaporation to integrated water vapor (E2Q) and for the contribution of sensible heat to the temperature in the sink region (which we call H2T).

## Components

1) **preprocessing.py**

   It reads the raw outputs of FLEXPART v11 (partoutputs) and iterates on dates to calculate the changes in specific humidity (*Delta_q*) of all the air parcels, or the changes in potential temperature (*Delta_theta*) for heat tracking. To this end, in each iteration, two files are read and their specific humidities (*qv*) are subtracted, that is, *Delta_q = qv[1,:]-qv[0,:]*. Subsequently, these *Delta_q* are corrected using observed evaporation and precipitation values. As a simple example, if on a given grid cell we have positive *Delta_q* values in the atmospheric column but the observed evaporation in that cell is equal to zero, we correct those *Delta_q* to zero. Likewise, if we have negative *Delta_q* but there is no rain, *Delta_q* is set to zero. This is one of the main improvements of HAMSTER v2 with respect to its previous version, since now the bias correction is applied directly to *Delta_q* (i.e. en route) and not a posteriori to correct the output fields (e.g. E2P). More specifically, previously, the bias correction for precipitation only scaled the E2P field to make it consistent with the rainfall in the sink region, but had no effect on the relative contributions of the sources. Now both the bias correction for evaporation and precipitation affect the spatial distribution of fields such as E2P and therefore the relative contributions of the sources. Once the *Delta_q* and/or *Delta_theta* have been corrected, the portoutputs files are updated with these new arrays.

3) **tracking.py**

   It reads the modified partouputs,  i.e., those containing the corrected *Delta_q* and/or *Delta_theta*. It also reads the sink region of interest, in which we select the air parcels to be tracked. It then applies the WaterSip algorithm to calculate the source fields (E2P, E2Q or H2T) using these corrected values. The arrays in this part of the code can be very heavy as the tracking is done in the 30 days prior to the study date. If we work with a time step of 3 hours and a total of 20 million parcels, this implies that *Delta_q*, for instance, will have the dimensions 240*20000000. This means that the code has a high demand for RAM. Furthermore, even when the arrays are loaded into RAM, the code may not be fast enough, especially if you want to analyze long periods of time. For this reason, this part of the code uses MPI parallelization to reduce computation time.

## Prerequisites 

It requires python 3 and the python modules shown below:

```
module purge
module load xarray/2023.9.0-gfbf-2023a
module load numba/0.58.1-foss-2023a
module load netcdf4-python/1.6.4-foss-2023a
module load mpi4py/3.1.4-gompi-2023a
module load h5py/3.9.0-foss-2023a
```
The example above shows how we load these modules on the HPC-UGent environment, but, of course, we could install them in our own environment and use other versions.

## Running HAMSTER v2

The *preprocessing.py* script only needs to be run once, while *tracking.py* has to be run for each of the sink regions we are interested in. On the other hand, *preprocessing.py* is not as RAM demanding as *tracking.py* is. That is why *preprocessing.py* runs serially and *tracking.py* is parallelized. Therefore, both scripts have to be run separately, first *preprocessing.py* and, once finished, *tracking.py*. An example of sending both *preprocessing.py* and *tracking.py* to the HPC-UGent queues can be found in the *example* folder. 

As an example, a submission script for *preprocessing.py* would look like this:

```
#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=08:00:00
#PBS -l mem=6GB
#PBS -A 2022_204

.
.
.

python $run_path/preprocessing.py
```

Whereas for *tracking.py*:

```
#!/bin/bash

#PBS -l nodes=1:ppn=32
#PBS -l walltime=06:00:00
#PBS -l mem=120GB
#PBS -A 2022_204

.
.
.

mpirun -np $PBS_NP python $run_path/tracking.py
```

## Control file

To make the program more user-friendly, the input variables are selected in a control file that looks like this:
```
[DATES]
start_date=set_dateini
end_date=set_datefin

[CONSTANTS]
num_days=30
dt=3
dx=0.5
dy=0.5
num_parc=19866694
atm_mass=set_mass

[PATHS]
path_data1=/XX/XX/set_path_to_partoutputs_folder
path_data2=/XX/XX/set_path_to_tracking_output
path_mask=/XX/XX/set_path_to_sink
path_obs=/XX/XX/set_path_to_observations

[NAMES]
obs_evap=evap_era5_3hourly
obs_heat=heat_era5_3hourly
obs_prec=prec_era5_3hourly

[FLAGS]
track_heat=false
obs_era5=true
save_e2q=true
save_diagnose=true
 ```
In the DATES section, the start_date and end_date of the period of interest are selected. In CONSTANTS, num_days would be the number of days we want to track the air parcels, dt, dx and dy are the temporal and spatial resolution of the data, respectively, num_parc is the total number of air parcels in the Lagrangian model simulations and atm_mass is the total atmospheric mass in these simulations. In the PATHS section we include the path to the folder containing the FLEXPART output files (path_data1), the path to the folder where we are going to store the HAMSTER outputs (path_data2), the path to the file containing the sink region (path_mask) and the path to the observational data files for bias correction (path_obs). NAMES stands for the names of the NetCDF files containing the observed evaporation (obs_evap), heat (obs_heat) and precipitation (obs_prec). Finally, in FLAGS, we include the option to activate or deactivate heat tracking (track_heat), and  also to save or not save E2Q (save_e2q), as well as the E and P diagnostics (save_diagnose). In addition, in this section, obs_era5 is used to indicate whether or not observed ERA5 data are used for bias correction.

## Backward and forward versions

HAMSTER is able to track moisture both backwards and forwards in time. What we have described so far is the backwards variant, for which the sink region is predefined and the changes in specific humidity of the air parcels that reach it during the rain event are tracked. Another option is to predefine the source and track the moisture that evaporates from it over the next 30 days. This is the forward version. In this case the output fields of HAMSTER are called PFE (precipitation from evaporation), QFE (integrated water vapor from evaporation) and HFS (sensible heat from source). Here is an example for rainfall from evaporation in North America in July 2021, i.e. PFE:

<div align="center">
  <img src="https://github.com/user-attachments/assets/020671cf-585c-4c89-b2fa-f64939225a23" width="500"/>
</div>

The forward version can be found in the *parallel-HAMSTER_forward* directory.

## Caveats

The code is only adapted to ingest data in a specific format. We have already given an example; if you want to use the outputs of a Lagrangian model other than FLEXPART v11, the code has to be adapted. Likewise, the target regions must have a specific format in terms of coordinates and variable name (*mask*). An example of a sink region can be seen in the *example* folder and an example of a source region in *parallel-HAMSTER_forward/example*. On the other hand, the code is only adapted to use observational data from ERA5. If you want to use another dataset for bias correction, the *preprocessing.py* script has to be modified accordingly. The time step for the observations must be the one set in *namelist.input*, which means that if dt=3, the observations must be 3-hourly. Furthermore, the dates in the observations must be centered (e.g. evaporation from 00 to 03 UTC has a date of 01:30 UTC). This can be done by modifying the ERA5 files using, for instance, the CDO settaxis function, or by directly  modifying *preprocessing.py*. In the output of *tracking.py*, the filename will contain the date of the end of the period, but the timestamp in the file is also centered. For example, the moisture source for rainfall between 00 and 03 UTC on June 1, 2015 will be saved as attribution_e2p_20150601030000.nc, but in reality the time variable will be set to 01:30 UTC. In short, it would be advisable to be familiar with the code and not just with its execution. This limits the chances of making mistakes that go unnoticed.

## References

***Sodemann, H., Schwierz, C., & Wernli, H.*** Interannual variability of Greenland winter precipitation sources: Lagrangian moisture diagnostic and North Atlantic Oscillation influence. J. Geophys. Res. D: Atmos., 113 (2008).

***Keune, J., Schumacher, D. L. & Miralles, D. G.*** A unified framework to estimate the origins of atmospheric moisture and heat using Lagrangian models. Geosci. Model Dev. 15, 1875–1898 (2022).

***Bakels, L., Tatsii, D., Tipka, A., Thompson, R., Dütsch, M., Blaschek, M., ... & Stohl, A.*** FLEXPART version 11: Improved accuracy, efficiency, and flexibility. Geosci. Model Dev., 17, 7595-7627 (2024).
