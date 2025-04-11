#!/usr/bin/python
"""
Created on 27/04/2025

Author: Damian Insua Costa
"""

#Import modules
import sys
import os
import xarray as xr
import time
import numpy as np
import tracemalloc
from datetime import datetime, timedelta
import pandas as pd
import json
from utilities import *
import configparser
config=configparser.ConfigParser()
config.read('namelist.input')
import h5py

c1 = time.time()
tracemalloc.start()

#0)
#----------------------------------------Pre-loop definitions-----------------------------------------

#Define some global variables    
dt=int(config['CONSTANTS']['dt'])#Temporal resolution
num_days=int(config['CONSTANTS']['num_days'])#Number of days to track forward
num_parc=int(config['CONSTANTS']['num_parc'])#Number of parcels
dx = float(config['CONSTANTS']['dx']); nx = int(360/dx) #Grid definition x direction
dy = float(config['CONSTANTS']['dy']); ny = int(180/dy)+1 #Grid definition y direction
atm_mass=float(config['CONSTANTS']['atm_mass']) #Total atmospheric mass
mass=atm_mass/num_parc #Atmospheric mass per parcel

#Define the grid to be used for output fields
grid_lon=np.arange(-180.,180.,dx).astype(np.float32)
grid_lat=np.arange(-90.,90.+dy,dy).astype(np.float32)
grid_lon,grid_lat=np.meshgrid(grid_lon,grid_lat)
lats1 = (grid_lat+(dy/2))*np.pi/180.  
lats2 = (grid_lat-(dy/2))*np.pi/180.
areas = dx*np.pi/180.0*(6371.*1000.)**2.*(np.sin(lats1)-np.sin(lats2))
areas[areas==0.0] = np.nan #Area of grid cells

#Define paths
path_data1=config['PATHS']['path_data1']
path_obs=config['PATHS']['path_obs']

#Define dates
start_date=datetime.strptime(config['DATES']['start_date'],'%Y-%m-%d_%H')+timedelta(hours=dt)
first_date=start_date
end_date=datetime.strptime(config['DATES']['end_date'],'%Y-%m-%d_%H')+timedelta(days=num_days)

while start_date<=end_date:
  #1)
  #----------------------------------------Definition of dates----------------------------------------

  #Dates
  arrival_date=start_date
  before_arrival=arrival_date-timedelta(hours=dt)
  middle_date=arrival_date-timedelta(hours=dt/2)
  dates=pd.date_range(start=before_arrival,end=arrival_date,freq=str(dt)+'H')

  #2)
  #---------------------------------------------Read data---------------------------------------------

  #List of files to open
  files=[path_data1+'/partoutputs/partoutput_'+d.strftime('%Y%m%d%H%M%S')+'.nc' for d in dates]

  #Initialise variables of interest
  lon = np.zeros((len(files), num_parc), dtype=np.float32)
  lat = np.zeros((len(files), num_parc), dtype=np.float32)
  qv = np.zeros((len(files), num_parc), dtype=np.float32)
  if json.loads(config['FLAGS']['track_heat'].lower()):
    temp = np.zeros((len(files), num_parc), dtype=np.float32)
    den = np.zeros((len(files), num_parc), dtype=np.float32)
  
  i = 0
  #Iterate through time to read the netcdf files one at a time using h5py
  for file in files:
    with h5py.File(file, 'r') as ds:
      lon[i, :] = ds['lon'][:num_parc].T
      lat[i, :] = ds['lat'][:num_parc].T
      lat[np.isnan(lat)] = -1.
      qv[i, :] = ds['sh'][:num_parc].T
      if json.loads(config['FLAGS']['track_heat'].lower()):
        temp[i, :] = ds['T'][:num_parc].T
        den[i, :] = ds['rho'][:num_parc].T
    i=i+1

  #Get potential temperature
  if json.loads(config['FLAGS']['track_heat'].lower()):
    theta=only_theta(qv,temp,den)
    del temp,den
  
  #Transform lon to -180,180
  lon[lon>180.]=lon[lon>180.]-360.
  
  #3)
  #---------------------------------------Load observations--------------------------------------

  #Some considerations:
  #*Times in the observations should also be centred, i.e. evaporation or precipitation from 00 to 06 should have time 03
  #*Here we use xarray for reading instead of h5py because it makes it easier to play with the coordinates
    
  file_eobs=path_obs+'/'+config['NAMES']['obs_evap']+'.nc'
  e_obs = xr.open_dataset(file_eobs).sel(time=middle_date)
  if json.loads(config['FLAGS']['obs_era5'].lower()):#correct latitudes and longitudes to match ours
    e_obs=e_obs.assign_coords({"longitude": (((e_obs.longitude + 180) % 360) - 180)})
    e_obs=e_obs.interp(latitude=np.arange(-90., 90.+dy, dy), longitude=np.arange(-180., 180., dx))
    
  file_pobs=path_obs+'/'+config['NAMES']['obs_prec']+'.nc'
  p_obs = xr.open_dataset(file_pobs).sel(time=middle_date)
  if json.loads(config['FLAGS']['obs_era5'].lower()):#correct latitudes and longitudes to match ours
    p_obs=p_obs.assign_coords({"longitude": (((p_obs.longitude + 180) % 360) - 180)})
    p_obs=p_obs.interp(latitude=np.arange(-90., 90.+dy, dy), longitude=np.arange(-180., 180., dx))

  if json.loads(config['FLAGS']['track_heat'].lower()):
    file_hobs=path_obs+'/'+config['NAMES']['obs_heat']+'.nc'
    h_obs = xr.open_dataset(file_hobs).sel(time=middle_date)
    if json.loads(config['FLAGS']['obs_era5'].lower()):#correct latitudes and longitudes to match ours
      h_obs=h_obs.assign_coords({"longitude": (((h_obs.longitude + 180) % 360) - 180)})
      h_obs=h_obs.interp(latitude=np.arange(-90., 90.+dy, dy), longitude=np.arange(-180., 180., dx))  
    
  #Load data into memory
  if json.loads(config['FLAGS']['obs_era5'].lower()):
    e_obs=e_obs.e.values.astype(np.float32)*-1000.
    e_obs[e_obs<0.]=0.
    p_obs=p_obs.tp.values.astype(np.float32)*-1000.
    p_obs[p_obs>0.]=0.
    if json.loads(config['FLAGS']['track_heat'].lower()):
      h_obs=h_obs.sshf.values.astype(np.float32)/(-dt*60.*60.)
      h_obs[h_obs<0.]=0.

  #4)
  #-------Determine, for each time interval, specific humidity changes due to evaporation and precipitation--------

  #Calculate the variation in specific humidity and average relative humidity along the trajectory
  Delta_q = qv[1,:]-qv[0,:]
  
  Delta_q[(Delta_q<=1.E-4)&(Delta_q>=-1.E-4)]=0.#This threshold does not have a great impact on the results but it does save computational cost later on

  #Calculate the indices (boolean array) of evaporation and precipitation as a function of the above variables
  ind_evap = Delta_q>0.
  ind_prec = Delta_q<0.

  #5)
  #-------Calculate the contributions of each parcel to evaporation (E) and (P) precipitation--------

  E=Delta_q[ind_evap]
  P=Delta_q[ind_prec]

  #6)
  #----------------------------------Replicte calculations for heat-----------------------------------

  if json.loads(config['FLAGS']['track_heat'].lower()):
    Delta_theta = theta[1,:]-theta[0,:]

    Delta_theta[(Delta_theta<=1.E-4)&(Delta_theta>=-1.E-4)]=0.
    
    ind_heat = Delta_theta>0.

    H=Delta_theta[ind_heat]

  #7)
  #-------------------Accumulate all contributions on a grid to construct 2D fields-------------------

  #Compute mid-point coordinates along trajectories
  lona=lon[1,:]
  lonb=lon[0,:]
  lata=lat[1,:]
  latb=lat[0,:]
  lonmidp,latmidp=midpoints(lona,lonb,lata,latb)
  del lon,lat,lona,lonb,lata,latb

  #Find indices corresponding to the parcels position in the grid           
  x_index = np.round((lonmidp+180.0)/dx).astype(int)
  x_index[x_index==nx] = 0
  y_index = np.round((latmidp+90.0)/dy).astype(int)
  
  #Clean
  del lonmidp,latmidp

  #Integrated water vapor
  x_index_iwv=np.copy(x_index)
  y_index_iwv=np.copy(y_index)
  iwv,num_par = ongrid1D(nx,ny,x_index_iwv,y_index_iwv,qv.mean(axis=0),mass)
  del qv

  #Evaporation
  x_index_evap=x_index[ind_evap]
  y_index_evap=y_index[ind_evap]
  evap,num_par_evap = ongrid1D(nx,ny,x_index_evap,y_index_evap,E,mass)
  del E

  #Precipitation
  x_index_prec=x_index[ind_prec]
  y_index_prec=y_index[ind_prec]
  prec,num_par_prec = ongrid1D(nx,ny,x_index_prec,y_index_prec,P,mass)
  del P

  #Heat
  if json.loads(config['FLAGS']['track_heat'].lower()):
    x_index_heat=x_index[ind_heat]
    y_index_heat=y_index[ind_heat]
    heat,num_par_heat = ongrid1D(nx,ny,x_index_heat,y_index_heat,H,mass)   
    del H

  #8)
  # ----------------------------Convert units and save to a netcdf file, if desired-------------------------------

  #Change units
  inv_density = 1000./997.#This transforms kg of water to mm
  cp = 1005.7 
  dts = dt*60.*60.
  iwv = iwv*inv_density/areas
  iwv = np.nan_to_num(iwv)#mm
  evap = evap*inv_density/areas
  evap = np.nan_to_num(evap)#mm
  prec = prec*inv_density/areas
  prec = np.nan_to_num(prec)#mm
  if json.loads(config['FLAGS']['track_heat'].lower()):
    heat = heat*cp/dts/areas
    heat = np.nan_to_num(heat)#W/m2

  #Create xarray dataset
  if json.loads(config['FLAGS']['save_diagnose'].lower()):
    if json.loads(config['FLAGS']['track_heat'].lower()):
      final=xr.Dataset(data_vars=dict(E=(["time","lat","lon"],np.reshape(evap,(1,evap.shape[0],evap.shape[1]))),E_n_part=(["time","lat","lon"],np.reshape(num_par_evap,(1,num_par_evap.shape[0],num_par_evap.shape[1]))),P=(["time","lat","lon"],np.reshape(prec,(1,prec.shape[0],prec.shape[1]))),P_n_part=(["time","lat","lon"],np.reshape(num_par_prec,(1,num_par_prec.shape[0],num_par_prec.shape[1]))),H=(["time","lat","lon"],np.reshape(heat,(1,heat.shape[0],heat.shape[1]))),H_n_part=(["time","lat","lon"],np.reshape(num_par_heat,(1,num_par_heat.shape[0],num_par_heat.shape[1]))),IWV=(["time","lat","lon"],np.reshape(iwv,(1,iwv.shape[0],iwv.shape[1]))),n_part=(["time","lat","lon"],np.reshape(num_par,(1,num_par.shape[0],num_par.shape[1])))),coords=dict(time=(["time"],[middle_date]),lat=(["lat"],grid_lat[:,0]),lon=(["lon"],grid_lon[0,:])))
    else:
      final=xr.Dataset(data_vars=dict(E=(["time","lat","lon"],np.reshape(evap,(1,evap.shape[0],evap.shape[1]))),E_n_part=(["time","lat","lon"],np.reshape(num_par_evap,(1,num_par_evap.shape[0],num_par_evap.shape[1]))),P=(["time","lat","lon"],np.reshape(prec,(1,prec.shape[0],prec.shape[1]))),P_n_part=(["time","lat","lon"],np.reshape(num_par_prec,(1,num_par_prec.shape[0],num_par_prec.shape[1]))),IWV=(["time","lat","lon"],np.reshape(iwv,(1,iwv.shape[0],iwv.shape[1]))),n_part=(["time","lat","lon"],np.reshape(num_par,(1,num_par.shape[0],num_par.shape[1])))),coords=dict(time=(["time"],[middle_date]),lat=(["lat"],grid_lat[:,0]),lon=(["lon"],grid_lon[0,:])))

    #Save diagnosis
    os.system('mkdir -p '+path_data1+'/diagnosis')
    final.to_netcdf(path_data1+'/diagnosis/diagnosis_'+arrival_date.strftime('%Y%m%d%H%M%S')+'.nc')

  #9)
  # -----------------------------Recalculating deltas from observations---------------------------

  #Select observed and diagnosed evaporation, precipitation and heat for indices corresponding to the parcels position in the grid
  e_obs_parcel = e_obs[y_index, x_index]
  e_diag_parcel = evap[y_index, x_index]

  p_obs_parcel = p_obs[y_index, x_index]
  p_diag_parcel = prec[y_index, x_index]

  if json.loads(config['FLAGS']['track_heat'].lower()):
    h_obs_parcel = h_obs[y_index, x_index]
    h_diag_parcel = heat[y_index, x_index]

  #Correct deltas
  Delta_q_old=np.copy(Delta_q)
  Delta_q[:]=0.
  Delta_q[ind_evap]=Delta_q_old[ind_evap]*(e_obs_parcel[ind_evap]/e_diag_parcel[ind_evap])
  Delta_q[ind_prec]=Delta_q_old[ind_prec]*(p_obs_parcel[ind_prec]/p_diag_parcel[ind_prec])
  Delta_q[np.isinf(Delta_q)]=0.
  Delta_q[np.isnan(Delta_q)]=0.
  del e_obs,evap,p_obs,prec,e_obs_parcel,e_diag_parcel,p_obs_parcel,p_diag_parcel,Delta_q_old

  if json.loads(config['FLAGS']['track_heat'].lower()):
    Delta_theta_old=np.copy(Delta_theta)
    Delta_theta[:]=0.
    Delta_theta[ind_heat]=Delta_theta_old[ind_heat]*(h_obs_parcel[ind_heat]/h_diag_parcel[ind_heat])  
    Delta_theta[Delta_theta_old<0.]=Delta_theta_old[Delta_theta_old<0.]
    Delta_theta[np.isinf(Delta_theta)]=0.
    Delta_theta[np.isnan(Delta_theta)]=0.
    del h_obs,heat,h_obs_parcel,h_diag_parcel,Delta_theta_old
        
  #10)
  # ---------------------Update partoutputs including bias corrected values---------------------

  #Some considerations:
  #*We start from a raw partoutput and create/delete variables to maintain the same structure
  #*You don't necessarily have to replace the original files, you can simply save new ones with a different name, for which you would have to make some small modifications in the following lines of code
  #*Another option is to link the original porouputs to the directory path_data1+'/partoutputs'. This way what we will replace will be the links and we will not lose the raw data
  #*Some exceptions are made depending on the dates (hence the if) because, for example, in the first interaction we cannot delete the original partoutputs of time steps 0 and 1, because 1 will be opened in the next interaction. This condition varies depending on whether our date is the start, end or an intermediate date
  
  #Create new dataset
  if start_date==first_date:
    ds1=xr.open_dataset(files[0]).sel(particle=slice(1, num_parc)).transpose()
    ds1=ds1.drop_vars(['z','hmix', 'pv', 'to', 'tro', 'm','rho','T','longitude','latitude'])
    ds1["Delta_q"]=(["time","particle"],np.zeros((1,Delta_q.shape[0])))
    if json.loads(config['FLAGS']['track_heat'].lower()):
      ds1["theta"]=(["time","particle"],np.reshape(theta[0,:],(1,theta[0,:].shape[0])))
      ds1["Delta_theta"]=(["time","particle"],np.zeros((1,Delta_theta.shape[0])))

    #Save new partoutput
    os.system('rm '+path_data1+'/partoutputs/partoutput_'+before_arrival.strftime('%Y%m%d%H%M%S')+'.nc')
    ds1.astype('float32').to_netcdf(path_data1+'/partoutputs/partoutput_'+before_arrival.strftime('%Y%m%d%H%M%S')+'.nc')
    
    ds2=xr.open_dataset(files[1]).sel(particle=slice(1, num_parc)).transpose()
    ds2=ds2.drop_vars(['z','hmix', 'pv', 'to', 'tro', 'm','rho','T','longitude','latitude'])
    ds2["Delta_q"]=(["time","particle"],np.reshape(Delta_q,(1,Delta_q.shape[0])))
    if json.loads(config['FLAGS']['track_heat'].lower()):
      ds2["theta"]=(["time","particle"],np.reshape(theta[1,:],(1,theta[1,:].shape[0])))
      ds2["Delta_theta"]=(["time","particle"],np.reshape(Delta_theta,(1,Delta_theta.shape[0])))

    #Save new partoutput
    ds2.astype('float32').to_netcdf(path_data1+'/partoutputs/partoutput2_'+arrival_date.strftime('%Y%m%d%H%M%S')+'.nc')
  elif start_date==end_date:
    ds2=xr.open_dataset(files[1]).sel(particle=slice(1, num_parc)).transpose()
    ds2=ds2.drop_vars(['z','hmix', 'pv', 'to', 'tro', 'm','rho','T','longitude','latitude'])
    ds2["Delta_q"]=(["time","particle"],np.reshape(Delta_q,(1,Delta_q.shape[0])))
    if json.loads(config['FLAGS']['track_heat'].lower()):
      ds2["theta"]=(["time","particle"],np.reshape(theta[1,:],(1,theta[1,:].shape[0])))
      ds2["Delta_theta"]=(["time","particle"],np.reshape(Delta_theta,(1,Delta_theta.shape[0])))
  
    #Save new partoutput
    os.system('rm '+path_data1+'/partoutputs/partoutput_'+before_arrival.strftime('%Y%m%d%H%M%S')+'.nc')
    os.system('rm '+path_data1+'/partoutputs/partoutput_'+arrival_date.strftime('%Y%m%d%H%M%S')+'.nc')
    os.system('mv '+path_data1+'/partoutputs/partoutput2_'+before_arrival.strftime('%Y%m%d%H%M%S')+'.nc '+path_data1+'/partoutputs/partoutput_'+before_arrival.strftime('%Y%m%d%H%M%S')+'.nc')
    ds2.astype('float32').to_netcdf(path_data1+'/partoutputs/partoutput_'+arrival_date.strftime('%Y%m%d%H%M%S')+'.nc')
  else:
    ds2=xr.open_dataset(files[1]).sel(particle=slice(1, num_parc)).transpose()
    ds2=ds2.drop_vars(['z','hmix', 'pv', 'to', 'tro', 'm','rho','T','longitude','latitude'])
    ds2["Delta_q"]=(["time","particle"],np.reshape(Delta_q,(1,Delta_q.shape[0])))
    if json.loads(config['FLAGS']['track_heat'].lower()):
      ds2["theta"]=(["time","particle"],np.reshape(theta[1,:],(1,theta[1,:].shape[0])))
      ds2["Delta_theta"]=(["time","particle"],np.reshape(Delta_theta,(1,Delta_theta.shape[0])))

    #Save new partoutput
    os.system('rm '+path_data1+'/partoutputs/partoutput_'+before_arrival.strftime('%Y%m%d%H%M%S')+'.nc')
    os.system('mv '+path_data1+'/partoutputs/partoutput2_'+before_arrival.strftime('%Y%m%d%H%M%S')+'.nc '+path_data1+'/partoutputs/partoutput_'+before_arrival.strftime('%Y%m%d%H%M%S')+'.nc')
    ds2.astype('float32').to_netcdf(path_data1+'/partoutputs/partoutput2_'+arrival_date.strftime('%Y%m%d%H%M%S')+'.nc')

  #Go to the new date
  start_date=start_date+timedelta(hours=dt)

  #---------------------------------------------------------------------------------------------------

#Print final messages
c2 = time.time()
print('Finishing observe_diagnose.py...')
print('RAM footprint '+str(tracemalloc.get_traced_memory()))
print('Total time '+str(c2-c1))  
tracemalloc.stop() 