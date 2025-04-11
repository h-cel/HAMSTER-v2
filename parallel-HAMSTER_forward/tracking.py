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
from scipy.interpolate import griddata
from utilities import *
import configparser
config = configparser.ConfigParser()
config.read('namelist.input')
from mpi4py import MPI  
import gc
import h5py

c1 = time.time()
tracemalloc.start()

#0)
#----------------------------------------Pre-loop definitions-----------------------------------------

#Define the MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
path_data2=config['PATHS']['path_data2']
path_mask=config['PATHS']['path_mask']

# =========================================================================================================

# In what follows, three functions are presented that will be called later in the main (at the end of the script)

# =========================================================================================================


#1)
#---------------------------------------------Read source---------------------------------------------

#This function reads the source region and returns a tuple of latitudes and longitudes for that region
def read_mask():

  #Open mask (source region)
  data_source= xr.open_dataset(path_mask+'/mask.nc') 
  try:
    data_source=data_source.isel(time=0)
  except:
    if rank == 0:
      print('Source mask has no temporal dimension')

  #Find a list of tuples with the coordinates (lon,lat) of all the points of the source (source==1.)
  lon_source=data_source.lon.values
  lat_source=data_source.lat.values
  lons_source,lats_source=np.meshgrid(lon_source,lat_source)
  lons_source=lons_source.flatten()
  lats_source=lats_source.flatten()
  source_flatten=data_source.mask.values.flatten()

  #Interpolate to a common source (0.25 degrees resolution). This allows you to enter a source of the resolution you want
  points=np.transpose((lons_source, lats_source))
  grid_lon_025=np.arange(-180.,180.,0.25)
  grid_lat_025=np.arange(-90.,90.25,0.25)
  grid_lon_025,grid_lat_025=np.meshgrid(grid_lon_025,grid_lat_025)
  new_source=griddata(points, source_flatten, (grid_lon_025, grid_lat_025), method='nearest')
  new_lons_source=grid_lon_025.flatten()
  new_lats_source=grid_lat_025.flatten()
  new_source_flatten=new_source.flatten()

  #Now get the aforementioned tuple
  new_lons_source=new_lons_source[new_source_flatten==1.]
  new_lats_source=new_lats_source[new_source_flatten==1.]
  lonlat_source=list(zip(new_lons_source,new_lats_source))
  
  return lonlat_source
  
#2)
#---------------------------------------------Read data---------------------------------------------

#This function receives the paths of the input files and reads them in parallel

#Some considerations:
#*It is here where we define the data chunks to be used for the different processes 
#*The output is the variables of interest already loaded into RAM 
#*We avoid using xarray and functions such as xr.open_mfdataset because they have a greater RAM footprint and lower performance also in terms of speed (for example, using isel for chuncking is slow)
def load_data_in_parallel(files):

  #Distribute the air parcels among the available processes, i.e. parallelisation is done in the particle dimension
  block_size = num_parc // size  

  #Determine the block (chunk) limits for each process
  start = rank * block_size
  end = (rank + 1) * block_size if rank != size - 1 else num_parc  
  
  #Initialise variables of interest
  lon_local = np.zeros((len(files), end - start), dtype=np.float32)
  lat_local = np.zeros((len(files), end - start), dtype=np.float32)
  qv_local = np.zeros((len(files), end - start), dtype=np.float32)
  Delta_q_local = np.zeros((len(files), end - start), dtype=np.float32)
  if json.loads(config['FLAGS']['track_heat'].lower()):
    theta_local = np.zeros((len(files), end - start), dtype=np.float32)
    Delta_theta_local = np.zeros((len(files), end - start), dtype=np.float32)
  else:
    theta_local = None
    Delta_theta_local = None
  
  i = 0
  #Iterate through time to read the netcdf files one at a time using h5py. For each process we only read its block of data
  for file in files:
    with h5py.File(file, 'r') as ds:
      lon_local[i, :] = ds['lon'][:,start:end]
      lat_local[i, :] = ds['lat'][:,start:end]
      lat_local[np.isnan(lat_local)] = -1.
      qv_local[i, :] = ds['sh'][:,start:end]
      Delta_q_local[i, :] = ds['Delta_q'][:,start:end]
      if json.loads(config['FLAGS']['track_heat'].lower()):
        theta_local[i, :] = ds['theta'][:,start:end]
        Delta_theta_local[i, :] = ds['Delta_theta'][:,start:end]
    i=i+1
    
  #Transform lon_local to -180,180
  lon_local[lon_local>180.]=lon_local[lon_local>180.]-360.

  return lon_local,lat_local,qv_local,Delta_q_local,theta_local,Delta_theta_local

#3)
#---------------------------------------------Calculations---------------------------------------------

#Function to process a block of data 
def process_data_in_parallel(datesmidp,lonlat_source,lon,lat,qv,Delta_q,theta,Delta_theta):

  #3.1)
  #--------Check which parcels are on the source between the two first steps---------
      
  #Mid-point coordinates of parcels in the first two sets 
  lona=lon[0,:]
  lata=lat[0,:]

  lonb=lon[1,:]
  latb=lat[1,:]

  lonmidp,latmidp=midpoints(lona,lonb,lata,latb)

  #Interpolate parcel coordinates to nearest neighbor in the source (rounds to nearest 0.25)
  lon_data=np.round(lonmidp*4.)/4.
  lat_data=np.round(latmidp*4.)/4.

  #Create a multi-level index with pandas from parcel coordinates
  midx=pd.MultiIndex.from_arrays([lon_data,lat_data],names=('longitudes','latitudes'))

  #Searches for matches (or not) between parcel coordinates and the source and generates a boolean array
  isornot=midx.isin(lonlat_source)
  
  #Clean
  del lona,lata,lonb,latb,lon_data,lat_data,midx
    
  #3.2)
  #--------Select only parcels that are on the source between the two first steps---------

  qv_new=qv[:,isornot]
  Delta_q_new=Delta_q[1:,isornot]

  #3.3)
  #--------Determine, for each time interval, moisture gains and losses due to evaporation and/or precipitation-------

  #Calculate the indices (boolean array) of moisture gains and moisture losses
  ind_evap = (Delta_q_new>0.)
  ind_prec = (Delta_q_new<0.)
  
  #3.4)
  #-----------------Select the parcels contributing to evaporation in the first time step------------------

  #Parcels contributing to evaporation according to dq in the first step
  dq_first = Delta_q_new[0,:]

  #Define boolean array and select these parcels 
  ind_starting_e=(dq_first>0.)

  ind_evap=ind_evap[:,ind_starting_e]
  ind_prec=ind_prec[:,ind_starting_e]
  Delta_q_new=Delta_q_new[:,ind_starting_e]
  qv_new=qv_new[:,ind_starting_e]
  dq_first=dq_first[ind_starting_e]

  #3.5)
  #---------------------Calculate precipitable water from evaporation over source (qfe)---------------------

  #Define qfe from Delta_q_new (they have the same structure)
  qfe = np.zeros(Delta_q_new.shape,dtype=np.float32)

  #Initialise qfe and select only parcels that gain moisture at the beginning
  qfe[0,:] = dq_first

  #Loop over after evaporation
  for t in range(1,qfe.shape[0]):
    #If don't loss, define qfe to be equal to the previous time step
    ind_loss=ind_prec[t,:]
    ind_gain=np.invert(ind_loss)
    qfe[t,ind_gain] = qfe[t-1,ind_gain]
    #If loss, define qfe based on that of the previous step and the fraction of moisture lost
    qfe[t,ind_loss] = qfe[t-1,ind_loss]*((qv_new[t,ind_loss]+Delta_q_new[t,ind_loss])/qv_new[t,ind_loss])
    #Avoid possible negative values or NaN values
    qfe[qfe<0.]=0.
    qfe[np.isinf(qfe)]=0.
    qfe[np.isnan(qfe)]=0.

  #3.6)
  #-----------------------Calculate precipitation from evaporation over source (pfe)-----------------------
  
  pfe = np.zeros(Delta_q_new.shape,dtype=np.float32)
  pfe[1:,:] = qfe[:-1,:]-qfe[1:,:]
    
  #Upscale to consider total precipitation
  correction_fraction=dq_first/np.sum(pfe,axis=0)
  correction_fraction=np.repeat(correction_fraction[np.newaxis,:],pfe.shape[0],axis=0)
  pfe = pfe*correction_fraction

  #Avoid possible negative values or NaN values
  pfe[pfe<0.]=0.
  pfe[np.isinf(pfe)]=0.
  pfe[np.isnan(pfe)]=0.
  
  #Clean
  del qv_new,Delta_q_new,correction_fraction

  #3.7)
  #-------------------------------------Replicte calculations for heat-------------------------------------

  if json.loads(config['FLAGS']['track_heat'].lower()):
  
    theta_new=theta[:,isornot]
    Delta_theta_new=Delta_theta[1:,isornot]

    #Calculate the indices (boolean array) of heat gains and heat losses
    ind_heat_gain = (Delta_theta_new>0.)
    ind_heat_loss = (Delta_theta_new<0.)

    #Parcels increasing potential temperature in the first step
    dt_first = Delta_theta_new[0,:]

    #Define boolean array and select these parcels 
    ind_starting_h=(dt_first>0.)

    ind_heat_gain=ind_heat_gain[:,ind_starting_h]
    ind_heat_loss=ind_heat_loss[:,ind_starting_h]
    Delta_theta_new=Delta_theta_new[:,ind_starting_h]
    theta_new=theta_new[:,ind_starting_h]
    dt_first=dt_first[ind_starting_h]

    #Define hfs from Delta_theta_new (they have the same structure)
    hfs = np.zeros(Delta_theta_new.shape,dtype=np.float32)

    #Initialise hfs and select only parcels that gain heat at the beginning
    hfs[0,:] = dt_first

    #Loop over after sensible heat gain
    for t in range(1,hfs.shape[0]):
      #If don't loss, define hfs to be equal to the previous time step
      ind_loss=ind_heat_loss[t,:]
      ind_gain=np.invert(ind_loss)
      hfs[t,ind_gain] = hfs[t-1,ind_gain]
      #If loss, define qfe based on that of the previous step and the fraction of theta lost
      hfs[t,ind_loss] = hfs[t-1,ind_loss]*((theta_new[t,ind_loss]+Delta_theta_new[t,ind_loss])/theta_new[t,ind_loss])
      #Avoid possible negative values or NaN values
      hfs[hfs<0.]=0.
      hfs[np.isinf(hfs)]=0.
      hfs[np.isnan(hfs)]=0.
        
    #Clean
    del theta_new,Delta_theta_new

  #3.8)
  #----------------------------------Assigns the above variables to a grid--------------------------------
  
  lon_new=lon[:,isornot]
  lat_new=lat[:,isornot]

  #Compute mid-point coordinates along trajectories
  lona=lon_new[1:,:]
  lonb=lon_new[:-1,:]
  lata=lat_new[1:,:]
  latb=lat_new[:-1,:]
  lonmidp,latmidp=midpoints(lona,lonb,lata,latb)
  
  #Clean
  del lon_new,lat_new,lona,lonb,lata,latb

  #Find indices corresponding to the parcels position in the grid           
  x_index = np.round((lonmidp+180.0)/dx).astype(int)
  x_index[x_index==nx] = 0
  y_index = np.round((latmidp+90.0)/dy).astype(int)

  #Calculate PFE
  PFE=ongrid(nx,ny,x_index[:,ind_starting_e],y_index[:,ind_starting_e],pfe,mass)
  del lonmidp,latmidp,pfe
  
  #Calculate QFE
  if json.loads(config['FLAGS']['save_qfe'].lower()):
    QFE=ongrid(nx,ny,x_index[:,ind_starting_e],y_index[:,ind_starting_e],qfe,mass)
  del qfe

  if json.loads(config['FLAGS']['track_heat'].lower()):
    #Calculate HFS
    HFS=ongrid(nx,ny,x_index[:,ind_starting_h],y_index[:,ind_starting_h],hfs,mass)
    del hfs

  #3.9)
  #----------------------------------Convert grid fields (PFE,QFE,HFS)--------------------------------
  
  #Resample the tlevel dimension to daily data
  #Some considerations:
  #*We need to resample here and not wait until before saving, because otherwise rank=0 would receive too much data (see comm.gather) which would cause problems with the RAM
  #*Initially we opted to create an xarray dataset for resampling, but in the end we found that this option is much more efficient despite the loop
  PFE = np.stack([PFE[datesmidp.date == d].sum(axis=0) for d in np.unique(datesmidp.date)], axis=0).astype(np.float32)
  if json.loads(config['FLAGS']['save_qfe'].lower()):
    QFE = np.stack([QFE[datesmidp.date == d].sum(axis=0) for d in np.unique(datesmidp.date)], axis=0).astype(np.float32)
  if json.loads(config['FLAGS']['track_heat'].lower()):
    HFS = np.stack([HFS[datesmidp.date == d].sum(axis=0) for d in np.unique(datesmidp.date)], axis=0).astype(np.float32)
    
  #Change units
  inv_density = 1000./997.#This transforms kg of water to mm
  cp = 1005.7 
  dts = dt*60.*60.
  PFE = PFE*inv_density/areas
  PFE = np.nan_to_num(PFE)#mm
  if json.loads(config['FLAGS']['save_qfe'].lower()):
    QFE = QFE*inv_density/areas
    QFE = np.nan_to_num(QFE)#mm
  if json.loads(config['FLAGS']['track_heat'].lower()):
    HFS = HFS*cp/dts/areas
    HFS = np.nan_to_num(HFS)#W/m2
    
  #3.10)
  #----------------------------------Gather all results and sum--------------------------------
  
  #Clean
  gc.collect()#Free up RAM before starting the collection
    
  #Synchronise processes before combining the files
  comm.Barrier()
  
  #Collect all the results in the rank=0 process
  PFEs = comm.gather(PFE, root=0)
  del PFE
  if json.loads(config['FLAGS']['save_qfe'].lower()):
    QFEs = comm.gather(QFE, root=0)
    del QFE
  if json.loads(config['FLAGS']['track_heat'].lower()):
    HFSs = comm.gather(HFS, root=0)
    del HFS

  #Sum all the results
  if rank == 0:
    pfe_final = np.sum(PFEs, axis=0)
    if json.loads(config['FLAGS']['save_qfe'].lower()):
      qfe_final = np.sum(QFEs, axis=0)
    else:
      qfe_final = None
    if json.loads(config['FLAGS']['track_heat'].lower()):
      hfs_final = np.sum(HFSs, axis=0)
    else:
      hfs_final = None

    return pfe_final, qfe_final, hfs_final
  else:
    return None, None, None
    
# =========================================================================================================

# Definition of functions is finisehd

# =========================================================================================================

if __name__ == '__main__':

  #Read mask
  lonlat_source=read_mask()

  #Define dates
  start_date=datetime.strptime(config['DATES']['start_date'],'%Y-%m-%d_%H')
  end_date=datetime.strptime(config['DATES']['end_date'],'%Y-%m-%d_%H')-timedelta(hours=dt)
      
  counter=0
  while start_date<=end_date: 
    
    #4)
    #----------------------------------------Definition of dates----------------------------------------

    #Dates 
    starting_date=start_date
    after_starting=starting_date+timedelta(hours=dt)
    arrival_date=starting_date+timedelta(days=num_days)
    dates=pd.date_range(start=starting_date,end=arrival_date,freq=str(dt)+'H')
    datesmidp=pd.date_range(start=(starting_date+timedelta(hours=dt/2)),end=(arrival_date-timedelta(hours=dt/2)),freq=str(dt)+'H')     

    #5)
    #------------------------Call functions to read and process data in parallel------------------------
    
    if counter==0:
      #Read data (first date)
      files=[path_data1+'/partoutputs/partoutput_'+d.strftime('%Y%m%d%H%M%S')+'.nc' for d in dates]
      lon,lat,qv,Delta_q,theta,Delta_theta = load_data_in_parallel(files)
    else:
      #Read data
      #Now we can simply read a single date and update the arrays (defined in counter==0)
      files=[path_data1+'/partoutputs/partoutput_'+d.strftime('%Y%m%d%H%M%S')+'.nc' for d in [dates[-1]]]
      lon_aux,lat_aux,qv_aux,Delta_q_aux,theta_aux,Delta_theta_aux=load_data_in_parallel(files)
      lon = np.concatenate([lon[1:],lon_aux], axis=0)
      lat = np.concatenate([lat[1:],lat_aux], axis=0)
      qv = np.concatenate([qv[1:],qv_aux], axis=0)
      Delta_q = np.concatenate([Delta_q[1:],Delta_q_aux], axis=0)
      if json.loads(config['FLAGS']['track_heat'].lower()):
        theta = np.concatenate([theta[1:],theta_aux], axis=0)
        Delta_theta = np.concatenate([Delta_theta[1:],Delta_theta_aux], axis=0)
      del lon_aux,lat_aux,qv_aux,Delta_q_aux,theta_aux,Delta_theta_aux
      gc.collect()

    #Process data
    PFE,QFE,HFS = process_data_in_parallel(datesmidp,lonlat_source,lon,lat,qv,Delta_q,theta,Delta_theta)
        
    #6)
    # -----------------------------------Save to a netcdf file--------------------------------------

    if rank == 0:
      
       #Create datasets
      final_pfe=xr.Dataset(data_vars=dict(PFE=(["time","tlevel","lat","lon"],np.reshape(PFE,(1,PFE.shape[0],PFE.shape[1],PFE.shape[2])))),coords=dict(time=(["time"],[after_starting-timedelta(hours=dt/2)]),tlevel=(["tlevel"],datesmidp.normalize().unique()),lat=(["lat"],grid_lat[:,0]),lon=(["lon"],grid_lon[0,:]))).astype(np.float32)
      if json.loads(config['FLAGS']['save_qfe'].lower()):
        final_qfe=xr.Dataset(data_vars=dict(QFE=(["time","tlevel","lat","lon"],np.reshape(QFE,(1,QFE.shape[0],QFE.shape[1],QFE.shape[2])))),coords=dict(time=(["time"],[after_starting-timedelta(hours=dt/2)]),tlevel=(["tlevel"],datesmidp.normalize().unique()),lat=(["lat"],grid_lat[:,0]),lon=(["lon"],grid_lon[0,:]))).astype(np.float32)
      if json.loads(config['FLAGS']['track_heat'].lower()):
        final_hfs=xr.Dataset(data_vars=dict(HFS=(["time","tlevel","lat","lon"],np.reshape(HFS,(1,HFS.shape[0],HFS.shape[1],HFS.shape[2])))),coords=dict(time=(["time"],[after_starting-timedelta(hours=dt/2)]),tlevel=(["tlevel"],datesmidp.normalize().unique()),lat=(["lat"],grid_lat[:,0]),lon=(["lon"],grid_lon[0,:]))).astype(np.float32)

      #Save
      os.system('mkdir -p '+path_data2+'/attribution')
      final_pfe.to_netcdf(path_data2+'/attribution/attribution_pfe_'+after_starting.strftime('%Y%m%d%H%M%S')+'.nc')
      if json.loads(config['FLAGS']['save_qfe'].lower()):
        final_qfe.to_netcdf(path_data2+'/attribution/attribution_qfe_'+after_starting.strftime('%Y%m%d%H%M%S')+'.nc')
      if json.loads(config['FLAGS']['track_heat'].lower()):
        final_hfs.to_netcdf(path_data2+'/attribution/attribution_hfs_'+after_starting.strftime('%Y%m%d%H%M%S')+'.nc')

    #Go to the new date
    start_date=start_date+timedelta(hours=dt)
    counter=counter+1

    #---------------------------------------------------------------------------------------------------

  #Print final messages
  if rank == 1:
    print('Finishing attribut_correct.py...')
    print('RAM footprint for rank=1 '+str(tracemalloc.get_traced_memory()))   
  if rank == 0:
    c2 = time.time()
    print('RAM footprint for rank=0 '+str(tracemalloc.get_traced_memory())) 
    print('Total time '+str(c2-c1))
    tracemalloc.stop() 