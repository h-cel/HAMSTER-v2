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
#---------------------------------------------Read sink---------------------------------------------

#This function reads the sink region and returns a tuple of latitudes and longitudes for that region
def read_mask():

  #Open mask (sink region)
  data_sink= xr.open_dataset(path_mask+'/mask.nc') 
  try:
    data_sink=data_sink.isel(time=0)
  except:
    if rank == 0:
      print('Source mask has no temporal dimension')

  #Find a list of tuples with the coordinates (lon,lat) of all the points of the sink (sink==1.)
  lon_sink=data_sink.lon.values
  lat_sink=data_sink.lat.values
  lons_sink,lats_sink=np.meshgrid(lon_sink,lat_sink)
  lons_sink=lons_sink.flatten()
  lats_sink=lats_sink.flatten()
  sink_flatten=data_sink.mask.values.flatten()

  #Interpolate to a common sink (0.25 degrees resolution). This allows you to enter a sink of the resolution you want
  points=np.transpose((lons_sink, lats_sink))
  grid_lon_025=np.arange(-180.,180.,0.25)
  grid_lat_025=np.arange(-90.,90.25,0.25)
  grid_lon_025,grid_lat_025=np.meshgrid(grid_lon_025,grid_lat_025)
  new_sink=griddata(points, sink_flatten, (grid_lon_025, grid_lat_025), method='nearest')
  new_lons_sink=grid_lon_025.flatten()
  new_lats_sink=grid_lat_025.flatten()
  new_sink_flatten=new_sink.flatten()

  #Now get the aforementioned tuple
  new_lons_sink=new_lons_sink[new_sink_flatten==1.]
  new_lats_sink=new_lats_sink[new_sink_flatten==1.]
  lonlat_sink=list(zip(new_lons_sink,new_lats_sink))
  
  return lonlat_sink
  
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
def process_data_in_parallel(datesmidp,lonlat_sink,lon,lat,qv,Delta_q,theta,Delta_theta):

  #3.1)
  #--------Check which parcels are on the sink between the two final steps---------
      
  #Mid-point coordinates of parcels in the last two sets 
  lona=lon[-1,:]
  lata=lat[-1,:]
  
  lonb=lon[-2,:]
  latb=lat[-2,:]

  lonmidp,latmidp=midpoints(lona,lonb,lata,latb)

  #Interpolate parcel coordinates to nearest neighbor in the sink (rounds to nearest 0.25)
  lon_data=np.round(lonmidp*4.)/4.
  lat_data=np.round(latmidp*4.)/4.

  #Create a multi-level index with pandas from parcel coordinates
  midx=pd.MultiIndex.from_arrays([lon_data,lat_data],names=('longitudes','latitudes'))

  #Searches for matches (or not) between parcel coordinates and the sink and generates a boolean array
  isornot=midx.isin(lonlat_sink)
  
  #Clean
  del lona,lata,lonb,latb,lon_data,lat_data,midx
    
  #3.2)
  #--------Select only parcels that are on the sink between the two last steps---------

  qv_new=qv[:,isornot]
  Delta_q_new=Delta_q[1:,isornot]

  #3.3)
  #--------Determine, for each time interval, moisture gains and losses due to evaporation and/or precipitation-------

  #Calculate the indices (boolean array) of moisture gains and moisture losses
  ind_evap = (Delta_q_new>0.)
  ind_prec = (Delta_q_new<0.)
  
  #3.4)
  #----------------------Calculate the contribution of evaporation to precipitable water (e2q)----------------------

  #Define e2q from Delta_q_new (they have te same structure)
  e2q = np.zeros(Delta_q_new.shape)    

  #Loop over times previous to precipitation
  for t in range(e2q.shape[0]-1):
    #If gain, define f to be Delta_q_new
    ind_gain=ind_evap[t,:]
    e2q[t,ind_gain] = Delta_q_new[t,ind_gain]
    # If loss, recompute previous contributions
    ind_loss=ind_prec[t,:]
    e2q[:t,ind_loss] = e2q[:t,ind_loss]*((qv_new[t,ind_loss]+Delta_q_new[t,ind_loss])/qv_new[t,ind_loss])
    #Avoid possible negative values or NaN values
    e2q[e2q<0.]=0.
    e2q[np.isinf(e2q)]=0.
    e2q[np.isnan(e2q)]=0.

  #3.5)
  #---------------------Calculate the contribution of evaporation to precipitation (e2p)---------------------

  #Parcels contributing to precipitation according to dq in the final step
  dq_end = Delta_q_new[-1,:]

  #Select these parcels 
  ind_arrival_p=(dq_end<0.)
  e2p=e2q[:,ind_arrival_p]

  #Take the fraction represented by the water loss in the last step
  e2p = -e2p*(Delta_q_new[-1,ind_arrival_p]/qv_new[-2,ind_arrival_p])
  
  #Upscale to consider total precipitation
  correction_fraction=Delta_q_new[-1,ind_arrival_p]/np.sum(e2p,axis=0)
  correction_fraction=np.repeat(correction_fraction[np.newaxis,:],e2p.shape[0],axis=0)
  e2p = -e2p*correction_fraction
  #Avoid possible negative values or NaN values
  e2p[e2p<0.]=0.
  e2p[np.isinf(e2p)]=0.
  e2p[np.isnan(e2p)]=0.
  
  #Clean
  del qv_new,Delta_q_new,correction_fraction

  #3.6)
  #---------------------------------Replicte calculations for heat---------------------------------
  
  if json.loads(config['FLAGS']['track_heat'].lower()):
  
    theta_new=theta[:,isornot]
    Delta_theta_new=Delta_theta[1:,isornot]
    
    #Calculate the indices (boolean array) of heat gains and heat losses
    ind_heat_gain = (Delta_theta_new>0.)
    ind_heat_loss = (Delta_theta_new<0.)

    #Define h2t from Delta_theta_new (they have te same structure)
    h2t = np.zeros(Delta_theta_new.shape)

    #Loop over times previous to arrival
    for t in range(h2t.shape[0]-1):
      #If gain, define h2t to be just Delta_theta_new
      ind_gain=ind_heat_gain[t,:]
      h2t[t,ind_gain] = Delta_theta_new[t,ind_gain]
      #If loss, recompute previous contributions
      ind_loss=ind_heat_loss[t,:]
      h2t[:t,ind_loss] = h2t[:t,ind_loss]*((theta_new[t,ind_loss]+Delta_theta_new[t,ind_loss])/theta_new[t,ind_loss])
      #Avoid possible negative values or NaN values
      h2t[h2t<0.]=0.
      h2t[np.isinf(h2t)]=0.
      h2t[np.isnan(h2t)]=0.
      
    #Clean
    del theta_new,Delta_theta_new

  #3.7)
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

  #Calculate E2P
  E2P=ongrid(nx,ny,x_index[:,ind_arrival_p],y_index[:,ind_arrival_p],e2p,mass)
  del lonmidp,latmidp,e2p
  
  #Calculate E2Q
  if json.loads(config['FLAGS']['save_e2q'].lower()):
    E2Q=ongrid(nx,ny,x_index,y_index,e2q,mass)
  del e2q

  if json.loads(config['FLAGS']['track_heat'].lower()):
    #Calculate H2T
    H2T=ongrid(nx,ny,x_index,y_index,h2t,mass)
    del h2t

  #3.8)
  #----------------------------------Convert grid fields (E2P,E2Q,H2T)--------------------------------
  
  #Resample the tlevel dimension to daily data
  #Some considerations:
  #*We need to resample here and not wait until before saving, because otherwise rank=0 would receive too much data (see comm.gather) which would cause problems with the RAM
  #*Initially we opted to create an xarray dataset for resampling, but in the end we found that this option is much more efficient despite the loop
  E2P = np.stack([E2P[datesmidp.date == d].sum(axis=0) for d in np.unique(datesmidp.date)], axis=0).astype(np.float32)
  if json.loads(config['FLAGS']['save_e2q'].lower()):
    E2Q = np.stack([E2Q[datesmidp.date == d].sum(axis=0) for d in np.unique(datesmidp.date)], axis=0).astype(np.float32)
  if json.loads(config['FLAGS']['track_heat'].lower()):
    H2T = np.stack([H2T[datesmidp.date == d].sum(axis=0) for d in np.unique(datesmidp.date)], axis=0).astype(np.float32)
    
  #Change units
  inv_density = 1000./997.#This transforms kg of water to mm
  cp = 1005.7 
  dts = 24*60.*60.
  E2P = E2P*inv_density/areas
  E2P = np.nan_to_num(E2P)#mm
  if json.loads(config['FLAGS']['save_e2q'].lower()):
    E2Q = E2Q*inv_density/areas
    E2Q = np.nan_to_num(E2Q)#mm
  if json.loads(config['FLAGS']['track_heat'].lower()):
    H2T = H2T*cp/dts/areas
    H2T = np.nan_to_num(H2T)#W/m2
    
  #3.9)
  #----------------------------------Gather all results and sum--------------------------------
  
  #Clean
  gc.collect()#Free up RAM before starting the collection
    
  #Synchronise processes before combining the files
  comm.Barrier()
  
  #Collect all the results in the rank=0 process
  E2Ps = comm.gather(E2P, root=0)
  del E2P
  if json.loads(config['FLAGS']['save_e2q'].lower()):
    E2Qs = comm.gather(E2Q, root=0)
    del E2Q
  if json.loads(config['FLAGS']['track_heat'].lower()):
    H2Ts = comm.gather(H2T, root=0)
    del H2T

  #Sum all the results
  if rank == 0:
    e2p_final = np.sum(E2Ps, axis=0)
    if json.loads(config['FLAGS']['save_e2q'].lower()):
      e2q_final = np.sum(E2Qs, axis=0)
    else:
      e2q_final = None
    if json.loads(config['FLAGS']['track_heat'].lower()):
      h2t_final = np.sum(H2Ts, axis=0)
    else:
      h2t_final = None

    return e2p_final, e2q_final, h2t_final
  else:
    return None, None, None
    
# =========================================================================================================

# Definition of functions is finisehd

# =========================================================================================================

if __name__ == '__main__':

  #Read mask
  lonlat_sink=read_mask()

  #Define dates
  start_date=datetime.strptime(config['DATES']['start_date'],'%Y-%m-%d_%H')+timedelta(hours=dt)
  end_date=datetime.strptime(config['DATES']['end_date'],'%Y-%m-%d_%H')
      
  counter=0
  while start_date<=end_date: 
    
    #4)
    #----------------------------------------Definition of dates----------------------------------------

    #Dates 
    arrival_date=start_date
    starting_date=arrival_date-timedelta(days=num_days)
    before_arrival=arrival_date-timedelta(hours=dt)
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
    E2P,E2Q,H2T = process_data_in_parallel(datesmidp,lonlat_sink,lon,lat,qv,Delta_q,theta,Delta_theta)
        
    #6)
    # -----------------------------------Save to a netcdf file--------------------------------------

    if rank == 0:
      
       #Create datasets 
      final_e2p=xr.Dataset(data_vars=dict(E2P=(["time","tlevel","lat","lon"],np.reshape(E2P,(1,E2P.shape[0],E2P.shape[1],E2P.shape[2])))),coords=dict(time=(["time"],[arrival_date-timedelta(hours=dt/2)]),tlevel=(["tlevel"],datesmidp.normalize().unique()),lat=(["lat"],grid_lat[:,0]),lon=(["lon"],grid_lon[0,:]))).astype(np.float32)
      if json.loads(config['FLAGS']['save_e2q'].lower()):
        final_e2q=xr.Dataset(data_vars=dict(E2Q=(["time","tlevel","lat","lon"],np.reshape(E2Q,(1,E2Q.shape[0],E2Q.shape[1],E2Q.shape[2])))),coords=dict(time=(["time"],[arrival_date-timedelta(hours=dt/2)]),tlevel=(["tlevel"],datesmidp.normalize().unique()),lat=(["lat"],grid_lat[:,0]),lon=(["lon"],grid_lon[0,:]))).astype(np.float32)
      if json.loads(config['FLAGS']['track_heat'].lower()):
        final_h2t=xr.Dataset(data_vars=dict(H2T=(["time","tlevel","lat","lon"],np.reshape(H2T,(1,H2T.shape[0],H2T.shape[1],H2T.shape[2])))),coords=dict(time=(["time"],[arrival_date-timedelta(hours=dt/2)]),tlevel=(["tlevel"],datesmidp.normalize().unique()),lat=(["lat"],grid_lat[:,0]),lon=(["lon"],grid_lon[0,:]))).astype(np.float32)

      #Save
      os.system('mkdir -p '+path_data2+'/attribution')
      final_e2p.to_netcdf(path_data2+'/attribution/attribution_e2p_'+arrival_date.strftime('%Y%m%d%H%M%S')+'.nc')
      if json.loads(config['FLAGS']['save_e2q'].lower()):
        final_e2q.to_netcdf(path_data2+'/attribution/attribution_e2q_'+arrival_date.strftime('%Y%m%d%H%M%S')+'.nc')
      if json.loads(config['FLAGS']['track_heat'].lower()):
        final_h2t.to_netcdf(path_data2+'/attribution/attribution_h2t_'+arrival_date.strftime('%Y%m%d%H%M%S')+'.nc')

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

