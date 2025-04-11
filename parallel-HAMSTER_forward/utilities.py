#!/usr/bin/python
"""
Created on 15/11/2024

Author: Damian Insua Costa
"""

#Import modules
import numpy as np
from numba import jit
import struct
import math

#-------------------------------------Definition of functions---------------------------------------

#Reads binary outputs from FLEXPART (by Jessica Keune and Dominik Schumacher)
def f2t_read_partposit(ifile, maxn):
  nvars = 12
  binstr= "2fi3fi8f"
  nbytes_per_parcel = 8 + 4 + nvars * 4
  if "gz" in ifile:
    with gzip.open(ifile, "rb") as strm:
      #skip header
      _ = strm.read(4)  #dummy
      _ = struct.unpack("i", strm.read(4))[0]  #time
      #grep full binary data set (ATTN: 60 bytes for FP-ERA-Int hardcoded)
      tdata = strm.read(int(maxn) * nbytes_per_parcel)
      #get number of parcels from length of tdata
      nparc = math.floor(len(tdata) / (nbytes_per_parcel))
      #decode binary data
      pdata = struct.unpack((nparc) * binstr, tdata[0 : ((nparc) * nbytes_per_parcel)])
      flist = list(pdata)
    strm.close()
  else:
    with open(ifile, "rb") as strm:
      #skip header
      _ = strm.read(4)  #dummy
      _ = struct.unpack("i", strm.read(4))[0]  #time
      #grep full binary data set (ATTN: 60 bytes for FP-ERA-Int hardcoded)
      tdata = strm.read(int(maxn) * nbytes_per_parcel)
      #get number of parcels from length of tdata
      nparc = math.floor(len(tdata) / (nbytes_per_parcel))
      #decode binary data
      pdata = struct.unpack((nparc) * binstr, tdata[0 : ((nparc) * nbytes_per_parcel)])
      flist = list(pdata)
    strm.close()
  pdata = np.reshape(flist, newshape=(nparc, nvars+3))[:, 2:]  
  #remove last line if data is bs (pid = -99999)
  if np.any(pdata[:, 0] < 0):
    pdata = np.delete(pdata, np.where(pdata[:, 0] < 0), axis=0)
  return pdata

#Compute mid-point coordinates
def midpoints(lona,lonb,lata,latb):
  xmidp = 0.5*np.cos(latb*np.pi/180.0)*np.cos(lonb*np.pi/180.0)
  xmidp = xmidp + 0.5*np.cos(lata*np.pi/180.0)*np.cos(lona*np.pi/180.0)
  ymidp = 0.5*np.cos(latb*np.pi/180.0)*np.sin(lonb*np.pi/180.0)
  ymidp = ymidp + 0.5*np.cos(lata*np.pi/180.0)*np.sin(lona*np.pi/180.0)
  zmidp = 0.5*(np.sin(latb*np.pi/180.0)+np.sin(lata*np.pi/180.0))
  lonmidp = np.arctan2(ymidp, xmidp)*180.0/np.pi
  latmidp = np.arctan2(zmidp, np.sqrt(xmidp*xmidp+ymidp*ymidp))*180.0/np.pi
  return lonmidp,latmidp
  
#Compute potential temperature
def only_theta(qv,temp,rho):
  R_d= 287.057                           
  R_w = 461.5                            
  eps = R_d/R_w
  w = qv/(1.0-qv)                          
  tv = temp/(1.0-w*(1.0-eps)/(w+eps))      
  p_pa = rho*R_d*tv                        
  p_hpa=p_pa/1.E2
  theta=temp*(1000./p_hpa)**(0.2854*(1.-0.00028*w))
  return theta

#Compute relative humidity and potential temperature
def RH_theta(qv,temp,rho):
  R_d= 287.057                           
  R_w = 461.5                            
  eps = R_d/R_w
  w = qv/(1.0-qv)                          
  tv = temp/(1.0-w*(1.0-eps)/(w+eps))      
  p_pa = rho*R_d*tv                        
  e = p_pa*w/(w+eps)                       
  e_s = 611.2*np.exp(17.67*(temp-273.15)/(temp-273.15+243.5))
  RH = 100.*e/e_s 
  p_hpa=p_pa/1.E2
  theta=temp*(1000./p_hpa)**(0.2854*(1.-0.00028*w))
  return RH,theta

#Loop for assigning moisture gains to grid cells and integrating vertically 
@jit(nopython=True)
def ongrid(nx,ny,x_index,y_index,f,mass):
  ntimes,numpart = f.shape
  #Define grid (ms)
  ms = np.zeros((ntimes,ny,nx), dtype=np.float32)
  for t in range(ntimes):
    for j in range(numpart):
      #ms[t,y_index[t,j],x_index[t,j]] = ms[t,y_index[t,j],x_index[t,j]] + f[t,j]*mass[t,j]
      ms[t,y_index[t,j],x_index[t,j]] = ms[t,y_index[t,j],x_index[t,j]] + f[t,j]*mass
  return ms
  
#1D version of ongrid function
@jit(nopython=True)
def ongrid1D(nx,ny,x_index,y_index,f,mass):
  #Define grid
  var1 = np.zeros((ny,nx), dtype=np.float32)
  var2 = np.zeros((ny,nx), dtype=np.float32)
  for j in range(f.shape[0]-1):
    #var1[y_index[j],x_index[j]] = var1[y_index[j],x_index[j]] + f[j]*mass[j]
    var1[y_index[j],x_index[j]] = var1[y_index[j],x_index[j]] + f[j]*mass
    var2[y_index[j],x_index[j]] = var2[y_index[j],x_index[j]] + 1
  return var1,var2
  
#Find height of the first parcel in the column
@jit(nopython=True)
def hmin(nx,ny,x_index,y_index,h1,h2):
  #Define grid
  var1 = np.zeros((ny,nx), dtype=np.float32)+30000.
  var2 = np.zeros((ny,nx), dtype=np.float32)+30000.
  for j in range(h1.shape[0]-1):
    var1[y_index[j],x_index[j]] = min(var1[y_index[j],x_index[j]],h1[j])
    var2[y_index[j],x_index[j]] = min(var2[y_index[j],x_index[j]],h2[j])
  return var1,var2 
  
#---------------------------------------------------------------------------------------------------
