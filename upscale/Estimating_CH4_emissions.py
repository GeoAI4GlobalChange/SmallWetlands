import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle
from tqdm import tqdm
import math
import time
import numpy as np
from skimage.transform import resize
from rasterio.transform import Affine
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm
import copy
import os
import warnings
import time
import math
warnings.filterwarnings("ignore", category=DeprecationWarning)
import cv2
import matplotlib.pyplot as plt
import numpy as np
def global_wetland_CH4_emissions(lats,lons,start_year,end_year,thresholds_list,input_wetland_dir,input_CH4_dir,output_dir):
    start=True
    for year in range(start_year,end_year+1):
        dir = input_wetland_dir
        target = 'extent'
        nc_file = dir + f'GWL_global_0.5degree_{year}.nc'
        data = Dataset(nc_file, mode='r')
        extent = np.array(data.variables[target][:])
        lons = data.variables['lon'][:]
        lats = data.variables['lat'][:]
        data.close()
        extent=extent[np.newaxis]
        if start:
            extent_all=extent
            start=False
        else:
            extent_all=np.concatenate((extent_all,extent),axis=0)
    dir_CH4_intensity=input_CH4_dir
    nc_file = dir_CH4_intensity + f'FCH4_upscale_ML.nc'
    data = Dataset(nc_file, mode='r')
    CH4_flux = np.array(data.variables['FCH4'][:])
    data.close()
    CH4_flux=CH4_flux*24*3600*pow(10,-9)*16#nmol/m2/s->gCH4/m2/day
    CH4_flux=CH4_flux[:,np.newaxis]
    CH4_flux=np.repeat(CH4_flux,extent_all.shape[1],axis=1)
    CH4_flux=CH4_flux.reshape((52,-1,extent_all.shape[1],len(lats),len(lons)))
    CH4_flux=CH4_flux*extent_all
    CH4_flux=CH4_flux.reshape((-1,extent_all.shape[1],len(lats),len(lons)))
    save_file = output_dir + f'global_wetland_ch4_emissions_0.5degree.nc'
    nc_fid2 = Dataset(save_file, 'w', format="NETCDF4")
    nc_fid2.createDimension('lat', len(lats))
    nc_fid2.createDimension('lon', len(lons))
    nc_fid2.createDimension('time', CH4_flux.shape[0])
    nc_fid2.createDimension('thres', CH4_flux.shape[1])
    latitudes = nc_fid2.createVariable('lat', 'f4', ('lat',))
    longitudes = nc_fid2.createVariable('lon', 'f4', ('lon',))
    thres_save = nc_fid2.createVariable("thres", "f4", ("thres",))
    time_day = nc_fid2.createVariable("time", "f4", ("time",))
    var_save = nc_fid2.createVariable('wetland_CH4_emissions', "f4", ("time","thres", "lat", "lon",))
    var_save.units = 'gCH4 m-2 day-1'
    time_day.units = "days since 1993-08-01 00:00:00.0"
    time_day.calendar = "gregorian"
    dates_all=[]
    for year_idx in range(start_year,end_year+1):
        dates = [datetime(year_idx, 1, 1) + relativedelta(days=+7*n) for n in range(52)]
        dates_all.extend(dates)
    time_day[:] = date2num(dates_all, units=time_day.units, calendar=time_day.calendar)
    thres_save[:] = np.array(thresholds_list[:])/pow(1000, 2)#unit: km2
    latitudes[:] = lats[:]
    longitudes[:] = lons[:]
    var_save[:] = CH4_flux[:]
    nc_fid2.close()
    print('done!')
if __name__=='__main__':
    # the spatial region studied
    lats = np.linspace(89.5, -89.5, 360)
    lons = np.linspace(-179.5, 179.5, 720)
    # the temporal period studied
    start_year = 2003
    end_year = 2022
    # wetland areal thresholds:
    # For example, here 1 km2 is used to define small wetlands (<1km2)
    thresholds_list = [pow(1000, 2)]
    # the directory of wetland data
    input_wetland_dir = r'../output/'
    # the directory of wetland CH4 flux data
    # note: the CH4 flux data is generated
    # using the causality-guided machine learning model from:
    # https://github.com/GeoAI4GlobalChange/GlobalUpscaling
    input_CH4_dir = r'../output/'
    # the output directory to save the estimated gridded CH4 emissions
    output_dir = f'../output/'
    global_wetland_CH4_emissions(lats,lons,start_year,end_year,thresholds_list,input_wetland_dir,input_CH4_dir,output_dir)