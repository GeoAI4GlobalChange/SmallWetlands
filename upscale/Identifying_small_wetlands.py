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
def calculate_areas(data_with_neighbours,delta_neighbour,binary_image,thresholds_list):
    binary_image[binary_image==1]=255
    # Threshold the image to create a binary image
    _, binary_image = cv2.threshold(binary_image, 128, 255, cv2.THRESH_BINARY)
    image_width=np.min(binary_image.shape)
    delta_xy=int(image_width/10)
    area_stats = np.full((len(thresholds_list), 10, 10), np.nan)
    thresholds_list=np.array(thresholds_list)/pow(30,2)
    # Find connected components with statistics for the 5x5 degree image
    num_labels_all, labels_all, stats_all, centroids_all = cv2.connectedComponentsWithStats(data_with_neighbours,
                                                                                            connectivity=8)
    for temp_lat_idx in range(0,image_width-delta_xy,delta_xy):
        for temp_lon_idx in range(0, image_width - delta_xy, delta_xy):
            if int(temp_lat_idx/delta_xy)<9:
                temp_binary_image=binary_image[temp_lat_idx:(temp_lat_idx+delta_xy)]
            else:
                temp_binary_image = binary_image[temp_lat_idx:]
            if int(temp_lon_idx/delta_xy) < 9:
                temp_binary_image = temp_binary_image[:,temp_lon_idx:(temp_lon_idx + delta_xy)]
            else:
                temp_binary_image = temp_binary_image[:,temp_lon_idx:]
            # Find connected components with statistics
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(temp_binary_image, connectivity=8)
            area_list=[]
            area_list_thres=[]
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                upleft_row_id=stats[label, cv2.CC_STAT_TOP]
                upleft_col_id = stats[label, cv2.CC_STAT_LEFT]
                com_width=stats[label, cv2.CC_STAT_WIDTH]
                com_hight=stats[label, cv2.CC_STAT_HEIGHT]
                downright_row_id=upleft_row_id+com_hight
                downright_col_id =upleft_col_id+com_width
                if (upleft_row_id==0 or upleft_col_id==0) or ((downright_row_id>=(temp_binary_image.shape[0])) or (downright_col_id>=(temp_binary_image.shape[1]))):
                    edge_old_com=labels[upleft_row_id:downright_row_id,upleft_col_id:downright_col_id]
                    edge_mask=(edge_old_com==label)
                    edge_new_com=labels_all[(delta_neighbour+temp_lat_idx+upleft_row_id):(delta_neighbour+temp_lat_idx+downright_row_id),(delta_neighbour+temp_lon_idx+upleft_col_id):(delta_neighbour+temp_lon_idx+downright_col_id)]
                    edge_new_label=edge_new_com[edge_mask][0]
                    edge_new_area=stats_all[edge_new_label,4]
                    area_list_thres.append(edge_new_area)
                else:
                    area_list_thres.append(area)
                area_list.append(area)
            area_list = np.array(area_list)
            area_list_thres = np.array(area_list_thres)
            for thre_idx in range(len(thresholds_list)):
                if thre_idx==0:
                    temp_area=np.sum(area_list[area_list_thres>=thresholds_list[thre_idx]])
                else:
                    temp_area = np.sum(area_list[((area_list_thres >= thresholds_list[thre_idx])&(area_list_thres < thresholds_list[thre_idx-1]))])
                area_stats[thre_idx,int(temp_lat_idx/delta_xy),int(temp_lon_idx/delta_xy)]=temp_area/(temp_binary_image.shape[0]*temp_binary_image.shape[1])
    return area_stats
def get_tile_edge_pixels(tif_data,yoff,xoff,delta_neighbour,dir):
    ##########################################
    ###Considering edge pixels for identifying spatially connected wetlands
    tiff_lat = int(yoff)
    tiff_lon = int(xoff)
    # each tile represents a 5degree*5degree area
    tile_size=5
    tiff_lat_down = tiff_lat - tile_size
    tiff_lat_up = tiff_lat + tile_size
    tiff_lon_left = tiff_lon - tile_size
    tiff_lon_right = tiff_lon + tile_size
    #innitialize the data
    data_with_neighbours = np.full(
        (tif_data.shape[0] + delta_neighbour * 2, tif_data.shape[1] + delta_neighbour * 2), np.nan)
    data_with_neighbours[delta_neighbour:(delta_neighbour + tif_data.shape[0]),
    delta_neighbour:(delta_neighbour + tif_data.shape[1])] = tif_data
    #########upper tile
    temp_tif_lat = tiff_lat_up
    temp_tif_lon = tiff_lon
    lat_txt = f'N{temp_tif_lat}' if temp_tif_lat >= 0 else f'S{np.abs(temp_tif_lat)}'
    lon_txt = f'E{temp_tif_lon}' if temp_tif_lon >= 0 else f'W{np.abs(temp_tif_lon)}'
    temp_tif_file = f'GWL_FCS30D_{year}Maps_{lon_txt}{lat_txt}.tif'
    if temp_tif_file in tiff_lst:
        temp_tif_file = dir + temp_tif_file
        src_temp = rasterio.open(temp_tif_file, "r")
        tif_data_temp = src_temp.read(1)
        end_tif_lon_idx = np.min((tif_data_temp.shape[1], tif_data.shape[1]))
        end_tif_lat_idx = np.min((tif_data_temp.shape[0], tif_data.shape[0]))
        data_with_neighbours[0:delta_neighbour, delta_neighbour:(delta_neighbour + end_tif_lon_idx)] = tif_data_temp[
                                                                                                       -delta_neighbour:,
                                                                                                       :end_tif_lon_idx]
    #########left tile
    temp_tif_lat = tiff_lat
    temp_tif_lon = tiff_lon_left
    lat_txt = f'N{temp_tif_lat}' if temp_tif_lat >= 0 else f'S{np.abs(temp_tif_lat)}'
    lon_txt = f'E{temp_tif_lon}' if temp_tif_lon >= 0 else f'W{np.abs(temp_tif_lon)}'
    temp_tif_file = f'GWL_FCS30D_{year}Maps_{lon_txt}{lat_txt}.tif'
    if temp_tif_file in tiff_lst:
        temp_tif_file = dir + temp_tif_file
        src_temp = rasterio.open(temp_tif_file, "r")
        tif_data_temp = src_temp.read(1)
        end_tif_lon_idx = np.min((tif_data_temp.shape[1], tif_data.shape[1]))
        end_tif_lat_idx = np.min((tif_data_temp.shape[0], tif_data.shape[0]))
        data_with_neighbours[delta_neighbour:(delta_neighbour + end_tif_lat_idx), :delta_neighbour] = tif_data_temp[
                                                                                                      :end_tif_lat_idx,
                                                                                                      -delta_neighbour:]
    #########down tile
    temp_tif_lat = tiff_lat_down
    temp_tif_lon = tiff_lon
    lat_txt = f'N{temp_tif_lat}' if temp_tif_lat >= 0 else f'S{np.abs(temp_tif_lat)}'
    lon_txt = f'E{temp_tif_lon}' if temp_tif_lon >= 0 else f'W{np.abs(temp_tif_lon)}'
    temp_tif_file = f'GWL_FCS30D_{year}Maps_{lon_txt}{lat_txt}.tif'
    if temp_tif_file in tiff_lst:
        temp_tif_file = dir + temp_tif_file
        src_temp = rasterio.open(temp_tif_file, "r")
        tif_data_temp = src_temp.read(1)
        end_tif_lon_idx = np.min((tif_data_temp.shape[1], tif_data.shape[1]))
        end_tif_lat_idx = np.min((tif_data_temp.shape[0], tif_data.shape[0]))
        data_with_neighbours[-delta_neighbour:, delta_neighbour:(delta_neighbour + end_tif_lon_idx)] = tif_data_temp[
                                                                                                       :delta_neighbour,
                                                                                                       :end_tif_lon_idx]
    #########right tile
    temp_tif_lat = tiff_lat
    temp_tif_lon = tiff_lon_right
    lat_txt = f'N{temp_tif_lat}' if temp_tif_lat >= 0 else f'S{np.abs(temp_tif_lat)}'
    lon_txt = f'E{temp_tif_lon}' if temp_tif_lon >= 0 else f'W{np.abs(temp_tif_lon)}'
    temp_tif_file = f'GWL_FCS30D_{year}Maps_{lon_txt}{lat_txt}.tif'
    if temp_tif_file in tiff_lst:
        temp_tif_file = dir + temp_tif_file
        src_temp = rasterio.open(temp_tif_file, "r")
        tif_data_temp = src_temp.read(1)
        end_tif_lon_idx = np.min((tif_data_temp.shape[1], tif_data.shape[1]))
        end_tif_lat_idx = np.min((tif_data_temp.shape[0], tif_data.shape[0]))
        data_with_neighbours[delta_neighbour:(delta_neighbour + end_tif_lat_idx),
        -delta_neighbour:] = tif_data_temp[:end_tif_lat_idx, :delta_neighbour]
    return data_with_neighbours

def filter_water_systems(tif_data,xoff,yoff,delta_neighbour,reso,data_with_neighbours):
    ####################
    ##only keep pixels labeled as wetlands
    lat_glwd_up = yoff + delta_neighbour * reso
    lat_glwd_down = yoff - (delta_neighbour + tif_data.shape[0]) * reso
    glwd_lat_delta = (84 + 56) / 33600
    glwd_lon_delta = 360 / 86400

    lat_glwd_up_idx = int((84 - lat_glwd_up) / glwd_lat_delta)
    lat_glwd_down_idx = int((84 - lat_glwd_down) / glwd_lat_delta)

    lon_glwd_left = xoff - delta_neighbour * reso
    lon_glwd_right = xoff + (delta_neighbour + tif_data.shape[1]) * reso

    lon_glwd_left_idx = int((lon_glwd_left + 180) / glwd_lon_delta)
    lon_glwd_right_idx = int((lon_glwd_right + 180) / glwd_lon_delta)

    if (lat_glwd_up_idx >= 0) & (lat_glwd_down_idx < 33600) & (lon_glwd_left_idx >= 0) & (lon_glwd_right_idx < 86400):
        nc_file = r'../data/samples/' + f'GLWD_v2_large_lakes_river_stream_reservoir_pc.nc'
        data = Dataset(nc_file, mode='r')
        glwd_lrsr = np.array(
            data.variables['area'][lat_glwd_up_idx:lat_glwd_down_idx, lon_glwd_left_idx:lon_glwd_right_idx])
        data.close()
        glwd_lrsr = resize(glwd_lrsr, (data_with_neighbours.shape[0], data_with_neighbours.shape[1]),
                           anti_aliasing=False, order=0, mode='edge')
        data_with_neighbours[((glwd_lrsr == 1) & (data_with_neighbours == 180))] = 0
        tif_data[((glwd_lrsr[delta_neighbour:-delta_neighbour, delta_neighbour:-delta_neighbour] == 1) & (
                    tif_data == 180))] = 0
    data_with_neighbours[((data_with_neighbours > 179) & (
                data_with_neighbours < 184))] = 1  ######the labels of the wetlands are shown in GWL_FCS30 (Zhang et al. 2023)
    data_with_neighbours[data_with_neighbours > 1] = 0
    data_with_neighbours[np.isnan(data_with_neighbours)] = 0
    data_with_neighbours = data_with_neighbours.astype(np.uint8)

    tif_data[((tif_data > 179) & (tif_data < 184))] = 1  ######the labels of the wetlands are shown in GWL_FCS30 (Zhang et al. 2023)
    tif_data[tif_data > 1] = 0
    return data_with_neighbours,tif_data
def identifying_small_wetlands(lats,lons,start_year,end_year,thresholds_list,dir,output_dir):
    for year in range(start_year,end_year+1):
            target_data = np.full((len(thresholds_list), len(lats), len(lons)), np.nan)
            tiff_lst = os.listdir(dir)
            for tiff in tqdm(tiff_lst[:]):
                # Record the start time
                start_time = time.time()
                tif_file=dir+tiff
                src = rasterio.open(tif_file,"r")
                meta = src.meta
                xoff=meta['transform'].xoff
                yoff=meta['transform'].yoff
                width=meta['width']
                height=meta['height']
                reso=np.abs(src.res[0])
                tif_data = src.read(1)

                delta_neighbour = 1000
                data_with_neighbours=get_tile_edge_pixels(tif_data, yoff, xoff, delta_neighbour, dir)
                data_with_neighbours,tif_data=filter_water_systems(tif_data, xoff, yoff, delta_neighbour, reso, data_with_neighbours)
                area_stats =calculate_areas(data_with_neighbours,delta_neighbour,tif_data,thresholds_list)
                lat_idx = int((90 - yoff) / 0.5)
                lon_idx = int((xoff + 180) / 0.5)
                target_data[:,lat_idx:(lat_idx+10),lon_idx:(lon_idx+10)]=area_stats
                # Record the end time
                end_time = time.time()
                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                # Print the elapsed time
                print(f"{tiff} Elapsed Time: {elapsed_time} seconds")
            save_file =output_dir+ f'GWL_global_0.5degree_{year}.nc'
            nc_fid2 = Dataset(save_file, 'w', format="NETCDF4")
            nc_fid2.createDimension('lat', len(lats))
            nc_fid2.createDimension('lon', len(lons))
            nc_fid2.createDimension('thres', target_data.shape[0])
            latitudes = nc_fid2.createVariable('lat', 'f4', ('lat',))
            longitudes = nc_fid2.createVariable('lon', 'f4', ('lon',))
            time_day = nc_fid2.createVariable("thres", "f4", ("thres",))
            var_save = nc_fid2.createVariable('extent', "f4", ("thres", "lat", "lon",), zlib=True, complevel=4)  # ,significant_digits=4
            time_day[:] =np.array(thresholds_list[:])/pow(1000, 2)#unit: km2
            latitudes[:] = lats[:]
            longitudes[:] = lons[:]
            var_save[:] = target_data[:]
            nc_fid2.close()
            print(year,'done!')
            time.sleep(60)
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
    # the directory of input data
    input_dir=r'../data/samples/'
    # the output directory
    output_dir=f'../output/'
    # identify and calculate the area of small wetlands
    identifying_small_wetlands(lats,lons,start_year,end_year,thresholds_list,input_dir,output_dir)
    # the computational time for the sample data is typically within 1 hour, which may vary depending on the computing platform used. 
