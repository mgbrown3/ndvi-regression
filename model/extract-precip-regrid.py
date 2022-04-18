import rioxarray as rxr
import xarray as xr
import numpy as np
import geopandas as gpd
import os, sys
import glob
from shapely.geometry import box
from scipy import interpolate

import datetime

## This script is an example to prepare imerg precip data as inputs for convLSTM
## The precip data is subset for podor, then regridded to ndvi resolution (16-day average & 250m xy-axis)
if __name__ == '__main__':

    # get bounding box of Senegal Podor 
    shp_file = "/adapt/nobackup/people/jli30/workspace/ndvi_MGBrown_notebooks/data/senegal_podor.shp"
    bnd = gpd.read_file(shp_file)

    # clip modis ndvi & extract coords
    modis_f =  "/css/modis/Collection6/L3/MOD13Q1-Vegetation/2020/033/MOD13Q1.A2020033.h16v07.006.2020050021554.hdf"
    var_ndvi = '250m 16 days NDVI'
    ndvi = rxr.open_rasterio(modis_f, masked=True, variable=var_ndvi).squeeze().to_array()
    ndvi_clip = ndvi.rio.clip([box(*bnd.total_bounds)],
                         all_touched = True, 
                         from_disk = True).squeeze()
    x_coords = ndvi_clip.x.values
    y_coords = ndvi_clip.y.values

    modis_crs = ndvi.rio.crs
    x_dense, y_dense = np.meshgrid(x_coords, y_coords)   ## target grids

    # process precip data
    sp_size=16   ## target (ndvi) time interval
    years = np.arange(2008, 2018) ## range of years
    days = np.arange(1, 365, sp_size) ## timestamps to match ndvi

    precip_path = "/css/imerg/daily_Late_V06"
    out_path = "/adapt/nobackup/people/jli30/workspace/ndvi_MGBrown_notebooks/data/timeseries" ## output path


    for year in years:
        fn = os.path.join(precip_path, str(year), '*.nc4')
        files = glob.glob(fn)
        files = sorted(files)
        f_splited = [files[x:x+sp_size] for x in range(0, len(files), sp_size)] ## group daily files into 16-day
        var_pr = 'precipitationCal'  ## precipitation variable
        
        for i, day in enumerate(days):
            print(f"Proc {year}  {day}")
            ds= xr.open_mfdataset(f_splited[i], variable=var_pr, engine="rasterio") ## open 16 files at once
            pr = ds.mean('time') ## calculate time average

            # add crs
            pr = pr.rio.write_crs(4326)
            # reproj
            pr_reproj = pr.rio.reproject(modis_crs)
            
            # clip
            pr_clip = pr_reproj.rio.clip([box(*bnd.total_bounds)],
                            all_touched = True, 
                            from_disk = True).squeeze()
            
            # regridding
            x_sparse, y_sparse = np.meshgrid(pr_clip.x.values, pr_clip.y.values)  ## precip coords
            sparse_points = np.stack([x_sparse.ravel(), y_sparse.ravel()], -1)
            z_sparse= pr_clip[var_pr].values  ## precip value
            z_dense = interpolate.griddata(sparse_points, z_sparse.ravel(), (x_dense, y_dense), method="nearest")

            # write out
            out_fn = f"precip_{str(year)}_{str(day).zfill(3)}_podor.npy"
            np.save(os.path.join(out_path, out_fn), z_dense)