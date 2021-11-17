import xarray as xr
import rioxarray as rxr
import numpy as np
import os, sys
import glob
import geopandas as gpd
from shapely.geometry import box
import datetime


def get_modis_crs():
    f = "/css/modis/Collection6/L3/MOD13Q1-Vegetation/2020/033/MOD13Q1.A2020033.h16v07.006.2020050021554.hdf"
    v = rxr.open_rasterio(f)
    return v.rio.crs

def clip_ndvi(infile, var, bbox, outfile):
    #load variable from hdf
    ndvi = rxr.open_rasterio(infile, masked=True, variable=var).squeeze().to_array()

    ndvi_clip = ndvi.rio.clip([box(*bbox.total_bounds)],
                           all_touched = True,
                           from_disk = True).squeeze()
    if outfile is not None:
        np.save(outfile, ndvi_clip.to_numpy())

def clip_precip(infile, var, bbox, outfile):
    #load variable from netcdf
    precip = rxr.open_rasterio(infile, masked=True, variable=var, decode_times=False).squeeze().to_array()
    #add crs
    precip = precip.rio.write_crs(4326)

    # reproject
    pr_reproj = precip.rio.reproject(bbox.crs)

    pr_clip = pr_reproj.rio.clip([box(*bbox.total_bounds)],
                           all_touched = True,
                           from_disk = True).squeeze()
    if outfile is not None:
        np.save(outfile, pr_clip.to_numpy())

def main():
    
    ndvi_path = "/css/modis/Collection6/L3/MOD13Q1-Vegetation"
    precip_path = "/css/imerg/daily_Late_V06"


    tile = "h16v07" # MODIS tile contains Senegal
    years = np.arange(2008, 2018) 
    days = np.arange(1, 365, 16)

    # MODIS CRS
    t_crs = get_modis_crs()

    # Shape file for Senegal Podor
    shp_file = "/att/nobackup/jli30/workspace/ndvi_MGBrown_notebooks/data/senegal_podor.shp"
    bnd = gpd.read_file(shp_file)
    # Reproject if needed
    if not bnd.crs == t_crs:
        bnd = bnd.to_crs(t_crs)
    
    # Proc NDVI
    for year in years:
        for day in days:

            print(f"Process MODIS Veg {str(year)} {str(day).zfill(3)}")
            #file name
            fn = os.path.join(ndvi_path, 
                  str(year),
                  str(day).zfill(3),
                  f"*{tile}*.hdf")
            file = glob.glob(fn)[0]

            #ndvi variable name
            var = ['250m 16 days NDVI']

            #clip ndvi & save
            out_path = "/att/nobackup/jli30/workspace/ndvi_MGBrown_notebooks/data/timeseries"
            out_fn = f"ndvi_{str(year)}_{str(day).zfill(3)}_podor.npy"
            ofile = os.path.join(out_path, out_fn)
            #clip_ndvi(file, var, bnd, ofile)

    # Proc Precip
    for year in years:
        fn = os.path.join(precip_path,
                 str(year),
                 "*.nc4")
        files = glob.glob(fn)
        var = ['precipitationCal']
        for f in files:
            print(f"Process Precipitation {str(year)} {os.path.basename(f)}")
            tstamp = os.path.basename(f).split('.')[-3].split('-')[0]
            #clip precipitation & save
            out_path = "/att/nobackup/jli30/workspace/ndvi_MGBrown_notebooks/data/timeseries"
            out_fn = f"precip_{tstamp}_podor.npy"
            ofile = os.path.join(out_path, out_fn)
            #clip_precip(f, var, bnd, ofile)



if __name__ == "__main__":
    sys.exit(main())

'''
tile = "h16v07"
ndvi_path = "/css/modis/Collection6/L3/MOD13Q1-Vegetation"
year = 2008
days = np.arange(1, 265, 16)
# example of one day
day = days[5]
fn = os.path.join(ndvi_path, 
                  str(year),
                  str(day).zfill(3),
                  f"*{tile}*.hdf")
file = glob.glob(fn)[0]
print(file)

var = ['250m 16 days NDVI']
ndvi = rxr.open_rasterio(file, masked=True, variable=var).squeeze().to_array()
print(ndvi)

shp_file = "/att/nobackup/jli30/workspace/ndvi_MGBrown_notebooks/data/senegal_podor.shp"
bnd = gpd.read_file(shp_file)
bnd_sin = bnd.to_crs(ndvi.rio.crs)

ndvi_clip = ndvi.rio.clip([box(*bnd_sin.total_bounds)],
                           all_touched = True,
                           from_disk = True).squeeze()

out_path = "/att/nobackup/jli30/workspace/ndvi_MGBrown_notebooks/data/timeseries"
out_fn = f"ndvi_{str(year)}_{str(day).zfill(3)}_podor.npy"

np.save(os.path.join(out_path, out_fn), ndvi_clip.to_numpy())
'''
