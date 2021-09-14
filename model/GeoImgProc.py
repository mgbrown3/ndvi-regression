import os
from osgeo import gdal
# -----------------------------------------------------------------------------
# class TifProcess
#
# This class hadels geoTiff preparation through
# GDAL reprojection, resampling, and cropping based on cutline 
# 
# -----------------------------------------------------------------------------

class GeoImgProc(object):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def warp(srcFile, dstFile, **kwargs):
        cmd = "gdalwarp"

        for k, v in kwargs.items():
            cmd += f" {k}  {v}"
        
        cmd += f" {srcFile} {dstFile}"

        res = os.system(cmd)

        if res !=0:
            raise RuntimeError(f"Command failed: {cmd}")

        if not os.path.exists(dstFile):
            raise RuntimeError(f"gdal did not produce output. Command: {cmd}")