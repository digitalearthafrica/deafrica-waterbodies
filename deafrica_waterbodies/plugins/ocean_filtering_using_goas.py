"""
Ocean filtering using Marine Regions Global Oceans and Seas v01
"""
import os

import geopandas as gpd
import numpy as np
import xarray as xr
from deafrica_tools.spatial import xr_rasterize

from deafrica_waterbodies.plugins.utils import erode_land_sea_mask

# File extensions to recognise as Parquet files.
PARQUET_EXTENSIONS = {".pq", ".parquet"}


def load_land_sea_mask(
    land_sea_mask_fp: str,
    wofs_alltime_summary_ds: xr.DataArray,
    buffer_dist_m: float = 500,
) -> xr.DataArray:
    """
    Load the Marine Regions Global Oceans and Seas v01 from the file path
    provided. Rasterize the vector data to match the loaded datacube WOfS
    All Time Summary data and transform the raster into
    a boolean mask where 0/False are ocean pixels and 1/True are land pixels.
    Erode the land pixels by the `buffer_dist_m` buffer distance.

    Parameters
    ----------
    land_sea_mask_fp : str
        File path to the Marine Regions Global Oceans and Seas v01 vector data.
    wofs_alltime_summary_ds : xr.DataArray
        Loaded datacube WOfS All Time Summary data to match to.
    buffer_dist_m : float
            Distance in meters to erode the land by in the land/sea mask.

    Returns
    -------
    xr.DataArray
        A boolean land and sea mask from the Marine Regions Global Oceans and Seas v01 data.
    """

    _, file_extension = os.path.splitext(land_sea_mask_fp)
    if file_extension not in PARQUET_EXTENSIONS:
        land_sea_mask_gdf = gpd.read_file(land_sea_mask_fp).to_crs(
            wofs_alltime_summary_ds.geobox.crs
        )
    else:
        land_sea_mask_gdf = gpd.read_parquet(land_sea_mask_fp).to_crs(
            wofs_alltime_summary_ds.geobox.crs
        )

    land_sea_mask_ds = xr_rasterize(land_sea_mask_gdf, wofs_alltime_summary_ds)
    boolean_land_sea_mask = np.logical_not(land_sea_mask_ds)

    # Erode the land in the land sea mask
    eroded_boolean_land_sea_mask = erode_land_sea_mask(boolean_land_sea_mask, buffer_dist_m)

    return eroded_boolean_land_sea_mask
