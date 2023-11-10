"""
Ocean filtering using Marine Regions Global Oceans and Seas v01
"""
import os

import geopandas as gpd
import numpy as np
import xarray as xr
from deafrica_tools.spatial import xr_rasterize

# File extensions to recognise as Parquet files.
PARQUET_EXTENSIONS = {".pq", ".parquet"}


def transform_hydrosheds_land_mask(hydrosheds_land_mask: xr.DataArray) -> xr.DataArray:
    """
    Function to transform the HydroSHEDs Land Mask into a boolean mask where
    0/False are ocean pixels and 1/True are land pixels.
    """
    # Indicator values: 1 = land, 2 = ocean sink, 3 = inland sink, 255 is no data.
    boolean_mask = (hydrosheds_land_mask != 255) & (hydrosheds_land_mask != 2)

    return boolean_mask


def load_land_sea_mask(
    land_sea_mask_fp: str,
    wofs_alltime_summary_ds: xr.DataArray,
) -> xr.DataArray:
    """
    Load the Marine Regions Global Oceans and Seas v01 from the file path
    provided. Rasterize the vector data to match the loaded datacube WOfS
    All Time Summary data and transform the raster into
    a boolean mask where 0/False are ocean pixels and 1/True are land pixels.

    Parameters
    ----------
    land_sea_mask_fp : str
        File path to the Marine Regions Global Oceans and Seas v01 vector data.
    wofs_alltime_summary_ds : xr.DataArray
        Loaded datacube WOfS All Time Summary data to match to.

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

    return boolean_land_sea_mask
