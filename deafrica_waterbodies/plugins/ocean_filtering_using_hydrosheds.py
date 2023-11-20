"""
Ocean filtering using HydroSHEDS Land Mask
"""
import xarray as xr
from datacube.testutils.io import rio_slurp_xarray

from deafrica_waterbodies.plugins.utils import erode_land_sea_mask


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
    buffer_dist_m: float = 500,
) -> xr.DataArray:
    """
    Load and reproject the HydroSHEDS Land Mask raster from the file path provided to
    match the loaded datacube WOfS All Time Summary data. Transform the loaded raster into
    a boolean mask where 0/False are ocean pixels and 1/True are land pixels and erode the land
    pixels by the `buffer_dist_m` buffer distance.

    Parameters
    ----------
    land_sea_mask_fp : str
        File path to the HydroSHEDS Land Mask raster.
    wofs_alltime_summary_ds : xr.DataArray
        Loaded datacube WOfS All Time Summary data to match to
    buffer_dist_m : float
            Distance in meters to erode the land by in the land/sea mask.
    Returns
    -------
    xr.DataArray
        A boolean land and sea mask from the HydroSHEDs Land Mask.
    """
    land_sea_mask_ds = rio_slurp_xarray(
        fname=land_sea_mask_fp,
        gbox=wofs_alltime_summary_ds.geobox,
        resampling="bilinear",
    )

    # Filter the land sea mask.
    boolean_land_sea_mask = transform_hydrosheds_land_mask(land_sea_mask_ds)

    # Erode the land in the land sea mask
    eroded_boolean_land_sea_mask = erode_land_sea_mask(boolean_land_sea_mask, buffer_dist_m)

    return eroded_boolean_land_sea_mask
