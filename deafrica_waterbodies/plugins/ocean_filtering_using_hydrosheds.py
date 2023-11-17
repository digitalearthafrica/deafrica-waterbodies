"""
Ocean filtering using HydroSHEDS Land Mask
"""
import skimage.morphology
import xarray as xr
from datacube.testutils.io import rio_slurp_xarray

buffer_pixels = 500 / 30


def transform_hydrosheds_land_mask(hydrosheds_land_mask: xr.DataArray) -> xr.DataArray:
    """
    Function to transform the HydroSHEDs Land Mask into a boolean mask where
    0/False are ocean pixels and 1/True are land pixels.
    """
    # Indicator values: 1 = land, 2 = ocean sink, 3 = inland sink, 255 is no data.
    boolean_mask = (hydrosheds_land_mask != 255) & (hydrosheds_land_mask != 2)

    return boolean_mask


def erode_land_sea_mask(boolean_land_sea_mask: xr.DataArray, buffer_pixels: float) -> xr.DataArray:
    """
    Shrink the land in the land/sea mask.

    Parameters
    ----------
    boolean_land_sea_mask : xr.DataArray
        Boolean mask where 0/False are ocean pixels and 1/True are land pixels.
    buffer_pixels : float
        Number of pixels to erode the land by.

    Returns
    -------
    xr.DataArray
        Eroded land sea mask where 0/False are ocean pixels and 1/True are land pixels.
    """
    eroded_boolean_land_sea_mask = xr.apply_ufunc(
        skimage.morphology.binary_erosion,
        boolean_land_sea_mask,
        skimage.morphology.disk(buffer_pixels),
    )
    return eroded_boolean_land_sea_mask


def load_land_sea_mask(
    land_sea_mask_fp: str,
    wofs_alltime_summary_ds: xr.DataArray,
) -> xr.DataArray:
    """
    Load and reproject the HydroSHEDS Land Mask raster from the file path provided to
    match the loaded datacube WOfS All Time Summary data and transform the raster into
    a boolean mask where 0/False are ocean pixels and 1/True are land pixels.

    Parameters
    ----------
    land_sea_mask_fp : str
        File path to the HydroSHEDS Land Mask raster.
    wofs_alltime_summary_ds : xr.DataArray
        Loaded datacube WOfS All Time Summary data to match to

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
    eroded_boolean_land_sea_mask = erode_land_sea_mask(boolean_land_sea_mask, buffer_pixels)

    return eroded_boolean_land_sea_mask
