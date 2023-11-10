"""
Ocean filtering using HydroSHEDS Land Mask
"""
import xarray as xr
from datacube.testutils.io import rio_slurp_xarray


def filter_land_sea_mask(hydrosheds_land_mask: xr.DataArray) -> xr.DataArray:
    """
    Function to filter the HydroSHEDs Land Mask into a boolean mask.
    """
    # Indicator values: 1 = land, 2 = ocean sink, 3 = inland sink, 255 is no data.
    boolean_mask = (hydrosheds_land_mask != 255) & (hydrosheds_land_mask != 2)

    return boolean_mask


def load_land_sea_mask(
    land_sea_mask_fp: str,
    wofs_alltime_summary_ds: xr.DataArray,
) -> xr.DataArray:
    """
    Load and reproject the HydroSHEDS Land Mask raster from the file path provided to
    match the loaded datacube WOfS All Time Summary data.

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
    boolean_land_sea_mask = filter_land_sea_mask(land_sea_mask_ds)

    return boolean_land_sea_mask
