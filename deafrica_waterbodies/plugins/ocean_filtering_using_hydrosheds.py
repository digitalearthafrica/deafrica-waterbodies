"""
Ocean filtering using HydroSHEDS Land Mask
"""
import xarray as xr
from datacube.testutils.io import rio_slurp_xarray

land_sea_mask_fp = "/g/data/deafrica-waterbodies/masks/af_msk_3s.tif"


def filter_land_sea_mask(hydrosheds_land_mask: xr.DataArray) -> xr.DataArray:
    """
    Function to filter the HydroSHEDs Land Mask into a boolean mask.
    """
    # Indicator values: 1 = land, 2 = ocean sink, 3 = inland sink, 255 is no data.
    boolean_mask = (hydrosheds_land_mask != 255) & (hydrosheds_land_mask != 2)
    return boolean_mask


def load_land_sea_mask(land_sea_mask_fp, wofs_alltime_summary_ds, resampling_method):
    land_sea_mask_ds = rio_slurp_xarray(
        fname=land_sea_mask_fp,
        gbox=wofs_alltime_summary_ds.geobox,
        resampling=resampling_method,
    )
    # Filter the land sea mask.
    boolean_land_sea_mask = filter_land_sea_mask(land_sea_mask_ds)

    return boolean_land_sea_mask
