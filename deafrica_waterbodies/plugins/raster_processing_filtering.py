"""
Raster processing
"""
import xarray as xr

land_sea_mask_fp = "/g/data/deafrica-waterbodies/masks/af_msk_3s.tif"


def filter_land_sea_mask(hydrosheds_land_mask: xr.DataArray) -> xr.DataArray:
    """
    Function to filter the HydroSHEDs Land Mask into a boolean mask.
    """
    # Indicator values: 1 = land, 2 = ocean sink, 3 = inland sink, 255 is no data.
    boolean_mask = (hydrosheds_land_mask != 255) & (hydrosheds_land_mask != 2)
    return boolean_mask
