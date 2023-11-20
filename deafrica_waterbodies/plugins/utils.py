"""
Matthew Alger, Vanessa Newey, Alex Leith
Geoscience Australia
2021
"""
import importlib.util
from pathlib import Path
from types import ModuleType

import skimage
import xarray as xr


def run_plugin(plugin_path: str | Path) -> ModuleType:
    """Run a Python plugin from a path.

    Arguments
    ---------
    plugin_path : str | Path
        Path to Python plugin file.

    Returns
    -------
    module
    """
    plugin_path = str(plugin_path)

    spec = importlib.util.spec_from_file_location("deafrica_waterbodies.plugin", plugin_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_plugin(plugin: ModuleType):
    """Check that a plugin declares required globals."""
    # Check globals.
    required_globals = [
        "load_land_sea_mask",
    ]
    for name in required_globals:
        if not hasattr(plugin, name):
            raise ValueError(f"Plugin missing {name}")

    # Check that functions are runnable.
    required_functions = ["load_land_sea_mask"]
    for name in required_functions:
        assert hasattr(getattr(plugin, name), "__call__")


def erode_land_sea_mask(boolean_land_sea_mask: xr.DataArray, buffer_dist_m: float) -> xr.DataArray:
    """
    Shrink the land in the land/sea mask.

    Parameters
    ----------
    boolean_land_sea_mask : xr.DataArray
        Boolean mask where 0/False are ocean pixels and 1/True are land pixels.
    buffer_dist_m : float
        Distance in meters to erode the land by in the land/sea mask.

    Returns
    -------
    xr.DataArray
        Eroded land sea mask where 0/False are ocean pixels and 1/True are land pixels.
    """
    buffer_pixels = buffer_dist_m / abs(boolean_land_sea_mask.geobox.resolution[0])

    eroded_boolean_land_sea_mask = xr.apply_ufunc(
        skimage.morphology.binary_erosion,
        boolean_land_sea_mask,
        skimage.morphology.disk(buffer_pixels),
    )
    return eroded_boolean_land_sea_mask
