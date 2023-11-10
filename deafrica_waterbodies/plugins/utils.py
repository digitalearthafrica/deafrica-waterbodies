"""
Matthew Alger, Vanessa Newey, Alex Leith
Geoscience Australia
2021
"""
import importlib.util
from pathlib import Path
from types import ModuleType


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
