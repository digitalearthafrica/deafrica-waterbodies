import logging
import os
from importlib import import_module

import click
import datacube
import fsspec
import geopandas as gpd
import pandas as pd

from deafrica_waterbodies.cli.logs import logging_setup
from deafrica_waterbodies.io import (
    check_dir_exists,
    check_file_exists,
    check_if_s3_uri,
    find_parquet_files,
)
from deafrica_waterbodies.make_polygons import (
    merge_polygons_at_tile_boundaries,
    process_raster_polygons,
    set_wetness_thresholds,
)
from deafrica_waterbodies.plugins.utils import run_plugin, validate_plugin
from deafrica_waterbodies.tiling import get_wofs_ls_summary_alltime_tiles


@click.command(
    "generate-polygons",
    no_args_is_help=True,
)
@click.option("-v", "--verbose", count=True)
@click.option(
    "--aoi-vector-file", default=None, type=str, help="Vector file defining the area interest."
)
@click.option(
    "--tile-size-factor",
    default=4,
    type=float,
    help="Factor by which to increase/decrease the WOfS All Time Summary product tiles/regions.",
)
@click.option(
    "--num-workers",
    default=8,
    type=int,
    help="Number of worker processes to use when filtering WOfS All Time Summary product tiles",
)
@click.option(
    "--detection-threshold",
    default=0.1,
    type=float,
    help="Threshold to define the location of the water body polygons.",
    show_default=True,
)
@click.option(
    "--extent-threshold",
    default=0.05,
    type=float,
    help="Threshold to define the extent/shape of the water body polygons.",
    show_default=True,
)
@click.option(
    "--min-valid-observations",
    default=60,
    type=int,
    help="Minimum number of observations for a pixel to be considered valid.",
    show_default=True,
)
@click.option(
    "--raster-processing-plugin-name",
    default=None,
    type=str,
    help="Name of the plugin containing the filtering functions to use in the raster processing space. Plugin file must be in the deafrica_waterbodies/plugins/ directory.",
)
@click.option(
    "--output-directory",
    type=str,
    help="Directory to write the water body polygons to.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun tiles that have already been processed.",
)
def generate_polygons(
    verbose,
    aoi_vector_file,
    tile_size_factor,
    num_workers,
    detection_threshold,
    extent_threshold,
    min_valid_observations,
    plugin_name,
    output_directory,
    overwrite,
):
    # Set up logger.
    logging_setup(verbose=verbose)
    _log = logging.getLogger(__name__)

    # Parameters to use when loading datasets.
    dask_chunks = {"x": 3200, "y": 3200, "time": 1}

    # Support pathlib Paths.
    if aoi_vector_file is not None:
        aoi_vector_file = str(aoi_vector_file)

    output_directory = str(output_directory)

    # Set the filesystem to use.
    if check_if_s3_uri(output_directory):
        fs = fsspec.filesystem("s3")
    else:
        fs = fsspec.filesystem("file")

    # Load the area of interest as a GeoDataFrame.
    if aoi_vector_file is not None:
        try:
            aoi_gdf = gpd.read_file(aoi_vector_file)
        except Exception as error:
            _log.exception(f"Could not read the file {aoi_vector_file}")
            raise error
    else:
        aoi_gdf = None

    # Get the tiles fo the wofs_ls_summary_alltime product.
    tiles, grid_workflow = get_wofs_ls_summary_alltime_tiles(
        aoi_gdf=aoi_gdf, tile_size_factor=tile_size_factor, num_workers=num_workers
    )

    # Directory to write generated waterbody polygons to.
    polygons_from_thresholds_dir = os.path.join(output_directory, "polygons_from_thresholds")

    # Check if the directory exists. If it does not, create it.
    if not check_dir_exists(polygons_from_thresholds_dir):
        fs.mkdirs(polygons_from_thresholds_dir, exist_ok=True)
        _log.info(f"Created directory {polygons_from_thresholds_dir}")

    # Set the wetness thresholds.
    min_wet_thresholds = set_wetness_thresholds(
        detection_threshold=detection_threshold, extent_threshold=extent_threshold
    )

    if plugin_name is not None:
        # Read the plugin as a Python module.
        module = import_module(f"deafrica_conflux.plugins.{plugin_name}")
        plugin_file = module.__file__
        plugin = run_plugin(plugin_file)
        _log.info(f"Using plugin {plugin_file}")
        validate_plugin(plugin)

    # Generate the first set of polygons for each of the tiles.
    for tile in tiles.items():
        tile_id = tile[0]
        raster_polygons_fp = os.path.join(
            polygons_from_thresholds_dir, f"{tile_id[0]}_{tile_id[1]}_raster_polygons.parquet"
        )

        if not overwrite:
            _log.info(f"Checking existence of {raster_polygons_fp}")
            exists = check_file_exists(raster_polygons_fp)
            if exists:
                _log.info(
                    f"{raster_polygons_fp} exists! \n Skipping generating water body polygons for {tile_id}."
                )

        if overwrite or not exists:
            try:
                _log.info(f"Generating water body polygons for tile {tile_id}.")
                raster_polygons = process_raster_polygons(
                    tile=tile,
                    grid_workflow=grid_workflow,
                    dask_chunks=dask_chunks,
                    min_valid_observations=min_valid_observations,
                    min_wet_thresholds=min_wet_thresholds,
                    land_sea_mask_fp=land_sea_mask_fp,
                    filter_land_sea_mask=filter_hydrosheds_land_mask,
                )
                if raster_polygons.empty:
                    _log.info(f"Tile {str(tile_id)} contains no water body polygons.")
                else:
                    # Drop the attributes column if it exists.
                    raster_polygons.drop(columns=["attribute"], errors="ignore", inplace=True)
                    # Write the polygons to parquet files.
                    raster_polygons.to_parquet(raster_polygons_fp)
                    _log.info(
                        f"Tile {str(tile_id)} water body polygons written to {raster_polygons_fp}"
                    )
            except Exception as error:
                _log.exception(f"\nTile {str(tile_id)} did not run. \n")
                _log.exception(error)
