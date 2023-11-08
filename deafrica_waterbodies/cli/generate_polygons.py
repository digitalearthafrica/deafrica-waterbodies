import logging
import math
import os
from importlib import import_module

import click
import fsspec
import geopandas as gpd
import pandas as pd

from deafrica_waterbodies.attributes import (
    add_area_and_perimeter_attributes,
    add_timeseries_attribute,
    assign_unique_ids,
)
from deafrica_waterbodies.cli.logs import logging_setup
from deafrica_waterbodies.filters import filter_by_area
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
    "--min-polygon-size",
    default=4500,
    show_default=True,
    help="Minimum area in m2 of the waterbody polygons to be included.",
)
@click.option(
    "--max-polygon-size",
    default=math.inf,
    show_default=True,
    help="Maximum area in m2 of the waterbody polygons to be included.",
)
@click.option(
    "--output-directory",
    type=str,
    help="Directory to write the water body polygons to.",
)
@click.option(
    "--product-version",
    type=str,
    default="0.0.2",
    show_default=True,
    help="Product version for the DE Africa Waterbodies product.",
)
@click.option(
    "--timeseries-bucket",
    type=str,
    help="The s3 bucket to containing the timeseries for the polygons.",
)
@click.option(
    "--file-name-prefix",
    type=str,
    help="File name for the final output",
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
    min_polygon_size,
    max_polygon_size,
    product_version,
    timeseries_bucket,
    file_name_prefix,
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

    # Set filters to apply during raster processing.
    if plugin_name is not None:
        # Read the plugin as a Python module.
        module = import_module(f"deafrica_waterbodies.plugins.{plugin_name}")
        plugin_file = module.__file__
        plugin = run_plugin(plugin_file)
        _log.info(f"Using plugin {plugin_file}")
        validate_plugin(plugin)

        land_sea_mask_fp = plugin.land_sea_mask_fp
        filter_land_sea_mask = plugin.filter_land_sea_mask
    else:
        land_sea_mask_fp = ""
        filter_land_sea_mask = None

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
                    filter_land_sea_mask=filter_land_sea_mask,
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

    # Get the extent for each tile.
    crs = grid_workflow.grid_spec.crs
    tile_ids = [tile[0] for tile in tiles.items()]
    tile_extents_geoms = [tile[1].geobox.extent.geom for tile in tiles.items()]
    tile_extents_gdf = gpd.GeoDataFrame(
        {"tile_id": tile_ids, "geometry": tile_extents_geoms}, crs=crs
    )

    tile_extents_fp = os.path.join(output_directory, "tile_boundaries.parquet")

    tile_extents_gdf.to_parquet(tile_extents_fp)
    _log.info(f"Tile boundaries written to {tile_extents_fp}")

    # Find all parquet files for the first set of polygons.
    raster_polygon_paths = find_parquet_files(
        path=polygons_from_thresholds_dir, pattern=".*raster_polygons.*"
    )
    _log.info(f"Found {len(raster_polygon_paths)} parquet files for the raster polygons.")

    # Load all polygons into a single GeoDataFrame.
    _log.info("Loading the raster polygons parquet files..")
    raster_polygon_polygons_list = []
    for path in raster_polygon_paths:
        gdf = gpd.read_parquet(path)
        raster_polygon_polygons_list.append(gdf)

    raster_polygons = pd.concat(raster_polygon_polygons_list, ignore_index=True)
    _log.info(f"Found {len(raster_polygons)} raster polygons.")

    _log.info("Merging raster waterbody polygons located at tile boundaries...")
    raster_polygons_merged = merge_polygons_at_tile_boundaries(raster_polygons, tile_extents_gdf)
    # Drop the attributes column if it exists.
    raster_polygons_merged.drop(columns=["attribute"], errors="ignore", inplace=True)
    _log.info(
        f"Raster polygons count after merging polygons at tile boundaries {len(raster_polygons_merged)}."
    )

    _log.info("Writing raster polygons merged at tile boundaries to disk..")
    raster_polygons_merged_fp = os.path.join(
        output_directory, "raster_polygons_merged_at_tile_boundaries.parquet"
    )

    raster_polygons_merged.to_parquet(raster_polygons_merged_fp)
    _log.info(f"Polygons written to {raster_polygons_merged_fp}")

    # Delete to conserve memeory
    del raster_polygons
    del tile_extents_gdf

    # Filter the polygons by area.
    area_filtered_raster_polygons = filter_by_area(
        raster_polygons_merged, min_polygon_size=min_polygon_size, max_polygon_size=max_polygon_size
    )
    area_filtered_raster_polygons.to_parquet(
        os.path.join(output_directory, "area_filtered_raster_polygons.parquet")
    )

    waterbodies_gdf = assign_unique_ids(polygons=area_filtered_raster_polygons)
    waterbodies_gdf = add_area_and_perimeter_attributes(polygons=waterbodies_gdf)
    waterbodies_gdf = add_timeseries_attribute(
        polygons=waterbodies_gdf,
        product_version=product_version,
        timeseries_bucket=timeseries_bucket,
    )
