import logging
import math
import os
from importlib import import_module

import click
import fsspec
import geopandas as gpd
import pandas as pd

from deafrica_waterbodies.attributes import (
    add_polygon_properties,
    add_timeseries_attribute,
    assign_unique_ids,
)
from deafrica_waterbodies.cli.logs import logging_setup
from deafrica_waterbodies.filters import filter_by_area, filter_by_length
from deafrica_waterbodies.group_polygons import split_polygons_by_region
from deafrica_waterbodies.io import (
    check_dir_exists,
    check_file_exists,
    check_if_s3_uri,
    find_parquet_files,
    write_waterbodies_to_file,
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
    "--min-valid-observations",
    default=60,
    type=int,
    help="Minimum number of observations for a pixel to be considered valid.",
    show_default=True,
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
    "--land-sea-mask-fp",
    default="",
    help="File path to vector/raster dataset to use to filter out ocean polygons.",
)
@click.option(
    "--raster-processing-plugin-name",
    default=None,
    type=str,
    help="Name of the plugin containing the filtering functions to use in the raster processing space."
    "Plugin file must be in the deafrica_waterbodies/plugins/ directory.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun tiles that have already been processed.",
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
    "--timeseries-directory",
    type=str,
    help="The path to the directory containing the timeseries for the polygons.",
)
@click.option(
    "--file-name-prefix",
    default="waterbodies",
    type=str,
    help="File name for the final output",
)
@click.option(
    "--group-by-wofs-ls-regions/--not-group-by-wofs-ls-regions",
    default=True,
    help="Group waterbody polygons by wofs_ls regions.",
)
@click.option(
    "--length-threshold-km",
    default=150,
    show_default=True,
    help="Length threshold in kilometers by which to filter out large polygons before grouping polygons by wofs_ls region.",
)
def generate_polygons(
    verbose,
    aoi_vector_file,
    tile_size_factor,
    num_workers,
    min_valid_observations,
    detection_threshold,
    extent_threshold,
    land_sea_mask_fp,
    raster_processing_plugin_name,
    overwrite,
    min_polygon_size,
    max_polygon_size,
    output_directory,
    timeseries_directory,
    file_name_prefix,
    group_by_wofs_ls_regions,
    length_threshold_km,
):
    """
    Generate water body polygons from WOfS All Time Summary data
    """
    # Set up logger.
    logging_setup(verbose=verbose)
    _log = logging.getLogger(__name__)

    # Parameters to use when loading datasetspolygons_split_by_region_dir.
    # Chunks selected based on size of WOfs scene.
    dask_chunks = {"x": 3200, "y": 3200, "time": 1}

    # Support pathlib Paths.
    if aoi_vector_file is not None:
        aoi_vector_file = str(aoi_vector_file)

    output_directory = str(output_directory)

    # Directory to write outputs from intermediate steps
    intermediate_outputs_dir = os.path.join(output_directory, "intermediate_outputs")
    # Directory to write generated first set of waterbody polygons to.
    polygons_from_thresholds_dir = os.path.join(
        intermediate_outputs_dir, "polygons_from_thresholds"
    )
    # Directory to write final output.
    final_outputs_dir = os.path.join(output_directory, "historical_extent")
    # Directory to store polygons split by region.
    polygons_split_by_region_dir = os.path.join(
        output_directory, "historical_extent_split_by_wofs_region"
    )

    # Set the filesystem to use.
    if check_if_s3_uri(output_directory):
        fs = fsspec.filesystem("s3")
    else:
        fs = fsspec.filesystem("file")

    if not check_dir_exists(intermediate_outputs_dir):
        fs.mkdirs(intermediate_outputs_dir, exist_ok=True)
        _log.info(f"Created directory {intermediate_outputs_dir}")

    if not check_dir_exists(polygons_from_thresholds_dir):
        fs.mkdirs(polygons_from_thresholds_dir, exist_ok=True)
        _log.info(f"Created directory {polygons_from_thresholds_dir}")

    if not check_dir_exists(final_outputs_dir):
        fs.mkdirs(final_outputs_dir, exist_ok=True)
        _log.info(f"Created directory {final_outputs_dir}")

    if group_by_wofs_ls_regions:
        if not check_dir_exists(polygons_split_by_region_dir):
            fs.mkdirs(polygons_split_by_region_dir, exist_ok=True)
            _log.info(f"Created directory {polygons_split_by_region_dir}")

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

    # Set the wetness thresholds.
    min_wet_thresholds = set_wetness_thresholds(
        detection_threshold=detection_threshold, extent_threshold=extent_threshold
    )

    # Set filters to apply during raster processing.
    if raster_processing_plugin_name is not None:
        # Read the plugin as a Python module.
        module = import_module(f"deafrica_waterbodies.plugins.{raster_processing_plugin_name}")
        plugin_file = module.__file__
        plugin = run_plugin(plugin_file)
        _log.info(f"Using plugin {plugin_file}")
        validate_plugin(plugin)
    else:
        plugin = None

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
                    plugin=plugin,
                    dask_chunks=dask_chunks,
                    min_valid_observations=min_valid_observations,
                    min_wet_thresholds=min_wet_thresholds,
                    land_sea_mask_fp=land_sea_mask_fp,
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

    tile_extents_fp = os.path.join(intermediate_outputs_dir, "tile_boundaries.parquet")

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
        intermediate_outputs_dir, "raster_polygons_merged_at_tile_boundaries.parquet"
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
        os.path.join(intermediate_outputs_dir, "area_filtered_raster_polygons.parquet")
    )

    waterbodies_gdf = assign_unique_ids(polygons=area_filtered_raster_polygons, precision=10)

    waterbodies_gdf = add_polygon_properties(polygons=waterbodies_gdf)

    waterbodies_gdf = add_timeseries_attribute(
        polygons=waterbodies_gdf,
        timeseries_directory=timeseries_directory,
        region_code="af-south-1",
    )

    # Reproject to EPSG:4326
    waterbodies_gdf_4326 = waterbodies_gdf.to_crs("EPSG:4326")

    # Write to disk.
    write_waterbodies_to_file(
        waterbodies_gdf=waterbodies_gdf_4326,
        output_directory=final_outputs_dir,
        file_name_prefix=file_name_prefix,
    )

    waterbodies_gdf_4326.to_parquet(os.path.join(final_outputs_dir, f"{file_name_prefix}.parquet"))

    if group_by_wofs_ls_regions:
        waterbodies_gdf_4326 = filter_by_length(
            polygons_gdf=waterbodies_gdf_4326, length_threshold_km=length_threshold_km
        )

        split_by_region_fps = split_polygons_by_region(  # noqa F841
            polygons_gdf=waterbodies_gdf_4326,
            output_directory=polygons_split_by_region_dir,
            product="wofs_ls",
        )
