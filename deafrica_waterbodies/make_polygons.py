"""
Make waterbody polygons from the Water Observations from Space all-time
summary.

Geoscience Australia - 2021
    Claire Krause
    Matthew Alger
"""

import logging
from pathlib import Path
from typing import Callable

import datacube
import datacube.model
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from datacube.testutils.io import rio_slurp_xarray
from deafrica_tools.spatial import xr_vectorize

from deafrica_waterbodies.filters import filter_by_intersection, filter_hydrosheds_land_mask

_log = logging.getLogger(__name__)


def set_wetness_thresholds(
    detection_threshold: int | float = 0.1, extent_threshold: int | float = 0.05
) -> list:
    """
    Function to set and validate the minimum frequency for water a pixel must have
    to be included.

    Parameters
    ----------
    detection_threshold : int | float
        Threshold used to set the location of the water body polygons.
    extent_threshold : int | float
        Threshold used to set the shape/extent of the water body polygons.

    Returns
    -------
    list
        A list containing the extent and detection thresholds with the extent
        threshold listed first.
    """
    # Check for correct value type.
    assert detection_threshold is not None
    assert extent_threshold is not None
    assert isinstance(detection_threshold, float) or isinstance(detection_threshold, int)
    assert isinstance(extent_threshold, float) or isinstance(extent_threshold, int)

    # Check values.
    assert 0 <= detection_threshold <= 1
    assert 0 <= extent_threshold <= 1

    if extent_threshold > detection_threshold:
        _log.error(
            f"Detection threshold {detection_threshold} is less than the extent threshold {extent_threshold}."
        )
        error_msg = """We will be running a hybrid wetness threshold.
        Please ensure that the detection threshold has a higher value than the extent threshold."""
        raise ValueError(error_msg)
    else:
        _log.info(
            f"""We will be running a hybrid wetness threshold.
        You have set {detection_threshold} as the location threshold, which will define the location of the water body polygons.
        You have set {extent_threshold} as the extent threshold, which will define the extent/shape of the water body polygons."""
        )
        return [extent_threshold, detection_threshold]


def merge_polygons_at_dataset_boundaries(waterbody_polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Function to merge waterbody polygons located at WOfS All Time Summary dataset boundaries.

    Parameters
    ----------
    waterbody_polygons : gpd.GeoDataFrame
        The waterbody polygons.

    Returns
    -------
    gpd.GeoDataFrame
        Waterbody polygons with polygons located at WOfS All Time Summary dataset boundaries merged.
    """
    # Get the dataset extents/regions for the WOfS All Time Summary product.
    ds_extents = gpd.read_file(
        "https://explorer.digitalearth.africa/api/regions/wofs_ls_summary_alltime"
    ).to_crs(waterbody_polygons.crs)

    # Add a 1 pixel (30 m) buffer to the dataset extents.
    buffered_30m_ds_extents_geom = ds_extents.boundary.buffer(
        30, cap_style="flat", join_style="mitre"
    )
    buffered_30m_ds_extents = gpd.GeoDataFrame(
        geometry=buffered_30m_ds_extents_geom, crs=waterbody_polygons.crs
    )

    # Get the polygons at the dataset boundaries.
    boundary_polygons, not_boundary_polygons = filter_by_intersection(
        gpd_data=waterbody_polygons,
        gpd_filter=buffered_30m_ds_extents,
        invert_mask=False,
        return_inverse=True,
    )

    # Now combine overlapping polygons in boundary_polygons.
    merged_boundary_polygons_geoms = shapely.ops.unary_union(boundary_polygons["geometry"])

    # `Explode` the multipolygon back out into individual polygons.
    merged_boundary_polygons = gpd.GeoDataFrame(
        crs=waterbody_polygons.crs, geometry=[merged_boundary_polygons_geoms]
    )
    merged_boundary_polygons = merged_boundary_polygons.explode(index_parts=True).reset_index(
        drop=True
    )

    # Then combine our merged_boundary_polygons with the not_boundary_polygons.
    all_polygons = gpd.GeoDataFrame(
        pd.concat([not_boundary_polygons, merged_boundary_polygons], ignore_index=True, sort=True)
    ).set_geometry("geometry")

    return all_polygons


def merge_polygons_at_tile_boundaries(
    waterbody_polygons: gpd.GeoDataFrame, tile_extents_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Function to merge waterbody polygons located at WOfS All Time Summary tile boundaries.

    Parameters
    ----------
    waterbody_polygons : gpd.GeoDataFrame
        The waterbody polygons.
    tile_exents_gdf: gpd.GeoDataFrame
        The extents of the tiles used to generate the waterbody polygons.

    Returns
    -------
    gpd.GeoDataFrame
        Waterbody polygons with polygons located at WOfS All Time Summary tile boundaries merged.
    """
    # Get the tiles for the WOfS All Time Summary product.
    tile_extents_gdf = tile_extents_gdf.to_crs(waterbody_polygons.crs)

    # Add a 1 pixel (30 m) buffer to the dataset extents.
    buffered_30m_tile_extents_geom = tile_extents_gdf.boundary.buffer(
        30, cap_style="flat", join_style="mitre"
    )
    buffered_30m_tile_extents = gpd.GeoDataFrame(
        geometry=buffered_30m_tile_extents_geom, crs=waterbody_polygons.crs
    )

    # Get the polygons at the tile boundaries.
    boundary_polygons, not_boundary_polygons = filter_by_intersection(
        gpd_data=waterbody_polygons,
        gpd_filter=buffered_30m_tile_extents,
        invert_mask=False,
        return_inverse=True,
    )

    # Now combine overlapping polygons in boundary_polygons.
    merged_boundary_polygons_geoms = shapely.ops.unary_union(boundary_polygons["geometry"])

    # `Explode` the multipolygon back out into individual polygons.
    merged_boundary_polygons = gpd.GeoDataFrame(
        crs=waterbody_polygons.crs, geometry=[merged_boundary_polygons_geoms]
    )
    merged_boundary_polygons = merged_boundary_polygons.explode(index_parts=True).reset_index(
        drop=True
    )

    # Then combine our merged_boundary_polygons with the not_boundary_polygons.
    all_polygons = gpd.GeoDataFrame(
        pd.concat([not_boundary_polygons, merged_boundary_polygons], ignore_index=True, sort=True)
    ).set_geometry("geometry")

    return all_polygons


def get_polygons_from_dataset(
    dataset_id: str,
    dask_chunks: dict[str, int] = {"x": 3200, "y": 3200, "time": 1},
    resolution: tuple[int, int] = (-30, 30),
    output_crs: str = "EPSG:6933",
    min_valid_observations: int = 128,
    primary_threshold: float = 0.1,
    secondary_threshold: float = 0.05,
    dc: datacube.Datacube | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Generate water body polygons by thresholding a WOfS All Time Summary dataset.

    Parameters
    ----------
    dataset_id : str
        The dataset id of a WOfs All Time summary dataset for which to
        generate waterbody polygons for.
    dask_chunks : dict, optional
        dask_chunks to use to load WOfS data, by default {"x": 3200, "y": 3200, "time": 1}
    resolution : tuple[int, int], optional
        Resolution to use to load WOfS data, by default (-30, 30)
    output_crs : str, optional
        CRS to load data and for the output waterbody polygons, by default "EPSG:6933"
    min_valid_observations : int, optional
        Threshold to use to mask out pixels based on the number of valid WOfS observations for each pixel, by default 128
    primary_threshold : float, optional
        Threshold to use to determine the location of the waterbody polygons, by default 0.1
    secondary_threshold : float, optional
        Threshold to use to determine the extent / shape of the waterbodies polygons, by default 0.05
    dc : datacube.Datacube | None, optional
        Datacube connection, by default None

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        A tuple containing GeoDataFrames of waterbody polygons generated from thresholding WOfS All Time Summary data
        using the primary and secondary thresholds.

    """
    # Set up the primary and secondary thresholds.
    minimum_wet_thresholds = [secondary_threshold, primary_threshold]

    # Create a datacube query object.
    query = dict(
        dask_chunks=dask_chunks,
        resolution=resolution,
        output_crs=output_crs,
    )

    # Connect to the datacube.
    if dc is None:
        dc = datacube.Datacube(app="GenerateWaterbodyPolygons")

    # Get the dataset.
    dataset = dc.index.datasets.get(dataset_id)

    # Generate the waterbody polygons using the primary and secondary thresholds,
    # from the dataset.
    try:
        _log.info(f"Generating water body polygons for dataset {dataset_id}")

        # Load the WOfS All-Time Summary dataset.
        wofs_alltime_summary = dc.load(datasets=[dataset], **query).squeeze()

        # Set the no-data values to nan.
        # Masking here is done using the frequency measurement because for multiple
        # areas NaN values are present in the frequency measurement but the
        # no data value -999 is not present in the count_clear and
        # count_wet measurements.
        # Note: it seems some pixels with NaN values in the frequency measurement
        # have a value of zero in the count_clear and/or the count_wet measurements.
        wofs_alltime_summary = wofs_alltime_summary.where(~np.isnan(wofs_alltime_summary.frequency))

        # Mask pixels not observed at least min_valid_observations times.
        wofs_alltime_summary_valid_clear_count = (
            wofs_alltime_summary.count_clear >= min_valid_observations
        )

        # Generate the polygons.
        generated_polygons = {}
        for threshold in minimum_wet_thresholds:
            # Mask any pixels whose frequency of water detection is less than the threshold.
            wofs_alltime_summary_valid_wetness = wofs_alltime_summary.frequency > threshold

            # Now find pixels that meet both the minimum valid observations
            # and minimum wet threshold criteria.
            wofs_alltime_summary_valid = wofs_alltime_summary_valid_wetness.where(
                wofs_alltime_summary_valid_wetness & wofs_alltime_summary_valid_clear_count
            )

            # Convert the raster to polygons.
            # We use a mask of '1' to only generate polygons around values of '1' (not NaNs).
            polygons_mask = wofs_alltime_summary_valid == 1

            polygons = xr_vectorize(
                wofs_alltime_summary_valid,
                mask=polygons_mask,
                crs=wofs_alltime_summary.geobox.crs,
            )

            # Combine any overlapping polygons.
            merged_polygon_geoms = shapely.ops.unary_union(polygons["geometry"])

            # Turn the combined multipolygon back into a GeoDataFrame.
            try:
                merged_polygons = gpd.GeoDataFrame(geometry=list(merged_polygon_geoms.geoms))
            except AttributeError:
                merged_polygons = gpd.GeoDataFrame(geometry=[merged_polygon_geoms])

            # We need to add the crs back onto the GeoDataFrame.
            merged_polygons.crs = wofs_alltime_summary.geobox.crs

            generated_polygons[threshold] = merged_polygons

    except Exception as error:
        _log.exception(
            f"\nDataset {str(dataset_id)} did not run. \n"
            "This is probably because there are no waterbodies present in this dataset."
        )
        _log.exception(error)

    primary_threshold_polygons = generated_polygons[primary_threshold]
    secondary_threshold_polygons = generated_polygons[secondary_threshold]

    return primary_threshold_polygons, secondary_threshold_polygons


def get_polygons_from_dataset_with_land_sea_mask_filtering(
    dataset_id: str,
    dask_chunks: dict[str, int] = {"x": 3200, "y": 3200, "time": 1},
    resolution: tuple[int, int] = (-30, 30),
    output_crs: str = "EPSG:6933",
    min_valid_observations: int = 128,
    primary_threshold: float = 0.1,
    secondary_threshold: float = 0.05,
    dc: datacube.Datacube | None = None,
    land_sea_mask_fp: str | Path = "",
    resampling_method: str = "bilinear",
    filter_land_sea_mask: Callable = filter_hydrosheds_land_mask,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Generate water body polygons by thresholding a WOfS All Time Summary dataset.
    Use a raster land/sea mask to mask out ocean pixels from the WOfS data before
    vectorizing the polygons.

    Parameters
    ----------
    dataset_id : str
        The dataset id of a WOfs All Time summary dataset for which to
        generate waterbody polygons for.
    dask_chunks : dict, optional
        dask_chunks to use to load WOfS data, by default {"x": 3200, "y": 3200, "time": 1}
    resolution : tuple[int, int], optional
        Resolution to use to load WOfS data, by default (-30, 30)
    output_crs : str, optional
        CRS to load data and for the output waterbody polygons, by default "EPSG:6933"
    min_valid_observations : int, optional
        Threshold to use to mask out pixels based on the number of valid WOfS observations for each pixel, by default 128
    primary_threshold : float, optional
        Threshold to use to determine the location of the waterbody polygons, by default 0.1
    secondary_threshold : float, optional
        Threshold to use to determine the extent / shape of the waterbodies polygons, by default 0.05
    dc : datacube.Datacube | None, optional
        Datacube connection, by default None
    land_sea_mask_fp: str | Path, optional
        File path to raster to use to mask ocean pixels in WOfS data, by default ""
    resampling_method: str, optional
        Resampling method to use when loading the land sea mask raster, by default "bilinear"
    filter_land_sea_mask: Callable, optional
        Function to apply to the land sea mask xr.DataArray to generate a boolean
        mask where pixels with a value of True are land pixels and pixels with a
        value of False are ocean pixels, by default `filter_hydrosheds_land_mask`

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        A tuple containing GeoDataFrames of waterbody polygons generated from thresholding WOfS All Time Summary data
        using the primary and secondary thresholds.

    """
    # Set up the primary and secondary thresholds.
    minimum_wet_thresholds = [secondary_threshold, primary_threshold]

    # Create a datacube query object.
    query = dict(
        dask_chunks=dask_chunks,
        resolution=resolution,
        output_crs=output_crs,
    )

    # Connect to the datacube.
    if dc is None:
        dc = datacube.Datacube(app="GenerateWaterbodyPolygons")

    # Get the dataset.
    dataset = dc.index.datasets.get(dataset_id)

    # Generate the waterbody polygons using the primary and secondary thresholds,
    # from the dataset.
    try:
        _log.info(f"Generating water body polygons for dataset {dataset_id}")

        # Load the WOfS All-Time Summary dataset.
        wofs_alltime_summary = dc.load(datasets=[dataset], **query).squeeze()

        # Load the land sea mask.
        if land_sea_mask_fp:
            land_sea_mask = rio_slurp_xarray(
                fname=land_sea_mask_fp,
                gbox=wofs_alltime_summary.geobox,
                resampling=resampling_method,
            )

            # Filter the land sea mask.
            boolean_land_sea_mask = filter_land_sea_mask(land_sea_mask)

            # Mask the WOfS All-Time Summary dataset using the boolean land sea mask.
            wofs_alltime_summary = wofs_alltime_summary.where(boolean_land_sea_mask)

        # Set the no-data values to nan.
        # Masking here is done using the frequency measurement because for multiple
        # areas NaN values are present in the frequency measurement but the
        # no data value -999 is not present in the count_clear and
        # count_wet measurements.
        # Note: it seems some pixels with NaN values in the frequency measurement
        # have a value of zero in the count_clear and/or the count_wet measurements.
        wofs_alltime_summary = wofs_alltime_summary.where(~np.isnan(wofs_alltime_summary.frequency))

        # Mask pixels not observed at least min_valid_observations times.
        wofs_alltime_summary_valid_clear_count = (
            wofs_alltime_summary.count_clear >= min_valid_observations
        )

        # Generate the polygons.
        generated_polygons = {}
        for threshold in minimum_wet_thresholds:
            # Mask any pixels whose frequency of water detection is less than the threshold.
            wofs_alltime_summary_valid_wetness = wofs_alltime_summary.frequency > threshold

            # Now find pixels that meet both the minimum valid observations
            # and minimum wet threshold criteria.
            wofs_alltime_summary_valid = wofs_alltime_summary_valid_wetness.where(
                wofs_alltime_summary_valid_wetness & wofs_alltime_summary_valid_clear_count
            )

            # Convert the raster to polygons.
            # We use a mask of '1' to only generate polygons around values of '1' (not NaNs).
            polygons_mask = wofs_alltime_summary_valid == 1

            polygons = xr_vectorize(
                wofs_alltime_summary_valid,
                mask=polygons_mask,
                crs=wofs_alltime_summary.geobox.crs,
            )

            # Combine any overlapping polygons.
            merged_polygon_geoms = shapely.ops.unary_union(polygons["geometry"])

            # Turn the combined multipolygon back into a GeoDataFrame.
            try:
                merged_polygons = gpd.GeoDataFrame(geometry=list(merged_polygon_geoms.geoms))
            except AttributeError:
                merged_polygons = gpd.GeoDataFrame(geometry=[merged_polygon_geoms])

            # We need to add the crs back onto the GeoDataFrame.
            merged_polygons.crs = wofs_alltime_summary.geobox.crs

            generated_polygons[threshold] = merged_polygons

    except Exception as error:
        _log.exception(
            f"\nDataset {str(dataset_id)} did not run. \n"
            "This is probably because there are no waterbodies present in this dataset."
        )
        _log.exception(error)

    primary_threshold_polygons = generated_polygons[primary_threshold]
    secondary_threshold_polygons = generated_polygons[secondary_threshold]

    return primary_threshold_polygons, secondary_threshold_polygons


def get_polygons_from_tile(
    tile: tuple[tuple[int, int], datacube.api.grid_workflow.Tile],
    grid_workflow: datacube.api.GridWorkflow,
    dask_chunks: dict[str, int] = {"x": 3200, "y": 3200, "time": 1},
    min_valid_observations: int = 128,
    primary_threshold: float = 0.1,
    secondary_threshold: float = 0.05,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Generate water body polygons by thresholding a WOfS All Time Summary tile.

    Parameters
    ----------
    tile : tuple[tuple[int,int], datacube.api.grid_workflow.Tile]
        The WOfs All Time summary Tile object for which to
        generate waterbody polygons for.
    grid_workflow: datacube.api.GridWorkflow,
        Grid Workflow used to generate the tiles and to be used to load the Tile object.
    dask_chunks : dict, optional
        dask_chunks to use to load WOfS data, by default {"x": 3200, "y": 3200, "time": 1}
    min_valid_observations : int, optional
        Threshold to use to mask out pixels based on the number of valid WOfS observations for each pixel, by default 128
    primary_threshold : float, optional
        Threshold to use to determine the location of the waterbody polygons, by default 0.1
    secondary_threshold : float, optional
        Threshold to use to determine the extent / shape of the waterbodies polygons, by default 0.05

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        A tuple containing GeoDataFrames of waterbody polygons generated from thresholding WOfS All Time Summary data
        using the primary and secondary thresholds.

    """
    # Set up the primary and secondary thresholds.
    minimum_wet_thresholds = [secondary_threshold, primary_threshold]

    # Get the tile id and tile object.
    tile_id = tile[0]
    tile_object = tile[1]

    # Generate the waterbody polygons using the primary and secondary thresholds,
    # from the tile.
    try:
        _log.info(f"Generating water body polygons for tile {tile_id}")

        # Load the data for the tile.
        wofs_alltime_summary = grid_workflow.load(tile_object, dask_chunks=dask_chunks).squeeze()

        # Set the no-data values to nan.
        # Masking here is done using the frequency measurement because for multiple
        # areas NaN values are present in the frequency measurement but the
        # no data value -999 is not present in the count_clear and
        # count_wet measurements.
        # Note: it seems some pixels with NaN values in the frequency measurement
        # have a value of zero in the count_clear and/or the count_wet measurements.
        wofs_alltime_summary = wofs_alltime_summary.where(~np.isnan(wofs_alltime_summary.frequency))

        # Mask pixels not observed at least min_valid_observations times.
        wofs_alltime_summary_valid_clear_count = (
            wofs_alltime_summary.count_clear >= min_valid_observations
        )

        # Generate the polygons.
        generated_polygons = {}
        for threshold in minimum_wet_thresholds:
            # Mask any pixels whose frequency of water detection is less than the threshold.
            wofs_alltime_summary_valid_wetness = wofs_alltime_summary.frequency > threshold

            # Now find pixels that meet both the minimum valid observations
            # and minimum wet threshold criteria.
            wofs_alltime_summary_valid = wofs_alltime_summary_valid_wetness.where(
                wofs_alltime_summary_valid_wetness & wofs_alltime_summary_valid_clear_count
            )

            # Convert the raster to polygons.
            # We use a mask of '1' to only generate polygons around values of '1' (not NaNs).
            polygons_mask = wofs_alltime_summary_valid == 1

            polygons = xr_vectorize(
                wofs_alltime_summary_valid,
                mask=polygons_mask,
                crs=wofs_alltime_summary.geobox.crs,
            )

            # Combine any overlapping polygons.
            merged_polygon_geoms = shapely.ops.unary_union(polygons["geometry"])

            # Turn the combined multipolygon back into a GeoDataFrame.
            try:
                merged_polygons = gpd.GeoDataFrame(geometry=list(merged_polygon_geoms.geoms))
            except AttributeError:
                merged_polygons = gpd.GeoDataFrame(geometry=[merged_polygon_geoms])

            # We need to add the crs back onto the GeoDataFrame.
            merged_polygons.crs = wofs_alltime_summary.geobox.crs

            generated_polygons[threshold] = merged_polygons

    except Exception as error:
        _log.exception(
            f"\nTile {str(tile_id)} did not run. \n"
            "This is probably because there are no waterbodies present in this tile."
        )
        _log.exception(error)

    primary_threshold_polygons = generated_polygons[primary_threshold]
    secondary_threshold_polygons = generated_polygons[secondary_threshold]

    return primary_threshold_polygons, secondary_threshold_polygons


def get_polygons_from_tile_with_land_sea_mask_filtering(
    tile: tuple[tuple[int, int], datacube.api.grid_workflow.Tile],
    grid_workflow: datacube.api.GridWorkflow,
    dask_chunks: dict[str, int] = {"x": 3200, "y": 3200, "time": 1},
    min_valid_observations: int = 128,
    primary_threshold: float = 0.1,
    secondary_threshold: float = 0.05,
    land_sea_mask_fp: str | Path = "",
    resampling_method: str = "bilinear",
    filter_land_sea_mask: Callable = filter_hydrosheds_land_mask,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Generate water body polygons by thresholding a WOfS All Time Summary tile.
    Use a raster land/sea mask to mask out ocean pixels from the WOfS data before
    vectorizing the polygons.

    Parameters
    ----------
    tile : tuple[tuple[int,int], datacube.api.grid_workflow.Tile]
        The WOfs All Time summary Tile object for which to
        generate waterbody polygons for.
    grid_workflow: datacube.api.GridWorkflow,
        Grid Workflow used to generate the tiles and to be used to load the Tile object.
    dask_chunks : dict, optional
        dask_chunks to use to load WOfS data, by default {"x": 3200, "y": 3200, "time": 1}
    min_valid_observations : int, optional
        Threshold to use to mask out pixels based on the number of valid WOfS observations for each pixel, by default 128
    primary_threshold : float, optional
        Threshold to use to determine the location of the waterbody polygons, by default 0.1
    secondary_threshold : float, optional
        Threshold to use to determine the extent / shape of the waterbodies polygons, by default 0.05
    land_sea_mask_fp: str | Path, optional
        File path to raster to use to mask ocean pixels in WOfS data, by default ""
    resampling_method: str, optional
        Resampling method to use when loading the land sea mask raster, by default "bilinear"
    filter_land_sea_mask: Callable, optional
        Function to apply to the land sea mask xr.DataArray to generate a boolean
        mask where pixels with a value of True are land pixels and pixels with a
        value of False are ocean pixels, by default `filter_hydrosheds_land_mask`

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        A tuple containing GeoDataFrames of waterbody polygons generated from thresholding WOfS All Time Summary data
        using the primary and secondary thresholds.

    """
    # Set up the primary and secondary thresholds.
    minimum_wet_thresholds = [secondary_threshold, primary_threshold]

    # Get the tile id and tile object.
    tile_id = tile[0]
    tile_object = tile[1]

    # Generate the waterbody polygons using the primary and secondary thresholds,
    # from the tile.
    try:
        _log.info(f"Generating water body polygons for tile {tile_id}")

        # Load the data for the tile.
        wofs_alltime_summary = grid_workflow.load(tile_object, dask_chunks=dask_chunks).squeeze()

        # Load the land sea mask.
        if land_sea_mask_fp:
            land_sea_mask = rio_slurp_xarray(
                fname=land_sea_mask_fp,
                gbox=wofs_alltime_summary.geobox,
                resampling=resampling_method,
            )

            # Filter the land sea mask.
            boolean_land_sea_mask = filter_land_sea_mask(land_sea_mask)

            # Mask the WOfS All-Time Summary dataset using the boolean land sea mask.
            wofs_alltime_summary = wofs_alltime_summary.where(boolean_land_sea_mask)

        # Set the no-data values to nan.
        # Masking here is done using the frequency measurement because for multiple
        # areas NaN values are present in the frequency measurement but the
        # no data value -999 is not present in the count_clear and
        # count_wet measurements.
        # Note: it seems some pixels with NaN values in the frequency measurement
        # have a value of zero in the count_clear and/or the count_wet measurements.
        wofs_alltime_summary = wofs_alltime_summary.where(~np.isnan(wofs_alltime_summary.frequency))

        # Mask pixels not observed at least min_valid_observations times.
        wofs_alltime_summary_valid_clear_count = (
            wofs_alltime_summary.count_clear >= min_valid_observations
        )

        # Generate the polygons.
        generated_polygons = {}
        for threshold in minimum_wet_thresholds:
            # Mask any pixels whose frequency of water detection is less than the threshold.
            wofs_alltime_summary_valid_wetness = wofs_alltime_summary.frequency > threshold

            # Now find pixels that meet both the minimum valid observations
            # and minimum wet threshold criteria.
            wofs_alltime_summary_valid = wofs_alltime_summary_valid_wetness.where(
                wofs_alltime_summary_valid_wetness & wofs_alltime_summary_valid_clear_count
            )

            # Convert the raster to polygons.
            # We use a mask of '1' to only generate polygons around values of '1' (not NaNs).
            polygons_mask = wofs_alltime_summary_valid == 1

            polygons = xr_vectorize(
                wofs_alltime_summary_valid,
                mask=polygons_mask,
                crs=wofs_alltime_summary.geobox.crs,
            )

            # Combine any overlapping polygons.
            merged_polygon_geoms = shapely.ops.unary_union(polygons["geometry"])

            # Turn the combined multipolygon back into a GeoDataFrame.
            try:
                merged_polygons = gpd.GeoDataFrame(geometry=list(merged_polygon_geoms.geoms))
            except AttributeError:
                merged_polygons = gpd.GeoDataFrame(geometry=[merged_polygon_geoms])

            # We need to add the crs back onto the GeoDataFrame.
            merged_polygons.crs = wofs_alltime_summary.geobox.crs

            generated_polygons[threshold] = merged_polygons

    except Exception as error:
        _log.exception(
            f"\nTile {str(tile_id)} did not run. \n"
            "This is probably because there are no waterbodies present in this tile."
        )
        _log.exception(error)

    primary_threshold_polygons = generated_polygons[primary_threshold]
    secondary_threshold_polygons = generated_polygons[secondary_threshold]

    return primary_threshold_polygons, secondary_threshold_polygons
