"""
Make waterbody polygons from the Water Observations from Space all-time
summary.

Geoscience Australia - 2021
    Claire Krause
    Matthew Alger
"""

import logging
from pathlib import Path
from types import ModuleType

import datacube
import datacube.model
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import shapely
import xarray as xr
from deafrica_tools.spatial import xr_vectorize
from skimage import measure, morphology
from skimage.segmentation import watershed

from deafrica_waterbodies.filters import filter_by_intersection

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
        Threshold used to set the location of the waterbody polygons.
    extent_threshold : int | float
        Threshold used to set the shape/extent of the waterbody polygons.

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
        You have set {detection_threshold} as the location threshold, which will define the location of the waterbody polygons.
        You have set {extent_threshold} as the extent threshold, which will define the extent/shape of the waterbody polygons."""
        )
        return [extent_threshold, detection_threshold]


def merge_polygons_at_tile_boundaries(
    waterbody_polygons: gpd.GeoDataFrame, tile_extents_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Function to merge waterbody polygons located at WOfS All Time Summary tile boundaries.

    Parameters
    ----------
    waterbody_polygons : gpd.GeoDataFrame
        The waterbody polygons.
    tile_extents_gdf: gpd.GeoDataFrame
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


def load_wofs_frequency(
    tile: tuple[tuple[int, int], datacube.api.grid_workflow.Tile],
    grid_workflow: datacube.api.GridWorkflow,
    plugin: ModuleType,
    dask_chunks: dict[str, int] = {
        "x": 3200,
        "y": 3200,
        "time": 1,
    },  # based on Landsat WOfS scene size
    min_valid_observations: int = 60,
    min_wet_thresholds: list[int | float] = [0.05, 0.1],
    land_sea_mask_fp: str | Path = "",
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Load the WOfS All-Time Summary frequency measurement for a tile and threshold the data
    using the extent and the detection thresholds.

    Parameters
    ----------
    tile : tuple[tuple[int,int], datacube.api.grid_workflow.Tile]
        The WOfS All Time Summary Tile object for which to
        generate waterbody polygons for.
    grid_workflow: datacube.api.GridWorkflow,
        Grid Workflow used to generate the tiles and to be used to load the Tile object.
    plugin: ModuleType
        A validated plugin to load filtering masks with.
    dask_chunks : dict, optional
        dask_chunks to use to load WOfS data, by default {"x": 3200, "y": 3200, "time": 1}
    min_valid_observations : int, optional
        Threshold to use to mask out pixels based on the number of valid WOfS
        observations for each pixel, by default 60
    min_wet_thresholds: list[int | float], optional
        A list containing the extent threshold and the detection threshold, with
        the extent threshold listed first, by default [0.05, 0.1]
    land_sea_mask_fp: str | Path, optional
        File path to raster to use to mask ocean pixels in WOfS data, by default ""

    Returns
    -------
    tuple[valid_detection: xr.DataArray, valid_extent: xr.DataArray]
        WOfS All Time Summary frequency measurement thresholded using the detection and extent
        thresholds.

    """
    if land_sea_mask_fp:
        assert plugin is not None

    # Set up the detection and extent thresholds.
    extent_threshold = min_wet_thresholds[0]
    detection_threshold = min_wet_thresholds[-1]

    # Get the tile id and tile object.
    tile_id = tile[0]  # noqa F841
    tile_object = tile[1]

    # Load the WOfS All-Time Summary data for the tile and threshold the data
    # using the extent and the detection thresholds.
    try:
        # Load the data for the tile.
        wofs_alltime_summary = grid_workflow.load(tile_object, dask_chunks=dask_chunks).squeeze()

        # Load the land sea mask.
        if land_sea_mask_fp:
            boolean_land_sea_mask = plugin.load_land_sea_mask(
                land_sea_mask_fp=land_sea_mask_fp, wofs_alltime_summary_ds=wofs_alltime_summary
            )

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

        # Threshold using the detection threshold.
        detection = wofs_alltime_summary.frequency > detection_threshold
        valid_detection = (detection > 0) & wofs_alltime_summary_valid_clear_count

        # Threshold the using the extent threshold.
        extent = wofs_alltime_summary.frequency > extent_threshold
        valid_extent = (extent > 0) & wofs_alltime_summary_valid_clear_count

    except Exception as error:
        _log.exception(error)
        raise error
    else:
        return valid_detection, valid_extent


def remove_small_waterbodies(waterbody_raster: np.ndarray, min_size: int = 6) -> np.ndarray:
    """
    Remove water bodies from the raster that are smaller than the specified number of pixels.

    Parameters
    ----------
    waterbody_raster : np.ndarray
        Raster image to filter.
    min_size : int, optional
        The smallest allowable waterbody size i.e. minimum number of pixels a waterbody must have, by default 6

    Returns
    -------
    np.ndarray
        Raster image with small waterbodies removed.
    """

    waterbodies_labelled = morphology.label(waterbody_raster, background=0)
    waterbodies_small_removed = morphology.remove_small_objects(
        waterbodies_labelled, min_size=min_size, connectivity=1
    )

    return waterbodies_small_removed


def select_waterbodies_for_segmentation(
    waterbodies_labelled: np.ndarray, min_size: int = 1000
) -> np.ndarray:
    """
    Select waterbodies larger than the specified number of pixels for segmentation.

    Parameters
    ----------
    waterbodies_labelled : np.ndarray
        Raster image to filter.
    min_size : int, optional
        Minimum number of pixels to classify a waterbody as large, by default 1000

    Returns
    -------
    np.ndarray
        Raster containing the large waterbodies to be segmented.
    """

    props = measure.regionprops(waterbodies_labelled)

    labels_to_keep = []
    for region_prop in props:
        count = region_prop.num_pixels
        label = region_prop.label

        if count > min_size:
            labels_to_keep.append(label)

    segment_image = np.where(np.isin(waterbodies_labelled, labels_to_keep), 1, 0)

    return segment_image


def generate_segmentation_markers(
    marker_source: np.ndarray, erosion_radius: int = 1, min_size: int = 100
) -> np.ndarray:
    """
    Create watershed segmentation markers.

    Parameters
    ----------
    marker_source : np.ndarray
        Raster image to generate watershed segmentation markers from.
    erosion_radius : int, optional
        Radius to use to generate footprint for erosion., by default 1
    min_size : int, optional
        The smallest allowable object size, by default 100

    Returns
    -------
    np.ndarray
        Watershed segmentation markers.
    """
    markers = morphology.erosion(marker_source, footprint=morphology.disk(radius=erosion_radius))
    markers_relabelled = morphology.label(markers, background=0)

    markers_acceptable_size = morphology.remove_small_objects(
        markers_relabelled, min_size=min_size, connectivity=1
    )

    return markers_acceptable_size


def run_watershed(
    waterbodies_for_segmentation: np.ndarray, segmentation_markers: np.ndarray
) -> np.ndarray:
    """
    Segment large waterbodies.

    Parameters
    ----------
    waterbodies_for_segmentation : np.ndarray
        Raster image containing the large waterbodies to be segmented.
    segmentation_markers : np.ndarray
        Raster image containing the watershed segmentation markers.

    Returns
    -------
    np.ndarray
        Raster image with the large waterbodies segmented.
    """
    distance = ndi.distance_transform_edt(waterbodies_for_segmentation)
    segmented = watershed(-distance, segmentation_markers, mask=waterbodies_for_segmentation)

    return segmented


def confirm_extent_contains_detection(extent: np.ndarray, detection: np.ndarray) -> np.ndarray:
    """
    Filter the waterbodies in the extent raster to keep only waterbodies that contain a waterbody pixel from the
    detection raster.

    Parameters
    ----------
    extent : np.ndarray
        Raster of the extent of the waterbodies.
    detection : np.ndarray
        Raster of the location of the waterbodies.

    Returns
    -------
    np.ndarray
        Filtered waterbodies in the extent raster.
    """

    def sum_intensity(regionmask, intensity_image):
        return np.sum(intensity_image[regionmask])

    props = measure.regionprops(
        extent, intensity_image=detection, extra_properties=(sum_intensity,)
    )

    labels_to_keep = []
    for region_prop in props:
        detection_count = region_prop.sum_intensity
        label = region_prop.label

        if detection_count > 0:
            labels_to_keep.append(label)

    extent_keep = np.where(np.isin(extent, labels_to_keep), extent, 0)

    return extent_keep


def process_raster_polygons(
    tile: tuple[tuple[int, int], datacube.api.grid_workflow.Tile],
    grid_workflow: datacube.api.GridWorkflow,
    plugin: ModuleType,
    dask_chunks: dict[str, int] = {
        "x": 3200,
        "y": 3200,
        "time": 1,
    },  # based on Landsat WOfS scene size
    min_valid_observations: int = 60,
    min_wet_thresholds: list[int | float] = [0.05, 0.1],
    land_sea_mask_fp: str | Path = "",
) -> gpd.GeoDataFrame:
    """
    Generate waterbody polygons by thresholding a WOfS All Time Summary tile.

    Parameters
    ----------
    tile : tuple[tuple[int,int], datacube.api.grid_workflow.Tile]
        The WOfS All Time Summary Tile object for which to
        generate waterbody polygons for.
    grid_workflow: datacube.api.GridWorkflow,
        Grid Workflow used to generate the tiles and to be used to load the Tile object.
    plugin: ModuleType
        A validated plugin to load masks with.
    dask_chunks : dict, optional
        dask_chunks to use to load WOfS data, by default {"x": 3200, "y": 3200, "time": 1}
    min_valid_observations : int, optional
        Threshold to use to mask out pixels based on the number of valid WOfS
        observations for each pixel, by default 60
    min_wet_thresholds: list[int | float], optional
        A list containing the extent threshold and the detection threshold, with
        the extent threshold listed first, by default [0.05, 0.1]
    land_sea_mask_fp: str | Path, optional
        File path to raster to use to mask ocean pixels in WOfS data, by default ""
    Returns
    -------
    gpd.GeoDataFrame
        waterbody polygons.
    """
    # Load and threshold the WOfS All Time Summary tile.
    xr_detection, xr_extent = load_wofs_frequency(
        tile=tile,
        grid_workflow=grid_workflow,
        plugin=plugin,
        dask_chunks=dask_chunks,
        min_valid_observations=min_valid_observations,
        min_wet_thresholds=min_wet_thresholds,
        land_sea_mask_fp=land_sea_mask_fp,
    )

    # Get the crs.
    try:
        output_crs = xr_detection.geobox.crs
    except Exception as error:
        _log.exception(error)
        output_crs = xr_extent.geobox.crs

    # Convert to numpy arrays for image processing.
    np_detection = xr_detection.to_numpy().astype(int)
    np_extent = xr_extent.to_numpy().astype(int)

    # Remove any objects of size 5 or less, as measured by connectivity=1
    np_extent_small_removed = remove_small_waterbodies(np_extent, min_size=6)

    # Identify waterbodies to apply segmentation to
    np_extent_segment = select_waterbodies_for_segmentation(np_extent_small_removed, min_size=1000)
    np_extent_nosegment = np.where(np_extent_segment > 0, 0, np_extent_small_removed)

    # Create watershed segmentation markers by taking the detection threshold pixels and eroding them by 1
    # Includes removal of any markers smaller than 100 pixels
    segmentation_markers = generate_segmentation_markers(
        np_detection, erosion_radius=1, min_size=100
    )

    # Run segmentation
    np_segmented_extent = run_watershed(np_extent_segment, segmentation_markers)

    # Combine segmented and non segmented back together
    np_combined_extent = np.where(np_segmented_extent > 0, np_segmented_extent, np_extent_nosegment)

    # Only keep extent areas that contain a detection pixel
    np_combined_extent_contains_detection = confirm_extent_contains_detection(
        np_combined_extent, np_detection
    )

    # Relabel and remove small objects
    np_combined_clean_label = remove_small_waterbodies(
        np_combined_extent_contains_detection, min_size=6
    )

    # Convert back to xarray
    xr_combined_extent = xr.DataArray(
        np_combined_clean_label, coords=xr_extent.coords, dims=xr_extent.dims, attrs=xr_extent.attrs
    )

    # Vectorize
    vector_combined_extent = xr_vectorize(
        xr_combined_extent, crs=output_crs, mask=xr_combined_extent.values > 0
    )

    return vector_combined_extent
