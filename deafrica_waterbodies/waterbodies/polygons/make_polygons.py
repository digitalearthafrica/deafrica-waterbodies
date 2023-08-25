"""
Make waterbody polygons from the Water Observations from Space all-time
summary.

Geoscience Australia - 2021
    Claire Krause
    Matthew Alger
"""

import datacube
import math
import logging
import shapely
import geohash as gh
import numpy as np
import pandas as pd
import geopandas as gpd

from datacube.utils.geometry import Geometry
from deafrica_tools.spatial import xr_vectorize

from .filters import filter_geodataframe_by_intersection, filter_waterbodies


_log = logging.getLogger(__name__)


def get_product_regions(product: str):
    """
    Returns a GeoDataFrame of all the tiles/regions of the DE Africa product.

    Parameters
    ----------
    product: str
        DE Africa product to get regions for.

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame

    """
    base_url = "https://explorer.digitalearth.africa/api/regions/"
    regions_url = f"{base_url}{product}"

    try:
        regions = gpd.read_file(regions_url).drop('count', axis=1)
        regions.set_index("region_code", inplace=True)
        return regions
    except Exception as error:
        _log.error(error)
        raise


def get_product_tiles(product="wofs_ls_summary_alltime", aoi_gdf=None):
    """
    Returns the regions/tiles of the DE Africa product that intersect with
    the area of interest GeoDataFrame.

    Parameters
    ----------
    product : str, optional
        Digital Earth Africa product, by default "wofs_ls_summary_alltime"
    aoi_gdf : geopandas.geodataframe.GeoDataFrame, optional
        GeoDataFrame of the area of interest, by default None

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        Regions/tiles of the DE Africa product that intersect with
        the area of interest GeoDataFrame.
    """

    # Load the product regions.
    regions = get_product_regions(product=product)

    if aoi_gdf is None:
        _log.info(f"Getting all {product} regions...")
        tiles = regions
    else:
        # Reproject the regions to match the area of interest.
        crs = aoi_gdf.crs
        regions = regions.to_crs(crs)

        tiles, _ = filter_geodataframe_by_intersection(regions,
                                                       aoi_gdf,
                                                       filtertype="intersects",
                                                       invert_mask=False,
                                                       return_inverse=False)
    _log.info(f"{len(tiles)} {product} tiles found.")
    return tiles


def check_wetness_thresholds(minimum_wet_thresholds):
    """
    Function to validate the wetness thresholds.
    """
    # Test whether the wetness threshold has been correctly set.

    if minimum_wet_thresholds[0] > minimum_wet_thresholds[-1]:
        error_msg = 'We will be running a hybrid wetness threshold. ' \
            'Please ensure that the primary threshold has a higher value than the ' \
            'secondary threshold. \n'
        raise ValueError(error_msg)
    else:
        print_msg = 'We will be running a hybrid wetness threshold. \n' \
            f'**You have set {minimum_wet_thresholds[-1]} as the ' \
            'primary threshold, which will define the location of the waterbody ' \
            f'polygons \n with {minimum_wet_thresholds[0]} set as the supplementary ' \
            'threshold, which will define the extent/shape of the waterbody polygons.**'
        return (print_msg)


def get_polygons_using_thresholds(
        input_gdf: gpd.geodataframe.GeoDataFrame,
        dask_chunks: dict[str, int] = {"x": 3000, "y": 3000, "time": 1},
        resolution: tuple[int, int] = (-30, 30),
        output_crs: str = "EPSG:6933",
        min_valid_observations: int = 128,
        primary_threshold: float = 0.1,
        secondary_threshold: float = 0.05,
                            ):
    """
    Generate polygons by thresholding WOfS All Time Summary data.

    Parameters
    ----------
    input_gdf : gpd.geodataframe.GeoDataFrame
        Area of interest GeoDataFrame
    dask_chunks : dict, optional
        dask_chunks to use to load WOfS data, by default {"x": 3000, "y": 3000, "time": 1}
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

    Returns
    -------
    primary_threshold_polygons, secondary_threshold_polygons
        Polygons generated from thresholding WOfS All Time Summary data
        using the primary and secondary thresholds.

    """
    # Check if the wetness thresholds have been set correctly.
    minimum_wet_thresholds = [secondary_threshold, primary_threshold]
    _log.info(check_wetness_thresholds(minimum_wet_thresholds))

    # Create a datacube query object.
    query = dict(
        dask_chunks=dask_chunks,
        resolution=resolution,
        output_crs=output_crs,
    )

    # Reproject the input_gdf.
    input_gdf = input_gdf.to_crs(output_crs)

    # Connect to the datacube.
    dc = datacube.Datacube(app='WaterbodiesPolygons')

    primary_threshold_polygons_list = []
    secondary_threshold_polygons_list = []
    for row in input_gdf.itertuples():
        row_id = row.Index
        _log.info(f"Generating polygons for tile: {row_id}")
        row_geom = Geometry(geom=row.geometry, crs=input_gdf.crs)

        # Update the datacube query with the row geometry.
        query.update(geopolygon=row_geom)

        try:
            # Load the WOfS All-Time Summary of clear and wet observations.
            wofs_alltime_summary = dc.load('wofs_ls_summary_alltime',
                                           **query).squeeze()

            # Set the no-data values to nan.
            # Masking here is done using the frequency measurement because for multiple
            # areas NaN values are present in the frequency measurement but the
            # no data value -999 is not present in the count_clear and
            # count_wet measurements.
            # Note: it seems some pixels with NaN values in the frequency measurement
            # have a value of zero in the count_clear and/or the count_wet measurements.
            wofs_alltime_summary = wofs_alltime_summary.where(~np.isnan(wofs_alltime_summary.frequency))

            # Mask pixels not observed at least min_valid_observations times.
            wofs_alltime_summary_valid_clear_count = wofs_alltime_summary.count_clear >= min_valid_observations

            # Generate the polygons.
            row_polygons = {}
            for threshold in minimum_wet_thresholds:
                # Mask any pixels whose frequency of water detection is less than the threshold.
                wofs_alltime_summary_valid_wetness = wofs_alltime_summary.frequency > threshold

                # Now find pixels that meet both the minimum valid observations
                # and minimum wet threshold criteria.
                wofs_alltime_summary_valid = wofs_alltime_summary_valid_wetness.where(wofs_alltime_summary_valid_wetness & wofs_alltime_summary_valid_clear_count)

                # Convert the raster to polygons.
                # We use a mask of '1' to only generate polygons around values of '1' (not NaNs).
                polygons_mask = wofs_alltime_summary_valid == 1

                polygons = xr_vectorize(wofs_alltime_summary_valid,
                                        mask=polygons_mask,
                                        crs=wofs_alltime_summary.geobox.crs,
                                        transform=wofs_alltime_summary.geobox.transform)

                # Combine any overlapping polygons.
                merged_polygon_geoms = shapely.ops.unary_union(polygons['geometry'])

                # Turn the combined multipolygon back into a GeoDataFrame.
                try:
                    merged_polygons = gpd.GeoDataFrame(geometry=list(merged_polygon_geoms.geoms))
                except AttributeError:
                    merged_polygons = gpd.GeoDataFrame(geometry=[merged_polygon_geoms])

                # We need to add the crs back onto the GeoDataFrame.
                merged_polygons.crs = wofs_alltime_summary.geobox.crs

                row_polygons[threshold] = merged_polygons

            # Append the row's waterbody polygons to the list.
            primary_threshold_polygons_list.append(row_polygons[primary_threshold])
            secondary_threshold_polygons_list.append(row_polygons[secondary_threshold])
        except Exception as error:
            _log.error(error)
            _log.error(
                f'\nTile {row_id} did not run. \n'
                'This is probably because there are no waterbodies present in this tile.'
            )
    primary_threshold_polygons = pd.concat(primary_threshold_polygons_list)
    secondary_threshold_polygons = pd.concat(secondary_threshold_polygons_list)

    return primary_threshold_polygons, secondary_threshold_polygons


def merge_polygons_at_tile_boundary(input_polygons, tiles):
    """ 
    Function to merge waterbody polygons located at tile boundaries.
    """

    assert input_polygons.crs == tiles.crs

    # Add a 1 pixel (30 m) buffer to the tiles boundary.
    buffered_30m_tiles = tiles.boundary.buffer(30, cap_style="flat", join_style="mitre")
    buffered_30m_tiles_gdf = gpd.GeoDataFrame(geometry=buffered_30m_tiles, crs=tiles.crs)

    # Get the polygons at the tile boundaries.
    boundary_polygons, _, not_boundary_polygons = filter_geodataframe_by_intersection(
        input_polygons,
        buffered_30m_tiles_gdf,
        invert_mask=False,
        return_inverse=True)

    # Now combine overlapping polygons in boundary_polygons.
    merged_boundary_polygons_geoms = shapely.ops.unary_union(boundary_polygons['geometry'])

    # `Explode` the multipolygon back out into individual polygons.
    merged_boundary_polygons = gpd.GeoDataFrame(crs=input_polygons.crs, geometry=[merged_boundary_polygons_geoms])
    merged_boundary_polygons = merged_boundary_polygons.explode(index_parts=True).reset_index(drop=True)

    # Then combine our merged_boundary_polygons with the not_boundary_polygons.
    all_polygons = gpd.GeoDataFrame(pd.concat([not_boundary_polygons, merged_boundary_polygons], ignore_index=True, sort=True)).set_geometry('geometry')

    return all_polygons


def assign_unique_ids(polygons):
    """ 
    Function to assign a unique ID to each waterbody polygon. 
    """

    crs = polygons.crs

    # Generate a unique id for each polygon.
    polygons_with_unique_ids = polygons.to_crs(epsg=4326)
    polygons_with_unique_ids['UID'] = polygons_with_unique_ids.apply(lambda x: gh.encode(x.geometry.centroid.y, x.geometry.centroid.x, precision=9), axis=1)

    # Check that our unique ID is in fact unique
    assert polygons_with_unique_ids['UID'].is_unique

    # Make an arbitrary numerical ID for each polygon. We will first sort the dataframe by geohash
    # so that polygons close to each other are numbered similarly.
    polygons_with_unique_ids_sorted = polygons_with_unique_ids.sort_values(by=['UID']).reset_index()
    polygons_with_unique_ids_sorted['WB_ID'] = polygons_with_unique_ids_sorted.index

    # The step above creates an 'index' column, which we don't actually want, so drop it.
    polygons_with_unique_ids_sorted.drop(columns=['index'], inplace=True)

    # Reproject to the same crs as the input polygons.
    polygons_with_unique_ids_sorted = polygons_with_unique_ids_sorted.to_crs(crs)

    return polygons_with_unique_ids_sorted


def get_waterbodies(
        aoi_gdf: gpd.geodataframe.GeoDataFrame,
        continental_run: bool = False,
        dask_chunks: dict[str, int] = {"x": 3000, "y": 3000, "time": 1},
        resolution: tuple[int, int] = (-30, 30),
        output_crs: str = "EPSG:6933",
        min_valid_observations: int = 128,
        primary_threshold: float = 0.1,
        secondary_threshold: float = 0.05,
        min_polygon_size: float = 4500,
        max_polygon_size: float = math.inf,
        filter_out_ocean_polygons: bool = False,
        land_sea_mask_fp: str = None,
        filter_out_major_rivers_polygons: bool = False,
        major_rivers_mask_fp: str = None,
        filter_out_urban_polygons: bool = False,
        urban_mask_fp: str = None,
        handle_large_polygons: str = "nothing",
        pp_test_threshold: float = 0.005,
):
    # Check if this is a continental run.
    if aoi_gdf is None and continental_run:
        _log.info("Running for all the WOfS All Time Summary tiles covering Africa...")
    elif aoi_gdf is not None and continental_run:
        raise ValueError("If setting an area of interest, set `continental_run=False`")
    elif aoi_gdf is not None and not continental_run:
        _log.info("Running for the WOfS All Time Summary tiles covering the defined area of interest...")

    # Get the tiles covering the area of interest.
    _log.info("Loading tiles..")
    aoi_gdf = aoi_gdf.to_crs(output_crs)
    tiles = get_product_tiles(product="wofs_ls_summary_alltime", aoi_gdf=aoi_gdf)

    _log.info("Generating the first temporary set of waterbody polygons.")
    temp_primary, temp_secondary = get_polygons_using_thresholds(input_gdf=tiles,
                                                                 dask_chunks=dask_chunks,
                                                                 resolution=resolution,
                                                                 output_crs=output_crs,
                                                                 min_valid_observations=min_valid_observations,
                                                                 primary_threshold=primary_threshold,
                                                                 secondary_threshold=secondary_threshold,
                                                                 )

    _log.info("Merging polygons at tile boundaries...")
    merged_temp_primary = merge_polygons_at_tile_boundary(input_polygons=temp_primary, tiles=tiles)
    merged_temp_secondary = merge_polygons_at_tile_boundary(input_polygons=temp_secondary, tiles=tiles)

    _log.info("Filtering waterbodies...")
    filtered_polygons = filter_waterbodies(
        primary_threshold_polygons=merged_temp_primary,
        secondary_threshold_polygons=merged_temp_secondary,
        min_polygon_size=min_polygon_size,
        max_polygon_size=max_polygon_size,
        filter_out_ocean_polygons=filter_out_ocean_polygons,
        land_sea_mask_fp=land_sea_mask_fp,
        filter_out_major_rivers_polygons=filter_out_major_rivers_polygons,
        major_rivers_mask_fp=major_rivers_mask_fp,
        filter_out_urban_polygons=filter_out_urban_polygons,
        urban_mask_fp=urban_mask_fp,
        handle_large_polygons=handle_large_polygons,
        pp_test_threshold=pp_test_threshold,
    )

    _log.info("Assigning unique ids to each polygon....")
    filtered_polygons_with_unique_ids = assign_unique_ids(filtered_polygons)

    return filtered_polygons_with_unique_ids
